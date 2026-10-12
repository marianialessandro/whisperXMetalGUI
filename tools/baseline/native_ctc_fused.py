"""Trial-only deferred wildcard preparation; wrapper tensors carry metadata and fail if numerical data is requested before native CTC."""

import contextlib
import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode


class DeferredWildcard(torch.Tensor):
    @staticmethod
    def __new__(cls, base, blank, kind):
        frames, vocabulary = base.shape
        shapes = {"masked": (frames, vocabulary - 1), "value": (frames,), "indices": (frames,), "column": (frames, 1), "expanded": (frames, vocabulary + 1)}
        return torch.Tensor._make_wrapper_subclass(cls, shapes[kind], dtype=torch.int64 if kind == "indices" else torch.float32, device="cpu", requires_grad=False)

    def __init__(self, base, blank, kind):
        self.base_emission, self.blank, self.kind = base, blank, kind

    def __repr__(self):
        return f"DeferredWildcard(kind={self.kind}, shape={tuple(self.shape)})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func == torch.ops.aten.detach.default and isinstance(args[0], cls):
            item = args[0]
            return cls(item.base_emission, item.blank, item.kind)
        if func == torch.ops.aten.max.dim:
            item = args[0]
            dimension = args[1] if len(args) > 1 else kwargs.get("dim")
            keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
            if item.kind == "masked" and dimension == 1 and not keepdim:
                return cls(item.base_emission, item.blank, "value"), cls(item.base_emission, item.blank, "indices")
        if func == torch.ops.aten.unsqueeze.default:
            item, dimension = args[:2]
            if item.kind == "value" and dimension == 1: return cls(item.base_emission, item.blank, "column")
        if func == torch.ops.aten.cat.default:
            tensors = args[0]
            dimension = args[1] if len(args) > 1 else kwargs.get("dim", 0)
            if len(tensors) == 2 and dimension == 1 and isinstance(tensors[1], cls) and tensors[1].kind == "column":
                base, column = tensors
                original = column.base_emission
                if not isinstance(base, cls) and base.shape == original.shape and base.stride() == original.stride() and base.data_ptr() == original.data_ptr():
                    return cls(base, column.blank, "expanded")
                raise ValueError("fused_wildcard_base_identity_mismatch")
        raise ValueError("fused_wildcard_tensor_operation_unsupported")


@contextlib.contextmanager
def fused_wildcard_preparation(vocabulary):
    class PrepareWildcard(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func == torch.ops.aten.index.Tensor:
                base, selectors = args[:2]
                if not isinstance(base, DeferredWildcard) and base.device.type == "cpu" and base.dtype == torch.float32 and base.ndim == 2 and not base.requires_grad and len(selectors) == 2 and selectors[0] is None:
                    mask = selectors[1]
                    if mask is not None and mask.device.type == "cpu" and mask.dtype == torch.bool and mask.ndim == 1 and len(mask) == base.shape[1]:
                        if base.shape[1] != vocabulary: raise ValueError("fused_wildcard_vocabulary_mismatch")
                        excluded = np.flatnonzero(~mask.detach().numpy())
                        if len(excluded) != 1: raise ValueError("fused_wildcard_mask_unsupported")
                        return DeferredWildcard(base, int(excluded[0]), "masked")
            return func(*args, **kwargs)
    with PrepareWildcard():
        yield
