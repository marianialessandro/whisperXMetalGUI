"""Scoped pinned MPS emission handoff to development native CTC; unsupported transport fails explicitly."""

import contextlib
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tools.baseline.native_ctc_fused import DeferredWildcard


@contextlib.contextmanager
def native_mps_ctc(alignment, context, records):
    if torch.__version__ != "2.8.0" or not isinstance(records, list): raise ValueError("native_mps_scope_runtime_or_records_invalid")
    original_trellis, original_backtrack = alignment.get_trellis, alignment.backtrack
    class EmissionHandoff(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.log_scores, self.emission = None, None

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func in (torch.ops.aten._log_softmax.default, torch.ops.aten.log_softmax.int) and args[0].device.type == "mps" and args[0].dtype == torch.float32 and args[0].ndim == 3 and args[1] in (-1, 2):
                result = func(*args, **kwargs)
                if result.shape[0] != 1 or result.dtype != torch.float32: raise ValueError("native_mps_logsoftmax_batch_or_dtype_not_supported")
                self.log_scores, self.emission = result, None
                return result
            if func == torch.ops.aten.select.int and args[0] is self.log_scores and tuple(args[1:]) == (0, 0) and not kwargs:
                self.emission = func(*args, **kwargs)
                return self.emission
            if func in (torch.ops.aten._to_copy.default, torch.ops.aten.to.dtype_layout) and args[0] is self.emission and self.emission is not None and kwargs.get("device") == torch.device("cpu"):
                if any(key not in ("device", "dtype", "layout", "non_blocking", "memory_format", "pin_memory", "copy") for key in kwargs) or kwargs.get("dtype", torch.float32) not in (None, torch.float32) or kwargs.get("layout", torch.strided) not in (None, torch.strided) or kwargs.get("non_blocking", False) or kwargs.get("pin_memory", False) or kwargs.get("copy", False) or kwargs.get("memory_format", torch.preserve_format) not in (None, torch.preserve_format): raise ValueError("native_mps_transport_options_not_supported")
                emission = self.emission
                self.log_scores, self.emission = None, None
                return DeferredWildcard(emission, 0, "base")
            return func(*args, **kwargs)

    class Trellis:
        def __init__(self, path, frames, count): self.path, self.shape = path, (frames + 1, count + 1)

        def size(self, dimension): return self.shape[dimension]

    def trellis(emission, tokens, blank_id=0):
        if not isinstance(emission, DeferredWildcard) or emission.kind not in ("base", "expanded") or emission.base_emission.device.type != "mps": raise ValueError("native_mps_emission_handoff_missing")
        if emission.kind == "expanded" and emission.blank != blank_id: raise ValueError("native_mps_blank_metadata_changed")
        path, record = context.compute(emission.base_emission, tokens, blank_id, wildcard=emission.kind == "expanded")
        records.append(record)
        return Trellis(path, emission.shape[0], len(tokens))

    def backtrack(trellis, emission, tokens, blank_id=0):
        if not isinstance(trellis, Trellis): raise ValueError("native_mps_trellis_not_supported")
        return None if trellis.path is None else [alignment.Point(token, frame, score) for token, frame, score in trellis.path]

    alignment.get_trellis, alignment.backtrack = trellis, backtrack
    try:
        with EmissionHandoff(): yield
    finally:
        alignment.get_trellis, alignment.backtrack = original_trellis, original_backtrack
