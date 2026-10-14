"""Observe ATen copy/scalar metadata without reading or retaining tensor values."""

from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
import functools

import torch
from torch.utils._python_dispatch import TorchDispatchMode


class TransferAudit(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.stage = ContextVar("transfer_stage", default="outside_observed_phases")
        self.records = defaultdict(lambda: {"calls": 0, "source_tensor_logical_bytes": 0, "destination_tensor_logical_bytes": 0})
        self.failed_dispatches = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        try:
            result = func(*args, **(kwargs or {}))
        except BaseException:
            self.failed_dispatches += 1
            raise
        source = None
        destination = None
        if name == "aten._to_copy.default" and args and isinstance(args[0], torch.Tensor) and isinstance(result, torch.Tensor):
            source, destination = args[0], result
        elif name == "aten.copy_.default" and len(args) >= 2 and all(isinstance(item, torch.Tensor) for item in args[:2]):
            destination, source = args[:2]
        elif name == "aten._local_scalar_dense.default" and args and isinstance(args[0], torch.Tensor):
            source = args[0]
        if source is not None:
            target = destination.device.type if destination is not None else "host_scalar"
            dtype = str(source.dtype)
            output_dtype = str(destination.dtype) if destination is not None else dtype
            key = (self.stage.get(), name, source.device.type, target, dtype, output_dtype)
            row = self.records[key]
            row["calls"] += 1
            row["source_tensor_logical_bytes"] += source.numel() * source.element_size()
            if destination is not None: row["destination_tensor_logical_bytes"] += destination.numel() * destination.element_size()
        return result

    def report(self):
        rows = [{"stage": key[0], "operation": key[1], "source_device": key[2], "destination_device": key[3], "source_dtype": key[4], "destination_dtype": key[5], **value} for key, value in sorted(self.records.items())]
        return {"scope": "python_thread_aten_copy_and_scalar_boundaries", "records": rows, "failed_dispatches": self.failed_dispatches, "physical_copy_bytes": None, "numerical_values_retained": False, "compiled_backend_internal_transfers_complete": False, "other_threads_observed": False, "mlx_transfers_observed": False, "gpu_dispatch_certified": False, "benchmark_timing_eligible": False}


@contextmanager
def phases(adapter, audit):
    original_boundaries = adapter.boundaries
    originals = []
    def boundaries(request):
        targets = original_boundaries(request)
        for owner, attribute, stage in targets:
            original = getattr(owner, attribute)
            originals.append((owner, attribute, original))
            @functools.wraps(original)
            def observed(*args, __call=original, __stage=stage, **kwargs):
                token = audit.stage.set(__stage)
                try:
                    return __call(*args, **kwargs)
                finally:
                    audit.stage.reset(token)
            setattr(owner, attribute, observed)
        return targets
    adapter.boundaries = boundaries
    try:
        yield
    finally:
        adapter.boundaries = original_boundaries
        for owner, attribute, original in reversed(originals): setattr(owner, attribute, original)
