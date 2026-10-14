"""Record tensor device metadata around unchanged ATen calls, never numerical values or credentials."""

from collections import Counter
import torch
from torch.utils._python_dispatch import TorchDispatchMode


def devices(value):
    if isinstance(value, torch.Tensor): return {value.device.type}
    if isinstance(value, dict): return set().union(*(devices(item) for item in value.values())) if value else set()
    if isinstance(value, (tuple, list)): return set().union(*(devices(item) for item in value)) if value else set()
    return set()


class DispatchAudit(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.operations = Counter()
        self.device_operations = Counter()
        self.scalar_readbacks = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        before = devices((args, kwargs))
        result = func(*args, **kwargs)
        after = devices(result)
        name = str(func)
        self.operations[name] += 1
        for device in before | after: self.device_operations[device] += 1
        if "_local_scalar_dense" in name: self.scalar_readbacks += 1
        return result

    def report(self):
        return {"aten_operations": dict(sorted(self.operations.items())), "tensor_device_operations": dict(sorted(self.device_operations.items())), "scalar_readbacks": self.scalar_readbacks, "numerical_values_retained": False, "internal_backend_cpu_work_certified_absent": False}
