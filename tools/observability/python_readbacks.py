"""Optional original Tensor method observations, kept separate from ATen counters."""

from collections import defaultdict
from contextlib import contextmanager
import functools
import threading

import torch

from tools.baseline import corpus


class PythonReadbacks:
    def __init__(self, stage):
        self.stage = stage
        self.owner = threading.get_ident()
        self.rows = defaultdict(lambda: {"calls": 0, "source_tensor_logical_bytes": 0, "destination_logical_bytes": 0, "identity_returns": 0})
        self.failed_calls = 0

    @contextmanager
    def observe(self):
        originals = []
        try:
            for method in ("cpu", "numpy"):
                original = getattr(torch.Tensor, method)
                originals.append((method, original))
                @functools.wraps(original)
                def observed(source, *args, __call=original, __method=method, **kwargs):
                    if threading.get_ident() != self.owner: return __call(source, *args, **kwargs)
                    try:
                        result = __call(source, *args, **kwargs)
                    except BaseException:
                        self.failed_calls += 1
                        raise
                    target = "cpu_tensor" if __method == "cpu" else "numpy_array"
                    key = (self.stage.get(), __method, source.device.type, target, str(source.dtype), str(result.dtype))
                    row = self.rows[key]
                    row["calls"] += 1
                    row["source_tensor_logical_bytes"] += source.numel() * source.element_size()
                    row["destination_logical_bytes"] += result.numel() * result.element_size() if __method == "cpu" else result.nbytes
                    row["identity_returns"] += result is source
                    return result
                setattr(torch.Tensor, method, observed)
            yield
        finally:
            for method, original in reversed(originals): setattr(torch.Tensor, method, original)

    def report(self):
        return {"scope": "python_tensor_methods_current_thread", "failed_calls": self.failed_calls, "records": [{"stage": key[0], "method": key[1], "source_device": key[2], "destination_kind": key[3], "source_dtype": key[4], "destination_dtype": key[5], **value} for key, value in sorted(self.rows.items())], "physical_copy_bytes": None, "numerical_values_retained": False, "compiled_backend_internal_calls_complete": False, "other_threads_observed": False, "aten_counters_additive": False}


def summarize(report, spans):
    corpus.require(report.get("scope") == "python_tensor_methods_current_thread" and report.get("physical_copy_bytes") is None and report.get("numerical_values_retained") is False and report.get("compiled_backend_internal_calls_complete") is False and report.get("other_threads_observed") is False and report.get("aten_counters_additive") is False, "python_readback_scope_invalid")
    corpus.require(type(report.get("failed_calls")) is int and report["failed_calls"] >= 0 and isinstance(report.get("records"), list), "python_readback_counters_invalid")
    stages = {row["stage"] for row in spans["spans"]} | {"outside_observed_phases"}
    counts = {"cpu_method_device_to_host_calls": 0, "cpu_method_source_logical_bytes": 0, "numpy_method_device_to_host_calls": 0, "numpy_method_source_logical_bytes": 0, "cpu_method_same_object_calls": 0, "numpy_method_calls": 0}
    seen = set()
    for row in report["records"]:
        corpus.require(row.get("stage") in stages and row.get("method") in ("cpu", "numpy") and row.get("source_device") in ("cpu", "mps", "cuda", "xpu", "meta") and row.get("destination_kind") == ("cpu_tensor" if row["method"] == "cpu" else "numpy_array"), "python_readback_boundary_invalid")
        corpus.require(all(type(row.get(key)) is int and row[key] >= 0 for key in ("calls", "source_tensor_logical_bytes", "destination_logical_bytes", "identity_returns")) and row["calls"] > 0 and row["identity_returns"] <= row["calls"], "python_readback_counters_invalid")
        corpus.require(all(isinstance(row.get(key), str) and 0 < len(row[key]) <= 64 for key in ("source_dtype", "destination_dtype")), "python_readback_dtype_invalid")
        key = tuple(row[name] for name in ("stage", "method", "source_device", "destination_kind", "source_dtype", "destination_dtype"))
        corpus.require(key not in seen, "python_readback_duplicate")
        seen.add(key)
        if row["source_device"] not in ("cpu", "meta"):
            prefix = row["method"] + "_method_"
            counts[prefix + "device_to_host_calls"] += row["calls"]
            counts[prefix + "source_logical_bytes"] += row["source_tensor_logical_bytes"]
        if row["method"] == "cpu": counts["cpu_method_same_object_calls"] += row["identity_returns"]
        else: counts["numpy_method_calls"] += row["calls"]
    return {"scope": report["scope"], **counts, "failed_calls": report["failed_calls"], "physical_copy_bytes": None, "aten_counters_additive": False, "full_transfer_coverage_certified": False}
