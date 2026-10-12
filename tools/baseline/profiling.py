"""Synchronized development measurements; timing and counters never prove GPU dispatch."""

import contextlib
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time
import math

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline.corpus import safe_target
from tools.baseline.manifest import sha256


class MeasurementError(ValueError):
    pass


OPERATOR_IDS = ("host_codec", "downmix", "resample", "whisper_stft", "whisper_mel", "whisper_encoder", "whisper_decoder", "whisper_logits_selection", "speaker_features", "wav2vec2", "alignment_log_softmax", "ctc_trellis", "ctc_backtrack", "word_reductions", "speaker_segmentation", "speaker_embedding", "embedding_normalization", "speaker_distances", "plda", "ahc", "vbx", "speaker_kmeans", "constrained_assignment", "window_aggregation", "word_speaker_intersections", "host_text_export")


def expected_operators():
    return [{"id": identifier, "target": "cpu_host" if identifier in ("host_codec", "host_text_export") else "gpu", "required": True} for identifier in OPERATOR_IDS]


def measure(call, drain, evaluate=lambda value: None, clock=time.perf_counter_ns, cpu_clock=time.process_time_ns, memory=None, reset_peak=lambda: None, cadence=0.05):
    if not isinstance(cadence, (int, float)) or not math.isfinite(cadence) or not 0 < cadence <= 60:
        raise MeasurementError("sample_cadence_invalid")
    try:
        drain()
        reset_peak()
    except Exception:
        raise MeasurementError("synchronization_failed") from None
    samples, stopped = [], threading.Event()
    def sample():
        try:
            samples.append({"sample_time_ns": time.perf_counter_ns(), **memory()})
        except Exception:
            samples.append({"sample_time_ns": time.perf_counter_ns(), "sample_unavailable": True})
    def sample_loop():
        while not stopped.wait(cadence):
            sample()
    sampler = None
    if memory is not None:
        sample()
        sampler = threading.Thread(target=sample_loop, daemon=True)
        sampler.start()
    started, cpu_started = clock(), cpu_clock()
    try:
        result = call()
        returned = clock()
        try:
            evaluate(result)
            drain()
        except Exception:
            raise MeasurementError("synchronization_failed") from None
        completed, cpu_completed = clock(), cpu_clock()
    finally:
        stopped.set()
        if sampler is not None:
            sampler.join()
            sample()
    if not started <= returned <= completed or cpu_completed < cpu_started:
        raise MeasurementError("clock_invalid")
    return result, {"status": "synchronized", "api_wall_ns": returned - started, "post_return_wait_ns": completed - returned, "completion_wall_ns": completed - started, "cpu_process_ns": cpu_completed - cpu_started, "gpu_dispatch_observed": False, "drain_calls": 2, "memory_samples": samples, "sample_cadence_ns": int(cadence * 1e9) if memory is not None else None, "memory": memory_summary(samples)}


def memory_summary(samples):
    domains = ("rss_peak_process", "mlx_active", "mlx_peak", "mlx_cache", "mps_allocated", "mps_driver")
    summary = {domain + "_sampled_max_bytes": max((sample[domain + "_bytes"] for sample in samples if domain + "_bytes" in sample), default=None) for domain in domains}
    summary.update(mps_peak_is_lower_bound=True, rss_is_process_lifetime_peak=True, unified_memory_domains_overlap=True, unavailable_samples=sum(bool(sample.get("sample_unavailable")) for sample in samples))
    return summary


class RuntimeProbe:
    def __init__(self):
        try:
            import mlx.core as mx
            self.mx = mx if mx.metal.is_available() else None
        except ImportError:
            self.mx = None
        try:
            import torch
            self.torch = torch if torch.backends.mps.is_available() else None
        except ImportError:
            self.torch = None
        self.streams = [self.mx.default_stream(self.mx.gpu)] if self.mx else []
        self.depth = 0

    @contextlib.contextmanager
    def track_streams(self):
        originals = []
        try:
            if self.mx:
                for name in ("new_stream", "new_thread_local_stream", "new_thread_unsafe_stream"):
                    original = getattr(self.mx, name, None)
                    if original is None:
                        continue
                    originals.append((name, original))
                    def tracked(*args, __original=original, **kwargs):
                        stream = __original(*args, **kwargs)
                        self.streams.append(stream)
                        return stream
                    setattr(self.mx, name, tracked)
            yield
        finally:
            for name, original in reversed(originals):
                setattr(self.mx, name, original)

    def drain(self):
        if self.mx:
            for stream in self.streams:
                self.mx.synchronize(stream)
        if self.torch:
            self.torch.mps.synchronize()

    def evaluate(self, result):
        arrays = []
        def visit(value):
            if self.mx and isinstance(value, self.mx.array):
                arrays.append(value)
            elif isinstance(value, dict):
                for item in value.values(): visit(item)
            elif isinstance(value, (list, tuple)):
                for item in value: visit(item)
        visit(result)
        if arrays:
            self.mx.eval(*arrays)

    def reset_peak(self):
        if self.mx:
            self.mx.reset_peak_memory()

    def memory(self):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        counters = {"rss_peak_process_bytes": int(rss if sys.platform == "darwin" else rss * 1024)}
        if self.mx:
            counters.update(mlx_active_bytes=self.mx.get_active_memory(), mlx_peak_bytes=self.mx.get_peak_memory(), mlx_cache_bytes=self.mx.get_cache_memory())
        if self.torch:
            counters.update(mps_allocated_bytes=self.torch.mps.current_allocated_memory(), mps_driver_bytes=self.torch.mps.driver_allocated_memory())
        return counters

    def profile(self, stage, call):
        outermost = self.depth == 0
        self.depth += 1
        try:
            result, report = measure(call, self.drain, self.evaluate, memory=self.memory, reset_peak=self.reset_peak if outermost else lambda: None)
        finally:
            self.depth -= 1
        report.update(stage=stage, pid=os.getpid(), registered_mlx_streams=len(self.streams), stream_inventory_complete=False, logical_readback_bytes=None, physical_copy_bytes=None)
        report["mlx_peak_scope"] = "current_outermost_span" if outermost else "enclosing_span_not_independent_peak"
        return result, report


def artifact_identity(root, name):
    path = safe_target(root, name)
    if path.is_file():
        if path.stat().st_size > 2 * 1024 * 1024 * 1024:
            raise MeasurementError("trace_budget_exceeded")
        return sha256(path)
    if not path.is_dir():
        raise MeasurementError("trace_missing")
    files, total = [], 0
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            resolved = item.resolve(strict=True)
            if not resolved.is_relative_to(path.resolve()) or not resolved.is_file():
                raise MeasurementError("trace_symlink_escape")
            files.append([str(item.relative_to(path)), "internal_link", str(resolved.relative_to(path.resolve())), sha256(resolved)])
            continue
        if item.is_file():
            total += item.stat().st_size
            if total > 2 * 1024 * 1024 * 1024:
                raise MeasurementError("trace_budget_exceeded")
            files.append([str(item.relative_to(path)), sha256(item)])
    if not files:
        raise MeasurementError("trace_empty")
    return hashlib.sha256(json.dumps(files, separators=(",", ":")).encode()).hexdigest()


def verify_trace(descriptor, root, run_id, pid, phase):
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0 or descriptor.get("run_id") != run_id or descriptor.get("pid") != pid or phase not in descriptor.get("phases", []) or descriptor.get("kind") != "diagnostic":
        raise MeasurementError("trace_identity_mismatch")
    if artifact_identity(root, descriptor["path"]) != descriptor.get("sha256"):
        raise MeasurementError("trace_hash_mismatch")
    return True


def placement_gate(entries, expected=None, target="native_gpu"):
    expected = expected_operators() if expected is None else expected
    if expected != expected_operators():
        raise MeasurementError("operator_expectation_changed")
    if target not in ("native_gpu", "baseline"):
        raise MeasurementError("placement_target_invalid")
    indexed = {entry["operator"]: entry for entry in entries}
    if len(indexed) != len(entries):
        raise MeasurementError("operator_duplicate")
    missing, residuals = [], []
    for operator in expected:
        entry = indexed.get(operator["id"], {})
        if not all(entry.get(key) is True for key in ("observed", "reviewed", "trace_verified")) or entry.get("evidence_kind") != "trace" or entry.get("backend") not in ("gpu", "cpu"):
            missing.append({"operator": operator["id"], "code": "dispatch_evidence_missing"})
        elif operator["target"] == "gpu" and entry["backend"] != "gpu":
            residuals.append(operator["id"])
            if target == "native_gpu":
                missing.append({"operator": operator["id"], "code": "numerical_cpu_residue"})
    return {"status": "incomplete" if missing else "observed", "missing": missing, "cpu_residuals": residuals, "target": target, "complete_gpu": target == "native_gpu" and not missing}
