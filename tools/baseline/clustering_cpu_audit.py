"""Bounded owner-thread API observations of original CPU clustering work."""

import contextlib
import hashlib
import importlib.metadata
import inspect
import sys
import time


SOURCE_HASHES = {"ahc_linkage": "dc3365e0f8faaf78258b9b38fe98350426c0fe1b2eb74d94f67ada04c5b7dcbd", "ahc_fcluster": "e997611514ac25f42f2e7ba018a956a4a3b4f8cb2e0377fda5837843474c20e9", "kmeans_fit_predict": "559a0b7eaf40016786760f88159de545c203e761ecd97084c025b060f025b408"}


def targets():
    import pyannote.audio.pipelines.clustering as clustering
    import scipy.cluster.hierarchy as hierarchy
    import scipy.optimize as optimize
    from sklearn.cluster import KMeans

    versions = {name: importlib.metadata.version(name) for name in ("pyannote-audio", "scipy", "scikit-learn")}
    if versions != {"pyannote-audio": "4.0.4", "scipy": "1.15.3", "scikit-learn": "1.9.1"}:
        raise ValueError("clustering_cpu_audit_runtime_not_supported")
    if clustering.linkage is not hierarchy.linkage or clustering.fcluster is not hierarchy.fcluster or clustering.KMeans is not KMeans or clustering.linear_sum_assignment is not optimize.linear_sum_assignment:
        raise ValueError("clustering_cpu_audit_reference_changed")
    if not inspect.isbuiltin(optimize.linear_sum_assignment) or optimize.linear_sum_assignment.__module__ != "scipy.optimize._lsap" or optimize.linear_sum_assignment.__name__ != "linear_sum_assignment":
        raise ValueError("clustering_cpu_audit_solver_not_supported")
    functions = {"ahc_linkage": hierarchy.linkage, "ahc_fcluster": hierarchy.fcluster, "kmeans_fit_predict": KMeans.fit_predict}
    codes = {}
    for name, function in functions.items():
        original = inspect.unwrap(function)
        if hashlib.sha256(inspect.getsource(original).encode()).hexdigest() != SOURCE_HASHES[name]:
            raise ValueError("clustering_cpu_audit_source_not_supported")
        codes[original.__code__] = name
    return codes, optimize.linear_sum_assignment


@contextlib.contextmanager
def clustering_cpu_audit(records):
    if not isinstance(records, list) or records:
        raise ValueError("clustering_cpu_audit_empty_records_required")
    previous = sys.getprofile()
    if previous is not None:
        raise ValueError("clustering_cpu_audit_profiler_already_active")
    codes, solver = targets()
    rows = {}
    for name in (*SOURCE_HASHES, "linear_sum_assignment"):
        row = {"operation": name, "scope": "owner_thread_clustering_api", "calls_started": 0, "calls_returned_or_unwound": 0, "c_exceptions": 0, "incomplete_calls": 0, "total_wall_ns": 0, "max_wall_ns": 0, "source_sha256": SOURCE_HASHES.get(name), "inclusive_wall_time": True, "instrumentation_overhead_included": True, "exclusive_cpu_time": False, "gpu_dispatch_verified": False, "python_exception_status_available": False}
        records.append(row)
        rows[name] = row
        row["observation_complete"] = False
    active = {}

    def profile(frame, event, arg):
        if event in ("call", "return"):
            name = codes.get(frame.f_code)
            if name is None:
                return
        elif event in ("c_call", "c_return", "c_exception") and arg is solver:
            name = "linear_sum_assignment"
        else:
            return
        row = rows[name]
        key = (id(frame), name)
        if event in ("call", "c_call"):
            row["calls_started"] += 1
            active[key] = time.perf_counter_ns()
        else:
            started = active.pop(key, None)
            if started is not None:
                elapsed = time.perf_counter_ns() - started
                row["calls_returned_or_unwound"] += 1
                row["c_exceptions"] += event == "c_exception"
                row["total_wall_ns"] += elapsed
                row["max_wall_ns"] = max(row["max_wall_ns"], elapsed)

    sys.setprofile(profile)
    try:
        yield
    finally:
        retained = sys.getprofile() is profile
        sys.setprofile(previous)
        for row in rows.values():
            row["incomplete_calls"] = row["calls_started"] - row["calls_returned_or_unwound"]
            row["observation_complete"] = retained and row["incomplete_calls"] == 0
        active.clear()
