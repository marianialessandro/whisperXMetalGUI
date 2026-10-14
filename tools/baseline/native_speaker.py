"""Experimental native Metal speaker intersections with exact bounded binary64 reductions and compact label readback."""

import argparse
import contextlib
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time

import numpy as np


class Metrics(ct.Structure):
    _fields_ = [("version", ct.c_uint32), ("kernels", ct.c_uint32), ("wall_ns", ct.c_uint64), ("gpu_seconds", ct.c_double), ("allocated_bytes", ct.c_uint64)]


def shader_source():
    root = Path(__file__).parent
    return (root / "native_ctc.metal").read_bytes() + b"\n" + (root / "native_speaker.metal").read_bytes()


def build_library(output):
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("speaker_build_output_exists")
    source = Path(__file__).with_suffix(".mm")
    with tempfile.TemporaryDirectory(prefix="speaker-build-", dir=output.parent) as temporary:
        staged = Path(temporary) / "speaker.dylib"
        subprocess.run(["xcrun", "clang++", "-std=c++20", "-O1", "-fobjc-arc", "-dynamiclib", "-Wno-deprecated-declarations", str(source), "-framework", "Foundation", "-framework", "Metal", "-o", str(staged)], check=True, capture_output=True, timeout=60)
        os.link(staged, output)
    return {"library_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "shader_sha256": hashlib.sha256(shader_source()).hexdigest(), "scope": "experimental_native_speaker_assignment"}


class NativeSpeakerAssignment:
    def __init__(self, library, expected_sha256):
        library = Path(library).resolve(strict=True)
        if hashlib.sha256(library.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError("speaker_library_hash_mismatch")
        self.lock = threading.RLock()
        self.context = ct.c_void_p()
        self.library = ct.CDLL(str(library))
        self.library.wxspeaker_create.argtypes = [ct.c_char_p, ct.c_uint64, ct.POINTER(ct.c_void_p), ct.c_char_p, ct.c_uint32]
        self.library.wxspeaker_create.restype = ct.c_int
        self.library.wxspeaker_assign.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_uint32, ct.c_uint32, ct.c_void_p, ct.POINTER(Metrics), ct.c_char_p, ct.c_uint32]
        self.library.wxspeaker_assign.restype = ct.c_int
        self.library.wxspeaker_destroy.argtypes = [ct.c_void_p]
        self.library.wxspeaker_destroy.restype = None
        shader = shader_source()
        error = ct.create_string_buffer(128)
        started = time.perf_counter_ns()
        if self.library.wxspeaker_create(shader, len(shader), ct.byref(self.context), error, len(error)):
            raise ValueError(error.value.decode("ascii"))
        self.initialization_wall_ns = time.perf_counter_ns() - started
        self.identity = {"library_sha256": expected_sha256, "shader_sha256": hashlib.sha256(shader).hexdigest()}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        with self.lock:
            if self.context:
                self.library.wxspeaker_destroy(self.context)
                self.context = ct.c_void_p()

    def assign(self, turns, labels, queries, speakers, fill_nearest=False):
        for times in (turns, queries):
            if not isinstance(times, np.ndarray) or times.dtype != np.float64 or times.ndim != 2 or times.shape[1] != 2 or not times.flags.c_contiguous:
                raise ValueError("speaker_times_format_invalid")
        if not isinstance(labels, np.ndarray) or labels.dtype != np.uint32 or labels.shape != (len(turns),) or not labels.flags.c_contiguous:
            raise ValueError("speaker_labels_format_invalid")
        if type(speakers) is not int or type(fill_nearest) is not bool or not 0 < speakers <= 256 or not 0 < len(turns) <= 4096 or not 0 < len(queries) <= 4096:
            raise ValueError("speaker_shape_budget_invalid")
        output = np.empty(len(queries), dtype=np.int32)
        metrics = Metrics()
        error = ct.create_string_buffer(128)
        with self.lock:
            if not self.context:
                raise ValueError("speaker_context_closed")
            started = time.perf_counter_ns()
            if self.library.wxspeaker_assign(self.context, turns.ctypes.data, labels.ctypes.data, queries.ctypes.data, len(turns), len(queries), speakers, int(fill_nearest), output.ctypes.data, ct.byref(metrics), error, len(error)):
                raise ValueError(error.value.decode("ascii"))
        if metrics.version != 1 or metrics.kernels != 2 or np.any(output < -1) or np.any(output >= speakers):
            raise ValueError("speaker_native_result_invalid")
        return output, {"scope": "experimental_native_speaker_assignment", "queries": len(queries), "turns": len(turns), "speakers": speakers, "fill_nearest": fill_nearest, "gpu_kernels": ["speaker_stable_order", "speaker_assign"], "command_buffers_per_run": 1, "completion_and_bridge_wall_ns": time.perf_counter_ns() - started, "native_wall_ns": metrics.wall_ns, "computation_gpu_seconds": metrics.gpu_seconds, "allocated_shared_buffer_bytes": metrics.allocated_bytes, "logical_input_upload_bytes": turns.nbytes + labels.nbytes + queries.nbytes, "logical_result_readback_bytes": output.nbytes, "physical_copy_bytes": None, "timestamp_arithmetic": "bounded_binary64_integer_emulation", "quality_certified": False, "full_pipeline_gpu_certified": False}


def assign_word_speakers(context, records, diarize_df, transcript_result, speaker_embeddings=None, fill_nearest=False):
    segments = transcript_result.get("segments", [])
    if not segments or diarize_df is None or len(diarize_df) == 0:
        return transcript_result
    if len(diarize_df) > 4096:
        raise ValueError("speaker_turn_budget_invalid")
    names, name_indices, labels, turns = [], {}, [], []
    for _, row in diarize_df.iterrows():
        name = row["speaker"]
        if not isinstance(name, str) or not name:
            raise ValueError("speaker_label_type_invalid")
        if name not in name_indices:
            name_indices[name] = len(names)
            names.append(name)
        labels.append(name_indices[name])
        turns.append((row["start"], row["end"]))
    queries, targets = [], []
    def timestamp(value):
        if not isinstance(value, (float, int, np.float64, np.int64, np.uint64)) or isinstance(value, (bool, np.bool_)):
            raise ValueError("speaker_query_timestamp_type_invalid_" + type(value).__name__)
        return value
    for segment in segments:
        queries.append((timestamp(segment.get("start", 0.0)), timestamp(segment.get("end", 0.0))))
        targets.append(segment)
        for word in segment.get("words", []):
            if "start" in word:
                queries.append((timestamp(word["start"]), timestamp(word.get("end", word["start"]))))
                targets.append(word)
    turns = np.asarray(turns, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.uint32)
    queries = np.asarray(queries, dtype=np.float64)
    if len(queries) > 1024 * 1024:
        raise ValueError("speaker_query_budget_invalid")
    choices = []
    for begin in range(0, len(queries), 4096):
        output, record = context.assign(turns, labels, np.ascontiguousarray(queries[begin:begin + 4096]), len(names), fill_nearest)
        records.append(record)
        choices.extend(output.tolist())
    for target, choice in zip(targets, choices):
        if choice >= 0:
            target["speaker"] = names[choice]
    if speaker_embeddings is not None:
        transcript_result["speaker_embeddings"] = speaker_embeddings
    return transcript_result


@contextlib.contextmanager
def native_speaker_assignment(task, context, records):
    original = task.assign_word_speakers
    def wrapped(diarize_df, transcript_result, speaker_embeddings=None, fill_nearest=False):
        return assign_word_speakers(context, records, diarize_df, transcript_result, speaker_embeddings, fill_nearest)
    task.assign_word_speakers = wrapped
    try:
        yield
    finally:
        task.assign_word_speakers = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True, help="Compile a new local native dylib at this path; existing paths are never overwritten.")
    args = parser.parse_args()
    print(json.dumps(build_library(args.build)))


if __name__ == "__main__":
    main()
