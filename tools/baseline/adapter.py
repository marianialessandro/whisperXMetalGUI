"""Process-local instrumentation of the selected original GUI worker; no algorithm replacement."""

import argparse
import contextlib
import functools
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline.profiling import RuntimeProbe
from tools.baseline.offline_worker import deny_python_network
from tools.baseline.reproducibility import seed_process


@contextlib.contextmanager
def observe(targets, spans, profiler=None):
    originals = []
    try:
        for owner, attribute, stage in targets:
            original = getattr(owner, attribute)
            originals.append((owner, attribute, original))
            @functools.wraps(original)
            def wrapper(*args, __original=original, __stage=stage, **kwargs):
                if profiler is not None:
                    result, report = profiler(__stage, lambda: __original(*args, **kwargs))
                    spans.append(report)
                    return result
                started = time.perf_counter_ns()
                status = "returned"
                try:
                    return __original(*args, **kwargs)
                except BaseException:
                    status = "failed"
                    raise
                finally:
                    spans.append({"stage": __stage, "api_wall_ns": time.perf_counter_ns() - started, "status": status, "gpu_completion_observed": False})
            setattr(owner, attribute, wrapper)
        yield
    finally:
        for owner, attribute, original in reversed(originals):
            setattr(owner, attribute, original)


def load_worker(worker_path, source_root):
    sys.path.insert(0, str(source_root))
    spec = importlib.util.spec_from_file_location("selected_gui_worker", worker_path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker


def boundaries(request):
    import whisperx
    from whisperx import utils
    targets = [(whisperx, "load_audio", "decode"), (whisperx, "load_model", "asr_model_load"), (whisperx, "load_align_model", "alignment_model_load"), (whisperx, "align", "alignment"), (utils, "get_writer", "export_writer"), (utils.ResultWriter, "__call__", "export_write")]
    try:
        import mlx_whisper
        targets.append((mlx_whisper, "transcribe", "asr_mlx"))
    except ImportError:
        pass
    if request.get("voices"):
        from whisperx.diarize import DiarizationPipeline
        import whisperx.diarize as diarize
        targets.extend([(DiarizationPipeline, "__call__", "diarization"), (diarize, "assign_word_speakers", "speaker_assignment")])
    return targets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--spans", type=Path, required=True, help="Dedicated new diagnostic file; API wall duration is not synchronized GPU completion.")
    parser.add_argument("--synchronized", action="store_true", help="Drain registered MLX streams and MPS, evaluate lazy arrays and sample separate memory domains; never a GPU dispatch claim.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, help="Optional paired diagnostic RNG initialization; same bootstrap seed as original entrypoint, no decoder policy changes.")
    parser.add_argument("--preserve-failure-message", action="store_true", help="Optional private contract diagnostic: preserve the original exception message with the original token redaction. Default still emits only worker_failed; never use unsanitized events as public evidence.")
    args = parser.parse_args(argv)
    request = json.load(sys.stdin)
    spans = []
    os.setpgrp()
    with deny_python_network(), open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        try:
            if args.seed is not None: seed_process(args.seed)
            worker = load_worker(args.worker, args.source_root)
            probe = RuntimeProbe() if args.synchronized else None
            with probe.track_streams() if probe else contextlib.nullcontext():
                with observe(boundaries(request), spans, probe.profile if probe else None):
                    worker.run(request)
            code = 0
        except Exception as error:
            message = "worker_failed"
            if args.preserve_failure_message:
                message = str(error) or "Il motore non è riuscito a completare la trascrizione."
                if request.get("token"): message = message.replace(request["token"], "[token]")
            sys.__stdout__.write(json.dumps({"event": "failed", "message": message}) + "\n")
            sys.__stdout__.flush()
            code = 2
    request.clear()
    with args.spans.open("x", encoding="utf-8") as output:
        json.dump({"schema_version": 1, "run_id": args.run_id or str(uuid.uuid4()), "pid": os.getpid(), "spans": spans, "timing_kind": "synchronized_completion" if args.synchronized else "api_wall_only", "gpu_placement_certified": False}, output)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
