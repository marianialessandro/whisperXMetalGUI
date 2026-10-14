"""Run one original alignment operation after setup, holding readback until the trace is saved."""

import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import freeze, reproducibility, offline_worker
from tools.tracing.dispatch_audit import DispatchAudit


def main():
    request = json.loads(sys.stdin.readline())
    source = Path(request["source_root"])
    sys.path.insert(0, str(source))
    os.setpgrp()
    with offline_worker.deny_python_network(), open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        reproducibility.seed_process(1729)
        import torch
        import whisperx
        if not torch.backends.mps.is_available() or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0": raise ValueError("isolated_mps_required")
        model, metadata = whisperx.load_align_model(request["language"], "mps")
        if model.training: raise ValueError("isolated_original_model_not_eval")
        samples = whisperx.load_audio(request["audio"])
        waveform = torch.from_numpy(samples[:min(len(samples), 8 * 16000)]).unsqueeze(0).to("mps")
        def forward():
            if metadata["type"] == "torchaudio": return model(waveform)[0]
            return model(waveform).logits
        def sha(tensor): return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        with torch.inference_mode():
            name = request["operator"]
            prepared = forward() if name != "wav2vec2" else None
            if name in ("ctc_trellis", "ctc_backtrack"):
                import whisperx.alignment as alignment
                emission = torch.log_softmax(prepared, dim=-1)[0].cpu().detach()
                dictionary = metadata["dictionary"]
                text = request["reference_text"].lower().replace(" ", "|")
                tokens = [dictionary[character] for character in text if character in dictionary][:64]
                if not tokens: raise ValueError("isolated_tokens_missing")
                blank = next((index for character, index in dictionary.items() if character in ("[pad]", "<pad>")), 0)
                trellis = alignment.get_trellis(emission, tokens, blank) if name == "ctc_backtrack" else None
                def operation():
                    if trellis is not None: return alignment.backtrack(trellis, emission, tokens, blank)
                    return alignment.get_trellis(emission, tokens, blank)
            else:
                def operation(): return torch.log_softmax(prepared, dim=-1) if prepared is not None else forward()
            def identity(value):
                if isinstance(value, torch.Tensor): return sha(value), list(value.shape), str(value.dtype), str(value.device), value.device.type
                if value is None: raise ValueError("isolated_path_missing")
                points = [{"token_index": point.token_index, "time_index": point.time_index, "score": point.score} for point in value]
                return freeze.digest(points), [len(points)], "python_points", "cpu", "cpu"
            expected = operation()
            torch.mps.synchronize()
            expected_sha256, expected_shape, expected_dtype, expected_device, expected_device_type = identity(expected)
            del expected
            torch.mps.synchronize()
            import mlx.core as mx
            mx.synchronize(mx.gpu)
            sys.__stdout__.write(json.dumps({"ready": True, "pid": os.getpid(), "run_id": request["run_id"]}) + "\n")
            sys.__stdout__.flush()
            time.sleep(5)
            audit = DispatchAudit()
            torch.mps.synchronize()
            started = time.perf_counter_ns()
            with audit: result = operation()
            torch.mps.synchronize()
            finished = time.perf_counter_ns()
            sys.__stdout__.write(json.dumps({"operator_completed": request["operator"], "pid": os.getpid(), "run_id": request["run_id"]}) + "\n")
            sys.__stdout__.flush()
            if sys.stdin.readline().strip() != "trace_saved": raise ValueError("isolated_release_invalid")
            result_sha256, shape, dtype, device, device_type = identity(result)
            if (result_sha256, shape, dtype, device_type) != (expected_sha256, expected_shape, expected_dtype, expected_device_type): raise ValueError("isolated_output_mismatch")
            report = {"status": "isolated_operator_completed", "pid": os.getpid(), "run_id": request["run_id"], "operator": request["operator"], "family": metadata["type"], "output_device": device, "output_device_type": device_type, "output_dtype": dtype, "output_shape": shape, "original_output_sha256": expected_sha256, "original_output_shape": expected_shape, "original_output_dtype": expected_dtype, "observed_output_sha256": result_sha256, "output_identical": True, "completion_wall_ns": finished - started, "dispatch_audit": audit.report(), "setup_and_original_reference_before_trace": True, "result_readback_after_trace_saved": True, "mps_fallback_allowed": False, "transcript_origin": "first_up_to_64_released_reference_characters_numeric_fixture_only" if name in ("ctc_trellis", "ctc_backtrack") else "not_used", "benchmark_timing_eligible": False}
            freeze.seal(Path(request["folder"]).resolve(), "operator.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
