"""Original alignment/export boundary fixtures; synthetic input, no ASR or quality claim."""

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import adapter
from tools.baseline.offline_worker import deny_python_network
from tools.baseline.reproducibility import seed_process


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--kind", choices=("unknown_chars", "out_of_range"), required=True)
    parser.add_argument("--instrumented", action="store_true")
    parser.add_argument("--spans", type=Path)
    args = parser.parse_args(argv)
    sys.path.insert(0, str(args.source_root))
    request = json.load(sys.stdin)
    os.setpgrp()
    spans = []
    try:
        with deny_python_network(), open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            seed_process(1729)
            import whisperx
            import whisperx.utils as utils
            import torch
            output = Path(request["output"])
            output.mkdir()
            scope = adapter.observe(adapter.boundaries(request), spans) if args.instrumented else contextlib.nullcontext()
            with scope:
                audio = whisperx.load_audio(request["audio"])
                duration = len(audio) / 16000
                segments = [{"start": 0.0, "end": min(2.0, duration), "text": "🧪🛰"}] if args.kind == "unknown_chars" else [{"start": duration + 1, "end": duration + 2, "text": "ciao"}]
                original = {"segments": segments, "language": "it"}
                model, metadata = whisperx.load_align_model("it", "mps" if torch.backends.mps.is_available() else "cpu")
                result = whisperx.align(segments, model, metadata, audio, next(model.parameters()).device.type)
                result["language"] = "it"
                for name, document in (("original.json", original), ("aligned.json", result), ("result.json", result)):
                    with (output / name).open("x") as destination: json.dump(document, destination, ensure_ascii=False)
                for extension in ("txt", "srt", "vtt"):
                    utils.get_writer(extension, str(output))(result, request["audio"], {"max_line_width": None, "max_line_count": None, "highlight_words": False})
        event = {"event": "completed", "output": str(output), "result": str(output / "result.json"), "warnings": []}
        code = 0
    except Exception:
        event, code = {"event": "failed", "message": "contract_leaf_failed"}, 2
    if args.spans:
        with args.spans.open("x") as destination: json.dump({"spans": spans, "scope": "synthetic_alignment_export_boundary", "asr_performed": False}, destination)
    print(json.dumps(event))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
