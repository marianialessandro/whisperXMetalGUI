"""Optional private turn capture around the unchanged diagnostic worker adapter."""

import argparse
from contextlib import contextmanager
import functools
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import adapter, freeze


def serialize_turns(result):
    frame = result[0] if isinstance(result, tuple) else result
    rows = frame[["start", "end", "speaker"]].to_dict("records")
    turns = []
    for row in rows:
        start, end, speaker = float(row["start"]), float(row["end"]), row["speaker"]
        if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end or not isinstance(speaker, str) or not speaker:
            raise ValueError("voice_turn_invalid")
        turns.append({"start": start, "end": end, "speaker": speaker})
    if not turns: raise ValueError("voice_turns_missing")
    return turns


@contextmanager
def capture(boundaries, retained):
    originals = []
    def selected(request):
        if request.get("voices") is not True: raise ValueError("voice_capture_requires_diarization")
        targets = boundaries(request)
        choices = [(owner, attribute) for owner, attribute, stage in targets if stage == "diarization"]
        if len(choices) != 1: raise ValueError("voice_capture_boundary_missing")
        owner, attribute = choices[0]
        original = getattr(owner, attribute)
        originals.append((owner, attribute, original))
        @functools.wraps(original)
        def observed(*args, **kwargs):
            result = original(*args, **kwargs)
            if retained: raise ValueError("voice_capture_repeated_boundary")
            retained.append(serialize_turns(result))
            return result
        setattr(owner, attribute, observed)
        return targets
    old = adapter.boundaries
    adapter.boundaries = selected
    try:
        yield
    finally:
        adapter.boundaries = old
        for owner, attribute, original in reversed(originals): setattr(owner, attribute, original)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__ + " Additional worker/source/spans/seed arguments are forwarded to the original adapter.")
    parser.add_argument("--speaker-turns", type=Path, required=True, help="New private JSON containing actual returned diarization turns, before word-speaker assignment.")
    args, remainder = parser.parse_known_args(argv)
    if args.speaker_turns.exists(): raise ValueError("voice_turn_destination_exists")
    retained = []
    with capture(adapter.boundaries, retained):
        status = adapter.main(remainder)
    if status != 0: return status
    if len(retained) != 1: raise ValueError("voice_capture_incomplete")
    freeze.seal(args.speaker_turns.parent.resolve(strict=True), args.speaker_turns.name, retained[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
