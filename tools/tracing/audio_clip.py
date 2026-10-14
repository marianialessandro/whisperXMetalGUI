"""Bounded PCM prefix for separate diagnostics; original licensed audio remains unchanged."""

import wave

from tools.baseline import corpus, manifest


def prefix(audio, target, seconds):
    corpus.require(type(seconds) is int and 1 <= seconds <= 120, "trace_audio_prefix_limit_invalid")
    with wave.open(str(audio), "rb") as source:
        corpus.require(source.getnchannels() == 1 and source.getsampwidth() == 2 and source.getframerate() == 16000 and source.getcomptype() == "NONE", "trace_audio_prefix_format_invalid")
        total = source.getnframes()
        frames = min(total, seconds * source.getframerate())
        corpus.require(frames > 0, "trace_audio_prefix_empty")
        payload = source.readframes(frames)
        corpus.require(len(payload) == frames * 2, "trace_audio_prefix_truncated")
    with target.open("xb") as output:
        with wave.open(output, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(payload)
    return {"kind": "licensed_source_prefix_diagnostic_only", "source_frames": total, "selected_frames": frames, "sample_rate": 16000, "offset_frames": 0, "requested_max_seconds": seconds, "sha256": manifest.sha256(target), "full_recording_trace": frames == total, "quality_sample": False, "benchmark_timing_eligible": False}
