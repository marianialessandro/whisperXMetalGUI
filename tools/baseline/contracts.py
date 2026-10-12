"""Brownfield GUI worker semantics, without imports of inference libraries."""

import copy
import math
from pathlib import PurePosixPath
import re


class ContractError(ValueError):
    pass


PROFILES = {"Rapida": {"mlx": "mlx-community/whisper-base-mlx", "cpu": "base"}, "Bilanciata": {"mlx": "mlx-community/whisper-small-mlx", "cpu": "small"}, "Massima": {"mlx": "mlx-community/whisper-large-v3-mlx", "cpu": "large-v3"}}


def require(condition, code):
    if not condition:
        raise ContractError(code)


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_timing(value, required):
    for key in ("start", "end"):
        require(key in value or not required, "timing_missing")
        if key in value: require(number(value[key]) and value[key] >= 0, "timing_invalid")
    if "start" in value and "end" in value:
        require(value["start"] <= value["end"], "timing_reversed")
    if "score" in value: require(number(value["score"]), "score_invalid")
    if "speaker" in value: require(value["speaker"] is None or isinstance(value["speaker"], str), "speaker_invalid")


def validate_word(word, text_key="word"):
    require(isinstance(word, dict) and isinstance(word.get(text_key), str), "word_invalid")
    validate_timing(word, False)


def validate_result(result):
    require(isinstance(result, dict) and isinstance(result.get("segments"), list), "segments_invalid")
    require("language" in result and (result["language"] is None or isinstance(result["language"], str) and bool(result["language"])), "language_invalid")
    for segment in result["segments"]:
        require(isinstance(segment, dict) and isinstance(segment.get("text"), str), "segment_invalid")
        validate_timing(segment, True)
        for key, text_key in (("words", "word"), ("chars", "char")):
            if key == "words" and key in segment: require(isinstance(segment[key], list), "words_invalid")
            if key in segment and segment[key] is not None:
                require(isinstance(segment[key], list), "words_invalid")
                for word in segment[key]: validate_word(word, text_key)
    if "word_segments" in result:
        require(isinstance(result["word_segments"], list), "words_invalid")
        for word in result["word_segments"]: validate_word(word)
    # Do not enforce non-overlap, audio bounds or word containment; the current oracle can retain out-of-range segments.
    return result


def validate_artifacts(artifacts):
    require("original.json" in artifacts and "result.json" in artifacts, "artifact_missing")
    for name in ("original.json", "aligned.json", "result.json"):
        if name in artifacts: validate_result(artifacts[name])
    require(not artifacts["original.json"]["segments"] or "aligned.json" in artifacts, "aligned_missing")


def validate_settings(settings):
    require(isinstance(settings, dict), "settings_invalid")
    require(settings.get("quality") in PROFILES, "quality_invalid")
    require(isinstance(settings.get("language"), str) and bool(re.fullmatch(r"auto|[a-z]{2,3}", settings["language"])), "language_invalid")
    require(isinstance(settings.get("voices"), bool), "voices_invalid")
    require(isinstance(settings.get("hints", ""), str), "hints_invalid")
    if "speaker_count" in settings:
        require(isinstance(settings["speaker_count"], int) and not isinstance(settings["speaker_count"], bool) and settings["speaker_count"] > 0, "speaker_count_invalid")


def validate_events(events, outcome, output_root=None):
    require(outcome in ("completed", "failed", "cancelled", "interrupted"), "outcome_invalid")
    require(isinstance(events, list), "events_invalid")
    terminal = None
    for event in events:
        require(isinstance(event, dict) and terminal is None, "event_after_terminal")
        kind = event.get("event")
        require(kind in ("stage", "completed", "failed"), "event_unknown")
        if kind in ("stage", "failed"):
            require(isinstance(event.get("message"), str) and bool(event["message"]), "message_invalid")
        if "backend" in event: require(isinstance(event["backend"], str), "backend_invalid")
        if kind == "completed":
            require(isinstance(event.get("warnings"), list) and all(isinstance(item, str) for item in event["warnings"]), "warnings_invalid")
            require(isinstance(event.get("output"), str) and isinstance(event.get("result"), str), "output_invalid")
            output, result = PurePosixPath(event["output"]), PurePosixPath(event["result"])
            require(".." not in output.parts and ".." not in result.parts and result == output / "result.json", "output_invalid")
            if output_root is not None: require(output == PurePosixPath(output_root), "output_escape")
        if kind in ("completed", "failed"): terminal = kind
    if outcome in ("completed", "failed"): require(terminal == outcome, "terminal_missing")
    else: require(terminal is None, "termination_conflict")


def normalize(bundle, output_root):
    result = copy.deepcopy(bundle)
    for event in result.get("events", []):
        if event.get("event") == "completed":
            require(event.get("output") == output_root and event.get("result") == str(PurePosixPath(output_root) / "result.json"), "output_invalid")
            event["output"], event["result"] = "$OUTPUT", "$OUTPUT/result.json"
    # Diagnostic fields are separate from semantic data and must be selected explicitly by callers.
    return result


def equivalent(left, right, left_root, right_root):
    return normalize(left, left_root) == normalize(right, right_root)
