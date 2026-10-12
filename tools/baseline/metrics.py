"""Fixed-policy quality metrics; natural accuracy requires independent annotations."""

import math
import unicodedata


class MetricError(ValueError):
    pass


NO_SPACE_LANGUAGES = frozenset(("ja", "zh"))
POLICY = {"version": 1, "unicode": "NFC", "case": "casefold", "punctuation": "replace_unicode_P_with_space", "numerals": "verbatim", "whitespace": "collapse", "cer_whitespace": "excluded", "wer_tokenizer": "whitespace_except_ja_zh_not_applicable", "edit_ties": "diagonal_delete_insert_small_hirschberg_leftmost_split_large", "percentiles": "nearest_rank", "collar": 0.0, "skip_overlap": False, "uem": "required", "speaker_mapping": "per_recording_hungarian_reused_for_words", "ambiguous_word_speakers": "exclude_and_report"}


def normalize(text):
    if not isinstance(text, str):
        raise MetricError("text_invalid")
    text = unicodedata.normalize("NFC", text).casefold()
    return " ".join("".join(" " if unicodedata.category(character).startswith("P") else character for character in text).split())


def distance_row(reference, hypothesis):
    if not reference: return list(range(len(hypothesis) + 1))
    masks = {}
    mask_bytes = 0
    for index, token in enumerate(reference):
        old = masks.get(token, 0)
        mask = old | (1 << index)
        mask_bytes += mask.bit_length() // 8 + 32 - (old.bit_length() // 8 + 32 if old else 0)
        if mask_bytes > 64 * 1024 * 1024:
            raise MetricError("edit_mask_budget_exceeded")
        masks[token] = mask
    full = (1 << len(reference)) - 1
    positive, negative, score = full, 0, len(reference)
    high = 1 << (len(reference) - 1)
    result = [score]
    for token in hypothesis:
        equal = masks.get(token, 0)
        vertical = equal | negative
        horizontal = (((equal & positive) + positive) ^ positive) | equal
        plus, minus = negative | ~(horizontal | positive), positive & horizontal
        score += int(bool(plus & high)) - int(bool(minus & high))
        plus, minus = (plus << 1) | 1, minus << 1
        positive, negative = (minus | ~(vertical | plus)) & full, (plus & vertical) & full
        result.append(score)
    return result


def edit_path(reference, hypothesis, reference_offset=0, hypothesis_offset=0):
    if not reference: return [(None, hypothesis_offset + index, "I") for index in range(len(hypothesis))]
    if not hypothesis: return [(reference_offset + index, None, "D") for index in range(len(reference))]
    if reference == hypothesis: return [(reference_offset + index, hypothesis_offset + index, "M") for index in range(len(reference))]
    if len(reference) * len(hypothesis) > 65_536 and len(reference) > 1 and len(hypothesis) > 1:
        midpoint = len(reference) // 2
        left = distance_row(reference[:midpoint], hypothesis)
        right = distance_row(reference[midpoint:][::-1], hypothesis[::-1])
        cut = min(range(len(hypothesis) + 1), key=lambda index: left[index] + right[len(hypothesis) - index])
        del left, right
        return edit_path(reference[:midpoint], hypothesis[:cut], reference_offset, hypothesis_offset) + edit_path(reference[midpoint:], hypothesis[cut:], reference_offset + midpoint, hypothesis_offset + cut)
    costs = [[0] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]
    for i in range(len(reference) + 1): costs[i][0] = i
    for j in range(len(hypothesis) + 1): costs[0][j] = j
    for i, expected in enumerate(reference, 1):
        for j, actual in enumerate(hypothesis, 1):
            costs[i][j] = min(costs[i - 1][j - 1] + (expected != actual), costs[i - 1][j] + 1, costs[i][j - 1] + 1)
    i, j = len(reference), len(hypothesis)
    path = []
    while i or j:
        if i and j and costs[i][j] == costs[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]):
            path.append((reference_offset + i - 1, hypothesis_offset + j - 1, "M" if reference[i - 1] == hypothesis[j - 1] else "S"))
            i, j = i - 1, j - 1
        elif i and costs[i][j] == costs[i - 1][j] + 1:
            path.append((reference_offset + i - 1, None, "D"))
            i -= 1
        else:
            path.append((None, hypothesis_offset + j - 1, "I"))
            j -= 1
    return list(reversed(path))


def edit_distance(reference, hypothesis):
    if len(reference) + len(hypothesis) > 250_000:
        raise MetricError("edit_token_budget_exceeded")
    path = edit_path(reference, hypothesis)
    substitutions = sum(operation == "S" for i, j, operation in path)
    insertions = sum(operation == "I" for i, j, operation in path)
    deletions = sum(operation == "D" for i, j, operation in path)
    matches = [[i, j] for i, j, operation in path if operation == "M"]
    return {"substitutions": substitutions, "insertions": insertions, "deletions": deletions, "reference_count": len(reference), "rate": (substitutions + insertions + deletions) / len(reference) if reference else None, "status": "measured" if reference else "not_applicable", "matches": matches}


def text_scores(reference, hypothesis, language):
    normalized_reference, normalized_hypothesis = normalize(reference), normalize(hypothesis)
    results = {}
    for mode, left, right in (("raw", reference, hypothesis), ("normalized", normalized_reference, normalized_hypothesis)):
        results[mode] = {"wer": None if language in NO_SPACE_LANGUAGES else edit_distance(left.split(), right.split()), "cer": edit_distance(list("".join(left.split())), list("".join(right.split())))}
    return {**results, "primary_metric": "cer" if language in NO_SPACE_LANGUAGES else "wer", "policy_version": POLICY["version"]}


def valid_time(start, end):
    return all(isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) for value in (start, end)) and 0 <= start <= end


def hallucinated_seconds(segments):
    intervals = []
    for segment in segments:
        start, end = segment.get("start"), segment.get("end")
        if not valid_time(start, end):
            raise MetricError("timing_invalid")
        intervals.append((start, end))
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]: merged[-1][1] = max(end, merged[-1][1])
        else: merged.append([start, end])
    return sum(end - start for start, end in merged)


def percentile(values, probability):
    return sorted(values)[max(0, math.ceil(len(values) * probability) - 1)] if values else None


def timestamp_scores(reference, hypothesis):
    alignment = edit_distance([normalize(word["word"]) for word in reference], [normalize(word["word"]) for word in hypothesis])
    starts, ends = [], []
    missing_hypothesis = missing_reference = 0
    for i, j in alignment["matches"]:
        left, right = reference[i], hypothesis[j]
        left_timed = "start" in left and "end" in left
        right_timed = "start" in right and "end" in right
        if left_timed and not valid_time(left["start"], left["end"]) or right_timed and not valid_time(right["start"], right["end"]):
            raise MetricError("timing_invalid")
        if not left_timed: missing_reference += 1
        if not right_timed: missing_hypothesis += 1
        if left_timed and right_timed:
            starts.append(abs(left["start"] - right["start"]))
            ends.append(abs(left["end"] - right["end"]))
    return {"reference_words": len(reference), "timed_matches": len(starts), "coverage": len(starts) / len(reference) if reference else None, "unmatched_reference": len(reference) - len(alignment["matches"]), "untimed_hypothesis_matches": missing_hypothesis, "untimed_reference_matches": missing_reference, "start_mae": sum(starts) / len(starts) if starts else None, "end_mae": sum(ends) / len(ends) if ends else None, "start_p50": percentile(starts, 0.5), "start_p95": percentile(starts, 0.95), "end_p50": percentile(ends, 0.5), "end_p95": percentile(ends, 0.95)}


def word_speaker_scores(reference, hypothesis, mapping):
    alignment = edit_distance([normalize(word["word"]) for word in reference], [normalize(word["word"]) for word in hypothesis])
    eligible, ambiguous = set(), 0
    for i, word in enumerate(reference):
        speakers = word.get("speakers", [word.get("speaker")])
        if len(speakers) == 1 and speakers[0] is not None: eligible.add(i)
        elif len(speakers) > 1: ambiguous += 1
    correct = assigned = unassigned = 0
    for i, j in alignment["matches"]:
        if i not in eligible: continue
        expected = reference[i].get("speakers", [reference[i].get("speaker")])[0]
        actual = hypothesis[j].get("speaker")
        if actual is None:
            unassigned += 1
        else:
            assigned += 1
            correct += mapping.get(actual) == expected
    return {"accuracy": correct / len(eligible) if eligible else None, "assigned_coverage": assigned / len(eligible) if eligible else None, "reference_words": len(reference), "eligible_reference": len(eligible), "ambiguous_reference": ambiguous, "correct": correct, "unassigned_matches": unassigned, "unmatched_reference": len(eligible - {i for i, j in alignment["matches"]})}


def diarization_scores(reference_turns, hypothesis_turns, uem, collar=0.0, skip_overlap=False):
    from pyannote.core import Annotation, Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
    if not uem or collar != POLICY["collar"] or skip_overlap != POLICY["skip_overlap"]:
        raise MetricError("diarization_policy_changed")
    def annotation(turns):
        result = Annotation()
        for index, turn in enumerate(turns):
            if not valid_time(turn.get("start"), turn.get("end")) or not isinstance(turn.get("speaker"), str):
                raise MetricError("turn_invalid")
            result[Segment(turn["start"], turn["end"]), index] = turn["speaker"]
        return result
    if not all(valid_time(start, end) and start < end for start, end in uem):
        raise MetricError("uem_invalid")
    reference, hypothesis = annotation(reference_turns), annotation(hypothesis_turns)
    timeline = Timeline([Segment(start, end) for start, end in uem]).support()
    der = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    reference_crop, hypothesis_crop = reference.crop(timeline), hypothesis.crop(timeline)
    report = {"mapping": der.optimal_mapping(reference, hypothesis, uem=timeline), "reference_speakers": len(reference_crop.labels()), "hypothesis_speakers": len(hypothesis_crop.labels()), "collar": collar, "skip_overlap": skip_overlap}
    report["speaker_count_error"] = abs(report["reference_speakers"] - report["hypothesis_speakers"])
    if not reference_crop.labels():
        report.update(der=None, jer=None, status="not_applicable", false_alarm_seconds=hypothesis_crop.get_timeline().support().duration())
    else:
        report.update(der=float(der(reference, hypothesis, uem=timeline)), jer=float(JaccardErrorRate(collar=collar, skip_overlap=skip_overlap)(reference, hypothesis, uem=timeline)), status="measured")
    return report


def evaluate_case(catalog, licenses, root, case_id, result_path, speaker_turns_path=None):
    import json
    from tools.baseline import corpus, contracts
    case = next((case for case in catalog["cases"] if case["id"] == case_id), None)
    if case is None or case.get("type") != "natural_annotated":
        raise MetricError("independent_ground_truth_required")
    asset = next(asset for asset in catalog["assets"] if asset["id"] == case["asset_id"])
    license = next(license for license in licenses["licenses"] if license["id"] == asset["license_id"])
    corpus.verify_license(asset, license, root, "download")
    if corpus.sha256(corpus.safe_target(root, asset["path"])) != asset["sha256"]:
        raise MetricError("evaluation_asset_mutated")
    annotation_path = corpus.safe_target(root, case["annotation_path"])
    if corpus.sha256(annotation_path) != case["annotation_sha256"]:
        raise MetricError("evaluation_annotation_mutated")
    annotation = json.loads(annotation_path.read_text())
    corpus.validate_annotation(annotation)
    if annotation["audio_sha256"] != asset["sha256"]:
        raise MetricError("evaluation_annotation_audio_mismatch")
    result = json.loads(corpus.safe_target(root, result_path).read_text())
    contracts.validate_result(result)
    text = " ".join(segment["text"] for segment in result["segments"])
    hypothesis_words = []
    for segment in result["segments"]:
        hypothesis_words.extend(segment["words"] if "words" in segment else [{"word": word} for word in segment["text"].split()])
    language = asset["language"]
    scores = text_scores(annotation["transcript"], text, language)
    timings = timestamp_scores(annotation["words"], hypothesis_words)
    diarization = {"status": "intermediate_unavailable"}
    speakers = {"accuracy": None, "assigned_coverage": None}
    if speaker_turns_path is not None:
        turns = json.loads(corpus.safe_target(root, speaker_turns_path).read_text())
        diarization = diarization_scores(annotation["turns"], turns, annotation["uem"])
        speakers = word_speaker_scores(annotation["words"], hypothesis_words, diarization["mapping"])
    no_reference = not normalize(annotation["transcript"])
    insertions = scores["normalized"]["cer" if language in NO_SPACE_LANGUAGES else "wer"]["insertions"]
    values = {"wer": scores["normalized"]["wer"]["rate"] if scores["normalized"]["wer"] is not None else None, "cer": scores["normalized"]["cer"]["rate"], "timestamp_start_mae": timings["start_mae"], "timestamp_end_mae": timings["end_mae"], "timestamp_coverage": timings["coverage"], "der": diarization.get("der"), "jer": diarization.get("jer"), "speaker_count_error": diarization.get("speaker_count_error"), "word_speaker_accuracy": speakers["accuracy"], "silence_insertions": insertions if no_reference else None, "hallucinated_seconds": hallucinated_seconds([segment for segment in result["segments"] if normalize(segment["text"])]) if no_reference else None}
    return {"schema_version": 1, "case_id": case_id, "type": "natural_annotated", "independent_annotation_verified": True, "language": language, "reference_word_count": len(annotation["words"]), "reference_char_count": len("".join(normalize(annotation["transcript"]).split())), "reference_speaker_word_count": speakers.get("eligible_reference", 0), "voices_requested": case.get("settings", {}).get("voices", False), "metrics": values, "text": scores, "timestamps": timings, "diarization": diarization, "word_speakers": speakers, "quality_certified": False, "policy_version": POLICY["version"]}
