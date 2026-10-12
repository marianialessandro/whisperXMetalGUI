"""Fail-closed calibration and immutable evidence sealing, never candidate-driven retuning."""

import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import uuid
import re
from datetime import datetime, timezone

from tools.baseline import corpus, metrics, profiling


class FreezeError(ValueError):
    pass


REQUIRED_METRICS = ("wer", "cer", "timestamp_start_mae", "timestamp_end_mae", "timestamp_coverage", "der", "jer", "speaker_count_error", "word_speaker_accuracy", "silence_insertions", "hallucinated_seconds")
LOWER_METRICS = frozenset(("timestamp_coverage", "word_speaker_accuracy"))
CALIBRATION_POLICY = {"seed": 17, "bootstrap_replicates": 2000, "confidence": 0.95, "minimum_recordings_per_split": 3, "minimum_recordings_per_stratum_per_split": 3, "minimum_replicates_per_recording": 3, "order": "case_id_recording_id_replica", "candidate_inputs": "forbidden", "resource_policy": "observed_maxima_not_unified_memory_sum", "metric_applicability": "reference_count_language_and_requested_voices"}


def canonical(document):
    try:
        return json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (ValueError, TypeError):
        raise FreezeError("nonfinite_or_invalid_document") from None


def digest(document):
    return hashlib.sha256(canonical(document)).hexdigest()


def seal(root, name, document):
    target = corpus.safe_target(root, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical(document)
    descriptor, temporary = tempfile.mkstemp(prefix=".freeze-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            raise FreezeError("freeze_destination_exists") from None
    finally:
        os.unlink(temporary)
    return hashlib.sha256(payload).hexdigest()


def verify_seal(root, name, expected_digest):
    target = corpus.safe_target(root, name)
    if profiling.artifact_identity(root, name) != expected_digest:
        raise FreezeError("freeze_mutated")
    canonical(json.loads(target.read_text()))
    return True


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def bootstrap_bound(recording_values, direction, seed=17):
    if len(recording_values) < CALIBRATION_POLICY["minimum_recordings_per_split"] or not all(finite(value) for value in recording_values) or direction not in ("upper", "lower"):
        raise FreezeError("insufficient_ground_truth")
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(recording_values, k=len(recording_values))) / len(recording_values) for _ in range(CALIBRATION_POLICY["bootstrap_replicates"]))
    probability = CALIBRATION_POLICY["confidence"] if direction == "upper" else 1 - CALIBRATION_POLICY["confidence"]
    return metrics.percentile(means, probability)


def evidence_gate(evidence, coverage, root=None, target="baseline"):
    try:
        corpus.validate_coverage(coverage, Path(__file__).resolve().parents[2])
    except ValueError:
        raise FreezeError("required_coverage_changed") from None
    missing = list(evidence.get("prerequisites", []))
    indexed = {item["case_id"]: item for item in evidence.get("cases", [])}
    if len(indexed) != len(evidence.get("cases", [])):
        raise FreezeError("case_evidence_duplicate")
    for expected in coverage["required_cases"]:
        case = indexed.get(expected["id"], {})
        if not case:
            missing.append({"case_id": expected["id"], "code": "required_case_missing"})
            continue
        if case.get("oracle_adapter_equivalent") is not True or case.get("measured") is not True:
            missing.append({"case_id": expected["id"], "code": "real_paired_evidence_missing"})
        if expected["requires_natural_quality"] and (case.get("type") != "natural_annotated" or case.get("independent_annotation_verified") is not True):
            missing.append({"case_id": expected["id"], "code": "insufficient_ground_truth"})
        if not case.get("source_digest") or not case.get("model_digest") or not case.get("runtime_digest"):
            missing.append({"case_id": expected["id"], "code": "source_model_identity_missing"})
    entries = []
    for entry in evidence.get("operators", []):
        entry = dict(entry, trace_verified=False)
        descriptor = entry.get("trace")
        if root is not None and descriptor:
            try:
                profiling.verify_trace(descriptor, root, entry["run_id"], entry["pid"], entry["phase"])
                entry["trace_verified"] = True
            except (ValueError, OSError, KeyError):
                pass
        entries.append(entry)
    placement = profiling.placement_gate(entries, target=target)
    missing.extend({"code": "operator_observation_missing", "operator": item["operator"]} for item in placement["missing"])
    artifacts = evidence.get("artifacts", [])
    required_roles = {"corpus", "licenses", "annotations", "models", "sources", "profiles", "heldout_evaluation"}
    if required_roles - {artifact.get("role") for artifact in artifacts}:
        missing.append({"code": "evidence_artifact_roles_missing"})
    for artifact in artifacts:
        try:
            if root is None or profiling.artifact_identity(root, artifact["path"]) != artifact["sha256"]:
                raise FreezeError("evidence_artifact_mutated")
        except (OSError, ValueError, KeyError):
            missing.append({"code": "evidence_artifact_unverified", "role": artifact.get("role")})
    roles = {artifact.get("role"): artifact for artifact in artifacts}
    if root is not None and "corpus" in roles and "licenses" in roles:
        try:
            catalog = json.loads(corpus.safe_target(root, roles["corpus"]["path"]).read_text())
            licenses = json.loads(corpus.safe_target(root, roles["licenses"]["path"]).read_text())
            ground_truth = corpus.preflight(catalog, coverage, licenses, root, root)
            missing.extend(ground_truth["missing"])
        except (OSError, ValueError, KeyError, TypeError):
            missing.append({"code": "ground_truth_artifact_invalid"})
    if evidence.get("profiles_synchronized") is not True or evidence.get("stream_inventory_reviewed") is not True:
        missing.append({"code": "synchronized_profile_or_stream_review_missing"})
    return {"status": "prerequisite_missing" if missing else "verified", "missing": missing, "placement": placement}


def calibrate(observations, draft):
    if draft.get("state") != "draft" or draft.get("policy") != metrics.POLICY or draft.get("calibration_policy") != CALIBRATION_POLICY:
        raise FreezeError("calibration_policy_missing_or_changed")
    if not observations or any(row.get("role") != "baseline" or row.get("type") != "natural_annotated" or row.get("independent_annotation_verified") is not True or row.get("measured") is not True or row.get("split") not in ("calibration", "evaluation") for row in observations):
        raise FreezeError("insufficient_ground_truth")
    coverage = corpus.expected_coverage(Path(__file__).resolve().parents[2])
    required_strata = sorted(case["id"] for case in coverage["required_cases"] if case["requires_natural_quality"])
    if draft.get("required_strata") != required_strata:
        raise FreezeError("calibration_strata_changed_or_missing")
    identities = {(row.get("source_digest"), row.get("runtime_digest"), row.get("model_digest"), row.get("hardware_digest")) for row in observations}
    if len(identities) != 1 or any(not identity for identity in next(iter(identities))):
        raise FreezeError("baseline_identity_mismatch")
    run_ids = [row.get("run_id") for row in observations]
    if not all(run_ids) or len(set(run_ids)) != len(run_ids):
        raise FreezeError("run_identity_missing_or_duplicate")
    try:
        corpus.validate_splits(observations)
    except ValueError:
        raise FreezeError("calibration_split_leakage") from None
    groups = {}
    replicas = {}
    for row in observations:
        if row.get("case_id") not in corpus.REQUIRED_IDS or not isinstance(row.get("reference_word_count"), int) or not isinstance(row.get("reference_char_count"), int) or not isinstance(row.get("voices_requested"), bool):
            raise FreezeError("metric_applicability_metadata_missing")
        key = (row["split"], row["recording_id"])
        groups.setdefault(key, []).append(row)
        replicas.setdefault((row["split"], row["case_id"], row["recording_id"]), []).append(row)
    if any(len(rows) < CALIBRATION_POLICY["minimum_replicates_per_recording"] for rows in replicas.values()):
        raise FreezeError("insufficient_replicates")
    for split in ("calibration", "evaluation"):
        if len([key for key in groups if key[0] == split]) < CALIBRATION_POLICY["minimum_recordings_per_split"]:
            raise FreezeError("insufficient_ground_truth")
        for stratum in required_strata:
            if len({row["recording_id"] for row in observations if row["split"] == split and row["case_id"] == stratum}) < CALIBRATION_POLICY["minimum_recordings_per_stratum_per_split"]:
                raise FreezeError("insufficient_stratum_ground_truth")
    def applicable(row, name):
        word_count, char_count = row["reference_word_count"], row["reference_char_count"]
        if name == "wer": return word_count > 0 and row.get("language") not in metrics.NO_SPACE_LANGUAGES
        if name == "cer": return char_count > 0
        if name.startswith("timestamp_"): return word_count > 0
        if name in ("der", "jer", "speaker_count_error"): return row["voices_requested"]
        if name == "word_speaker_accuracy": return row["voices_requested"] and row.get("reference_speaker_word_count", 0) > 0
        if name in ("silence_insertions", "hallucinated_seconds"): return char_count == 0
        return True
    thresholds, variability = {}, {}
    for name in REQUIRED_METRICS:
        calibration, heldout = [], []
        deltas = []
        for (split, recording), rows in sorted(groups.items()):
            eligible = [row for row in rows if applicable(row, name)]
            if not eligible:
                continue
            values = [row.get("metrics", {}).get(name) for row in eligible]
            if not all(finite(value) and value >= 0 and (value <= 1 if name in LOWER_METRICS or name == "jer" else True) for value in values):
                raise FreezeError("metric_missing_or_nonfinite")
            (calibration if split == "calibration" else heldout).append(sum(values) / len(values))
            if split == "calibration": deltas.append(max(values) - min(values))
        direction = "lower" if name in LOWER_METRICS else "upper"
        if len(heldout) < CALIBRATION_POLICY["minimum_recordings_per_split"]:
            raise FreezeError("insufficient_metric_heldout_ground_truth")
        thresholds[name] = {"direction": direction, "bound": bootstrap_bound(calibration, direction), "baseline_mean": sum(calibration) / len(calibration), "heldout_mean": sum(heldout) / len(heldout), "calibration_recordings": len(calibration), "heldout_recordings": len(heldout), "not_applicable_observations": sum(not applicable(row, name) for row in observations)}
        variability[name] = bootstrap_bound(deltas, "upper")
    strata_thresholds = {}
    for stratum in required_strata:
        per_metric = {}
        for name in REQUIRED_METRICS:
            per_split = {"calibration": {}, "evaluation": {}}
            for row in observations:
                if row["case_id"] == stratum and applicable(row, name):
                    per_split[row["split"]].setdefault(row["recording_id"], []).append(row["metrics"][name])
            if not per_split["calibration"] and not per_split["evaluation"]:
                per_metric[name] = {"status": "not_applicable"}
                continue
            values = {split: [sum(samples) / len(samples) for samples in records.values()] for split, records in per_split.items()}
            if len(values["evaluation"]) < CALIBRATION_POLICY["minimum_recordings_per_stratum_per_split"]:
                raise FreezeError("insufficient_stratum_metric_ground_truth")
            direction = "lower" if name in LOWER_METRICS else "upper"
            per_metric[name] = {"status": "measured", "direction": direction, "bound": bootstrap_bound(values["calibration"], direction), "baseline_mean": sum(values["calibration"]) / len(values["calibration"]), "heldout_mean": sum(values["evaluation"]) / len(values["evaluation"])}
        strata_thresholds[stratum] = per_metric
    resources = {}
    for name in ("completion_wall_ns", "rss_peak_process_bytes"):
        values = [row.get("resources", {}).get(name) for row in observations]
        if not all(finite(value) and value >= 0 for value in values):
            raise FreezeError("resource_evidence_missing")
        resources[name] = max(values)
    return {"schema_version": 1, "policy": metrics.POLICY, "calibration_policy": CALIBRATION_POLICY, "required_strata": required_strata, "state": "calibrated", "contract_id": str(uuid.uuid4()), "calibration_role": "baseline", "thresholds": thresholds, "thresholds_by_stratum": strata_thresholds, "degradation_delta": variability, "resources": resources, "source_digest": next(iter(identities))[0], "runtime_digest": next(iter(identities))[1], "model_digest": next(iter(identities))[2], "hardware_digest": next(iter(identities))[3], "baseline_run_ids": sorted(run_ids), "heldout_sanity_evaluated": True, "calibration_date": datetime.now(timezone.utc).isoformat(), "rationale": "Recording-level baseline variability; no candidate-derived margins. Bounds are per recording mean and are not absolute natural-quality certification."}


def validate_contract(contract):
    if contract.get("state") not in ("calibrated", "frozen"):
        raise FreezeError("contract_not_calibrated")
    allowed = {"schema_version", "policy", "calibration_policy", "required_strata", "state", "contract_id", "calibration_role", "thresholds", "thresholds_by_stratum", "degradation_delta", "resources", "source_digest", "runtime_digest", "model_digest", "hardware_digest", "baseline_run_ids", "heldout_sanity_evaluated", "calibration_date", "rationale", "evidence_digest", "evidence", "coverage_digest", "frozen_at", "complete_gpu"}
    if set(contract) - allowed:
        raise FreezeError("contract_unknown_fields")
    if contract.get("state") not in ("calibrated", "frozen") or contract.get("calibration_role") != "baseline" or contract.get("policy") != metrics.POLICY or contract.get("calibration_policy") != CALIBRATION_POLICY or contract.get("heldout_sanity_evaluated") is not True:
        raise FreezeError("contract_not_calibrated")
    if not contract.get("contract_id") or not contract.get("baseline_run_ids") or any(not contract.get(key) for key in ("source_digest", "runtime_digest", "model_digest", "hardware_digest")):
        raise FreezeError("contract_identity_missing")
    try:
        uuid.UUID(contract["contract_id"])
        for run_id in contract["baseline_run_ids"]: uuid.UUID(run_id)
    except (ValueError, TypeError, AttributeError):
        raise FreezeError("contract_run_identity_invalid") from None
    if any(not isinstance(contract[key], str) or not re.fullmatch(r"[0-9a-f]{64}", contract[key]) for key in ("source_digest", "runtime_digest", "model_digest", "hardware_digest")):
        raise FreezeError("contract_digest_invalid")
    required_strata = sorted(case["id"] for case in corpus.expected_coverage(Path(__file__).resolve().parents[2])["required_cases"] if case["requires_natural_quality"])
    if contract.get("required_strata") != required_strata:
        raise FreezeError("contract_strata_incomplete")
    if set(contract.get("thresholds_by_stratum", {})) != set(required_strata):
        raise FreezeError("contract_stratum_thresholds_missing")
    for stratum in required_strata:
        for name in REQUIRED_METRICS:
            threshold = contract["thresholds_by_stratum"][stratum].get(name, {})
            if threshold.get("status") == "not_applicable":
                continue
            if threshold.get("status") != "measured" or not all(finite(threshold.get(key)) for key in ("bound", "baseline_mean", "heldout_mean")):
                raise FreezeError("contract_stratum_threshold_invalid")
    for name in REQUIRED_METRICS:
        threshold = contract.get("thresholds", {}).get(name, {})
        if threshold.get("direction") != ("lower" if name in LOWER_METRICS else "upper") or not all(finite(threshold.get(key)) for key in ("bound", "baseline_mean", "heldout_mean")) or not finite(contract.get("degradation_delta", {}).get(name)):
            raise FreezeError("contract_threshold_missing_or_invalid")
    if not all(finite(contract.get("resources", {}).get(key)) for key in ("completion_wall_ns", "rss_peak_process_bytes")):
        raise FreezeError("resource_budget_missing")
    canonical(contract)
    return True


def freeze_contract(contract, evidence, coverage, root, name):
    validate_contract(contract)
    gate = evidence_gate(evidence, coverage, root)
    if gate["missing"]:
        raise FreezeError("freeze_prerequisites_missing")
    def private_field(value):
        if isinstance(value, dict):
            return any(key.lower() in ("token", "password", "secret", "credential", "authorization", "environment", "request") or private_field(item) for key, item in value.items())
        if isinstance(value, list): return any(private_field(item) for item in value)
        if isinstance(value, str): return bool(re.search(r"hf_[A-Za-z0-9]+|/Users/|/home/|Bearer\s", value))
        return False
    if private_field(evidence):
        raise FreezeError("private_metadata_rejected")
    if sorted(evidence.get("baseline_run_ids", [])) != sorted(contract["baseline_run_ids"]):
        raise FreezeError("freeze_run_identity_mismatch")
    for case in evidence["cases"]:
        if any(case.get(key) != contract[key] for key in ("source_digest", "runtime_digest", "model_digest")):
            raise FreezeError("freeze_source_model_identity_mismatch")
    frozen = {**contract, "state": "frozen", "evidence_digest": digest(evidence), "evidence": evidence, "coverage_digest": digest(coverage), "frozen_at": datetime.now(timezone.utc).isoformat(), "complete_gpu": gate["placement"]["complete_gpu"]}
    return {"contract_id": frozen["contract_id"], "sha256": seal(root, name, frozen), "state": "frozen", "complete_gpu": frozen["complete_gpu"]}


def verify_frozen(root, name, expected_digest, coverage):
    verify_seal(root, name, expected_digest)
    contract = json.loads(corpus.safe_target(root, name).read_text())
    validate_contract(contract)
    if contract.get("state") != "frozen" or digest(coverage) != contract.get("coverage_digest") or digest(contract.get("evidence")) != contract.get("evidence_digest"):
        raise FreezeError("freeze_policy_or_evidence_mutated")
    gate = evidence_gate(contract["evidence"], coverage, root)
    if gate["missing"]:
        raise FreezeError("freeze_evidence_unverified")
    return {"state": "frozen", "contract_id": contract["contract_id"], "complete_gpu": gate["placement"]["complete_gpu"]}
