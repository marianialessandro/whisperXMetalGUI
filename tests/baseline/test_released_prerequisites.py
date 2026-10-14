from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import freeze
from tools.certification import released


class ReleasedPrerequisitesTests(unittest.TestCase):
    def document(self):
        return {"schema_version": 1, "reference_policy": "released-dataset", "evidence": [{"id": name, "kind": kind, "snapshot": name, "sha256": "a" * 64} for name, kind in released.ROLES.items()]}

    def verifiers(self):
        def isolated(root, snapshot, checksum):
            return {"operator": snapshot, "individual_operator_dispatch_verified": True, "gpu_activity_observed": not snapshot.startswith("ctc_")}
        return {"text": lambda *args: {"profiles": ["Rapida", "Bilanciata", "Massima"], "jobs": 540}, "voices": lambda *args: {"profiles": ["Rapida", "Bilanciata", "Massima"], "jobs": 54}, "pipeline": lambda *args: {"gpu_activity_observed": True}, "isolated": isolated}

    def fixture(self, root):
        (root / "pipeline").mkdir()
        (root / "pipeline/context.json").write_text(json.dumps({"settings": {"voices": True}}))

    def test_missing_checksum_and_deleted_role_cannot_seal(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(released.VERIFIERS, self.verifiers()):
            root = Path(directory)
            self.fixture(root)
            document = self.document()
            document["evidence"][0]["sha256"] = None
            self.assertEqual(released.inspect(root, document)["status"], "incomplete")
            with self.assertRaises(ValueError): released.seal(root, "report.json", document)
            self.assertFalse((root / "report.json").exists())
            document["evidence"].pop()
            with self.assertRaises(ValueError): released.inspect(root, document)

    def test_invalid_archive_is_collected_without_credit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            verifiers = self.verifiers()
            verifiers["voices"] = lambda *args: (_ for _ in ()).throw(ValueError("altered archive"))
            with patch.dict(released.VERIFIERS, verifiers):
                report = released.inspect(root, self.document())
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(next(row for row in report["evidence"] if row["id"] == "voices")["status"], "invalid_or_missing")

    def test_cpu_evidence_stays_cpu_and_archive_rechecked_after_seal(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(released.VERIFIERS, self.verifiers()):
            root = Path(directory)
            self.fixture(root)
            checksum = released.seal(root, "report.json", self.document())
            report = released.verify(root, "report.json", checksum)
            self.assertFalse(report["promotion_allowed"])
            self.assertFalse(report["local_human_review_required"])
            ctc = next(row for row in report["evidence"] if row["id"] == "ctc_trellis")
            self.assertEqual(ctc["observed_tensor_backend"], "cpu")
            self.assertFalse(ctc["gpu_activity_observed"])
            with self.assertRaises(freeze.FreezeError): released.seal(root, "report.json", self.document())
            (root / "pipeline/context.json").write_text(json.dumps({"settings": {"voices": False}}))
            with self.assertRaises(ValueError): released.verify(root, "report.json", checksum)

    def test_manifest_rejects_traversal_duplicates_bad_hash_and_wrong_operator(self):
        document = self.document()
        for field, value in (("snapshot", "../outside"), ("sha256", "bad"), ("kind", "isolated")):
            invalid = copy.deepcopy(document)
            invalid["evidence"][0][field] = value
            with self.assertRaises(ValueError): released.validate(invalid)
        invalid = copy.deepcopy(document)
        invalid["evidence"][-1] = invalid["evidence"][0]
        with self.assertRaises(ValueError): released.validate(invalid)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            verifiers = self.verifiers()
            verifiers["isolated"] = lambda *args: {"operator": "another", "individual_operator_dispatch_verified": True}
            with patch.dict(released.VERIFIERS, verifiers):
                self.assertEqual(released.inspect(root, document)["status"], "incomplete")

    def test_rapid_only_archive_does_not_replace_complete_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            verifiers = self.verifiers()
            verifiers["text"] = lambda *args: {"profiles": ["Rapida"], "jobs": 180}
            with patch.dict(released.VERIFIERS, verifiers):
                report = released.inspect(root, self.document())
            self.assertEqual(report["status"], "incomplete")

    def test_valid_trace_archive_without_target_gpu_activity_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            verifiers = self.verifiers()
            verifiers["pipeline"] = lambda *args: {"gpu_activity_observed": False}
            with patch.dict(released.VERIFIERS, verifiers):
                report = released.inspect(root, self.document())
            self.assertEqual(report["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
