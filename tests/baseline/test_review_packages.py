import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import corpus, review_packages


class ReviewPackageTests(unittest.TestCase):
    def inputs(self, root):
        (root / "source.wav").write_bytes(b"fixture audio")
        (root / "license.txt").write_bytes(b"Test permission")
        asset = {"id": "source", "path": "source.wav", "sha256": hashlib.sha256(b"fixture audio").hexdigest(), "license_id": "test", "type": "parity_only", "duration": 1, "language": "it", "transcript": "ciao"}
        license = {"id": "test", "verified": True, "revision": "test", "source": "https://example.test/license", "text_path": "license.txt", "text_sha256": hashlib.sha256(b"Test permission").hexdigest(), "attribution": "Test fixture", "rights": {"share": True}, "asset_ids": ["source"]}
        return {"assets": [asset]}, {"licenses": [license]}, corpus.expected_coverage(Path(__file__).resolve().parents[2])

    def test_draft_is_not_ground_truth_and_preparation_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self.inputs(root)
            report = review_packages.prepare(root, "review", *inputs)
            self.assertFalse(report["quality_certified"])
            folder = root / "review/recordings/source"
            draft = json.loads((folder / "annotation.json").read_text())
            self.assertFalse(draft["reviewed"])
            self.assertEqual(draft["words"], [])
            self.assertEqual(draft["reviewers"], [])
            self.assertEqual(json.loads((folder / "source-reference.json").read_text())["transcript"], "ciao")
            self.assertEqual(review_packages.check(root, "review")["pending"], ["source"])
            with self.assertRaises(ValueError): review_packages.prepare(root, "review", *inputs)
            self.assertEqual((root / "source.wav").read_bytes(), b"fixture audio")

    def test_complete_review_requires_audio_bounds_and_stable_speaker_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self.inputs(root)
            review_packages.prepare(root, "review", *inputs)
            folder = root / "review/recordings/source"
            annotation = {"audio_sha256": inputs[0]["assets"][0]["sha256"], "provenance": "independent_human", "reviewed": True, "reviewers": ["person-pass1", "person-pass2"], "uncertainties": [], "transcript": "ciao", "words": [{"word": "ciao", "start": 0, "end": 1}], "turns": [{"speaker": "person", "start": 0, "end": 1}], "uem": [[0, 1]]}
            assignment = {"recording_id": "source", "source_group": "source-session", "speaker_ids": ["person"], "split": "calibration", "case_ids": ["it_short"]}
            (folder / "annotation.json").write_text(json.dumps(annotation))
            (folder / "assignment.json").write_text(json.dumps(assignment))
            report = review_packages.check(root, "review")
            self.assertEqual(report["completed"], ["source"])
            self.assertFalse(report["quality_certified"])
            annotation["uem"] = [[0, 2]]
            (folder / "annotation.json").write_text(json.dumps(annotation))
            self.assertEqual(review_packages.check(root, "review")["pending"], ["source"])
            annotation["uem"] = [[0, 1]]
            (folder / "annotation.json").write_text(json.dumps(annotation))
            assignment["speaker_ids"] = ["invented"]
            (folder / "assignment.json").write_text(json.dumps(assignment))
            self.assertEqual(review_packages.check(root, "review")["pending"], ["source"])

    def test_unlicensed_corrupt_or_linked_sources_leave_no_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog, licenses, coverage = self.inputs(root)
            licenses["licenses"][0]["rights"]["share"] = False
            with self.assertRaises(ValueError): review_packages.prepare(root, "review", catalog, licenses, coverage)
            self.assertFalse((root / "review").exists())
            licenses["licenses"][0]["rights"]["share"] = True
            (root / "source.wav").write_bytes(b"changed")
            with self.assertRaises(ValueError): review_packages.prepare(root, "review", catalog, licenses, coverage)
            self.assertFalse((root / "review").exists())

    def test_optional_source_split_proposals_require_disjoint_verified_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog, licenses, coverage = self.inputs(root)
            asset = catalog["assets"][0]
            asset.update(proposed_split="calibration", source_group="session", speaker_ids=["person"])
            review_packages.prepare(root, "default", catalog, licenses, coverage)
            self.assertEqual(json.loads((root / "default/recordings/source/assignment.json").read_text())["split"], "")
            review_packages.prepare(root, "proposed", catalog, licenses, coverage, use_proposed_splits=True)
            assignment = json.loads((root / "proposed/recordings/source/assignment.json").read_text())
            self.assertEqual(assignment["split"], "calibration")
            self.assertFalse(json.loads((root / "proposed/recordings/source/annotation.json").read_text())["reviewed"])
            asset["speaker_ids"] = []
            with self.assertRaisesRegex(ValueError, "split_identity_missing"):
                review_packages.prepare(root, "invalid", catalog, licenses, coverage, use_proposed_splits=True)
            self.assertFalse((root / "invalid").exists())
