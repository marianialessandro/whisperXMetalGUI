import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import corpus


class CorpusTests(unittest.TestCase):
    def asset(self, root):
        text = b"Synthetic permission text for offline testing only."
        (root / "license.txt").write_bytes(text)
        data = b"synthetic-media"
        asset = {"id": "asset-1", "path": "audio/sample.wav", "sha256": hashlib.sha256(data).hexdigest(), "max_bytes": 100, "url": "https://huggingface.co/datasets/org/data/resolve/" + "a" * 40 + "/sample.wav", "revision": "a" * 40, "license_id": "fixture-license"}
        license = {"id": "fixture-license", "verified": True, "revision": "a" * 40, "source": "https://huggingface.co/datasets/org/data", "text_path": "license.txt", "text_sha256": hashlib.sha256(text).hexdigest(), "attribution": "Test only", "rights": {"download": True, "derive": True, "share": True}, "asset_ids": ["asset-1"]}
        return asset, license, data

    def test_download_requires_exact_license_before_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset, license, data = self.asset(root)
            called = []
            def transport(url):
                called.append(url)
                return io.BytesIO(data)
            with self.assertRaises(corpus.CorpusError):
                corpus.acquire(asset, dict(license, verified=False), root, root, transport)
            self.assertEqual(called, [])
            license["asset_ids"] = []
            with self.assertRaises(corpus.CorpusError):
                corpus.acquire(asset, license, root, root, transport)
            self.assertEqual(called, [])

    def test_hash_budget_and_overwrite_preserve_external_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset, license, data = self.asset(root)
            external = root / "external.wav"
            external.write_bytes(b"preserve")
            with self.assertRaises(corpus.CorpusError):
                corpus.acquire(asset, license, root, root, lambda url: io.BytesIO(b"wrong"))
            self.assertFalse((root / asset["path"]).exists())
            with self.assertRaises(corpus.CorpusError):
                corpus.acquire(dict(asset, max_bytes=1), license, root, root, lambda url: io.BytesIO(data))
            target = corpus.acquire(asset, license, root, root, lambda url: io.BytesIO(data))
            with self.assertRaises(corpus.CorpusError):
                corpus.acquire(asset, license, root, root, lambda url: io.BytesIO(data))
            self.assertEqual(target.read_bytes(), data)
            self.assertEqual(external.read_bytes(), b"preserve")

    def test_traversal_and_symlink_fail_before_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset, license, data = self.asset(root)
            (root / "link").symlink_to(root, target_is_directory=True)
            for path in ("../external.wav", "/absolute.wav", "link/sample.wav"):
                calls = []
                with self.assertRaises(corpus.CorpusError):
                    corpus.acquire(dict(asset, path=path), license, root, root, lambda url: calls.append(url))
                self.assertEqual(calls, [])

    def test_expected_matrix_cannot_be_weakened_by_removing_rows(self):
        coverage = corpus.expected_coverage(Path(__file__).resolve().parents[2])
        report = corpus.preflight({"assets": [], "cases": []}, coverage, {"licenses": []}, Path("."), Path("."))
        self.assertEqual(report["status"], "prerequisite_missing")
        self.assertIn("italian_natural_diarization_ground_truth", [item["code"] for item in report["missing"]])
        coverage["required_cases"].pop()
        with self.assertRaises(corpus.CorpusError):
            corpus.validate_coverage(coverage, Path(__file__).resolve().parents[2])

    def test_every_alignment_mapping_and_ui_language_is_recorded(self):
        root = Path(__file__).resolve().parents[2]
        coverage = corpus.expected_coverage(root)
        corpus.validate_coverage(coverage, root)
        rows = coverage["languages"]
        self.assertEqual(len([row for row in rows if row["model"]]), 41)
        self.assertTrue(all(row["status"] == "untested" for row in rows))
        self.assertTrue(any(row["language"] == "no" and "1b" in row["model"] for row in rows))

    def test_synthetic_and_unreviewed_annotations_are_not_ground_truth(self):
        for kind in ("synthetic_contract", "derived_audio", "parity_only"):
            self.assertFalse(corpus.quality_eligible({"type": kind, "annotation": {"reviewed": True}}))
        self.assertFalse(corpus.quality_eligible({"type": "natural_annotated", "annotation": {"reviewed": False}}))

    def test_annotation_requires_independent_two_pass_review_and_valid_uem(self):
        annotation = {"audio_sha256": "a" * 64, "provenance": "independent_human", "reviewers": ["reviewer-1", "reviewer-2"], "reviewed": True, "uncertainties": [], "transcript": "ciao", "uem": [[0, 1]], "words": [{"word": "ciao", "start": 0, "end": 1}], "turns": [{"speaker": "speaker-1", "start": 0, "end": 1}]}
        corpus.validate_annotation(annotation)
        for patch in ({"reviewers": ["reviewer-1"]}, {"reviewers": ["reviewer-1", " REVIEWER-1 "]}, {"reviewers": [" ", "r2"]}, {"provenance": "oracle"}, {"uem": [[1, 0]]}, {"uncertainties": ["unresolved boundary"]}):
            with self.assertRaises(corpus.CorpusError):
                corpus.validate_annotation(dict(annotation, **patch))

    def test_split_leakage_detects_recording_and_speaker(self):
        cases = [{"split": "calibration", "recording_id": "a", "source_group": "a", "speaker_ids": ["same"]}, {"split": "evaluation", "recording_id": "b", "source_group": "b", "speaker_ids": ["same"]}]
        with self.assertRaises(corpus.CorpusError):
            corpus.validate_splits(cases)

    def test_split_identities_require_named_people_and_not_an_iterable_string(self):
        valid = {"split": "calibration", "recording_id": "recording", "source_group": "session", "speaker_ids": ["person"]}
        for change in ({"recording_id": " "}, {"source_group": []}, {"speaker_ids": "person"}, {"speaker_ids": [" person"]}, {"speaker_ids": [" "]}, {"speaker_ids": ["person", "person"]}):
            with self.subTest(change=change), self.assertRaises(corpus.CorpusError): corpus.validate_splits([dict(valid, **change)])

    def test_review_flag_cannot_hide_missing_word_annotation(self):
        annotation = {"audio_sha256": "a" * 64, "provenance": "independent_human", "reviewers": ["r1", "r2"], "reviewed": True, "uncertainties": [], "transcript": "ciao", "uem": [[0, 1]], "words": [], "turns": [{"speaker": "A", "start": 0, "end": 1}]}
        with self.assertRaises(corpus.CorpusError):
            corpus.validate_annotation(annotation)

    def test_duplicate_case_names_cannot_fake_long_overlap_and_quality(self):
        settings = {"quality": "Bilanciata", "language": "it", "voices": False, "hints": ""}
        asset = {"language": "it", "duration": 10}
        annotation = {"turns": [{"speaker": "A", "start": 0, "end": 1}]}
        for identifier in ("it_long", "it_overlap", "profile_large_v3", "it_auto", "voices_exact"):
            with self.subTest(case=identifier), self.assertRaises(ValueError):
                corpus.validate_case_capabilities({"id": identifier, "settings": settings}, asset, annotation)

    def test_derived_audio_records_recipe_and_preserves_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset, license, _ = self.asset(root)
            source = root / "audio.wav"
            with wave.open(str(source), "wb") as audio:
                audio.setparams((1, 2, 16000, 4, "NONE", "not compressed"))
                audio.writeframes(b"\0" * 8)
            original = source.read_bytes()
            asset.update(path="audio.wav", sha256=hashlib.sha256(original).hexdigest())
            report = corpus.derive_concat([asset, asset], {license["id"]: license}, root, root, "derived.wav", 100)
            self.assertEqual(report["type"], "derived_audio")
            self.assertFalse(report["quality_eligible"])
            self.assertEqual(report["recipe"]["source_hashes"], [asset["sha256"]] * 2)
            with wave.open(str(root / "derived.wav")) as audio:
                self.assertEqual(audio.getnframes(), 8)
            self.assertEqual(source.read_bytes(), original)
            with self.assertRaises(corpus.CorpusError):
                corpus.derive_concat([asset], {license["id"]: license}, root, root, "too_big.wav", 45)
            self.assertFalse((root / "too_big.wav").exists())

    def test_imported_annotations_are_hash_bound_and_never_overwrite(self):
        annotation = {"audio_sha256": "a" * 64, "provenance": "independent_human", "reviewers": ["r1", "r2"], "reviewed": True, "uncertainties": [], "transcript": "", "uem": [[0, 1]], "words": [], "turns": []}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "original.json"
            payload = json.dumps(annotation).encode()
            source.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            target = corpus.annotation_import(source, root, "annotation.json", digest, "a" * 64)
            self.assertEqual(target.read_bytes(), payload)
            self.assertEqual(source.read_bytes(), payload)
            with self.assertRaises(corpus.CorpusError):
                corpus.annotation_import(source, root, "annotation.json", digest, "a" * 64)
            with self.assertRaises(corpus.CorpusError):
                corpus.annotation_import(source, root, "wrong.json", digest, "b" * 64)
            self.assertFalse((root / "wrong.json").exists())

    def test_archive_extracts_only_selected_regular_files_with_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset, license, _ = self.asset(root)
            archive = root / "archive.tar.gz"
            with tarfile.open(archive, "w:gz") as output:
                for name, data in [("dev/one.wav", b"one"), ("dev/two.wav", b"two")]:
                    member = tarfile.TarInfo(name)
                    member.size = len(data)
                    output.addfile(member, io.BytesIO(data))
            asset.update(path="archive.tar.gz", sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
            records = corpus.extract_selected(asset, license, root, root, {"dev/one.wav": "selected/one.wav"}, 100)
            self.assertEqual((root / "selected/one.wav").read_bytes(), b"one")
            self.assertEqual(records[0]["archive_sha256"], asset["sha256"])
            self.assertFalse((root / "selected/two.wav").exists())
            with self.assertRaises(corpus.CorpusError):
                corpus.extract_selected(asset, license, root, root, {"dev/one.wav": "selected/one.wav"}, 100)

    def test_archive_link_missing_member_and_budget_do_not_publish_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset, license, _ = self.asset(root)
            archive = root / "archive.tar.gz"
            with tarfile.open(archive, "w:gz") as output:
                member = tarfile.TarInfo("link.wav")
                member.type = tarfile.SYMTYPE
                member.linkname = "../private"
                output.addfile(member)
            asset.update(path="archive.tar.gz", sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
            for selected in ({"link.wav": "out.wav"}, {"absent.wav": "out.wav"}, {"../private": "out.wav"}):
                with self.assertRaises(corpus.CorpusError):
                    corpus.extract_selected(asset, license, root, root, selected, 100)
                self.assertFalse((root / "out.wav").exists())


if __name__ == "__main__":
    unittest.main()
