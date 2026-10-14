import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import corpus, dataset_reference, manifest, import_ami, run


class DatasetReferenceTests(unittest.TestCase):
    def inputs(self, root):
        folder = root / "corpus-fleurs"
        (folder / "metadata").mkdir(parents=True)
        (folder / "licenses").mkdir()
        (folder / "sample.wav").write_bytes(b"fixture audio")
        (folder / "licenses/permission.txt").write_bytes(b"test license")
        card = folder / "licenses/fleurs-card.md"
        card.write_bytes(b"fixture card")
        metadata = folder / "metadata/it-dev.tsv"
        metadata.write_text("1\tsample.wav\tCiao Marco\tciao marco\n")
        revision = dataset_reference.acquire_fleurs.REVISION
        inventory = {"archive": {"sha256": "a" * 64, "revision": revision}, "metadata_sha256": manifest.sha256(metadata), "license": {"source": "https://huggingface.co/datasets/google/fleurs"}}
        (folder / "it-acquisition.json").write_text(json.dumps(inventory))
        asset = {"id": "fleurs-it-sample", "path": "corpus-fleurs/sample.wav", "sha256": manifest.sha256(folder / "sample.wav"), "license_id": "test", "language": "it", "type": "parity_only", "revision": revision, "archive_sha256": "a" * 64, "transcript": "Ciao Marco", "duration": 2}
        license = {"id": "test", "verified": True, "revision": revision, "source": inventory["license"]["source"], "text_path": "corpus-fleurs/licenses/permission.txt", "text_sha256": manifest.sha256(folder / "licenses/permission.txt"), "attribution": "Test fixture", "rights": {"download": True}, "asset_ids": [asset["id"]]}
        return {"assets": [asset]}, {"licenses": [license]}, manifest.sha256(card)

    def test_released_reference_scores_real_errors_without_manufacturing_word_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog, licenses, card = self.inputs(root)
            with patch.object(dataset_reference.acquire_fleurs, "CARD_HASH", card):
                prepared = dataset_reference.prepare(root, "released", catalog, licenses)
                result = {"segments": [{"text": "Ciao Mario", "start": 0, "end": 2}], "language": "it"}
                (root / "result.json").write_text(json.dumps(result))
                report = dataset_reference.evaluate(root, prepared, licenses, "fleurs-it-sample", "result.json")
                self.assertEqual(report["text"]["normalized"]["wer"]["rate"], 0.5)
                self.assertEqual(report["dataset_timestamp_agreement"]["status"], "not_available")
                self.assertFalse(report["local_human_review_required"])
                self.assertFalse(report["local_human_review_performed"])
                self.assertFalse(report["independent_timestamp_accuracy_certified"])
                with self.assertRaisesRegex(ValueError, "destination_exists"): dataset_reference.prepare(root, "released", catalog, licenses)

    def test_changed_official_text_metadata_or_audio_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog, licenses, card = self.inputs(root)
            asset = catalog["assets"][0]
            with patch.object(dataset_reference.acquire_fleurs, "CARD_HASH", card):
                asset["transcript"] = "invented model text"
                with self.assertRaisesRegex(ValueError, "fleurs_reference_text_changed"): dataset_reference.prepare(root, "released", catalog, licenses)
                self.assertFalse((root / "released").exists())
                asset["transcript"] = "Ciao Marco"
                (root / "corpus-fleurs/metadata/it-dev.tsv").write_text("changed metadata")
                with self.assertRaisesRegex(ValueError, "fleurs_reference_identity_changed"): dataset_reference.prepare(root, "released", catalog, licenses)
                (root / asset["path"]).write_bytes(b"changed audio")
                with self.assertRaisesRegex(ValueError, "dataset_audio_changed"): dataset_reference.prepare(root, "released", catalog, licenses)

    def test_corrupted_sealed_reference_and_wrong_language_cannot_be_scored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog, licenses, card = self.inputs(root)
            with patch.object(dataset_reference.acquire_fleurs, "CARD_HASH", card):
                prepared = dataset_reference.prepare(root, "released", catalog, licenses)
                (root / "result.json").write_text(json.dumps({"language": "en", "segments": []}))
                with self.assertRaisesRegex(ValueError, "dataset_result_language_mismatch"):
                    dataset_reference.evaluate(root, prepared, licenses, "fleurs-it-sample", "result.json")
                path = root / prepared["cases"][0]["reference_path"]
                path.write_text("{}")
                with self.assertRaisesRegex(ValueError, "dataset_reference_changed"):
                    dataset_reference.evaluate(root, prepared, licenses, "fleurs-it-sample", "result.json")

    def test_runner_released_policy_is_explicit_and_default_still_requires_independent_annotation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog, licenses, card = self.inputs(root)
            with patch.object(dataset_reference.acquire_fleurs, "CARD_HASH", card):
                dataset_reference.prepare(root, "released", catalog, licenses)
                (root / "coverage.json").write_text("{}")
                (root / "models.json").write_text("[]")
                (root / "result.json").write_text(json.dumps({"language": "it", "segments": [{"text": "Ciao Marco", "start": 0, "end": 2}]}))
                arguments = ["--baseline-source", "gui-checkout", "--source-root", str(root), "--runtime", sys.executable, "--reference-root", str(root), "--corpus", str(root / "released/catalog.json"), "--licenses", str(root / "released/licenses.json"), "--coverage", str(root / "coverage.json"), "--model-specs", str(root / "models.json"), "--evidence-root", str(root), "evaluate", "--case-id", "fleurs-it-sample", "--result", "result.json", "--output", "score.json"]
                self.assertEqual(run.main(arguments), 2)
                self.assertFalse((root / "score.json").exists())
                self.assertEqual(run.main(arguments + ["--reference-policy", "released-dataset"]), 0)
                report = json.loads((root / "score.json").read_text())
                self.assertEqual(report["text"]["normalized"]["wer"]["rate"], 0)
                self.assertFalse(report["local_human_review_required"])

    def test_ami_released_reference_keeps_source_timings_without_claiming_local_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "corpus-ami"
            folder.mkdir()
            archive = folder / "manual-1.6.2.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("corpusResources/meetings.xml", '<root><meeting observation="ES2002a"><speaker nxt_agent="A" global_name="source-person"/></meeting></root>')
                output.writestr("words/ES2002a.A.words.xml", '<root><w starttime="0" endtime="1">hello</w></root>')
                output.writestr("segments/ES2002a.A.segments.xml", '<root><segment transcriber_start="0" transcriber_end="1"/></root>')
            audio = folder / "audio.wav"
            audio.write_bytes(b"fixture audio")
            annotation = import_ami.import_manual(archive, manifest.sha256(archive), "ES2002a", manifest.sha256(audio), 2)
            source = folder / "reference.json"
            source.write_text(json.dumps(annotation))
            asset = {"id": "ami-ES2002a", "path": "corpus-ami/audio.wav", "sha256": manifest.sha256(audio), "language": "en", "duration": 2, "review_reference_path": "corpus-ami/reference.json", "review_reference_sha256": manifest.sha256(source)}
            released = dataset_reference.reference(root, asset)
            self.assertEqual(released["words"][0]["word"], "hello")
            self.assertFalse(released["local_human_review_performed"])
            self.assertFalse(released["independent_word_boundary_review_verified"])
            self.assertEqual(released["origin"]["timing_basis"], "dataset_supplied_not_locally_adjudicated")
            annotation["transcript"] = "invented"
            source.write_text(json.dumps(annotation))
            asset["review_reference_sha256"] = manifest.sha256(source)
            with self.assertRaisesRegex(ValueError, "ami_reference_not_original_release"): dataset_reference.reference(root, asset)
