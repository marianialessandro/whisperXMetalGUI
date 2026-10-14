import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.evaluation import voice_pair
from tools.baseline import manifest, metrics


class VoicePairTests(unittest.TestCase):
    def test_private_token_goes_only_in_child_request_and_actual_turns_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            calls = []
            def execute(runtime, arguments, request, timeout, **kwargs):
                self.assertFalse(any("fixture-secret" in str(argument) for argument in arguments))
                self.assertEqual(request["token"], "fixture-secret")
                calls.append(request.copy())
                if "--speaker-turns" in arguments:
                    target = arguments[arguments.index("--speaker-turns") + 1]
                    target.write_text(json.dumps([{"start": 1, "end": 3, "speaker": "A"}]))
                return [{"backend": "MLX / Metal"}, {"warnings": []}]
            with patch.object(voice_pair.run, "execute", side_effect=execute), patch.object(voice_pair.run, "semantic_bundle", return_value={}), patch.object(voice_pair.contracts, "equivalent", return_value=True):
                report = voice_pair.paired(root, root, root / "audio.wav", {"voices": True}, root / "first/pair", "fixture-secret", root, root, 60)
                self.assertTrue(report["actual_turns_retained"])
                self.assertFalse(report["original_turn_parity_measured"])
                self.assertEqual(len(calls), 2)
                with self.assertRaisesRegex(ValueError, "destination_exists"): voice_pair.paired(root, root, root / "audio.wav", {"voices": True}, root / "first/pair", "fixture-secret", root, root, 60)
                self.assertFalse(any("fixture-secret" in path.read_text() for path in root.rglob("*") if path.is_file()))

    def test_warning_or_semantic_mismatch_never_becomes_a_measured_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for mismatch in (True, False):
                events = [{"backend": "MLX / Metal"}, {"warnings": ["incomplete"] if not mismatch else []}]
                with patch.object(voice_pair.run, "execute", return_value=events), patch.object(voice_pair.run, "semantic_bundle", return_value={}), patch.object(voice_pair.contracts, "equivalent", return_value=not mismatch):
                    with self.assertRaisesRegex(ValueError, "pair_mismatch" if mismatch else "pipeline_incomplete"):
                        voice_pair.paired(root, root, root / "audio.wav", {"voices": True}, root / str(mismatch) / "pair", "fixture-secret", root, root, 60)

    def test_unsupported_reference_is_rejected_before_reading_private_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            catalog = {"assets": [{"id": "fleurs-it-a", "language": "it"}], "cases": [{"id": "case", "asset_id": "fleurs-it-a"}]}
            with patch.object(voice_pair.provision, "read_credential") as credential:
                with self.assertRaisesRegex(ValueError, "reference_unsupported"):
                    voice_pair.execute(root, "snapshot", catalog, {}, "case", "Rapida", None, root, root, root, [], root, 60)
                credential.assert_not_called()
                self.assertFalse((root / "snapshot").exists())


    def test_successful_reference_scoring_retains_evidence_and_never_serializes_credential(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = Path(__file__).resolve().parents[2]
            words = [{"word": "Hello", "start": 0, "end": 1, "speaker": "reference"}]
            reference = {"words": words, "turns": [{"start": 0, "end": 2, "speaker": "reference"}], "uem": [[0, 2]]}
            (root / "ref.json").write_text(json.dumps(reference))
            (root / "audio.wav").write_bytes(b"fixture audio")
            asset = {"id": "ami-fixture", "language": "en", "path": "audio.wav", "sha256": manifest.sha256(root / "audio.wav"), "license_id": "license"}
            case = {"id": "case", "asset_id": asset["id"], "reference_path": "ref.json", "reference_sha256": manifest.sha256(root / "ref.json")}
            catalog = {"assets": [asset], "cases": [case]}
            licenses = {"licenses": [{"id": "license"}]}
            fingerprint = {"fixture": True}
            context = {"model_inventory": [{"roles": ["speaker_segmentation", "speaker_embedding", "speaker_plda", "alignment_en"]}]}
            def paired(runtime, source, audio, settings, destination, token, hf, torch, timeout):
                self.assertEqual(token, "fixture-secret")
                destination.mkdir()
                result = {"segments": [], "word_segments": [{"word": "Hello", "start": 0, "end": 1, "speaker": "hypothesis"}]}
                (destination / "instrumented").mkdir()
                (destination / "instrumented/result.json").write_text(json.dumps(result))
                return {"status": "paired_equivalent", "actual_turns_retained": True}
            score = {"dataset_turn_agreement": {"mapping": {"hypothesis": "reference"}, "der": 0, "jer": 0}}
            with patch.object(voice_pair.corpus, "verify_license"), patch.object(voice_pair.dataset_reference, "reference", return_value=reference), patch.object(voice_pair.released_baseline, "identity", return_value=(fingerprint, context)), patch.object(voice_pair.run, "model_cache_roots", return_value=(root, root)), patch.object(voice_pair.provision, "read_credential", return_value="fixture-secret"), patch.object(voice_pair, "paired", side_effect=paired), patch.object(voice_pair.dataset_reference, "evaluate", return_value=score):
                report, checksum = voice_pair.execute(root, "snapshot", catalog, licenses, "case", "Rapida", 1, source, root, root, [], root, 60)
                self.assertEqual(report["score"]["dataset_word_speaker_agreement"]["accuracy"], 1)
                self.assertEqual(manifest.sha256(root / "snapshot/measurement.json"), checksum)
                self.assertFalse(any("fixture-secret" in path.read_text() for path in (root / "snapshot").rglob("*") if path.is_file()))
                self.assertEqual(report["settings"]["speaker_count"], 1)


if __name__ == "__main__":
    unittest.main()
