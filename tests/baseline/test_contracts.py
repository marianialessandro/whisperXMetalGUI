import copy
import ast
import json
import importlib.util
import io
from pathlib import Path
import sys
import unittest
import tempfile
import types
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import contracts


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads((Path(__file__).resolve().parents[1] / "fixtures/baseline/contracts.json").read_text())

    def test_valid_results_keep_untimed_words_and_overlap(self):
        for case in self.fixture["valid_results"]:
            contracts.validate_result(case)
        contracts.validate_artifacts({"original.json": self.fixture["valid_results"][0], "result.json": self.fixture["valid_results"][0]})

    def test_invalid_results_fail(self):
        for case in self.fixture["invalid_results"]:
            with self.subTest(case=case), self.assertRaises(contracts.ContractError):
                contracts.validate_result(case)
        invalid = {"segments": [{"start": float("nan"), "end": 1, "text": "invalid"}], "language": "it"}
        with self.assertRaises(contracts.ContractError):
            contracts.validate_result(invalid)

    def test_event_stream_requires_exactly_one_terminal_and_correct_paths(self):
        events = self.fixture["completed_events"]
        contracts.validate_events(events, "completed", output_root="/sandbox/run")
        contracts.validate_events([{"event": "failed", "message": "Accesso negato"}], "failed")
        contracts.validate_events(events[:1], "cancelled")
        contracts.validate_events(events[:1], "interrupted")
        for invalid in [[], events + [events[-1]], events + [events[0]], [{"event": "ready"}], [{"event": "failed"}]]:
            with self.assertRaises(contracts.ContractError):
                contracts.validate_events(invalid, "completed", output_root="/sandbox/run")
        changed = copy.deepcopy(events)
        changed[-1]["result"] = "/sandbox/run/../other/result.json"
        with self.assertRaises(contracts.ContractError):
            contracts.validate_events(changed, "completed", output_root="/sandbox/run")

    def test_semantic_comparison_only_normalizes_declared_paths(self):
        left = {"events": self.fixture["completed_events"], "result": self.fixture["valid_results"][1], "exports": {"txt": "ciao\n"}}
        right = copy.deepcopy(left)
        right["events"][-1]["output"] = "/different/run"
        right["events"][-1]["result"] = "/different/run/result.json"
        self.assertTrue(contracts.equivalent(left, right, left_root="/sandbox/run", right_root="/different/run"))
        for mutation in ["language", "text", "score", "speaker", "warnings", "export"]:
            changed = copy.deepcopy(right)
            if mutation == "language": changed["result"]["language"] = "en"
            elif mutation == "text": changed["result"]["segments"][0]["text"] = "diverso"
            elif mutation == "score": changed["result"]["segments"][0]["words"][0]["score"] = 0.1
            elif mutation == "speaker": changed["result"]["segments"][0]["speaker"] = "SPEAKER_99"
            elif mutation == "warnings": changed["events"][-1]["warnings"] = ["Allineamento non disponibile"]
            else: changed["exports"]["txt"] = "ciao\r\n"
            self.assertFalse(contracts.equivalent(left, changed, left_root="/sandbox/run", right_root="/different/run"), mutation)

    def test_profile_and_request_contract(self):
        self.assertEqual(contracts.PROFILES["Massima"], {"mlx": "mlx-community/whisper-large-v3-mlx", "cpu": "large-v3"})
        base = {"language": "auto", "quality": "Bilanciata", "voices": False, "hints": "è Unicode"}
        contracts.validate_settings(base)
        contracts.validate_settings(dict(base, voices=True, speaker_count=3))
        for patch in [{"quality": "turbo"}, {"voices": "yes"}, {"speaker_count": 0}, {"speaker_count": True}, {"language": ""}]:
            with self.assertRaises(contracts.ContractError):
                contracts.validate_settings(dict(base, **patch))

    def test_actual_worker_profile_maps_match_frozen_contract(self):
        root = Path(__file__).resolve().parents[2]
        tree = ast.parse((root / "macOS/Worker/transcribe_worker.py").read_text())
        maps = [ast.literal_eval(node.value) for node in ast.walk(tree) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "models" for target in node.targets)]
        self.assertEqual(len(maps), 2)
        for backend in ("mlx", "cpu"):
            self.assertIn({quality: mapping[backend] for quality, mapping in contracts.PROFILES.items()}, maps)

    def test_real_writers_match_hand_authored_export_bytes(self):
        spec = importlib.util.spec_from_file_location("baseline_writer_oracle", Path(__file__).resolve().parents[2] / "whisperx/utils.py")
        writers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(writers)
        options = {"max_line_width": None, "max_line_count": None, "highlight_words": False}
        for case in self.fixture["exports"]:
            for extension in ("txt", "srt", "vtt"):
                with self.subTest(extension=extension), tempfile.TemporaryDirectory() as directory:
                    writers.get_writer(extension, directory)(case["result"], "Audio è.wav", options)
                    self.assertEqual((Path(directory) / ("Audio è." + extension)).read_bytes(), case[extension].encode())
        result = copy.deepcopy(self.fixture["exports"][1]["result"])
        result["segments"][0]["words"] = [{"word": "Ciao"}, {"word": "è"}]
        for extension in ("srt", "vtt"):
            stream = io.StringIO()
            writers.get_writer(extension, "unused").write_result(result, stream, options)
            self.assertEqual(stream.getvalue(), self.fixture["exports"][1][extension])

    def test_original_worker_preserves_optional_failure_and_exact_count(self):
        root = Path(__file__).resolve().parents[2]
        worker_spec = importlib.util.spec_from_file_location("baseline_worker_contract", root / "macOS/Worker/transcribe_worker.py")
        worker = importlib.util.module_from_spec(worker_spec)
        worker_spec.loader.exec_module(worker)
        writer_spec = importlib.util.spec_from_file_location("whisperx.utils", root / "whisperx/utils.py")
        writers = importlib.util.module_from_spec(writer_spec)
        writer_spec.loader.exec_module(writers)
        calls, events = [], []
        torch = types.ModuleType("torch")
        torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
        engine = types.ModuleType("whisperx")
        engine.__path__ = []
        engine.load_audio = lambda _: "synthetic_audio"
        result = {"segments": [{"start": 0, "end": 1, "text": "ciao"}], "language": "it"}
        def load_model(name, device, **options):
            calls.append((name, device, options))
            return types.SimpleNamespace(transcribe=lambda audio, batch_size: copy.deepcopy(result))
        engine.load_model = load_model
        def alignment_failure(**options):
            raise RuntimeError("sentinel-private-credential")
        engine.load_align_model = alignment_failure
        diarize = types.ModuleType("whisperx.diarize")
        class Pipeline:
            def __init__(self, **options): pass
            def __call__(self, audio, **bounds):
                calls.append(bounds)
                raise RuntimeError("sentinel-private-credential")
        diarize.DiarizationPipeline = Pipeline
        diarize.assign_word_speakers = lambda speakers, result: result
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"torch": torch, "whisperx": engine, "whisperx.utils": writers, "whisperx.diarize": diarize}), patch.object(worker, "emit", lambda kind, **values: events.append({"event": kind, **values})):
            audio = Path(directory) / "audio.wav"
            audio.write_bytes(b"synthetic")
            output = Path(directory) / "output"
            worker.run({"audio": str(audio), "output": str(output), "quality": "Massima", "language": "auto", "voices": True, "speaker_count": 3, "hints": "Giulia", "token": "sentinel-private-credential"})
            contracts.validate_events(events, "completed", str(output))
            artifacts = {name: json.loads((output / name).read_text()) for name in ("original.json", "aligned.json", "result.json")}
            contracts.validate_artifacts(artifacts)
            self.assertEqual(calls[0], ("large-v3", "cpu", {"compute_type": "int8", "language": None, "vad_method": "silero", "asr_options": {"initial_prompt": "Giulia"}}))
            self.assertEqual(calls[1], {"min_speakers": 3, "max_speakers": 3})
            self.assertEqual(len(events[-1]["warnings"]), 2)
            self.assertNotIn("sentinel-private-credential", json.dumps(events) + json.dumps(artifacts))


if __name__ == "__main__":
    unittest.main()
