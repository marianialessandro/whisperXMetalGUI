import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import os
import signal
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import run, corpus


class RunnerTests(unittest.TestCase):
    def group_lifecycle(self, interrupt):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = Path(__file__).resolve().parents[1] / "fixtures/baseline/sleeping_group.py"
            children = []
            original = run.subprocess.Popen
            stop = threading.Event()
            def launch(*args, **kwargs):
                child = original(*args, **kwargs)
                children.append(child)
                return child
            def cancellation():
                deadline = time.monotonic() + 5
                while not stop.is_set() and time.monotonic() < deadline:
                    if (root / "parent-ready").exists():
                        if not stop.is_set(): os.kill(os.getpid(), signal.SIGINT)
                        return
                    stop.wait(0.01)
            thread = threading.Thread(target=cancellation, daemon=True) if interrupt else None
            if thread is not None: thread.start()
            try:
                with patch.object(run.subprocess, "Popen", launch), self.assertRaisesRegex(run.RunError, "job_cancelled" if interrupt else "job_interrupted"):
                    run.execute(Path(sys.executable), [fixture, "--markers", root], {"output": str(root / "output")}, 10 if interrupt else 1)
            finally:
                stop.set()
                if thread is not None: thread.join(timeout=6)
                for child in children:
                    if child.poll() is None: run.stop_child(child)
            self.assertTrue((root / "parent-stopped").exists())
            self.assertTrue((root / "grandchild-stopped").exists())
            self.assertTrue(all(child.poll() is not None for child in children))
            self.assertTrue(all(child.stdin.closed and child.stdout.closed for child in children))

    def test_timeout_terminates_the_real_process_group_and_closes_private_pipes(self):
        self.group_lifecycle(False)

    def test_interruption_terminates_the_real_process_group_and_closes_private_pipes(self):
        self.group_lifecycle(True)

    def test_timeout_cleans_descendants_even_after_the_group_leader_has_exited(self):
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = Path(__file__).resolve().parents[1] / "fixtures/baseline/sleeping_group.py"
            with self.assertRaisesRegex(run.RunError, "job_interrupted"):
                run.execute(Path(sys.executable), [fixture, "--markers", root, "--exit-parent"], {"output": str(root / "output")}, 1)
            descendant = int((root / "grandchild-ready").read_text())
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                status = subprocess.run(["/bin/ps", "-o", "stat=", "-p", str(descendant)], capture_output=True, text=True).stdout.strip()
                if not status or status.startswith("Z"): break
                time.sleep(0.01)
            self.assertTrue(not status or status.startswith("Z"), "descendant remains alive after leader exit")

    def test_component_roles_count_only_after_file_and_configuration_verification(self):
        spec = {"repository": "pyannote/speaker-diarization-community-1", "role": "speaker_segmentation", "cache_root": "/private/cache", "revision": "a" * 40, "license_verified": True, "license_source": "https://example.org/license"}
        checked = {"status": "files_verified", "roles": ["speaker_segmentation", "speaker_embedding", "speaker_plda"]}
        with tempfile.TemporaryDirectory() as directory, patch.object(run.manifest, "contained", side_effect=OSError()):
            arguments = ({"assets": [], "cases": []}, corpus.expected_coverage(Path(__file__).resolve().parents[2]), {"licenses": []}, Path(directory), [spec])
            with patch.object(run.provision, "inspect_spec", return_value=checked):
                report = run.preflight(*arguments)
            missing_roles = {item.get("role") for item in report["missing"] if item["code"] == "required_model_role_undeclared"}
            self.assertNotIn("speaker_embedding", missing_roles)
            self.assertNotIn("speaker_plda", missing_roles)
            self.assertIn("vad", missing_roles)
            with patch.object(run.provision, "inspect_spec", return_value={**checked, "status": "prerequisite_missing", "code": "model_component_config_mismatch"}):
                report = run.preflight(*arguments)
            self.assertIn({"code": "required_model_role_undeclared", "role": "speaker_embedding"}, report["missing"])

    def test_cache_routes_are_transport_specific_and_inventory_order_independent(self):
        specs = [{"transport": "torch_hub", "cache_root": "/private/torch"}, {"cache_root": "/private/hf"}, {"cache_root": "/private/hf"}]
        expected = (Path("/private/hf").resolve(), Path("/private/torch").resolve())
        self.assertEqual(run.model_cache_roots(specs), expected)
        self.assertEqual(run.model_cache_roots(list(reversed(specs))), expected)
        self.assertEqual(run.model_cache_roots(specs[:1]), (None, expected[1]))
        with self.assertRaisesRegex(run.RunError, "model_cache_topology_unresolved"):
            run.model_cache_roots(specs + [{"cache_root": "/private/other-hf"}])
        with self.assertRaisesRegex(run.RunError, "model_cache_transport_unsupported"):
            run.model_cache_roots([{"transport": "unknown", "cache_root": "/private/cache"}])

    def test_execute_passes_both_cache_roots_to_the_offline_child(self):
        import types
        received = []
        event = {"event": "completed", "output": "/private/output", "result": "/private/output/result.json", "warnings": []}
        child = types.SimpleNamespace(communicate=lambda payload, timeout: (json.dumps(event).encode(), None), returncode=0)
        def launch(arguments, **options):
            received.append(options["env"])
            return child
        with patch.object(run.subprocess, "Popen", launch):
            run.execute(Path("/runtime/bin/python"), [], {"output": "/private/output"}, 3, Path("/private/hf"), Path("/private/torch"))
        self.assertEqual(received[0]["HF_HUB_CACHE"], "/private/hf")
        self.assertEqual(received[0]["TORCH_HOME"], "/private/torch")

    def test_full_preflight_keeps_missing_corpus_and_models_explicit(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            report = run.preflight({"assets": [], "cases": []}, corpus.expected_coverage(root), {"licenses": []}, Path(directory), [])
        self.assertEqual(report["status"], "prerequisite_missing")
        self.assertIn({"code": "model_inventory_missing"}, report["missing"])
        self.assertFalse(report["gpu_placement_certified"])

    def test_torch_selection_requires_live_inventory_and_never_uses_huggingface_refs(self):
        spec = {"transport": "torch_hub", "repository": "torchaudio/BUNDLE", "role": "alignment_it", "cache_root": "/private/torch", "files": ["weights.pth"], "license_verified": True, "license_source": "https://example.org/license", "torchaudio_version": "2.8.0", "bundle_config": {"layers": 2}, "labels": ["-", "a"]}
        selection = {"status": "selection_inventory_verified", "models": [{**spec, "checkpoint": "weights.pth", "checkpoint_loaded": False, "status": "default_selection_and_config_verified"}]}
        with tempfile.TemporaryDirectory() as directory, patch.object(run.provision, "inspect_spec", return_value={"status": "files_verified"}), patch.object(run.manifest, "contained", side_effect=AssertionError("Torch inventories have no Hugging Face refs")):
            arguments = ({"assets": [], "cases": []}, corpus.expected_coverage(Path(__file__).resolve().parents[2]), {"licenses": []}, Path(directory), [spec])
            missing = run.preflight(*arguments)["missing"]
            self.assertIn({"code": "worker_alignment_selection_unverified", "repository": spec["repository"]}, missing)
            verified = run.preflight(*arguments, alignment_selection=selection)
            self.assertNotIn({"code": "worker_alignment_selection_unverified", "repository": spec["repository"]}, verified["missing"])
            self.assertEqual(verified["status"], "prerequisite_missing")
            self.assertIn({"case_id": "it_short", "code": "case_missing"}, verified["missing"])

    def test_paired_output_comparison_detects_export_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = {"quality": "Rapida", "language": "it", "voices": False, "hints": ""}
            audio = root / "sample.wav"
            audio.write_bytes(b"synthetic")
            def execute(runtime, arguments, request, timeout, cache_root=None):
                output = Path(request["output"])
                for name in ("original.json", "result.json"):
                    (output / name).write_text(json.dumps({"segments": [], "language": "it"}))
                for extension in ("txt", "srt", "vtt"):
                    (output / ("sample." + extension)).write_text("")
                if "instrumented" in output.parts:
                    (output / "sample.txt").write_text("changed")
                return [{"event": "completed", "output": str(output), "result": str(output / "result.json"), "warnings": []}]
            with patch.object(run, "execute", execute), self.assertRaises(run.RunError) as error:
                run.paired_run(Path("runtime"), Path("worker"), root, audio, settings, root / "pair")
            self.assertEqual(str(error.exception), "paired_semantic_mismatch")

    def test_missing_credential_never_launches_worker(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(run, "execute") as execute:
            root = Path(directory)
            with self.assertRaises(run.RunError):
                run.paired_run(Path("runtime"), Path("worker"), root, root / "sample.wav", {"quality": "Rapida", "language": "it", "voices": True}, root / "pair")
            execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
