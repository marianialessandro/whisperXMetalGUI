import json
from pathlib import Path
import sys
import types
import subprocess
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline.adapter import observe


class AdapterTests(unittest.TestCase):
    def test_hooks_preserve_arguments_result_identity_and_restore_on_success(self):
        result, calls, spans = object(), [], []
        def original(*args, **kwargs):
            calls.append((args, kwargs))
            return result
        module = types.SimpleNamespace(run=original)
        with observe([(module, "run", "asr")], spans):
            self.assertIs(module.run("audio", language="it", hints="Giulia"), result)
        self.assertIs(module.run, original)
        self.assertEqual(calls, [(('audio',), {"language": "it", "hints": "Giulia"})])
        self.assertEqual(spans[0]["status"], "returned")
        self.assertFalse(spans[0]["gpu_completion_observed"])
        self.assertGreaterEqual(spans[0]["api_wall_ns"], 0)

    def test_exception_is_unchanged_hooks_restore_and_spans_never_log_it(self):
        exception = RuntimeError("sentinel-private-credential")
        def original():
            raise exception
        module = types.SimpleNamespace(run=original)
        spans = []
        with self.assertRaises(RuntimeError) as raised:
            with observe([(module, "run", "alignment")], spans):
                module.run()
        self.assertIs(raised.exception, exception)
        self.assertIs(module.run, original)
        self.assertEqual(spans[0]["status"], "failed")
        self.assertNotIn("sentinel-private-credential", json.dumps(spans))

    def test_partial_hook_setup_failure_restores_previous_hook(self):
        original = lambda: None
        module = types.SimpleNamespace(run=original)
        with self.assertRaises(AttributeError):
            with observe([(module, "run", "asr"), (module, "missing", "missing")], []):
                pass
        self.assertIs(module.run, original)

    def test_synchronized_profiler_path_preserves_result_and_hook_lifetime(self):
        value = object()
        original = lambda: value
        module = types.SimpleNamespace(run=original)
        spans, stages = [], []
        def profiler(stage, call):
            stages.append(stage)
            return call(), {"stage": stage, "status": "synchronized", "gpu_dispatch_observed": False}
        with observe([(module, "run", "asr")], spans, profiler):
            self.assertIs(module.run(), value)
        self.assertIs(module.run, original)
        self.assertEqual(stages, ["asr"])
        self.assertEqual(spans[0]["status"], "synchronized")

    def test_private_failure_message_flag_redacts_credentials_and_default_stays_generic(self):
        source = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "worker.py"
            worker.write_text('def run(request):\n    raise RuntimeError("test failure " + request["token"])\n')
            for preserve in (False, True):
                spans = root / ("private.json" if preserve else "default.json")
                arguments = [sys.executable, "-I", "-B", str(source / "tools/baseline/adapter.py"), "--source-root", str(source), "--worker", str(worker), "--spans", str(spans)]
                if preserve: arguments.append("--preserve-failure-message")
                result = subprocess.run(arguments, input=json.dumps({"token": "test-private-token", "voices": False}), capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 2)
                event = json.loads(result.stdout)
                self.assertEqual(event["message"], "test failure [token]" if preserve else "worker_failed")
                self.assertNotIn("test-private-token", result.stdout + result.stderr + spans.read_text())


if __name__ == "__main__":
    unittest.main()
