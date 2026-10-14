import tempfile
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from whisperx.mlx_asr import MlxPipeline


class MlxCLITests(unittest.TestCase):
    def setUp(self):
        self.mx = MagicMock()
        self.engine = MagicMock()
        self.hub = MagicMock()
        self.hub.snapshot_download.return_value = "/cache/model"
        self.modules = patch.dict("sys.modules", {"mlx": MagicMock(core=self.mx), "mlx.core": self.mx, "mlx_whisper": self.engine, "huggingface_hub": self.hub})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_offline_snapshot_and_decoder_options_reach_metal_engine(self):
        pipeline = MlxPipeline("base", language="it", download_root="/cache", local_files_only=True, asr_options={"initial_prompt": "Antartide", "temperatures": [0.0, 0.2]})
        self.hub.snapshot_download.assert_called_once_with("mlx-community/whisper-base-mlx", cache_dir="/cache", local_files_only=True)
        audio = object()
        self.assertIs(pipeline.transcribe(audio), self.engine.transcribe.return_value)
        args, options = self.engine.transcribe.call_args
        self.assertIs(args[0], audio)
        self.assertEqual(options["path_or_hf_repo"], "/cache/model")
        self.assertEqual(options["temperature"], [0.0, 0.2])
        self.assertEqual(options["initial_prompt"], "Antartide")
        self.assertEqual(options["language"], "it")
        self.assertTrue(options["fp16"])
        self.mx.set_default_device.assert_called_once_with(self.mx.gpu)

    def test_local_model_and_float32_do_not_download(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = MlxPipeline(directory, compute_type="float32")
            pipeline.transcribe(object())
            self.assertEqual(pipeline.model_path, str(Path(directory).resolve()))
        self.hub.snapshot_download.assert_not_called()
        self.assertFalse(self.engine.transcribe.call_args.kwargs["fp16"])

    def test_no_gpu_and_unsupported_precision_fail_without_cpu_fallback(self):
        self.mx.metal.is_available.return_value = False
        with self.assertRaisesRegex(RuntimeError, "Metal GPU"):
            MlxPipeline("base")
        self.mx.metal.is_available.return_value = True
        with self.assertRaisesRegex(ValueError, "int8"):
            MlxPipeline("base", compute_type="int8")
        self.hub.snapshot_download.assert_not_called()
        self.engine.transcribe.assert_not_called()

    def test_incompatible_text_options_are_explicit_errors(self):
        for options in [{"hotwords": "example"}, {"suppress_numerals": True}]:
            with self.assertRaisesRegex(ValueError, "initial_prompt"):
                MlxPipeline("base", asr_options=options)
        pipeline = MlxPipeline("base")
        with self.assertRaisesRegex(ValueError, "interleaved_context"):
            pipeline.transcribe(object(), interleaved_context=True)
        self.engine.transcribe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
