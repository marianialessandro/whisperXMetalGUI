"""Apple Metal ASR adapter for the standalone WhisperX CLI."""

from pathlib import Path


class MlxPipeline:
    def __init__(self, model_name, language=None, task="transcribe", compute_type="default", download_root=None, local_files_only=False, asr_options=None):
        import mlx.core as mx
        import mlx_whisper
        from huggingface_hub import snapshot_download

        if not mx.metal.is_available():
            raise RuntimeError("The MLX ASR backend requires an Apple Metal GPU.")
        if compute_type not in {"default", "float16", "float32"}:
            raise ValueError("MLX supports --compute_type default, float16 or float32; int8 is not supported.")
        options = dict(asr_options or {})
        if options.get("hotwords") or options.get("suppress_numerals"):
            raise ValueError("MLX does not support --hotwords or --suppress_numerals; use --initial_prompt instead.")
        mx.set_default_device(mx.gpu)
        self.mx = mx
        self.engine = mlx_whisper
        self.fp16 = compute_type != "float32"
        self.language = language
        self.task = task
        local_model = Path(model_name).expanduser()
        repository = model_name if "/" in model_name else f"mlx-community/whisper-{model_name}-mlx"
        self.model_path = str(local_model.resolve()) if local_model.is_dir() else snapshot_download(repository, cache_dir=download_root, local_files_only=local_files_only)
        self.options = {"temperature": options.get("temperatures", [0.0]), "compression_ratio_threshold": options.get("compression_ratio_threshold", 2.4), "logprob_threshold": options.get("log_prob_threshold", -1.0), "no_speech_threshold": options.get("no_speech_threshold", 0.6), "condition_on_previous_text": options.get("condition_on_previous_text", False), "initial_prompt": options.get("initial_prompt"), "suppress_tokens": options.get("suppress_tokens", [-1])}

    def transcribe(self, audio, batch_size=8, chunk_size=30, print_progress=False, verbose=False, interleaved_context=False):
        if interleaved_context:
            raise ValueError("MLX does not support --interleaved_context.")
        print("ASR: MLX / Metal (GPU Apple)", flush=True)
        return self.engine.transcribe(audio, path_or_hf_repo=self.model_path, language=self.language, task=self.task, fp16=self.fp16, verbose=verbose if verbose else None, **self.options)

    def unload(self):
        from mlx_whisper.transcribe import ModelHolder
        ModelHolder.model = None
        ModelHolder.model_path = None
        self.mx.clear_cache()
