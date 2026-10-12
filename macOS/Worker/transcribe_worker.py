"""Private worker for the macOS app; requests arrive on stdin and events on stdout."""

import argparse
import contextlib
import gc
import json
import os
from pathlib import Path
import sys


def emit(kind, **values):
    sys.__stdout__.write(json.dumps({"event": kind, **values}, ensure_ascii=False) + "\n")
    sys.__stdout__.flush()


def save_result(path, result):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def run(request):
    import torch
    import whisperx
    from whisperx.utils import get_writer

    source = Path(request["audio"]).resolve(strict=True)
    output = Path(request["output"])
    output.mkdir(parents=True, exist_ok=True)
    language = request["language"] if request["language"] != "auto" else None
    quality = request["quality"]
    token = request.get("token")
    if request["voices"] and not token:
        raise ValueError("Collega Hugging Face per distinguere le voci, oppure disattiva questa opzione.")

    emit("stage", message="Preparo l’audio…")
    audio = whisperx.load_audio(str(source))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    use_mlx = False
    if device == "mps":
        try:
            import mlx.core as mx
            import mlx_whisper
            use_mlx = mx.metal.is_available()
        except ImportError:
            pass

    if use_mlx:
        models = {"Rapida": "mlx-community/whisper-base-mlx", "Bilanciata": "mlx-community/whisper-small-mlx", "Massima": "mlx-community/whisper-large-v3-mlx"}
        emit("stage", message="Trascrivo su GPU Apple. Al primo utilizzo scarico il modello…", backend="MLX / Metal")
        options = {"path_or_hf_repo": models[quality], "language": language, "verbose": False, "condition_on_previous_text": False}
        if request.get("hints"):
            options["initial_prompt"] = request["hints"]
        result = mlx_whisper.transcribe(audio, **options)
        from mlx_whisper.transcribe import ModelHolder
        ModelHolder.model = None
        ModelHolder.model_path = None
        gc.collect()
        mx.clear_cache()
    else:
        models = {"Rapida": "base", "Bilanciata": "small", "Massima": "large-v3"}
        emit("stage", message="Trascrivo su CPU. Al primo utilizzo scarico il modello…", backend="CTranslate2 / CPU")
        model = whisperx.load_model(models[quality], "cpu", compute_type="int8", language=language, vad_method="silero", asr_options={"initial_prompt": request.get("hints") or None})
        result = model.transcribe(audio, batch_size=4)
        del model
        gc.collect()

    language = result.get("language") or language
    result["language"] = language
    save_result(output / "original.json", result)
    warnings = []
    if result.get("segments"):
        emit("stage", message="Rendo più precisi i tempi delle parole…")
        try:
            model, metadata = whisperx.load_align_model(language_code=language, device=device)
            result = whisperx.align(result["segments"], model, metadata, audio, device)
            result["language"] = language
            del model, metadata
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
        except Exception:
            warnings.append("Allineamento non disponibile: ho mantenuto i tempi dei segmenti.")
        save_result(output / "aligned.json", result)

        if request["voices"]:
            emit("stage", message="Distinguo le voci. Al primo utilizzo scarico il modello…")
            try:
                from whisperx.diarize import DiarizationPipeline, assign_word_speakers
                pipeline = DiarizationPipeline(token=token, device=device)
                bounds = {}
                if request.get("speaker_count"):
                    bounds = {"min_speakers": request["speaker_count"], "max_speakers": request["speaker_count"]}
                speakers = pipeline(audio, **bounds)
                result = assign_word_speakers(speakers, result)
                del pipeline
            except Exception:
                warnings.append("Non sono riuscito a distinguere le voci. La trascrizione è comunque disponibile; verifica accesso al modello e connessione.")

    emit("stage", message="Salvo trascrizione e sottotitoli…")
    result["language"] = language
    save_result(output / "result.json", result)
    options = {"max_line_width": None, "max_line_count": None, "highlight_words": False}
    for format_name in ["txt", "srt", "vtt"]:
        get_writer(format_name, str(output))(result, str(source), options)
    emit("completed", result=str(output / "result.json"), output=str(output), warnings=warnings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, help="Optional WhisperX checkout used by a local development build.")
    args = parser.parse_args()
    bundled_sources = Path(__file__).resolve().parent
    if (bundled_sources / "whisperx").is_dir():
        sys.path.insert(0, str(bundled_sources))
    if args.source_root:
        sys.path.insert(0, str(args.source_root))
    request = json.load(sys.stdin)
    os.setpgrp()
    # Libraries write their diagnostics to stderr so that stdout remains a structured event stream.
    with contextlib.redirect_stdout(sys.stderr):
        try:
            run(request)
        except Exception as error:
            message = str(error)
            token = request.get("token")
            if token:
                message = message.replace(token, "[token]")
            emit("failed", message=message or "Il motore non è riuscito a completare la trascrizione.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
