"""Check the installed runtime without downloads or development dependencies."""

import json
import math
import struct
import wave
import os
from pathlib import Path
import subprocess
import sys
import tempfile

app = Path(sys.argv[1]).resolve()
runtime = app / "Contents/Runtime"
sys.path.insert(0, str(app / "Contents/Resources"))
assert Path(sys.prefix).resolve() == runtime
import torch
import torchaudio
import torchvision
import ctranslate2
import faster_whisper
import mlx.core
import mlx_whisper
import pyannote.audio
import whisperx
from whisperx.alignment import load_align_model
from whisperx.diarize import DiarizationPipeline
from torchcodec.decoders import AudioDecoder

with tempfile.TemporaryDirectory(prefix="WhisperX audio ") as folder:
    audio = Path(folder) / "prova è audio.m4a"
    source = Path(folder) / "source.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setparams((1, 2, 16000, 16000, "NONE", "not compressed"))
        wav.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * sample / 16000))) for sample in range(16000)))
    subprocess.run([str(runtime / "bin/ffmpeg"), "-v", "error", "-i", str(source), "-c:a", "aac", str(audio)], check=True)
    decoded = whisperx.load_audio(str(audio))
    assert 15000 <= len(decoded) <= 18000
    assert AudioDecoder(str(audio)).get_all_samples().data.numel() > 0
    print(json.dumps({"runtime": "ready", "python": sys.version.split()[0], "torch": torch.__version__, "audioSamples": len(decoded), "metal": mlx.core.metal.is_available()}))
