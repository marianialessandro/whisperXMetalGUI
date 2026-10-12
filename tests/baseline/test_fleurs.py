import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import acquire_fleurs as fleurs


class FleursTests(unittest.TestCase):
    def test_actual_acquisition_flow_preserves_raw_text_and_stays_parity_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = io.BytesIO()
            with wave.open(audio, "wb") as output:
                output.setparams((1, 2, 16000, 8, "NONE", "not compressed"))
                output.writeframes(b"\0" * 16)
            archive = io.BytesIO()
            with tarfile.open(fileobj=archive, mode="w:gz") as output:
                member = tarfile.TarInfo("dev/123.wav")
                member.size = len(audio.getvalue())
                output.addfile(member, io.BytesIO(audio.getvalue()))
            payload = archive.getvalue()
            card = b"license: cc-by-4.0"
            metadata = b"1\t123.wav\tCiao, Giulia!\tciao giulia\tcharacters\t8\tFEMALE\n"
            def read(url, limit=0):
                if url.endswith("README.md"): return card
                if url.endswith("license.shtml"): return b"Attribution 4.0"
                if url.endswith("dev.tsv"): return metadata
                return json.dumps([{"path": "data/it_it/audio/dev.tar.gz", "size": len(payload), "lfs": {"oid": hashlib.sha256(payload).hexdigest()}}]).encode()
            def acquire(asset, license, evidence, licenses):
                from tools.baseline.corpus import acquire as original
                return original(asset, license, evidence, licenses, lambda _: io.BytesIO(payload))
            info = {"sample_rate": 16000, "channels": 1, "duration": 8 / 16000, "codec": "pcm_s16le"}
            with patch.object(fleurs, "CARD_HASH", hashlib.sha256(card).hexdigest()), patch.object(fleurs, "read_public", read), patch.object(fleurs, "acquire", acquire), patch.object(fleurs, "audio_info", return_value=info):
                result = fleurs.acquire_language("it", root, 1)
                again = fleurs.acquire_language("it", root, 1)
            self.assertEqual(result, again)
            self.assertEqual(result["audio"][0]["transcript"], "Ciao, Giulia!")
            self.assertEqual(result["audio"][0]["normalized_transcript"], "ciao giulia")
            self.assertEqual(result["audio"][0]["type"], "parity_only")
            self.assertFalse(result["quality_certified"])

    def test_changed_license_card_stops_before_audio_acquisition(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(fleurs, "read_public", return_value=b"unverified"), patch.object(fleurs, "acquire") as acquisition:
            with self.assertRaises(ValueError):
                fleurs.acquire_language("it", Path(directory), 1)
            acquisition.assert_not_called()
