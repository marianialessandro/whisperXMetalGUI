import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import acquire_ami, corpus, manifest


class Response(io.BytesIO):
    def __init__(self, payload, location):
        super().__init__(payload)
        self.location = location
        self.headers = {"Content-Length": str(len(payload))}

    def geturl(self):
        return self.location


class AMIAcquisitionTests(unittest.TestCase):
    def archive(self, root, leakage=False):
        meetings = []
        for index, meeting in enumerate(name for names in acquire_ami.MEETINGS.values() for name in names):
            identity = "person-0" if leakage and index == 3 else "person-" + str(index)
            meetings.append('<meeting observation="' + meeting + '"><speaker nxt_agent="A" global_name="' + identity + '"/></meeting>')
        path = root / "manual.zip"
        with zipfile.ZipFile(path, "w") as output:
            output.writestr("corpusResources/meetings.xml", "<root>" + "".join(meetings) + "</root>")
            for meeting in (name for names in acquire_ami.MEETINGS.values() for name in names):
                output.writestr("words/" + meeting + ".A.words.xml", '<root><w starttime="0" endtime="1">hello</w></root>')
                output.writestr("segments/" + meeting + ".A.segments.xml", '<root><segment transcriber_start="0" transcriber_end="1"/></root>')
        return path, manifest.sha256(path)

    def test_offline_plan_requires_disjoint_actual_metadata_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, digest = self.archive(root)
            report = acquire_ami.plan(archive, digest)
            self.assertEqual(len(report["recordings"]), 6)
            self.assertEqual([row["split"] for row in report["recordings"]], ["calibration"] * 3 + ["evaluation"] * 3)
            self.assertFalse(report["quality_certified"])
            archive, digest = self.archive(root, leakage=True)
            with self.assertRaisesRegex(ValueError, "split_leakage"): acquire_ami.plan(archive, digest)

    def test_unverified_license_and_existing_destination_stop_before_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, digest = self.archive(root)
            legal = root / "license.html"
            legal.write_bytes(b"Wrong scope Attribution 4.0")
            with patch.object(acquire_ami, "download", side_effect=AssertionError("unexpected download")):
                with self.assertRaisesRegex(ValueError, "ami_license_scope_invalid"):
                    acquire_ami.acquire(root, "snapshot", archive, digest, legal, manifest.sha256(legal))
                self.assertFalse((root / "snapshot").exists())
                (root / "snapshot").mkdir()
                with self.assertRaisesRegex(ValueError, "destination_exists"):
                    acquire_ami.acquire(root, "snapshot", archive, digest, legal, manifest.sha256(legal))

    def test_download_rejects_unverified_redirects_and_budget_overruns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            url = acquire_ami.BASE + "ES2002a/audio/ES2002a.Mix-Headset.wav"
            with self.assertRaisesRegex(ValueError, "ami_redirect_unverified"):
                acquire_ami.download(url, root / "wrong.wav", 64, lambda _: Response(b"a" * 64, "https://unverified.test/file"))
            self.assertFalse((root / "wrong.wav").exists())
            truncated = Response(b"a" * 64, url)
            truncated.headers["Content-Length"] = "65"
            with self.assertRaisesRegex(ValueError, "ami_response_truncated"):
                acquire_ami.download(url, root / "truncated.wav", 100, lambda _: truncated)
            with self.assertRaisesRegex(ValueError, "ami_audio_budget_exceeded"):
                acquire_ami.download(url, root / "large.wav", 64, lambda _: Response(b"a" * 65, url))
            acquire_ami.download(url, root / "valid.wav", 64, lambda _: Response(b"a" * 64, url))
            self.assertEqual((root / "valid.wav").stat().st_size, 64)
            with self.assertRaises(FileExistsError): acquire_ami.download(url, root / "valid.wav", 64, lambda _: Response(b"b" * 64, url))

    def test_concurrent_acquisition_publishes_only_verified_files_and_references(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, digest = self.archive(root)
            legal = root / "license.html"
            legal.write_bytes(b"The AMI corpus and its annotations are released under Attribution 4.0")
            def download(url, target, maximum): target.write_bytes(b"fixture audio " + url.encode())
            info = {"duration": 600, "sample_rate": 16000, "channels": 1, "codec": "pcm_s16le"}
            cache = root / "cache"
            cache.mkdir()
            with patch.object(acquire_ami, "download", download), patch.object(acquire_ami.acquire_fleurs, "audio_info", return_value=info):
                report = acquire_ami.acquire(root, "snapshot", archive, digest, legal, manifest.sha256(legal), download_workers=3, verified_audio_cache=cache)
            self.assertEqual(len(report["assets"]), 6)
            with patch.object(acquire_ami, "download", side_effect=AssertionError("must reuse verified cache")), patch.object(acquire_ami.acquire_fleurs, "audio_info", return_value=info):
                resumed = acquire_ami.acquire(root, "resumed", archive, digest, legal, manifest.sha256(legal), download_workers=3, verified_audio_cache=cache)
            self.assertEqual([item["sha256"] for item in report["assets"]], [item["sha256"] for item in resumed["assets"]])
            self.assertFalse(report["full_baseline_gate_passed"])
            for asset in report["assets"]:
                self.assertEqual(manifest.sha256(root / asset["path"]), asset["sha256"])
                reference = json.loads((root / asset["review_reference_path"]).read_text())
                self.assertFalse(reference["reviewed"])
                self.assertEqual(reference["reviewers"], [])
            with patch.object(acquire_ami, "download", download), patch.object(acquire_ami.acquire_fleurs, "audio_info", return_value=dict(info, duration=599)):
                with self.assertRaisesRegex(ValueError, "ami_audio_format_invalid"):
                    acquire_ami.acquire(root, "invalid", archive, digest, legal, manifest.sha256(legal), download_workers=2)
            self.assertFalse((root / "invalid").exists())

    def test_failed_group_retains_verified_cache_and_retry_downloads_only_missing_meeting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, digest = self.archive(root)
            legal = root / "license.html"
            legal.write_bytes(b"The AMI corpus and its annotations are released under Attribution 4.0")
            cache = root / "cache"
            cache.mkdir()
            calls = []
            def download(url, target, maximum):
                calls.append(url)
                target.write_bytes(b"fixture audio " + url.encode())
            info = {"duration": 600, "sample_rate": 16000, "channels": 1, "codec": "pcm_s16le"}
            def probe(target): return dict(info, duration=599) if target.name.startswith("ES2004a") else info
            with patch.object(acquire_ami, "download", download), patch.object(acquire_ami.acquire_fleurs, "audio_info", probe):
                with self.assertRaisesRegex(ValueError, "ami_audio_format_invalid"):
                    acquire_ami.acquire(root, "snapshot", archive, digest, legal, manifest.sha256(legal), download_workers=3, verified_audio_cache=cache)
            self.assertFalse((root / "snapshot").exists())
            self.assertTrue((cache / "ES2002a/receipt.json").exists())
            self.assertFalse((cache / "ES2004a").exists())
            calls.clear()
            with patch.object(acquire_ami, "download", download), patch.object(acquire_ami.acquire_fleurs, "audio_info", return_value=info):
                report = acquire_ami.acquire(root, "snapshot", archive, digest, legal, manifest.sha256(legal), download_workers=3, verified_audio_cache=cache)
            self.assertEqual(len(report["assets"]), 6)
            self.assertEqual(calls, [acquire_ami.BASE + "ES2004a/audio/ES2004a.Mix-Headset.wav"])
            def bad_copy(source, target): target.write_bytes(source.read_bytes() + b"changed during copy")
            with patch.object(acquire_ami.shutil, "copyfile", bad_copy), patch.object(acquire_ami.acquire_fleurs, "audio_info", return_value=info):
                with self.assertRaisesRegex(ValueError, "ami_cached_copy_changed"):
                    acquire_ami.acquire(root, "changed-copy", archive, digest, legal, manifest.sha256(legal), download_workers=3, verified_audio_cache=cache)
            self.assertFalse((root / "changed-copy").exists())
            cached_audio = cache / "ES2002a/ES2002a.Mix-Headset.wav"
            cached_audio.write_bytes(b"corrupt")
            with patch.object(acquire_ami, "download", side_effect=AssertionError("corrupt cache must not silently redownload")), patch.object(acquire_ami.acquire_fleurs, "audio_info", return_value=info):
                with self.assertRaisesRegex(ValueError, "ami_cache_audio_changed"):
                    acquire_ami.acquire(root, "corrupt", archive, digest, legal, manifest.sha256(legal), download_workers=3, verified_audio_cache=cache)
            self.assertFalse((root / "corrupt").exists())
