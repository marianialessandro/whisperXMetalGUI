import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import corpus, import_ami


class AMIManualImportTests(unittest.TestCase):
    def archive(self, root, entity=False, missing_identity=False):
        path = root / "manual.zip"
        identity = "" if missing_identity else 'global_name="person-a"'
        metadata = '<root><meeting observation="ES2002a"><speaker nxt_agent="A" ' + identity + '/><speaker nxt_agent="B" global_name="person-b"/></meeting></root>'
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("corpusResources/meetings.xml", '<!DOCTYPE root [<!ENTITY x "bad">]>' + metadata if entity else metadata)
            for agent, start, end in [("A", 2, 4), ("B", 3, 5)]:
                archive.writestr("words/ES2002a." + agent + ".words.xml", '<root><w starttime="' + str(start) + '" endtime="' + str(start + .2) + '">Hello</w><w punc="true">.</w></root>')
                archive.writestr("segments/ES2002a." + agent + ".segments.xml", '<root><segment transcriber_start="' + str(start) + '" transcriber_end="' + str(end) + '"/></root>')
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_overlap_and_global_speaker_identity_are_preserved_without_certification(self):
        with tempfile.TemporaryDirectory() as directory:
            path, digest = self.archive(Path(directory))
            reference = import_ami.import_manual(path, digest, "ES2002a", "b" * 64, 10.)
            self.assertEqual([(turn["start"], turn["end"]) for turn in reference["turns"]], [(2., 4.), (3., 5.)])
            self.assertEqual(reference["speaker_ids"], sorted("ami_person_" + hashlib.sha256(identity.encode()).hexdigest()[:16] for identity in ("person-a", "person-b")))
            self.assertEqual(reference["source_group"], "ami_ES2002")
            self.assertFalse(reference["reviewed"])
            self.assertFalse(reference["full_baseline_gate_passed"])
            with self.assertRaises(corpus.CorpusError): corpus.validate_annotation(reference)

    def test_entities_and_missing_speaker_identity_are_rejected(self):
        for entity, missing in [(True, False), (False, True)]:
            with self.subTest(entity=entity, missing=missing), tempfile.TemporaryDirectory() as directory:
                path, digest = self.archive(Path(directory), entity, missing)
                with self.assertRaises(corpus.CorpusError): import_ami.import_manual(path, digest, "ES2002a", "b" * 64, 10.)

    def test_hash_drift_and_audio_interval_overrun_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, digest = self.archive(Path(directory))
            with self.assertRaises(corpus.CorpusError): import_ami.import_manual(path, "c" * 64, "ES2002a", "b" * 64, 10.)
            with self.assertRaises(corpus.CorpusError): import_ami.import_manual(path, digest, "ES2002a", "b" * 64, 3.)


if __name__ == "__main__":
    unittest.main()
