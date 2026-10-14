import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import adapter
from tools.evaluation import voice_adapter


class Frame:
    def __init__(self, rows): self.rows = rows
    def __getitem__(self, columns):
        return Frame([{key: row[key] for key in columns} for row in self.rows])
    def to_dict(self, orientation):
        assert orientation == "records"
        return self.rows


class VoiceCaptureTests(unittest.TestCase):
    def test_retains_actual_overlapping_turns_and_restores_boundary_without_mutation(self):
        rows = [{"start": 0, "end": 2, "speaker": "A", "probability": 0.9}, {"start": 1, "end": 3, "speaker": "B", "probability": 0.8}]
        frame = Frame(rows)
        class Pipeline:
            def call(self): return frame
        original = Pipeline.call
        retained = []
        with voice_adapter.capture(lambda request: [(Pipeline, "call", "diarization")], retained):
            adapter.boundaries({"voices": True})
            self.assertIs(Pipeline().call(), frame)
        self.assertIs(Pipeline.call, original)
        self.assertEqual(retained, [[{"start": 0.0, "end": 2.0, "speaker": "A"}, {"start": 1.0, "end": 3.0, "speaker": "B"}]])
        self.assertEqual(rows[0]["probability"], 0.9)

    def test_incomplete_and_invalid_turns_fail_without_inventing_dominant_labels(self):
        with self.assertRaisesRegex(ValueError, "turns_missing"): voice_adapter.serialize_turns(Frame([]))
        for row in ({"start": 0, "end": float("nan"), "speaker": "A"}, {"start": 2, "end": 1, "speaker": "A"}, {"start": 0, "end": 1, "speaker": ""}):
            with self.assertRaisesRegex(ValueError, "turn_invalid"): voice_adapter.serialize_turns(Frame([row]))
        with voice_adapter.capture(lambda request: [], []):
            with self.assertRaisesRegex(ValueError, "requires_diarization"): adapter.boundaries({"voices": False})
            with self.assertRaisesRegex(ValueError, "boundary_missing"): adapter.boundaries({"voices": True})

    def test_completed_adapter_captures_once_exclusively_and_failed_adapter_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "turns.json"
            with patch.object(adapter, "main", return_value=2):
                self.assertEqual(voice_adapter.main(["--speaker-turns", str(target)]), 2)
                self.assertFalse(target.exists())
            class Pipeline:
                def call(self): return Frame([{"start": 0, "end": 2, "speaker": "A"}])
            def execute(arguments):
                adapter.boundaries({"voices": True})
                Pipeline().call()
                return 0
            with patch.object(adapter, "boundaries", return_value=[(Pipeline, "call", "diarization")]), patch.object(adapter, "main", side_effect=execute):
                self.assertEqual(voice_adapter.main(["--speaker-turns", str(target)]), 0)
                self.assertEqual(json.loads(target.read_text()), [{"start": 0.0, "end": 2.0, "speaker": "A"}])
                with self.assertRaisesRegex(ValueError, "destination_exists"): voice_adapter.main(["--speaker-turns", str(target)])


if __name__ == "__main__":
    unittest.main()
