import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baseline import profiling, metrics, freeze


class EvaluationDocumentTests(unittest.TestCase):
    def test_checked_in_expectations_match_code_and_remain_unmeasured(self):
        root = Path(__file__).resolve().parents[2] / "evaluation/baseline"
        operators = json.loads((root / "operators.json").read_text())
        self.assertEqual(operators["operators"], profiling.expected_operators())
        contract = json.loads((root / "evaluation-contract.json").read_text())
        self.assertEqual(contract["policy"], metrics.POLICY)
        self.assertEqual(contract["calibration_policy"], freeze.CALIBRATION_POLICY)
        self.assertEqual(contract["state"], "draft")
        self.assertIsNone(contract["thresholds"])
        self.assertIsNone(contract["resources"])
        with self.assertRaises(freeze.FreezeError):
            freeze.validate_contract(contract)


if __name__ == "__main__":
    unittest.main()
