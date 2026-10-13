import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circe.feature_bank import load_feature_bank, save_feature_bank
from circe.matching import aggregate, match_query


class CoreTests(unittest.TestCase):
    def test_bank_roundtrip_and_matching(self):
        vectors = np.array([[1, 0], [0.9, 0.4358899], [0, 1]], dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        rows = [{"component_id": "component_A", "patch_path": "a"}, {"component_id": "component_A", "patch_path": "b"}, {"component_id": "component_B", "patch_path": "c"}]
        with tempfile.TemporaryDirectory() as temporary:
            bank, meta = Path(temporary) / "bank.npz", Path(temporary) / "metadata.json"
            save_feature_bank(vectors, rows, bank, meta)
            saved, saved_rows = load_feature_bank(bank, meta)
        result = match_query(np.array([1, 0], dtype=np.float32), saved, saved_rows, ["component_A", "component_B"], "top_k_mean", 3)
        self.assertEqual(result["predicted_component"], "component_A")
        self.assertAlmostEqual(aggregate(np.array([0.2, 0.9]), "max", 3), 0.9)


if __name__ == "__main__":
    unittest.main()
