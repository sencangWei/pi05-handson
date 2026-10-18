"""Boundary checks: reweight transport without changing action windows or approvals."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_phase_tempo import phase_groups


class PhaseTempoTests(unittest.TestCase):
    def setUp(self):
        clips = [{"source_directory": f"corrections/source{i}", "metadata_sha256": str(i),
                  "start": 67 if i == 5 else 0, "stop": 97 if i == 5 else 30}
                 for i in range(19)]
        self.manifest = {"selected_source_ranges": clips, "original_train_episodes": 37,
                         "original_train_frames": 4400, "train_frames": 4400 + 19 * 30}
        self.selection = {"retained_old_clips": 5, "clips": [
            {"clip_index": i, "source_directory": clips[i]["source_directory"],
             "metadata_sha256": str(i), "raw_transport_ranges": [[72, 92]] if i == 5 else []}
            for i in range(5, 19)]}

    def test_full_horizon_stays_inside_reviewed_transport_and_source_offset(self):
        groups = phase_groups(self.manifest, self.selection)
        # Source clip5 starts at raw67. Only raw72..82 have all10 actions in72..91.
        low = {i for start, stop in groups[-1]["ranges"] for i in range(start, stop)}
        self.assertEqual(low, set(range(4555, 4566)))
        self.assertNotIn(4566, low)  # Its window would include raw92, outside transport.
        self.assertEqual(groups[1]["ranges"], [[4400, 4550]])  # Old corrections untouched.

    def test_exact_disjoint_cover_and_preserved_mixture(self):
        groups = phase_groups(self.manifest, self.selection)
        all_starts = []
        for group in groups:
            all_starts.extend(i for start, stop in group["ranges"] for i in range(start, stop))
        self.assertEqual(len(all_starts), len(set(all_starts)))
        self.assertEqual(set(all_starts), set(range(self.manifest["train_frames"])))
        self.assertEqual([g["virtual_count"] / 200000 for g in groups], [.70, .10, .19, .01])

    def test_rejects_approval_boundary_crossing_wrong_source_and_duplicate(self):
        cases = []
        bad = copy.deepcopy(self.selection)
        bad["clips"][0]["raw_transport_ranges"] = [[66, 80]]
        cases.append(bad)
        bad = copy.deepcopy(self.selection)
        bad["clips"][0]["raw_transport_ranges"] = [[90, 100]]
        cases.append(bad)
        bad = copy.deepcopy(self.selection)
        bad["clips"][0]["source_directory"] = "draft/not-approved"
        cases.append(bad)
        bad = copy.deepcopy(self.selection)
        bad["clips"].append(bad["clips"][0])
        cases.append(bad)
        for value in cases:
            with self.subTest(value=value["clips"][0]), self.assertRaises(ValueError):
                phase_groups(self.manifest, value)


if __name__ == "__main__":
    unittest.main()
