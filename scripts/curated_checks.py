"""Observation gates for reviewed partial corrections, separate from task success."""
import numpy as np

from correction_data import approval_checks

KINDS = {"placement_release", "lift_clearance", "regrasp_lift"}


def curated_clip_checks(kind, frames):
    if kind not in KINDS or len(frames) < 10:
        raise ValueError("Unknown correction phase or fewer than ten frames")
    if kind == "placement_release":
        return {"kind": kind, "checks": approval_checks(frames),
                "meaning": "Placement, opening and retreat observation gates; not an independent benchmark run"}

    before = np.asarray([frame["state"] for frame in frames], dtype=float)
    after = np.asarray([frame["post_state"] for frame in frames], dtype=float)
    bowl = np.asarray([frame["post_bowl_pos"] for frame in frames], dtype=float)
    actions = np.asarray([frame["action"] for frame in frames], dtype=float)
    if (before.shape != (len(frames), 8) or after.shape != before.shape
            or bowl.shape != (len(frames), 3) or actions.shape != (len(frames), 7)
            or not all(np.isfinite(value).all() for value in (before, after, bowl, actions))):
        raise ValueError("Invalid finite partial-correction state/action shapes")
    gap = np.abs(after[:, 6:8]).sum(axis=1)
    if kind == "lift_clearance":
        measures = {
            "eef_lift_m": float(after[-1, 2] - before[0, 2]),
            "final_eef_bowl_distance_m": float(np.linalg.norm(after[-1, :3] - bowl[-1])),
            "bowl_height_change_m": float(bowl[-1, 2] - bowl[0, 2]),
            "tail_bowl_position_range_m": float(np.linalg.norm(np.ptp(bowl[-10:], axis=0))),
        }
        checks = {
            "upward_commands": bool(np.all(actions[:, 2] > 0)),
            "eef_lift_at_least_4cm": measures["eef_lift_m"] >= .04,
            "eef_clears_bowl_at_least_9cm": measures["final_eef_bowl_distance_m"] >= .09,
            "bowl_not_lifted": measures["bowl_height_change_m"] <= .005,
            "bowl_stable_last_ten": measures["tail_bowl_position_range_m"] <= .002,
        }
        meaning = "Upward disengagement while bowl stays behind; does not prove an autonomous failure detector"
    else:
        closes = np.flatnonzero(actions[:, 6] > 0)
        first_close = int(closes[0]) if len(closes) else 0
        reference = max(0, first_close - 1)
        measures = {
            "max_finger_gap_m": float(gap.max()),
            "bowl_lift_since_before_close_m": float(bowl[-1, 2] - bowl[reference, 2]),
            "eef_lift_since_close_m": float(after[-1, 2] - before[first_close, 2]),
            "tail_relative_position_range_m": float(np.linalg.norm(np.ptp(after[-10:, :3] - bowl[-10:], axis=0))),
        }
        checks = {
            "opens_before_close": bool(first_close > 0 and np.all(actions[:first_close, 6] < 0)),
            "fingers_opened_at_least_6cm": measures["max_finger_gap_m"] >= .06,
            "closed_commands_last_ten": bool(np.all(actions[-10:, 6] > 0)),
            "fingers_closed_last_ten": bool(np.all(gap[-10:] <= .04)),
            "eef_lift_at_least_2cm": measures["eef_lift_since_close_m"] >= .02,
            "bowl_lift_at_least_2cm": measures["bowl_lift_since_before_close_m"] >= .02,
            "bowl_moves_with_eef_last_ten": measures["tail_relative_position_range_m"] <= .01,
        }
        meaning = "Open, close and short object lift; not long-duration grasp stability or completed placement"
    if not all(checks.values()):
        raise ValueError(f"{kind} observation gates failed: " + ", ".join(key for key, value in checks.items() if not value))
    return {"kind": kind, "checks": checks, "measures": measures, "meaning": meaning,
            "official_task_success": False}
