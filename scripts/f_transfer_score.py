from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


WINDOW = 10


def number(row: dict[str, str], key: str) -> float | None:
    try:
        value = float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def episode_score(row: dict[str, str]) -> dict[str, float] | None:
    keys = (
        "iter",
        "ep_len_mean",
        "loss_rate",
        "crash_rate",
        "ep_wez_steps",
        "ep_reward_bridge_entry",
        "ep_reward_bridge_entry_once",
        "ep_reward_bridge_stalemate",
        "ep_reward_incoming_threat",
        "ep_reward_incoming_damage",
        "ep_reward_damage",
        "ep_reward_merge_cross",
        "ep_reward_merge_handoff_once",
        "action_sat_rate",
    )
    values = {key: number(row, key) for key in keys}
    if any(value is None for value in values.values()):
        return None

    length = max(values["ep_len_mean"], 1.0)
    transfer = clamp(values["ep_reward_bridge_entry_once"] / 12.0)
    transfer_dwell = clamp(values["ep_reward_bridge_entry"] / (0.25 * 40.0))
    threat_clear = 1.0 - clamp(
        abs(values["ep_reward_incoming_threat"]) / (0.8 * length)
    )
    incoming_clear = 1.0 - clamp(
        abs(values["ep_reward_incoming_damage"]) / (240.0 * 0.05)
    )
    survival = clamp(1.0 - values["loss_rate"] - values["crash_rate"])
    wez = clamp(values["ep_wez_steps"] / 12.0)
    positive_net_damage = clamp(max(0.0, values["ep_reward_damage"]) / (180.0 * 0.05))
    offense = 0.65 * wez + 0.35 * positive_net_damage
    saturation_clear = clamp(1.0 - values["action_sat_rate"])
    stalemate_clear = 1.0 - clamp(
        abs(values["ep_reward_bridge_stalemate"]) / max(0.15 * length, 1e-6)
    )

    # Cross and the old post-cross attack handoff are diagnostics only. F owns
    # safe frontal passage and delivery into the proven B/C selector envelope.
    cross = clamp(values["ep_reward_merge_cross"] / 6.0)
    old_handoff = clamp(values["ep_reward_merge_handoff_once"] / 16.0)
    score = (
        0.24 * threat_clear
        + 0.24 * incoming_clear
        + 0.16 * survival
        + 0.16 * transfer
        + 0.06 * transfer_dwell
        + 0.06 * offense
        + 0.04 * saturation_clear
        + 0.04 * stalemate_clear
    )
    return {
        "iter": values["iter"],
        "score": score,
        "transfer": transfer,
        "transfer_dwell": transfer_dwell,
        "threat_clear": threat_clear,
        "incoming_clear": incoming_clear,
        "survival": survival,
        "offense": offense,
        "saturation_clear": saturation_clear,
        "stalemate_clear": stalemate_clear,
        "cross_diagnostic": cross,
        "old_handoff_diagnostic": old_handoff,
    }


def mean(items: list[dict[str, float]], key: str) -> float:
    return sum(item[key] for item in items) / len(items)


def summarize(path: Path, threshold: float, min_iteration: int) -> dict[str, object]:
    if not path.is_file():
        return {"ready": False, "reason": "training log missing", "path": str(path)}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        scored = [
            score
            for row in csv.DictReader(handle)
            if (score := episode_score(row)) is not None
        ]
    if not scored:
        return {"ready": False, "reason": "no completed F-transfer metrics", "path": str(path)}

    metric_keys = tuple(key for key in scored[0] if key != "iter")
    windows: list[dict[str, float]] = []
    for end in range(WINDOW, len(scored) + 1):
        sample = scored[end - WINDOW : end]
        windows.append(
            {
                "iter": sample[-1]["iter"],
                **{key: mean(sample, key) for key in metric_keys},
            }
        )

    current_iter = int(scored[-1]["iter"])
    if not windows:
        return {
            "ready": False,
            "reason": f"need {WINDOW} completed episode metrics",
            "current_iter": current_iter,
            "valid_episodes": len(scored),
        }
    eligible = [item for item in windows if item["iter"] >= min_iteration]
    best = max(eligible or windows, key=lambda item: item["score"])
    current = windows[-1]
    safety_ok = (
        current["survival"] >= 0.95
        and current["incoming_clear"] >= 0.95
        and current["threat_clear"] >= 0.90
    )
    quality_ok = (
        current["score"] >= threshold
        and current["transfer"] >= 0.35
        and current["transfer_dwell"] >= 0.20
        and safety_ok
    )
    return {
        "ready": True,
        "current_iter": current_iter,
        "valid_episodes": len(scored),
        "current": current,
        "best": best,
        "plateau_age": current_iter - int(best["iter"]),
        "quality_ok": quality_ok,
        "safety_ok": safety_ok,
        "threshold": threshold,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Score an F-series safe-transfer run.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--threshold", type=float, default=0.82)
    parser.add_argument("--min-iteration", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(summarize(args.csv, args.threshold, args.min_iteration), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
