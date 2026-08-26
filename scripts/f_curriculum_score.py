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


def episode_score(
    row: dict[str, str],
    *,
    cross_reward: float,
    handoff_bonus: float,
    turnin_scale: float,
    threat_penalty: float,
    incoming_damage_scale: float,
    damage_scale: float,
    stalemate_penalty: float,
) -> dict[str, float] | None:
    keys = (
        "iter", "ep_len_mean", "loss_rate", "crash_rate", "ep_wez_steps",
        "ep_reward_merge_cross", "ep_reward_merge_handoff_once",
        "ep_reward_merge_turnin", "ep_reward_merge_stalemate",
        "ep_reward_incoming_threat", "ep_reward_incoming_damage",
        "ep_reward_damage", "action_sat_rate",
    )
    values = {key: number(row, key) for key in keys}
    if any(value is None for value in values.values()):
        return None

    length = max(values["ep_len_mean"], 1.0)
    cross = clamp(values["ep_reward_merge_cross"] / cross_reward)
    handoff = clamp(values["ep_reward_merge_handoff_once"] / handoff_bonus)
    turnin = clamp(values["ep_reward_merge_turnin"] / (turnin_scale * 40.0))
    threat_clear = 1.0 - clamp(
        abs(values["ep_reward_incoming_threat"]) / (threat_penalty * length)
    )
    # Five percent own-health loss maps to zero.  The promotion gate below is
    # deliberately much stricter and therefore rejects even small mutual hits.
    incoming_clear = 1.0 - clamp(
        abs(values["ep_reward_incoming_damage"])
        / (incoming_damage_scale * 0.05)
    )
    wez = clamp(values["ep_wez_steps"] / 12.0)
    positive_net_damage = clamp(
        max(0.0, values["ep_reward_damage"]) / (damage_scale * 0.05)
    )
    offense = 0.65 * wez + 0.35 * positive_net_damage
    survival = clamp(1.0 - values["loss_rate"] - values["crash_rate"])
    saturation_clear = clamp(1.0 - values["action_sat_rate"])
    stalemate_clear = 1.0 - clamp(
        abs(values["ep_reward_merge_stalemate"])
        / max(stalemate_penalty * length, 1e-6)
    )
    score = (
        0.12 * cross
        + 0.18 * handoff
        + 0.10 * turnin
        + 0.18 * threat_clear
        + 0.18 * incoming_clear
        + 0.10 * offense
        + 0.10 * survival
        + 0.02 * saturation_clear
        + 0.02 * stalemate_clear
    )
    return {
        "iter": values["iter"],
        "score": score,
        "cross": cross,
        "handoff": handoff,
        "turnin": turnin,
        "threat_clear": threat_clear,
        "incoming_clear": incoming_clear,
        "offense": offense,
        "survival": survival,
        "saturation_clear": saturation_clear,
        "stalemate_clear": stalemate_clear,
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
            if (score := episode_score(
                row,
                cross_reward=6.0,
                handoff_bonus=16.0,
                turnin_scale=0.45,
                threat_penalty=0.8,
                incoming_damage_scale=240.0,
                damage_scale=180.0,
                stalemate_penalty=0.25,
            )) is not None
        ]
    if not scored:
        return {"ready": False, "reason": "no completed F metrics", "path": str(path)}

    metric_keys = tuple(key for key in scored[0] if key != "iter")
    windows: list[dict[str, float]] = []
    for end in range(WINDOW, len(scored) + 1):
        sample = scored[end - WINDOW:end]
        windows.append({
            "iter": sample[-1]["iter"],
            **{key: mean(sample, key) for key in metric_keys},
        })

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
        and current["cross"] >= 0.50
        and current["handoff"] >= 0.35
        and current["offense"] >= 0.10
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
    parser = argparse.ArgumentParser(description="Score an F-series frontal-merge run.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--threshold", type=float, default=0.74)
    parser.add_argument("--min-iteration", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(summarize(args.csv, args.threshold, args.min_iteration), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
