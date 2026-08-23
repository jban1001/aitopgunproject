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


def episode_score(row: dict[str, str], danger_penalty: float) -> dict[str, float] | None:
    required = {
        key: number(row, key)
        for key in (
            "iter",
            "ep_len_mean",
            "loss_rate",
            "crash_rate",
            "final_ata_deg",
            "final_aa_deg",
            "ep_reward_danger_exposure",
            "ep_reward_damage",
        )
    }
    if any(value is None for value in required.values()):
        return None

    own_ata = abs(required["final_ata_deg"])
    target_ata = abs(180.0 - abs(required["final_aa_deg"]))
    exposure_steps = max(0.0, -required["ep_reward_danger_exposure"] / danger_penalty)
    exposure_clear = 1.0 - clamp(exposure_steps / max(required["ep_len_mean"], 1.0))
    threat_clear = clamp(target_ata / 25.0)
    handoff = clamp((120.0 - own_ata) / 60.0)
    survival = clamp(1.0 - required["loss_rate"] - required["crash_rate"])
    raw_net_damage = required["ep_reward_damage"] / 60.0
    damage = clamp(0.5 + 2.0 * raw_net_damage)
    score = (
        0.35 * threat_clear
        + 0.20 * handoff
        + 0.20 * exposure_clear
        + 0.20 * survival
        + 0.05 * damage
    )
    return {
        "iter": required["iter"],
        "score": score,
        "threat_clear": threat_clear,
        "handoff": handoff,
        "exposure_clear": exposure_clear,
        "survival": survival,
        "damage": damage,
    }


def mean(items: list[dict[str, float]], key: str) -> float:
    return sum(item[key] for item in items) / len(items)


def summarize(
    path: Path,
    threshold: float,
    danger_penalty: float,
    min_iteration: int,
) -> dict[str, object]:
    if not path.is_file():
        return {"ready": False, "reason": "training log missing", "path": str(path)}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        scored = [
            score
            for row in csv.DictReader(handle)
            if (score := episode_score(row, danger_penalty)) is not None
        ]
    if not scored:
        return {"ready": False, "reason": "no completed episode metrics", "path": str(path)}

    windows: list[dict[str, float]] = []
    for end in range(WINDOW, len(scored) + 1):
        sample = scored[end - WINDOW : end]
        windows.append(
            {
                "iter": sample[-1]["iter"],
                **{key: mean(sample, key) for key in (
                    "score", "threat_clear", "handoff", "exposure_clear", "survival", "damage"
                )},
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
    safety_ok = current["survival"] >= 0.95
    quality_ok = (
        current["score"] >= threshold
        and current["threat_clear"] >= 0.70
        and current["exposure_clear"] >= 0.75
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
    parser = argparse.ArgumentParser(description="Score a C-series crossing-defense run.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--threshold", type=float, default=0.72)
    parser.add_argument("--danger-penalty", type=float, default=1.5)
    parser.add_argument("--min-iteration", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(
        summarize(args.csv, args.threshold, args.danger_penalty, args.min_iteration),
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
