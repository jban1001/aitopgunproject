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
    entry_scale: float,
    entry_bonus: float,
    stalemate_penalty: float,
) -> dict[str, float] | None:
    required = {
        key: number(row, key)
        for key in (
            "iter",
            "ep_len_mean",
            "loss_rate",
            "crash_rate",
            "final_ata_deg",
            "final_aa_deg",
            "ep_reward_bridge_entry",
            "ep_reward_bridge_entry_once",
            "ep_reward_bridge_stalemate",
            "action_sat_rate",
        )
    }
    if any(value is None for value in required.values()):
        return None

    own_ata = abs(required["final_ata_deg"])
    target_ata = abs(180.0 - abs(required["final_aa_deg"]))
    success = clamp(required["ep_reward_bridge_entry_once"] / entry_bonus)
    dwell = clamp(required["ep_reward_bridge_entry"] / (entry_scale * 40.0))
    gun_quality = clamp((70.0 - own_ata) / 40.0)
    position_quality = 0.5 * (
        clamp((100.0 - own_ata) / 40.0)
        + clamp((target_ata - 45.0) / 45.0)
    )
    final_handoff = max(gun_quality, position_quality)
    survival = clamp(1.0 - required["loss_rate"] - required["crash_rate"])
    saturation_clear = clamp(1.0 - required["action_sat_rate"])
    stalemate_clear = 1.0 - clamp(
        abs(required["ep_reward_bridge_stalemate"])
        / max(stalemate_penalty * required["ep_len_mean"], 1e-6)
    )
    score = (
        0.35 * success
        + 0.20 * dwell
        + 0.15 * final_handoff
        + 0.20 * survival
        + 0.05 * saturation_clear
        + 0.05 * stalemate_clear
    )
    return {
        "iter": required["iter"],
        "score": score,
        "success": success,
        "dwell": dwell,
        "final_handoff": final_handoff,
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
            if (
                score := episode_score(
                    row,
                    entry_scale=0.25,
                    entry_bonus=12.0,
                    stalemate_penalty=0.15,
                )
            )
            is not None
        ]
    if not scored:
        return {"ready": False, "reason": "no completed episode metrics", "path": str(path)}

    keys = (
        "score",
        "success",
        "dwell",
        "final_handoff",
        "survival",
        "saturation_clear",
        "stalemate_clear",
    )
    windows: list[dict[str, float]] = []
    for end in range(WINDOW, len(scored) + 1):
        sample = scored[end - WINDOW : end]
        windows.append({"iter": sample[-1]["iter"], **{key: mean(sample, key) for key in keys}})

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
        and current["success"] >= 0.45
        and current["dwell"] >= 0.25
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
    parser = argparse.ArgumentParser(description="Score a B-series selector-bridge run.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--threshold", type=float, default=0.68)
    parser.add_argument("--min-iteration", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(summarize(args.csv, args.threshold, args.min_iteration), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
