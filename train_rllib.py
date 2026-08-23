from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

from ray.tune.registry import register_env

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
_pythonpath_entries = [str(ROOT), str(SRC)]
_existing_pythonpath = os.environ.get("PYTHONPATH", "")
if _existing_pythonpath:
    _pythonpath_entries.append(_existing_pythonpath)
os.environ["PYTHONPATH"] = os.pathsep.join(_pythonpath_entries)
_RAY_RAYLET_START_WAIT_TIME_S = "60"

from DogFightEnvWrapper import DogFightWrapper
from dogfight.ai.checkpoint_io import (
    apply_lightweight_policy_bundle,
    save_lightweight_policy_bundle,
)
from dogfight.ai.dashboard_logger import (
    DashboardJsonlLogger,
    copy_experiment_yaml,
    load_experiment_metadata,
)
from dogfight.ai.engagement_replay_logger import EngagementReplayLogger
from dogfight.ai.policy_probe_logger import PolicyProbeLogger
from dogfight.ai.rllib_utils import build_algorithm_config, normalize_algorithm_name
from dogfight.ai.rllib_utils import build_algorithm_from_bundle
from dogfight.ai.rl_action_provider import RLActionProvider
from dogfight.ai.student_hooks import load_observation_hook, load_reward_hook
from dogfight.ai.training.config_io import deep_update, load_experiment_env_config
from dogfight.ai.training_record import save_training_record
from dogfight.envs.initial_scenario import describe_initial_scenario
from dogfight.envs.observation import (
    observation_size as builtin_observation_size,
)


def _ensure_ray_runtime_env() -> None:
    """Restart Ray with local project paths available to worker actors."""
    import ray

    # 2026-05-29: Give slower PCs more time for raylet/GCS startup after shutdown.
    os.environ.setdefault("RAY_raylet_start_wait_time_s", _RAY_RAYLET_START_WAIT_TIME_S)
    ray.shutdown()
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        runtime_env={
            "env_vars": {
                "PYTHONPATH": os.environ["PYTHONPATH"],
                "RAY_raylet_start_wait_time_s": os.environ[
                    "RAY_raylet_start_wait_time_s"
                ],
            }
        },
    )


def env_creator(env_config):
    cfg = dict(env_config)
    cfg["_runner_index"] = getattr(
        env_config,
        "worker_index",
        cfg.get("_runner_index", "local"),
    )
    cfg["_env_index"] = getattr(
        env_config,
        "vector_index",
        cfg.get("_env_index", 0),
    )
    reward_fn = None
    observation_hook = None
    reward_module = str(cfg.get("reward_module", "")).strip()
    if reward_module:
        reward_fn, reward_config = load_reward_hook(reward_module)
        cfg.setdefault("reward", reward_config)
    observation_module = str(cfg.get("observation_module", "")).strip()
    if observation_module:
        observation_hook = load_observation_hook(observation_module)
        cfg["observation_mode"] = observation_hook["mode"]
        cfg["observation_module"] = observation_module
        cfg["observation_summary"] = observation_hook["description"]
    # Viewer-BT parity takes precedence over a frozen self-play opponent.
    target_action_provider = None
    ownship_preroll_provider = None
    if bool(cfg.get("target_viewer_bt", False)):
        if cfg.get("selfplay_opponent_pool") or cfg.get("selfplay_opponent_bundle"):
            raise ValueError("target_viewer_bt cannot be combined with a self-play opponent pool")
        from dogfight.ai.viewer_bt_action_provider import ViewerBTActionProvider

        target_action_provider = ViewerBTActionProvider(
            dll_name=str(cfg["target_behavior_dll"]),
            decision_period=int(cfg.get("target_bt_decision_period", 6)),
            rule_xml=str(cfg.get("target_behavior_rule_xml", "")).strip() or None,
            workspace_root=str(ROOT),
            reuse_tree_across_resets=bool(
                cfg.get("viewer_bt_reuse_tree_across_resets", False)
            ),
        )
    preroll = cfg.get("ownship_bt_preroll") or {}
    if bool(preroll.get("enabled", False)):
        if target_action_provider is None:
            raise ValueError("ownship_bt_preroll requires target_viewer_bt=true")
        from dogfight.ai.viewer_bt_action_provider import ViewerBTActionProvider

        ownship_preroll_provider = ViewerBTActionProvider(
            dll_name=str(preroll["dll_name"]),
            decision_period=int(preroll.get("decision_period", 9)),
            rule_xml=str(preroll.get("rule_xml", "")).strip() or None,
            workspace_root=str(ROOT),
            reuse_tree_across_resets=bool(
                preroll.get("reuse_tree_across_resets", False)
            ),
        )
    # selfplay_opponent_pool (list) = LEAGUE (random past gen per episode, stops cycling);
    # selfplay_opponent_bundle (str) = single frozen opponent.
    opp_pool = cfg.get("selfplay_opponent_pool") or []
    if isinstance(opp_pool, str):
        opp_pool = [s.strip() for s in opp_pool.split(",") if s.strip()]
    opp_bundle = str(cfg.get("selfplay_opponent_bundle", "")).strip()
    if opp_bundle and not opp_pool:
        opp_pool = [opp_bundle]
    if opp_pool and target_action_provider is None:
        root = Path(__file__).resolve().parent
        resolved = [str(p if (p := Path(b)).is_absolute() else root / b) for b in opp_pool]
        # Use a PLAIN-torch opponent (no RLlib Algorithm, no Ray): build_algorithm_from_bundle
        # calls ray.shutdown() and builds a full Algorithm, which cannot run inside a training
        # env-runner actor.  LightweightOpponentProvider reconstructs the SAC actor MLP and runs
        # a forward pass (verified bit-identical to the RLlib provider).
        from dogfight.ai.lightweight_opponent import LightweightOpponentProvider
        target_action_provider = LightweightOpponentProvider(
            bundle_dir=resolved,
            rl_pitch_sign=float(cfg.get("rl_pitch_sign", -1.0)),
        )

    env = DogFightWrapper(
        cfg,
        ownship_preroll_provider=ownship_preroll_provider,
        target_action_provider=target_action_provider,
        reward_fn=reward_fn,
        observation_fn=observation_hook["build_observation"] if observation_hook else None,
        observation_size=observation_hook["size"] if observation_hook else None,
        observation_low=observation_hook["low"] if observation_hook else None,
        observation_high=observation_hook["high"] if observation_hook else None,
    )
    if reward_module and "reward" in cfg:
        env.config["reward"] = dict(cfg["reward"])
    return env


# ── Metric helpers ────────────────────────────────────────────────────────────

def _extract_learner_stats(result: dict) -> dict:
    """Extract algorithm-level stats from RLlib result (new & old API)."""
    keys = (
        "policy_loss", "vf_loss", "entropy", "kl", "clip_frac", "explained_var",
        "actor_loss", "critic_loss", "alpha_loss", "alpha", "target_entropy",
        "curriculum_actor_frozen", "curriculum_fresh_steps",
        "curriculum_actor_update_applied", "curriculum_learner_updates",
        "curriculum_actor_anchor_mse", "curriculum_actor_anchor_loss",
        "curriculum_gunline_fraction", "curriculum_actor_anchor_fraction",
        "curriculum_residual_gate_fraction",
        "curriculum_residual_mean_abs_delta",
        "curriculum_counter_rate_fraction",
        "curriculum_counter_rate_logit_error",
        "curriculum_counter_rate_logit_loss",
        "replay_buffer_size", "replay_buffer_memory_mb", "env_steps_per_sec",
        "learner_steps_per_sec", "iteration_time_s",
    )
    stats = {k: "n/a" for k in keys}

    # New API (RLlib 2.x): result["learners"][module_id][...]
    learners = result.get("learners", {})
    if learners:
        ps = next(iter(learners.values()), {})
        stats["policy_loss"]   = ps.get("policy_loss", "n/a")
        stats["vf_loss"]       = ps.get("vf_loss", "n/a")
        stats["entropy"]       = ps.get("entropy", "n/a")
        stats["kl"]            = ps.get("mean_kl_loss", ps.get("kl", "n/a"))
        stats["clip_frac"]     = ps.get("clip_frac", "n/a")
        stats["explained_var"] = ps.get("vf_explained_var", "n/a")
        _fill_optional_learner_stats(stats, ps, result)
        return stats

    # Old API: result["info"]["learner"][policy_id][...]
    ps = next(iter(result.get("info", {}).get("learner", {}).values()), {})
    if ps:
        stats["policy_loss"]   = ps.get("policy_loss", "n/a")
        stats["vf_loss"]       = ps.get("vf_loss", "n/a")
        stats["entropy"]       = ps.get("entropy", "n/a")
        stats["kl"]            = ps.get("kl", "n/a")
        stats["clip_frac"]     = ps.get("clip_frac", "n/a")
        stats["explained_var"] = ps.get("vf_explained_var", "n/a")
        _fill_optional_learner_stats(stats, ps, result)
    else:
        _fill_optional_learner_stats(stats, {}, result)
    return stats


def _iter_nested_items(value: Any, prefix: str = "", depth: int = 0):
    """Yield flattened result items without expanding large arrays."""
    if depth > 8:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            path = f"{prefix}/{key_text}" if prefix else key_text
            yield from _iter_nested_items(item, path, depth + 1)
    elif isinstance(value, (list, tuple)) and len(value) <= 16:
        for index, item in enumerate(value):
            path = f"{prefix}/{index}" if prefix else str(index)
            yield from _iter_nested_items(item, path, depth + 1)
    else:
        yield prefix, value


def _find_nested_metric(source: Any, *names: str):
    """Find the first scalar metric whose final path segment matches names."""
    wanted = set(names)
    for path, value in _iter_nested_items(source):
        if not path:
            continue
        path_parts = {part.lower() for part in path.split("/")}
        if "config" in path_parts or "replay_buffer_config" in path_parts:
            continue
        key = path.rsplit("/", 1)[-1]
        if key not in wanted or value is None:
            continue
        if isinstance(value, (int, float)):
            return value
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        if hasattr(value, "shape") and getattr(value, "shape", None) == ():
            try:
                return float(value)
            except Exception:
                pass
    return "n/a"


def _fill_optional_learner_stats(stats: dict, policy_stats: dict, result: dict) -> None:
    """Populate SAC/performance stats when RLlib exposes them."""

    def first_present(*names: str):
        for source in (policy_stats, result):
            for name in names:
                if name in source and source[name] is not None:
                    return source[name]
        nested = _find_nested_metric(result, *names)
        if nested != "n/a":
            return nested
        return "n/a"

    stats["actor_loss"] = first_present("actor_loss", "policy_loss")
    stats["critic_loss"] = first_present("critic_loss", "qf_loss", "q_loss")
    stats["alpha_loss"] = first_present("alpha_loss")
    stats["alpha"] = first_present("alpha", "alpha_value")
    if stats["alpha"] == "n/a":
        log_alpha = first_present("log_alpha_value", "curr_log_alpha")
        if isinstance(log_alpha, (int, float)):
            stats["alpha"] = math.exp(float(log_alpha))
    stats["target_entropy"] = first_present("target_entropy")
    stats["curriculum_actor_frozen"] = first_present(
        "curriculum_actor_frozen"
    )
    stats["curriculum_fresh_steps"] = first_present(
        "curriculum_fresh_steps"
    )
    stats["curriculum_actor_update_applied"] = first_present(
        "curriculum_actor_update_applied"
    )
    stats["curriculum_learner_updates"] = first_present(
        "curriculum_learner_updates"
    )
    stats["curriculum_actor_anchor_mse"] = first_present(
        "curriculum_actor_anchor_mse"
    )
    stats["curriculum_actor_anchor_loss"] = first_present(
        "curriculum_actor_anchor_loss"
    )
    stats["curriculum_gunline_fraction"] = first_present(
        "curriculum_gunline_fraction"
    )
    stats["curriculum_actor_anchor_fraction"] = first_present(
        "curriculum_actor_anchor_fraction"
    )
    stats["curriculum_residual_gate_fraction"] = first_present(
        "curriculum_residual_gate_fraction"
    )
    stats["curriculum_residual_mean_abs_delta"] = first_present(
        "curriculum_residual_mean_abs_delta"
    )
    stats["curriculum_counter_rate_fraction"] = first_present(
        "curriculum_counter_rate_fraction"
    )
    stats["curriculum_counter_rate_logit_error"] = first_present(
        "curriculum_counter_rate_logit_error"
    )
    stats["curriculum_counter_rate_logit_loss"] = first_present(
        "curriculum_counter_rate_logit_loss"
    )
    stats["replay_buffer_size"] = first_present(
        "replay_buffer_size", "num_steps_trained_this_iter"
    )
    stats["env_steps_per_sec"] = first_present(
        "env_steps_per_sec", "num_env_steps_sampled_throughput_per_sec"
    )
    stats["learner_steps_per_sec"] = first_present(
        "learner_steps_per_sec", "num_env_steps_trained_throughput_per_sec"
    )
    stats["iteration_time_s"] = first_present("time_this_iter_s")


def _print_learner_result_debug(result: dict, iteration: int) -> None:
    """Print compact learner/result key hints when SAC loss metrics are missing."""
    interesting = (
        "learner",
        "learners",
        "loss",
        "alpha",
        "td_error",
        "num_env_steps_trained",
        "num_module_steps_trained",
    )
    matches = []
    for path, value in _iter_nested_items(result):
        lower_path = path.lower()
        if any(token in lower_path for token in interesting):
            shape = getattr(value, "shape", None)
            if shape is not None:
                summary = f"shape={tuple(shape)}"
            elif isinstance(value, (str, int, float, bool)) or value is None:
                summary = repr(value)
            else:
                summary = type(value).__name__
            matches.append(f"{path}={summary}")
        if len(matches) >= 80:
            break
    print(
        "[DogFightEnv][RLlibResult][LEARNER_KEYS] "
        f"iteration={iteration} count={len(matches)}"
    )
    for item in matches:
        print(f"[DogFightEnv][RLlibResult][LEARNER_KEYS] {item}")


def _fill_algorithm_runtime_stats(stats: dict, algorithm: Any) -> None:
    """Fill direct-loop stats that RLlib does not always place in result."""
    replay_buffer = getattr(algorithm, "local_replay_buffer", None)
    if replay_buffer is None:
        return

    if stats.get("replay_buffer_memory_mb") == "n/a":
        stats["replay_buffer_memory_mb"] = _estimate_object_memory_mb(replay_buffer)

    if stats.get("replay_buffer_size") != "n/a":
        return

    for getter_name in ("get_num_timesteps", "get_num_episodes"):
        getter = getattr(replay_buffer, getter_name, None)
        if getter is None:
            continue
        try:
            stats["replay_buffer_size"] = getter()
            return
        except Exception:
            pass

    try:
        stats["replay_buffer_size"] = len(replay_buffer)
    except Exception:
        pass


def _estimate_object_memory_mb(obj: Any) -> Any:
    """Estimate Python object memory recursively, returned in MiB."""
    if obj is None:
        return "n/a"

    seen: set[int] = set()

    def sizeof(value: Any, depth: int = 0) -> int:
        obj_id = id(value)
        if obj_id in seen:
            return 0
        seen.add(obj_id)

        size = sys.getsizeof(value, 0)
        nbytes = getattr(value, "nbytes", None)
        if isinstance(nbytes, int):
            size += nbytes
            return size
        if hasattr(value, "numel") and hasattr(value, "element_size"):
            try:
                size += int(value.numel()) * int(value.element_size())
                return size
            except Exception:
                return size
        if depth >= 8:
            return size

        if isinstance(value, dict):
            for key, item in value.items():
                size += sizeof(key, depth + 1)
                size += sizeof(item, depth + 1)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                size += sizeof(item, depth + 1)
        elif hasattr(value, "__dict__"):
            size += sizeof(vars(value), depth + 1)
        return size

    try:
        return sizeof(obj) / (1024.0 * 1024.0)
    except Exception:
        return "n/a"


def _parse_target_entropy(value):
    """"auto" stays a string; anything else becomes a float."""
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"--target-entropy must be 'auto' or a number, got {value!r}")


class _GrowingDictWriter:
    """csv.DictWriter that accepts keys it has not seen before.

    Reward components are discovered at runtime from whatever the reward function emits, so
    a fixed fieldname list guarantees that a newly added term is dropped without warning.
    New columns are appended to the header and the file is rewritten in place, which is
    cheap at one row per training iteration.
    """

    def __init__(self, handle, fields):
        self._handle = handle
        self._fields = list(fields)
        self._rows: list[dict] = []

    def writeheader(self) -> None:
        self._flush()

    def writerow(self, row: dict) -> None:
        new = [k for k in row if k not in self._fields]
        self._rows.append(dict(row))
        if new:
            self._fields.extend(sorted(new))
            self._flush()                      # header changed: rewrite everything
        else:
            self._write_row(row)
            self._handle.flush()

    def _write_row(self, row: dict) -> None:
        csv.DictWriter(self._handle, fieldnames=self._fields,
                       extrasaction="ignore").writerow(row)

    def _flush(self) -> None:
        self._handle.seek(0)
        self._handle.truncate()
        w = csv.DictWriter(self._handle, fieldnames=self._fields, extrasaction="ignore")
        w.writeheader()
        for r in self._rows:
            w.writerow(r)
        self._handle.flush()


def _extract_custom_metrics(result: dict) -> dict:
    """Extract DogFightCallbacks custom metrics from result (mean values)."""
    env_metrics = result.get("env_runners", {})
    cm = env_metrics.get("custom_metrics", {})

    def metric(name: str, default: Any = "n/a"):
        for metrics in (cm, result.get("custom_metrics", {})):
            for key in (f"{name}_mean", name):
                if key in metrics and metrics[key] is not None:
                    return metrics[key]
        return default

    collected = {
        "win_rate":             metric("win"),
        "loss_rate":            metric("loss"),
        "timeout_rate":         metric("timeout"),
        "crash_rate":           metric("crash"),
        "ep_wez_steps":         metric("ep_wez_steps"),
        "ep_mean_distance":     metric("ep_mean_distance"),
        "ep_min_distance":      metric("ep_min_distance"),
        "ep_reward_pursuit":    metric("ep_reward_pursuit"),
        "ep_reward_damage":     metric("ep_reward_damage"),
        "ep_reward_safety":     metric("ep_reward_safety"),
        "ep_reward_energy_throttle": metric("ep_reward_energy_throttle"),
        "ep_reward_survival":   metric("ep_reward_survival"),
        "ep_reward_merge_cross": metric("ep_reward_merge_cross"),
        "ep_reward_merge_handoff": metric("ep_reward_merge_handoff"),
        "ep_reward_merge_handoff_once": metric("ep_reward_merge_handoff_once"),
        "ep_reward_merge_turnin": metric("ep_reward_merge_turnin"),
        "ep_reward_reacquire_turnin": metric("ep_reward_reacquire_turnin"),
        "ep_reward_merge_stalemate": metric("ep_reward_merge_stalemate"),
        "ep_reward_attack_band": metric("ep_reward_attack_band"),
        "ep_reward_front_hold": metric("ep_reward_front_hold"),
        "ep_reward_aim_hold": metric("ep_reward_aim_hold"),
        "ep_reward_angle_advantage": metric("ep_reward_angle_advantage"),
        "ep_reward_angle_advantage_progress": metric("ep_reward_angle_advantage_progress"),
        "ep_reward_cutin_success": metric("ep_reward_cutin_success"),
        "ep_reward_cutin_control": metric("ep_reward_cutin_control"),
        "ep_reward_terminal_angle_gain": metric("ep_reward_terminal_angle_gain"),
        "ep_reward_terminal_ata_gain": metric("ep_reward_terminal_ata_gain"),
        "ep_reward_range_progress": metric("ep_reward_range_progress"),
        "ep_reward_lost": metric("ep_reward_lost"),
        "ep_altitude_penalty_steps": metric("ep_altitude_penalty_steps"),
        "initial_scenario_index": metric("initial_scenario_index"),
        "initial_alpha_deg":    metric("initial_alpha_deg"),
        "initial_ata_deg":      metric("initial_ata_deg"),
        "initial_aa_deg":       metric("initial_aa_deg"),
        "initial_distance_m":   metric("initial_distance_m"),
        "final_ata_deg":        metric("final_ata_deg"),
        "final_aa_deg":         metric("final_aa_deg"),
        "headon_guard_fail":    metric("headon_guard_fail"),
        "cutin_handback_success": metric("cutin_handback_success"),
        "bt_preroll_detected": metric("bt_preroll_detected"),
        "action_roll_mean":     metric("action_roll_mean"),
        "action_pitch_mean":    metric("action_pitch_mean"),
        "action_rudder_mean":   metric("action_rudder_mean"),
        "action_throttle_mean": metric("action_throttle_mean"),
        "action_roll_std":      metric("action_roll_std"),
        "action_pitch_std":     metric("action_pitch_std"),
        "action_rudder_std":    metric("action_rudder_std"),
        "action_throttle_std":  metric("action_throttle_std"),
        "action_sat_rate":      metric("action_saturation_rate"),
    }

    # DogFightCallbacks logs one custom metric per reward component, whatever the reward
    # function happens to emit. The list above is hand-maintained, so a newly added term is
    # silently invisible until someone remembers to extend it -- a whole run was once
    # analysed with the aiming terms reading "n/a" for exactly that reason. Pick up any
    # ep_reward_* that is present and not already named.
    for source in (cm, result.get("custom_metrics", {})):
        for key in source:
            name = key[:-5] if key.endswith("_mean") else key
            if name.startswith("ep_reward_") and name not in collected:
                collected[name] = metric(name)
    return collected


def _extract_progress_metrics(result: dict) -> dict:
    """Extract rollout progress metrics that exist before episodes complete."""
    env_metrics = result.get("env_runners", {})

    def first_present(mapping: dict, keys: tuple[str, ...], default="n/a"):
        for key in keys:
            if key in mapping and mapping[key] is not None:
                return mapping[key]
        return default

    return {
        "sampled_steps": first_present(
            env_metrics,
            ("num_env_steps_sampled_lifetime", "num_agent_steps_sampled_lifetime"),
        ),
        "episodes": first_present(
            env_metrics,
            ("num_episodes_lifetime", "num_episodes"),
        ),
    }


def _fmt(val, fmt=".4f"):
    return f"{val:{fmt}}" if isinstance(val, (int, float)) else str(val)


def _build_tune_progress_reporter(algorithm_name: str):
    """Build a Ray Tune table reporter with stable RLlib 2.x metric paths."""
    from ray.tune import CLIReporter

    metric_columns = {
        "training_iteration": "iter",
        "time_total_s": "time_s",
        "env_runners/num_env_steps_sampled_lifetime": "steps",
        "env_runners/num_episodes_lifetime": "eps",
        "env_runners/episode_return_mean": "reward",
        "env_runners/episode_len_mean": "len",
        "env_runners/custom_metrics/win_mean": "win",
        "env_runners/custom_metrics/crash_mean": "crash",
        "env_runners/custom_metrics/ep_wez_steps_mean": "wez",
    }
    if algorithm_name == "sac":
        metric_columns.update({
            "learners/default_policy/actor_loss": "actor",
            "learners/default_policy/critic_loss": "critic",
            "learners/default_policy/alpha_value": "alpha",
        })
    else:
        metric_columns.update({
            "learners/default_policy/entropy": "entropy",
            "learners/default_policy/vf_loss": "vf_loss",
            "learners/default_policy/mean_kl_loss": "kl",
        })
    return CLIReporter(
        metric_columns=metric_columns,
        max_report_frequency=2,
        print_intermediate_tables=True,
    )


def _console_header(algorithm_name: str) -> str:
    base = (
        f"{'Iter':>6} | {'Steps':>10} | {'Eps':>5} | "
        f"{'Reward':>10} | {'WinRate':>8} | {'WEZ_ep':>7} | "
    )
    if algorithm_name == "sac":
        return (
            base
            + f"{'Actor':>9} | {'Critic':>9} | {'Alpha':>8} | {'ReplayMB':>9} |"
        )
    return base + f"{'Entropy':>8} | {'VF_loss':>8} | {'KL':>7} |"


def _console_row(
    algorithm_name: str,
    iteration: int,
    progress: dict,
    reward_mean: Any,
    custom: dict,
    learner_stats: dict,
) -> str:
    base = (
        f"iter=[{iteration}] | "
        f"Steps=[{_fmt(progress['sampled_steps'], '.0f')}] | "
        f"Eps=[{_fmt(progress['episodes'], '.0f')}] | "
        f"Reward=[{_fmt(reward_mean, '.4f')}] | "
        f"WinRate=[{_fmt(custom['win_rate'], '.3f')}] | "
        f"WEZ_ep=[{_fmt(custom['ep_wez_steps'], '.1f')}] | "
    )
    if algorithm_name == "sac":
        return (
            base
            + f"Actor=[{_fmt(learner_stats['actor_loss'], '.4f')}] | "
            f"Critic=[{_fmt(learner_stats['critic_loss'], '.4f')}] | "
            f"Alpha=[{_fmt(learner_stats['alpha'], '.4f')}] | "
            f"ReplayMem=[{_fmt(learner_stats['replay_buffer_memory_mb'], '.1f')}MB]"
        )
    return (
        base
        + f"Entropy=[{_fmt(learner_stats['entropy'], '.4f')}] | "
        f"VF_loss=[{_fmt(learner_stats['vf_loss'], '.4f')}] | "
        f"KL=[{_fmt(learner_stats['kl'], '.4f')}]"
    )


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _resolve_dashboard_root(path: str) -> Path:
    root = Path(path)
    return root if root.is_absolute() else ROOT / root


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a single-agent dogfight policy with RLlib."
    )
    parser.add_argument(
        "--algorithm",
        choices=["ppo", "sac"],
        default="ppo",
        help="RLlib algorithm to use.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of training iterations.",
    )
    parser.add_argument(
        "--framework",
        default="torch",
        choices=["torch"],
        help="Deep learning framework.",
    )
    parser.add_argument(
        "--num-env-runners",
        type=int,
        default=1,
        help="Number of RLlib env runners.",
    )
    parser.add_argument(
        "--num-envs-per-env-runner",
        type=int,
        default=1,
        help="Number of vectorized envs per env runner.",
    )
    parser.add_argument(
        "--rollout-fragment-length",
        default="auto",
        help="RLlib rollout fragment length, or 'auto'.",
    )
    parser.add_argument(
        "--min-sample-timesteps-per-iteration",
        type=int,
        default=0,
        help="Minimum fresh environment timesteps collected by each train iteration.",
    )
    parser.add_argument(
        "--min-train-timesteps-per-iteration",
        type=int,
        default=0,
        help="Minimum learner timesteps completed by each train iteration.",
    )
    parser.add_argument(
        "--batch-mode",
        default="truncate_episodes",
        choices=["truncate_episodes", "complete_episodes"],
    )
    parser.add_argument(
        "--explore",
        dest="explore",
        action="store_true",
        default=None,
        help="Sample stochastic policy actions in training rollouts.",
    )
    parser.add_argument(
        "--no-explore",
        dest="explore",
        action="store_false",
        help="Use deterministic policy actions in training rollouts.",
    )
    parser.add_argument(
        "--observation-mode",
        default="tactical16",
        choices=["classic12", "relative14", "tactical16", "custom"],
    )
    parser.add_argument(
        "--observation-module",
        default="",
        help="Optional module with custom observation size and build_observation(...).",
    )
    parser.add_argument(
        "--target-mode",
        default="behavior_tree",
        choices=["behavior_tree", "fixed", "loiter", "maneuver", "autopilot"],
    )
    parser.add_argument("--target-behavior-dll", default="AIP_BASE_target.dll")
    parser.add_argument(
        "--reward-module",
        default="",
        help="Optional module with MY_REWARD_CONFIG and compute_reward(...).",
    )
    parser.add_argument("--max-engage-time", type=float, default=300.0)
    parser.add_argument("--episode-step-limit", type=int, default=18000)
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Write one console progress row every N training iterations.",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--actor-lr",
        type=float,
        default=None,
        help="Optional SAC actor learning-rate override.",
    )
    parser.add_argument(
        "--critic-lr",
        type=float,
        default=None,
        help="Optional SAC critic learning-rate override.",
    )
    parser.add_argument(
        "--alpha-lr",
        type=float,
        default=None,
        help="Optional SAC entropy-temperature learning-rate override.",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train-batch-size", type=int, default=4096)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-param", type=float, default=0.2)
    # PPO 전용. RLlib 기본값 10.0 은 가치함수 손실을 10 으로 자르는데, 이 환경의 에피소드
    # 리턴은 -537 ~ +1240 이다(N1 400 iter 실측). 기본값이면 가치함수가 학습되지 않는다.
    # 기본값을 RLlib 과 같게 두어 기존 SAC 실행에는 영향이 없다.
    parser.add_argument("--vf-clip-param", type=float, default=10.0)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--target-entropy", default="auto")
    parser.add_argument(
        "--training-intensity",
        type=float,
        default=None,
        help="SAC replayed-training timesteps per sampled environment timestep.",
    )
    parser.add_argument(
        "--initial-alpha",
        type=float,
        default=None,
        help="Optional SAC entropy coefficient used for a fresh learner.",
    )
    parser.add_argument(
        "--replay-buffer-capacity",
        type=int,
        default=None,
        help="SAC replay buffer capacity. Ignored by PPO.",
    )
    parser.add_argument(
        "--replay-buffer-config-json",
        default="",
        help="Optional JSON mapping passed to SAC replay_buffer_config.",
    )
    parser.add_argument(
        "--sac-actor-freeze-fresh-steps",
        type=int,
        default=0,
        help=(
            "After the first learner update, train only the SAC critic for this "
            "many fresh environment steps before releasing actor and alpha."
        ),
    )
    parser.add_argument(
        "--sac-actor-update-interval",
        type=int,
        default=1,
        help="Apply one SAC actor/alpha update per N critic updates.",
    )
    parser.add_argument(
        "--sac-actor-anchor-coeff",
        type=float,
        default=0.0,
        help=(
            "Penalize drift of the SAC actor mean from its post-restore "
            "snapshot on replay observations."
        ),
    )
    parser.add_argument(
        "--sac-actor-anchor-action-weights",
        default="",
        help=(
            "Comma-separated per-action anchor weights. For "
            "roll,pitch,rudder,throttle use 1,1,1,0 to free throttle."
        ),
    )
    parser.add_argument(
        "--sac-actor-anchor-bundle",
        default="",
        help=(
            "Optional lightweight bundle whose pi_encoder/pi remain the actor "
            "anchor across a resumed transfer run."
        ),
    )
    parser.add_argument(
        "--sac-gunline-logit-config-json",
        default="",
        help=(
            "JSON configuration for terminal gunline actor-logit soft limiting. "
            "This is a training loss, not an inference action cap."
        ),
    )
    parser.add_argument(
        "--sac-counter-rate-logit-config-json",
        default="",
        help=(
            "JSON configuration for the upright terminal counter-rate roll "
            "logit training target. This does not alter inference controls."
        ),
    )
    parser.add_argument(
        "--sac-gated-residual-config-json",
        default="",
        help=(
            "JSON configuration for a frozen champion SAC actor plus a "
            "state-gated trainable residual logit adapter."
        ),
    )
    parser.add_argument(
        "--sac-reset-alpha-after-restore",
        type=float,
        default=None,
        help="Reset SAC alpha to this value after native checkpoint restore.",
    )
    parser.add_argument(
        "--sac-reset-optimizers-after-restore",
        action="store_true",
        help="Clear restored SAC Adam moments while preserving network weights.",
    )
    parser.add_argument(
        "--model-fcnet-hiddens",
        default=None,
        help="Comma-separated RLlib model hidden sizes, e.g. 512,256,128.",
    )
    parser.add_argument(
        "--model-fcnet-activation",
        default=None,
        help="RLlib model encoder activation, e.g. relu or tanh.",
    )
    parser.add_argument(
        "--model-head-fcnet-hiddens",
        default=None,
        help="Comma-separated RLlib model head hidden sizes, or empty for none.",
    )
    parser.add_argument(
        "--model-head-fcnet-activation",
        default=None,
        help="RLlib model head activation, e.g. relu or tanh.",
    )
    parser.add_argument(
        "--model-vf-share-layers",
        default=None,
        help="Whether PPO value function shares layers: true or false.",
    )
    parser.add_argument(
        "--network-spec-json",
        default="",
        help=(
            "JSON object for DogFight sequence_v1 network layout. "
            "Usually supplied by scripts/run_experiment.py from algo.network."
        ),
    )
    parser.add_argument(
        "--use-lstm",
        action="store_true",
        help="Enable RLlib DefaultModelConfig LSTM for non-SAC algorithms such as PPO.",
    )
    parser.add_argument(
        "--use-lstm-sac",
        action="store_true",
        help="Enable the patched Ray 2.54 SAC actor-LSTM path.",
    )
    parser.add_argument(
        "--lstm-scope",
        choices=["actor_only", "actor_critic"],
        default="actor_only",
        help="SAC LSTM scope: actor_only or actor_critic recurrent Q.",
    )
    parser.add_argument(
        "--lstm-cell-size",
        type=int,
        default=64,
        help="LSTM hidden state size for --use-lstm-sac.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=8,
        help="Replay/train sequence length for --use-lstm-sac.",
    )
    parser.add_argument(
        "--debug-io",
        dest="debug_io",
        action="store_true",
        help="Print recurrent SAC/RLlib debug I/O shape checks.",
    )
    parser.add_argument(
        "--debug-lstm-io",
        dest="debug_io",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--use-lstm-prioritized-replay",
        action="store_true",
        help=(
            "Use patched PrioritizedEpisodeReplayBuffer sequence sampling for "
            "--use-lstm-sac. Requires the RLLibLstm replay patch."
        ),
    )
    parser.add_argument("--output-name", default="f16_single_agent")
    parser.add_argument("--output-tag", default="latest")
    parser.add_argument(
        "--notes",
        default="",
        help="Optional free-text notes for this training run.",
    )
    parser.add_argument(
        "--save-lightweight-bundle",
        dest="save_lightweight_bundle",
        action="store_true",
        default=True,
        help="Save lightweight policy bundles for inference.",
    )
    parser.add_argument(
        "--no-save-lightweight-bundle",
        dest="save_lightweight_bundle",
        action="store_false",
        help="Disable lightweight policy bundle saves.",
    )
    parser.add_argument(
        "--lightweight-bundle-frequency",
        type=int,
        default=0,
        help=(
            "Save a lightweight bundle every N direct-loop iterations. "
            "0 means final bundle only."
        ),
    )
    parser.add_argument(
        "--save-native-checkpoint",
        action="store_true",
        help="Also save the full RLlib checkpoint.",
    )
    parser.add_argument(
        "--restore-checkpoint",
        default="",
        help="Restore a full RLlib native checkpoint before training.",
    )
    parser.add_argument(
        "--init-bundle",
        "--restart-from-bundle",
        dest="init_bundle",
        default="",
        help="Load lightweight policy bundle weights before fresh training.",
    )
    parser.add_argument(
        "--use-tune",
        action="store_true",
        help="Run training through Ray Tune/AIR.",
    )
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=0,
        help=(
            "Legacy native/Tune checkpoint frequency in training iterations. "
            "Prefer --native-checkpoint-frequency for direct training."
        ),
    )
    parser.add_argument(
        "--native-checkpoint-frequency",
        type=int,
        default=None,
        help=(
            "Save an RLlib native checkpoint every N direct-loop iterations. "
            "0 means final native checkpoint only when enabled."
        ),
    )
    parser.add_argument(
        "--dashboard-logdir",
        default="artifacts/dashboard",
        help="Dashboard JSONL root directory.",
    )
    parser.add_argument(
        "--disable-dashboard-log",
        action="store_true",
        help="Disable dashboard metrics.jsonl output.",
    )
    parser.add_argument(
        "--policy-probe-interval",
        type=int,
        default=0,
        help=(
            "Log fixed policy probe actions every N iterations. "
            "0 disables policy_probe.csv/jsonl."
        ),
    )
    parser.add_argument(
        "--policy-probe-steps",
        type=int,
        default=4,
        help="Number of recurrent inference steps per policy probe.",
    )
    parser.add_argument(
        "--no-policy-probe-print",
        action="store_true",
        help="Write policy probe files without console summaries.",
    )
    parser.add_argument(
        "--engagement-log-interval",
        type=int,
        default=0,
        help=(
            "Run a short policy-vs-target replay every N iterations and save "
            "Tacview CSV logs. 0 disables engagement_replays/."
        ),
    )
    parser.add_argument(
        "--engagement-log-steps",
        type=int,
        default=600,
        help="Maximum environment steps per engagement replay episode.",
    )
    parser.add_argument(
        "--engagement-log-episodes",
        type=int,
        default=1,
        help="Number of replay episodes to save at each engagement-log interval.",
    )
    parser.add_argument(
        "--stop-file",
        default="",
        help=(
            "Optional file checked after each completed iteration. When it exists, "
            "training exits through the normal final-save and cleanup path."
        ),
    )
    parser.add_argument(
        "--no-engagement-log-print",
        action="store_true",
        help="Write engagement replay files without console summaries.",
    )
    parser.add_argument(
        "--experiment-yaml",
        default="",
        help="Optional YAML experiment definition; env_config is deep-merged.",
    )
    args = parser.parse_args()
    if args.restore_checkpoint and args.init_bundle:
        parser.error("--restore-checkpoint and --init-bundle are mutually exclusive.")
    return args


def _build_model_config_args(args) -> dict[str, Any]:
    model_config = {
        "fcnet_hiddens": args.model_fcnet_hiddens,
        "fcnet_activation": args.model_fcnet_activation,
        "head_fcnet_hiddens": args.model_head_fcnet_hiddens,
        "head_fcnet_activation": args.model_head_fcnet_activation,
        "vf_share_layers": args.model_vf_share_layers,
    }
    if args.network_spec_json:
        model_config["network_spec"] = json.loads(args.network_spec_json)
    model_config["enabled"] = any(value is not None for value in model_config.values())
    return model_config


def _build_algorithm_args(args) -> dict:
    return {
        "framework": args.framework,
        "num_env_runners": args.num_env_runners,
        "num_envs_per_env_runner": args.num_envs_per_env_runner,
        "rollout_fragment_length": args.rollout_fragment_length,
        "min_sample_timesteps_per_iteration": (
            args.min_sample_timesteps_per_iteration
        ),
        "min_train_timesteps_per_iteration": (
            args.min_train_timesteps_per_iteration
        ),
        "batch_mode": args.batch_mode,
        "explore": args.explore,
        "lr": args.lr,
        "actor_lr": args.actor_lr if args.actor_lr is not None else args.lr,
        "critic_lr": args.critic_lr if args.critic_lr is not None else args.lr,
        "alpha_lr": args.alpha_lr if args.alpha_lr is not None else args.lr,
        "initial_alpha": args.initial_alpha,
        "gamma": args.gamma,
        "train_batch_size": args.train_batch_size,
        "minibatch_size": args.minibatch_size,
        "gae_lambda": args.gae_lambda,
        "clip_param": args.clip_param,
        "vf_clip_param": args.vf_clip_param,
        "tau": args.tau,
        # argparse declares --target-entropy without a type, so a numeric value arrives as
        # a string. SAC then fails deep inside the torch learner
        # ("unsupported operand" on target_entropy), which points nowhere near the cause.
        # "auto" is the one value that must stay a string.
        "target_entropy": _parse_target_entropy(args.target_entropy),
        "training_intensity": args.training_intensity,
        "replay_buffer_capacity": args.replay_buffer_capacity,
        "replay_buffer_config": (
            json.loads(args.replay_buffer_config_json)
            if args.replay_buffer_config_json
            else {}
        ),
        "sac_actor_freeze_fresh_steps": args.sac_actor_freeze_fresh_steps,
        "sac_actor_update_interval": args.sac_actor_update_interval,
        "sac_actor_anchor_coeff": args.sac_actor_anchor_coeff,
        "sac_actor_anchor_action_weights": (
            [
                float(value)
                for value in args.sac_actor_anchor_action_weights.split(",")
                if value.strip()
            ]
            if args.sac_actor_anchor_action_weights
            else None
        ),
        "sac_gunline_logit_config": (
            json.loads(args.sac_gunline_logit_config_json)
            if args.sac_gunline_logit_config_json
            else {}
        ),
        "sac_counter_rate_logit_config": (
            json.loads(args.sac_counter_rate_logit_config_json)
            if args.sac_counter_rate_logit_config_json
            else {}
        ),
        "sac_gated_residual_config": (
            json.loads(args.sac_gated_residual_config_json)
            if args.sac_gated_residual_config_json
            else {}
        ),
        "model_config": _build_model_config_args(args),
        "network_spec": args.network_spec_json,
        "use_lstm": args.use_lstm,
        "use_lstm_sac": args.use_lstm_sac,
        "use_lstm_prioritized_replay": args.use_lstm_prioritized_replay,
        "lstm_scope": args.lstm_scope,
        "lstm_cell_size": args.lstm_cell_size,
        "max_seq_len": args.max_seq_len,
        "debug_io": args.debug_io,
    }


def _sync_lstm_args_from_init_bundle(args) -> None:
    """Align SAC LSTM architecture args with a lightweight bundle before build."""

    if not args.init_bundle:
        return

    bundle_path = Path(args.init_bundle)
    metadata_path = bundle_path / "metadata.json"
    if not metadata_path.exists():
        return

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    bundle_meta = payload.get("metadata", {})
    saved_model_config = bundle_meta.get("model_config") or {}
    bundle_uses_lstm = bool(
        bundle_meta.get("use_lstm_sac")
        or saved_model_config.get("use_lstm")
    )
    if not bundle_uses_lstm:
        return

    algorithm_name = normalize_algorithm_name(args.algorithm)
    lstm_scope = (
        saved_model_config.get("dogfight_lstm_scope")
        or bundle_meta.get("lstm_scope")
        or "actor_only"
    )
    lstm_cell_size = int(
        saved_model_config.get("lstm_cell_size")
        or bundle_meta.get("lstm_cell_size")
        or args.lstm_cell_size
    )
    max_seq_len = int(
        saved_model_config.get("max_seq_len")
        or bundle_meta.get("max_seq_len")
        or args.max_seq_len
    )
    network_spec = (
        saved_model_config.get("dogfight_network_spec")
        or bundle_meta.get("network_spec")
    )

    if algorithm_name != "sac":
        changed = (
            not args.use_lstm
            or args.lstm_cell_size != lstm_cell_size
            or args.max_seq_len != max_seq_len
            or (
                network_spec is not None
                and args.network_spec_json
                != json.dumps(network_spec, ensure_ascii=False, separators=(",", ":"))
            )
        )
        args.use_lstm = True
        args.lstm_cell_size = lstm_cell_size
        args.max_seq_len = max_seq_len
        if network_spec is not None:
            args.network_spec_json = json.dumps(
                network_spec, ensure_ascii=False, separators=(",", ":")
            )
        if changed:
            print(
                "[DogFightEnv][LSTM_RESUME] "
                f"init_bundle={bundle_path} use_lstm=True "
                f"lstm_cell_size={lstm_cell_size} max_seq_len={max_seq_len} "
                f"network_type={(network_spec or {}).get('type')}"
            )
        return

    changed = (
        not args.use_lstm_sac
        or args.lstm_scope != lstm_scope
        or args.lstm_cell_size != lstm_cell_size
        or args.max_seq_len != max_seq_len
        or (
            network_spec is not None
            and args.network_spec_json
            != json.dumps(network_spec, ensure_ascii=False, separators=(",", ":"))
        )
    )
    args.use_lstm_sac = True
    args.lstm_scope = lstm_scope
    args.lstm_cell_size = lstm_cell_size
    args.max_seq_len = max_seq_len
    if network_spec is not None:
        args.network_spec_json = json.dumps(
            network_spec, ensure_ascii=False, separators=(",", ":")
        )
    if changed:
        print(
            "[DogFightEnv][LSTM_RESUME] "
            f"init_bundle={bundle_path} use_lstm_sac=True "
            f"lstm_scope={lstm_scope} lstm_cell_size={lstm_cell_size} "
            f"max_seq_len={max_seq_len} "
            f"network_type={(network_spec or {}).get('type')}"
        )


def _native_checkpoint_frequency(args) -> int:
    """Return the native checkpoint interval, including the legacy alias."""
    if args.native_checkpoint_frequency is not None:
        return max(0, int(args.native_checkpoint_frequency))
    return max(0, int(args.checkpoint_frequency))


def _build_observation_bundle_metadata(
    env_config: dict,
    fallback_mode: str,
) -> dict[str, Any]:
    """Return serializable observation metadata for lightweight bundles."""
    mode = str(env_config.get("observation_mode", fallback_mode))
    summary = env_config.get("observation_summary")
    if isinstance(summary, dict):
        observation_summary = dict(summary)
    else:
        observation_summary = None

    observation_size = None
    if observation_summary is not None:
        observation_size = _to_positive_int(observation_summary.get("size"))
    if observation_size is None:
        observation_size = _to_positive_int(env_config.get("observation_size"))
    if observation_size is None:
        observation_size = _to_positive_int(builtin_observation_size(mode))

    return {
        "obs_mode": mode,
        "observation_mode": mode,
        "observation_module": env_config.get("observation_module", ""),
        "observation_size": observation_size,
        "observation_summary": observation_summary,
    }


def _to_positive_int(value: Any) -> int | None:
    """Return a positive integer value or None if unavailable."""
    if value is None or isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _build_bundle_metadata(
    args,
    algorithm_name: str,
    env_config: dict,
    record_dir: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build metadata shared by final and periodic lightweight bundles."""
    observation_metadata = _build_observation_bundle_metadata(
        env_config,
        args.observation_mode,
    )
    metadata = {
        "model_name": args.output_name,
        "algorithm": algorithm_name,
        **observation_metadata,
        "action_dim": 4,
        "env_class": "DogFightWrapper",
        "target_mode": env_config.get("target_mode", args.target_mode),
        "initial_scenario_summary": describe_initial_scenario(env_config.get("initial_scenario")),
        "record_dir": str(record_dir),
        "use_lstm": args.use_lstm,
        "use_lstm_sac": args.use_lstm_sac,
        "use_lstm_prioritized_replay": (
            args.use_lstm_prioritized_replay if args.use_lstm_sac else None
        ),
        "lstm_cell_size": (
            args.lstm_cell_size if args.use_lstm or args.use_lstm_sac else None
        ),
        "max_seq_len": (
            args.max_seq_len if args.use_lstm or args.use_lstm_sac else None
        ),
        "lstm_scope": args.lstm_scope if args.use_lstm_sac else None,
        "network_spec": (
            json.loads(args.network_spec_json) if args.network_spec_json else None
        ),
        "gated_residual_sac": (
            json.loads(args.sac_gated_residual_config_json)
            if args.sac_gated_residual_config_json
            else None
        ),
    }
    if extra:
        metadata.update(extra)
    return metadata


def _save_lightweight_bundle(
    algorithm,
    bundle_dir: Path,
    args,
    algorithm_name: str,
    env_config: dict,
    record_dir: Path,
    *,
    label: str,
    iteration: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a lightweight policy bundle with common metadata."""
    metadata_extra = dict(extra or {})
    if iteration is not None:
        metadata_extra["iteration"] = iteration
    save_lightweight_policy_bundle(
        algorithm,
        bundle_dir,
        metadata=_build_bundle_metadata(
            args,
            algorithm_name,
            env_config,
            record_dir,
            extra=metadata_extra,
        ),
    )
    print(f"{label} lightweight bundle saved to {bundle_dir}")


def _save_native_checkpoint(algorithm, checkpoint_dir: Path, *, label: str) -> None:
    """Save an RLlib native checkpoint to the requested directory."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = algorithm.save(str(checkpoint_dir))
    print(f"{label} rllib checkpoint saved to {checkpoint_path}")


def _save_tune_outputs(args, algorithm_name: str, config, env_config: dict, result_grid) -> None:
    from ray.rllib.algorithms.algorithm import Algorithm

    result = result_grid[0]
    checkpoint = getattr(result, "checkpoint", None)
    if checkpoint is None:
        try:
            checkpoint = result.get_best_checkpoint(
                metric="env_runners/episode_return_mean",
                mode="max",
            )
        except Exception:
            checkpoint = None
    if checkpoint is None:
        raise RuntimeError("Tune run finished without a checkpoint to export.")

    algorithm = Algorithm.from_checkpoint(checkpoint)
    try:
        bundle_dir = ROOT / "artifacts" / "models" / args.output_name / args.output_tag
        record_dir = ROOT / "artifacts" / "records" / args.output_name / args.output_tag
        if args.save_lightweight_bundle:
            _save_lightweight_bundle(
                algorithm,
                bundle_dir,
                args,
                algorithm_name,
                env_config,
                record_dir,
                label="final",
                extra={"tune_checkpoint": str(checkpoint)},
            )
        else:
            print("lightweight bundle save skipped by --no-save-lightweight-bundle")

        metrics = _json_safe(getattr(result, "metrics", {}) or {})
        if not args.disable_dashboard_log:
            experiment_config = load_experiment_metadata(
                args.experiment_yaml,
                script_name="train_rllib",
                cli_argv=sys.argv[1:],
            )
            logger = DashboardJsonlLogger(
                _resolve_dashboard_root(args.dashboard_logdir),
                f"{args.output_name}_{args.output_tag}",
                config={
                    **experiment_config,
                    "algorithm": algorithm_name,
                    "cli_args": vars(args),
                    "env_config": env_config,
                },
            )
            env_metrics = metrics.get("env_runners", {})
            row = {
                "iter": metrics.get("training_iteration", args.iterations),
                "sampled_steps": env_metrics.get("num_env_steps_sampled_lifetime"),
                "episodes": env_metrics.get("num_episodes_lifetime"),
                "reward_mean": env_metrics.get("episode_return_mean"),
                "ep_len_mean": env_metrics.get("episode_len_mean"),
            }
            logger.write_row(row)
            print(f"dashboard log saved to {logger.metrics_path}")
        save_training_record(
            output_dir=record_dir,
            algorithm_name=algorithm_name,
            cli_args=vars(args),
            env_config=env_config,
            algorithm_config=_json_safe(config.to_dict() if hasattr(config, "to_dict") else {}),
            result_history=[{"iteration": "tune_final", **metrics}],
            workspace_root=ROOT,
        )
        copy_experiment_yaml(args.experiment_yaml, record_dir)
        print(f"training record saved to {record_dir}")
    finally:
        algorithm.stop()


def _run_with_tune(args, algorithm_name: str, config, env_config: dict) -> None:
    from ray import air, tune
    from ray.air import CheckpointConfig

    tune_dir = ROOT / "artifacts" / "tune" / args.output_name
    trainable = algorithm_name.upper()
    checkpoint_config = CheckpointConfig(
        checkpoint_frequency=_native_checkpoint_frequency(args),
        checkpoint_at_end=True,
        num_to_keep=2,
    )
    run_config = air.RunConfig(
        name=args.output_tag,
        storage_path=str(tune_dir),
        stop={"training_iteration": args.iterations},
        checkpoint_config=checkpoint_config,
        progress_reporter=_build_tune_progress_reporter(algorithm_name),
        verbose=1,
    )
    tuner = tune.Tuner(
        trainable,
        param_space=config.to_dict() if hasattr(config, "to_dict") else dict(config),
        run_config=run_config,
    )
    print(
        f"starting Tune run: {trainable}, "
        f"env_runners={args.num_env_runners}, "
        f"envs_per_runner={args.num_envs_per_env_runner}"
    )
    result_grid = tuner.fit()
    _save_tune_outputs(args, algorithm_name, config, env_config, result_grid)


def main():
    args = parse_args()
    algorithm_name = normalize_algorithm_name(args.algorithm)
    if args.use_tune and (args.restore_checkpoint or args.init_bundle):
        raise RuntimeError(
            "Checkpoint/bundle restart is supported by the direct training loop, "
            "not --use-tune."
        )
    _sync_lstm_args_from_init_bundle(args)

    env_config = {
        "observation_mode": args.observation_mode,
        "target_mode": args.target_mode,
        "target_behavior_dll": args.target_behavior_dll,
        "ownship_control_mode": "rl",
        "max_engage_time": args.max_engage_time,
        "episode_step_limit": args.episode_step_limit,
    }
    deep_update(env_config, load_experiment_env_config(args.experiment_yaml, ROOT))
    if args.reward_module:
        env_config["reward_module"] = args.reward_module
    if args.observation_module:
        env_config["observation_module"] = args.observation_module
    env_preview = env_creator(env_config)
    env_config["reward"] = dict(env_preview.config["reward"])
    env_config["wez"] = dict(env_preview.config["wez"])
    if args.observation_module:
        env_config["observation_mode"] = env_preview.config["observation_mode"]
        env_config["observation_module"] = args.observation_module
        env_config["observation_summary"] = dict(env_preview.config["observation_summary"])
    obs_shape = getattr(env_preview.observation_space, "shape", (0,))
    action_shape = getattr(env_preview.action_space, "shape", (4,))
    probe_obs_dim = int(obs_shape[0]) if obs_shape else 0
    probe_action_dim = int(action_shape[0]) if action_shape else 4
    env_preview.close()

    env_name = "dogfight-single-agent-v0"
    register_env(env_name, env_creator)

    config = build_algorithm_config(
        algorithm_name=algorithm_name,
        env_name=env_name,
        env_config=env_config,
        args=_build_algorithm_args(args),
    )

    # RLlib pads every iteration up to min_time_s_per_iteration (1 s for SAC/DQN). Measured:
    # the training step itself took ~8 ms inside a 1.000 s iteration, so sampling ran at
    # ~100 env steps/s while the simulator delivers well over a thousand -- roughly a 10x
    # waste of wall clock. Clearing the floor and asking for a fixed number of sampled steps
    # instead makes each iteration real work.
    #
    # Opt in per experiment: env_config.min_sample_timesteps_per_iteration. Left unset the
    # behaviour is unchanged, so older YAMLs keep reproducing.
    _min_sample = env_config.get("min_sample_timesteps_per_iteration")
    if _min_sample:
        config = config.reporting(
            min_time_s_per_iteration=0,
            min_sample_timesteps_per_iteration=int(_min_sample),
        )
        print(f"[train] {int(_min_sample)} sampled steps per iteration, no time floor")

    _ensure_ray_runtime_env()

    if args.use_tune:
        _run_with_tune(args, algorithm_name, config, env_config)
        return

    # A native Windows build can occasionally spend a long time inside RLlib's
    # constructor without emitting output. Keep startup observable so a real
    # deadlock can be distinguished from slow initialization.
    import faulthandler

    print("[train] building RLlib algorithm", flush=True)
    faulthandler.dump_traceback_later(60, repeat=True)
    try:
        algorithm = config.build_algo()
    finally:
        faulthandler.cancel_dump_traceback_later()
    print("[train] RLlib algorithm ready", flush=True)
    restored_policy = False
    if args.restore_checkpoint:
        # RLlib hands the path to pyarrow, which parses it as a URI and rejects anything
        # relative with "URI has empty scheme". Resolve here so an experiment YAML can carry
        # the natural repo-relative path.
        checkpoint_path = Path(args.restore_checkpoint).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"restore checkpoint not found: {checkpoint_path}")
        print(f"restoring native RLlib checkpoint from {checkpoint_path}")
        algorithm.restore(checkpoint_path.as_posix())
        restored_policy = True
        if (
            algorithm_name == "sac"
            and (
                args.sac_reset_alpha_after_restore is not None
                or args.sac_reset_optimizers_after_restore
                # Actor anchoring needs a snapshot of the restored actor before
                # the first update.  It is independent of alpha/optimizer reset.
                or args.sac_actor_anchor_coeff > 0.0
                # A hard transfer freeze also needs the exact restored actor
                # parameters so shared critic updates can be rolled back.
                or args.sac_actor_freeze_fresh_steps > 0
            )
        ):
            from dogfight.ai.conservative_sac_learner import (
                apply_post_restore_sac_transfer,
            )
            from dogfight.ai.checkpoint_io import load_lightweight_policy_bundle

            actor_anchor_weights = None
            actor_anchor_bundle = str(args.sac_actor_anchor_bundle).strip()
            if actor_anchor_bundle:
                anchor_path = Path(actor_anchor_bundle).resolve()
                if not anchor_path.exists():
                    raise FileNotFoundError(
                        f"actor anchor bundle not found: {anchor_path}"
                    )
                _, actor_anchor_weights = load_lightweight_policy_bundle(anchor_path)

            transfer_result = apply_post_restore_sac_transfer(
                algorithm,
                reset_alpha=args.sac_reset_alpha_after_restore,
                reset_optimizers=args.sac_reset_optimizers_after_restore,
                actor_anchor_weights=actor_anchor_weights,
            )
            print(
                "[DogFightEnv][SAC_TRANSFER] "
                f"alpha={args.sac_reset_alpha_after_restore} "
                f"reset_optimizers={args.sac_reset_optimizers_after_restore} "
                f"actor_freeze_fresh_steps={args.sac_actor_freeze_fresh_steps} "
                f"actor_update_interval={args.sac_actor_update_interval} "
                f"actor_anchor_coeff={args.sac_actor_anchor_coeff} "
                f"actor_anchor_bundle={actor_anchor_bundle or None} "
                f"learners={transfer_result}"
            )
    elif args.init_bundle:
        bundle_path = Path(args.init_bundle)
        if not bundle_path.exists():
            raise FileNotFoundError(f"lightweight bundle not found: {bundle_path}")
        print(f"loading lightweight bundle weights from {bundle_path}")
        apply_lightweight_policy_bundle(algorithm, bundle_path)
        restored_policy = True

    if restored_policy:
        # Native restore updates the Learner first.  The local EnvRunner owns a
        # separate inference module and otherwise keeps its freshly initialized
        # weights until the first train cycle.  That made pre-train replay
        # probes and early collection non-reproducible for the same checkpoint.
        env_runner_group = getattr(algorithm, "env_runner_group", None)
        learner_group = getattr(algorithm, "learner_group", None)
        if env_runner_group is None or learner_group is None:
            raise RuntimeError(
                "Restored policy could not be synchronized to the local EnvRunner"
            )
        env_runner_group.sync_weights(
            from_worker_or_learner_group=learner_group,
            timeout_seconds=30.0,
            inference_only=True,
        )
        print("[DogFightEnv][RESTORE_SYNC] learner weights synced to EnvRunner")

    native_checkpoint_dir = None
    result_history = []

    # CSV log setup
    log_dir = ROOT / "artifacts" / "logs" / args.output_name / args.output_tag
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "training_log.csv"
    _CSV_FIELDS = [
        "iter", "sampled_steps", "episodes", "reward_mean", "ep_len_mean",
        "win_rate", "loss_rate", "timeout_rate", "crash_rate",
        "ep_wez_steps", "ep_mean_distance", "ep_min_distance",
        "ep_reward_pursuit", "ep_reward_damage", "ep_reward_safety",
        "ep_reward_energy_throttle",
        "ep_reward_survival", "ep_reward_merge_cross",
        "ep_reward_merge_handoff", "ep_reward_merge_handoff_once",
        "ep_reward_merge_turnin", "ep_reward_reacquire_turnin",
        "ep_reward_merge_stalemate", "ep_reward_attack_band",
        "ep_reward_front_hold", "ep_reward_aim_hold",
        "ep_reward_angle_advantage", "ep_reward_angle_advantage_progress",
        "ep_reward_cutin_success", "ep_reward_cutin_control",
        "ep_reward_terminal_angle_gain", "ep_reward_terminal_ata_gain",
        "ep_reward_range_progress",
        "ep_reward_lost", "ep_altitude_penalty_steps",
        "initial_scenario_index",
        "initial_alpha_deg", "initial_ata_deg", "initial_aa_deg",
        "initial_distance_m", "final_ata_deg", "final_aa_deg",
        "headon_guard_fail", "cutin_handback_success", "bt_preroll_detected",
        "action_roll_mean", "action_pitch_mean", "action_rudder_mean",
        "action_throttle_mean", "action_roll_std", "action_pitch_std",
        "action_rudder_std", "action_throttle_std",
        "action_sat_rate",
        "policy_loss", "vf_loss", "entropy", "kl", "clip_frac", "explained_var",
        "actor_loss", "critic_loss", "alpha_loss", "alpha", "target_entropy",
        "curriculum_actor_frozen", "curriculum_fresh_steps",
        "curriculum_actor_update_applied", "curriculum_learner_updates",
        "curriculum_actor_anchor_mse", "curriculum_actor_anchor_loss",
        "curriculum_gunline_fraction", "curriculum_actor_anchor_fraction",
        "curriculum_residual_gate_fraction",
        "curriculum_residual_mean_abs_delta",
        "curriculum_counter_rate_fraction",
        "curriculum_counter_rate_logit_error",
        "curriculum_counter_rate_logit_loss",
        "replay_buffer_size", "replay_buffer_memory_mb", "env_steps_per_sec",
        "learner_steps_per_sec", "iteration_time_s",
    ]
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    # extrasaction="ignore" would silently drop any reward component not in the list above,
    # which is how a whole run ended up with its aiming terms missing. The writer is
    # wrapped instead so unknown keys are added to the header on first sight.
    csv_writer = _GrowingDictWriter(csv_file, _CSV_FIELDS)
    csv_writer.writeheader()
    policy_probe_logger = PolicyProbeLogger(
        log_dir,
        obs_dim=probe_obs_dim,
        action_dim=probe_action_dim,
        interval=args.policy_probe_interval,
        sequence_steps=args.policy_probe_steps,
        print_to_console=not args.no_policy_probe_print,
    )
    engagement_replay_logger = EngagementReplayLogger(
        log_dir,
        env_factory=env_creator,
        env_config=env_config,
        interval=args.engagement_log_interval,
        max_steps=args.engagement_log_steps,
        episodes=args.engagement_log_episodes,
        print_to_console=not args.no_engagement_log_print,
    )
    dashboard_logger = None
    if not args.disable_dashboard_log:
        experiment_config = load_experiment_metadata(
            args.experiment_yaml,
            script_name="train_rllib",
            cli_argv=sys.argv[1:],
        )
        dashboard_logger = DashboardJsonlLogger(
            _resolve_dashboard_root(args.dashboard_logdir),
            f"{args.output_name}_{args.output_tag}",
            config={
                **experiment_config,
                "algorithm": algorithm_name,
                "cli_args": vars(args),
                "env_config": env_config,
                "csv_path": str(csv_path),
            },
        )

    try:
        policy_probe_logger.__enter__()
        engagement_replay_logger.__enter__()
        bundle_root = (
            ROOT / "artifacts" / "models" / args.output_name / args.output_tag
        )
        checkpoint_root = (
            ROOT / "artifacts" / "checkpoints" / args.output_name / args.output_tag
        )
        record_dir = (
            ROOT / "artifacts" / "records" / args.output_name / args.output_tag
        )
        bundle_frequency = max(0, int(args.lightweight_bundle_frequency))
        native_frequency = _native_checkpoint_frequency(args)

        # A restored checkpoint must be evaluated BEFORE the first train() call, otherwise
        # "iteration 0" already contains a PPO update and a frozen-policy probe is not
        # actually frozen.
        if os.getenv("DOGFIGHT_EVAL_BEFORE_TRAIN", "0") == "1":
            print("[frozen-probe] evaluating the restored policy before any train()", flush=True)
            engagement_replay_logger.maybe_log(
                algorithm,
                iteration=0,
                sampled_steps=0,
            )

        for iteration in range(args.iterations):
            result = algorithm.train()
            env_metrics   = result.get("env_runners", {})
            reward_mean      = env_metrics.get("episode_return_mean", "n/a")
            episode_len_mean = env_metrics.get("episode_len_mean", "n/a")
            progress = _extract_progress_metrics(result)
            learner_stats = _extract_learner_stats(result)
            _fill_algorithm_runtime_stats(learner_stats, algorithm)
            if (
                args.debug_io
                and args.use_lstm_sac
                and learner_stats.get("actor_loss") == "n/a"
                and iteration >= 4
                and not getattr(algorithm, "_dogfight_printed_learner_keys", False)
            ):
                _print_learner_result_debug(result, iteration)
                setattr(algorithm, "_dogfight_printed_learner_keys", True)
            custom        = _extract_custom_metrics(result)

            row = {
                "iter":              iteration,
                "sampled_steps":     progress["sampled_steps"],
                "episodes":          progress["episodes"],
                "reward_mean":       reward_mean,
                "ep_len_mean":       episode_len_mean,
                **custom,
                **learner_stats,
            }
            csv_writer.writerow(row)
            csv_file.flush()
            if dashboard_logger is not None:
                dashboard_logger.write_row(row)
            policy_probe_logger.maybe_log(
                algorithm,
                iteration=iteration,
                sampled_steps=progress["sampled_steps"],
            )
            engagement_replay_logger.maybe_log(
                algorithm,
                iteration=iteration,
                sampled_steps=progress["sampled_steps"],
            )

            result_history.append({
                "iteration":       iteration,
                "reward_mean":     reward_mean,
                "episode_len_mean": episode_len_mean,
                **custom,
                **learner_stats,
            })

            # Console row
            print(_console_row(
                algorithm_name,
                iteration,
                progress,
                reward_mean,
                custom,
                learner_stats,
            ))
            iteration_number = iteration + 1
            if args.save_lightweight_bundle and bundle_frequency > 0:
                if iteration_number % bundle_frequency == 0:
                    periodic_bundle_dir = bundle_root / f"bundle_{iteration_number:06d}"
                    _save_lightweight_bundle(
                        algorithm,
                        periodic_bundle_dir,
                        args,
                        algorithm_name,
                        env_config,
                        record_dir,
                        label=f"periodic iter {iteration_number}",
                        iteration=iteration_number,
                    )
            if args.save_native_checkpoint and native_frequency > 0:
                if iteration_number % native_frequency == 0:
                    checkpoint_dir = (
                        checkpoint_root / f"checkpoint_{iteration_number:06d}"
                    )
                    _save_native_checkpoint(
                        algorithm,
                        checkpoint_dir,
                        label=f"periodic iter {iteration_number}",
                    )
            if args.stop_file:
                stop_path = Path(args.stop_file)
                if not stop_path.is_absolute():
                    stop_path = ROOT / stop_path
                if stop_path.exists():
                    print(
                        f"[DogFightEnv][STOP_FILE] graceful stop after "
                        f"iteration {iteration_number}: {stop_path}",
                        flush=True,
                    )
                    break
        csv_file.close()
        print(f"training log saved to {csv_path}")
        if dashboard_logger is not None:
            print(f"dashboard log saved to {dashboard_logger.metrics_path}")
        if policy_probe_logger.enabled:
            print(f"policy probe CSV saved to {policy_probe_logger.csv_path}")
            print(f"policy probe JSONL saved to {policy_probe_logger.jsonl_path}")

        if args.save_lightweight_bundle:
            _save_lightweight_bundle(
                algorithm,
                bundle_root,
                args,
                algorithm_name,
                env_config,
                record_dir,
                label="final",
            )
        else:
            print("lightweight bundle save skipped by --no-save-lightweight-bundle")

        save_training_record(
            output_dir=record_dir,
            algorithm_name=algorithm_name,
            cli_args=vars(args),
            env_config=env_config,
            algorithm_config=_json_safe(config.to_dict() if hasattr(config, "to_dict") else {}),
            result_history=result_history,
            workspace_root=ROOT,
        )
        copy_experiment_yaml(args.experiment_yaml, record_dir)
        print(f"training record saved to {record_dir}")

        if args.save_native_checkpoint:
            _save_native_checkpoint(
                algorithm,
                checkpoint_root / "checkpoint_final",
                label="final",
            )
    finally:
        policy_probe_logger.__exit__(None, None, None)
        engagement_replay_logger.__exit__(None, None, None)
        if policy_probe_logger.enabled:
            print(f"policy probe CSV saved to {policy_probe_logger.csv_path}")
            print(f"policy probe JSONL saved to {policy_probe_logger.jsonl_path}")
        if engagement_replay_logger.enabled:
            print(
                "engagement replay index saved to "
                f"{engagement_replay_logger.csv_path}"
            )
            print(
                "engagement replay JSONL saved to "
                f"{engagement_replay_logger.jsonl_path}"
            )
        algorithm.stop()


if __name__ == "__main__":
    main()
