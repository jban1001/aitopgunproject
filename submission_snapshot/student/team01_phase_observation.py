"""Self-contained build of the team01_phase35 observation.

WHY THIS EXISTS
---------------
`student/team01_phase_observation.py` declares 35 channels but only constructs 3
of them.  The other 32 come from `dogfight.envs.observation._build_tactical32`,
the state layout comes from `dogfight.sim.state_schema`, and the phase table
comes from `dogfight.config` -- and NONE of those three exist in the stock
Release.  Checked against Release_260526_ori:

    src/dogfight/envs/observation.py   no tactical32 at all
    src/dogfight/sim/state_schema.py   no U,V,W,P,Q,R,AOA,AOS,THROTTLE,
                                       VERTICAL_SPEED,NZ,NY
    src/dogfight/config.py             no wez_phases

So the observation contract only resolves because our modified `src/` ships
alongside `student/`.  That is fine when the whole Release folder is handed over,
and it is exactly what breaks if `student/` is ever dropped into a clean Release
-- the failure would be an immediate dimension/attribute error at inference.

This module removes that coupling: it depends on numpy and on the GeometryInfo
object handed to it, nothing else.  The channel arithmetic is a line-for-line
copy of the shipped implementation, and `scratch_obs_equivalence.py` checks it
bit-for-bit against the original over randomized states, so swapping to it
cannot move the policy.
"""
from __future__ import annotations

import numpy as np

OBSERVATION_MODE = "team01_phase35"
OBSERVATION_SIZE = 35
OBSERVATION_LOW = -1.0
OBSERVATION_HIGH = 1.0

MATCH_SECONDS = 200.0

# --- state layout (mirrors dogfight.sim.state_schema.StateIndex) --------------
_N, _E, _D = 0, 1, 2
_ROLL, _PITCH, _YAW = 3, 4, 5
_U, _V, _W = 6, 7, 8
_P, _Q, _R = 9, 10, 11
_KCAS = 12
_AOA, _AOS = 13, 14
_THROTTLE = 21
_VERTICAL_SPEED = 30
_NZ = 31
_SIM_TIME = 41
_ALT = 44
_HEALTH = 45

# --- competition damage schedule (mirrors dogfight.config wez_phases) ---------
# Kept as `<feet> * FEET_TO_METER` rather than the rounded metre value so the
# float is bit-identical to the shipped table (3000 ft -> 914.4000000000001).
_FEET_TO_METER = 0.30480
PHASES = [
    {"start_s": 0.0, "angle_deg": 2.0,
     "min_range_m": 500 * _FEET_TO_METER, "max_range_m": 3000 * _FEET_TO_METER,
     "coefficient": 1.0},
    {"start_s": 100.0, "angle_deg": 4.0,
     "min_range_m": 500 * _FEET_TO_METER, "max_range_m": 3500 * _FEET_TO_METER,
     "coefficient": 0.3},
    {"start_s": 150.0, "angle_deg": 6.0,
     "min_range_m": 500 * _FEET_TO_METER, "max_range_m": 4000 * _FEET_TO_METER,
     "coefficient": 0.1},
]
_MAX_HALF_ANGLE = max(p["angle_deg"] for p in PHASES) / 2.0
_MAX_RANGE = max(p["max_range_m"] for p in PHASES)


def normalize(value: float, minimum: float, maximum: float) -> float:
    if maximum <= minimum:
        return 0.0
    clipped = float(np.clip(value, minimum, maximum))
    midpoint = (maximum + minimum) / 2.0
    half_range = (maximum - minimum) / 2.0
    return (clipped - midpoint) / half_range


def _unit(value: float, scale: float) -> float:
    return float(np.clip(value / max(scale, 1.0e-6), -1.0, 1.0))


def _world_velocity_ned(state) -> np.ndarray:
    """Convert JSBSim body-axis u/v/w into NED velocity."""
    roll, pitch, yaw = np.radians(np.asarray(state[3:6], dtype=float))
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    forward = np.array([cp * cy, cp * sy, -sp], dtype=float)
    right = np.array(
        [sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, sr * cp],
        dtype=float,
    )
    down = np.array(
        [cr * sp * cy + sr * sy, cr * sp * sy - sr * cy, cr * cp],
        dtype=float,
    )
    return (
        float(state[_U]) * forward
        + float(state[_V]) * right
        + float(state[_W]) * down
    )


def _build_tactical32(ownship_state, target_state, geo_info, wez_config=None) -> np.ndarray:
    """Memory-free tactical state; line-for-line copy of the shipped builder."""
    own = np.asarray(ownship_state, dtype=float)
    target = np.asarray(target_state, dtype=float)
    delta = target[:3] - own[:3]
    distance = float(np.linalg.norm(delta))
    distance = max(distance, 1.0)
    relative_velocity = _world_velocity_ned(target) - _world_velocity_ned(own)
    closure = -float(np.dot(delta / distance, relative_velocity))
    relative_speed = float(np.linalg.norm(relative_velocity))
    ata = abs(float(geo_info._get_antenna_train_angle(own, target, False)))
    aspect = abs(float(geo_info._get_aspect_angle(own, target, False)))
    azimuth, elevation = geo_info._get_los_angle(own, target)
    azimuth_rad = np.radians(float(azimuth))
    elevation_rad = np.radians(float(elevation))
    roll_rad = np.radians(float(own[_ROLL]))
    pitch_rad = np.radians(float(own[_PITCH]))
    relative_heading_rad = np.radians(
        (float(target[_YAW]) - float(own[_YAW]) + 180.0) % 360.0 - 180.0
    )

    in_wez = False
    if wez_config is not None:
        in_wez = (
            float(wez_config["min_range_m"]) <= distance <= float(wez_config["max_range_m"])
            and ata <= float(wez_config["angle_deg"]) / 2.0
        )

    obs = np.asarray([
        np.sin(roll_rad), np.cos(roll_rad), np.sin(pitch_rad), np.cos(pitch_rad),
        _unit(own[_U], 450.0), _unit(own[_V], 120.0), _unit(own[_W], 120.0),
        _unit(own[_P], 180.0), _unit(own[_Q], 180.0), _unit(own[_R], 180.0),
        normalize(float(own[_KCAS]), 0.0, 450.0),
        normalize(float(own[_ALT]), 0.0, 12000.0),
        _unit(own[_VERTICAL_SPEED], 6000.0),
        _unit(own[_AOA], 45.0), _unit(own[_AOS], 30.0), _unit(own[_NZ], 12.0),
        _unit(own[_THROTTLE], 1.0),
        float(np.clip(2.0 * np.log1p(distance) / np.log1p(20000.0) - 1.0, -1.0, 1.0)),
        _unit(closure, 300.0),
        np.cos(elevation_rad) * np.cos(azimuth_rad),
        np.cos(elevation_rad) * np.sin(azimuth_rad),
        -np.sin(elevation_rad),
        np.cos(np.radians(ata)), np.cos(np.radians(aspect)),
        np.sin(relative_heading_rad), np.cos(relative_heading_rad),
        normalize(float(target[_KCAS]), 0.0, 450.0),
        _unit(float(target[_ALT]) - float(own[_ALT]), 5000.0),
        normalize(float(target[_HEALTH]), 0.0, 1.0),
        normalize(float(own[_HEALTH]), 0.0, 1.0),
        1.0 if in_wez else -1.0,
        _unit(relative_speed, 600.0),
    ], dtype=np.float32)
    return np.nan_to_num(np.clip(obs, -1.0, 1.0), nan=0.0, posinf=1.0, neginf=-1.0)


def active_phase(elapsed_s: float) -> dict:
    """Widest phase whose clock has arrived."""
    active = [p for p in PHASES if elapsed_s >= float(p.get("start_s", 0.0))]
    return dict(active[-1] if active else PHASES[0])


def build_observation(ownship_state, target_state, geo_info, wez_config=None):
    own = np.asarray(ownship_state, dtype=np.float32).copy()
    target = np.asarray(target_state, dtype=np.float32).copy()
    # The Viewer transmits neither damage nor normal load. Hold these identical in
    # training and deployment rather than letting the actor learn a dependency that
    # cannot exist in a real match.
    own[_HEALTH] = 1.0
    target[_HEALTH] = 1.0
    own[_NZ] = 0.0

    elapsed_s = float(own[_SIM_TIME])
    phase = active_phase(elapsed_s)
    effective_wez = wez_config if wez_config is not None else phase

    base = _build_tactical32(own, target, geo_info, effective_wez)

    extra = np.array([
        normalize(max(0.0, MATCH_SECONDS - elapsed_s), 0.0, MATCH_SECONDS),
        normalize(float(phase["angle_deg"]) / 2.0, 0.0, _MAX_HALF_ANGLE),
        normalize(float(phase["max_range_m"]), 0.0, _MAX_RANGE),
    ], dtype=np.float32)
    return np.concatenate([base, extra]).astype(np.float32)


def describe_observation():
    return {
        "mode": OBSERVATION_MODE,
        "size": OBSERVATION_SIZE,
        "features": [
            "sin_roll", "cos_roll", "sin_pitch", "cos_pitch",
            "body_u", "body_v", "body_w", "roll_rate", "pitch_rate", "yaw_rate",
            "own_speed", "own_altitude", "vertical_speed", "aoa", "sideslip",
            "normal_g", "throttle",
            "log_distance", "closure", "los_body_x", "los_body_y", "los_body_z",
            "cos_ata", "cos_aspect", "sin_relative_heading", "cos_relative_heading",
            "target_speed", "target_altitude_delta", "target_health", "own_health",
            "in_wez", "relative_speed",
            "time_remaining", "phase_half_angle", "phase_max_range",
        ],
        "description": (
            "Markov tactical state for MLP SAC: body flight dynamics, relative motion, "
            "LOS geometry, energy, health, and WEZ. All values are clipped to [-1, 1]."
        ),
        "phases": [dict(p) for p in PHASES],
        "viewer_fixed_channels": {
            "ownship_health": 1.0,
            "target_health": 1.0,
            "ownship_nz": 0.0,
        },
    }
