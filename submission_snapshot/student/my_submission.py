"""
[학생 작성 파일] 경진대회 제출 — Unreal 서버 연결
=====================================================
학습한 모델을 경진대회 서버에 연결합니다.
BUNDLE_DIR 경로와 팀 이름을 설정한 뒤 이 파일을 실행하세요.

커맨드라인으로 직접 실행하는 방법 (권장)
-------------------------------------------
  # RL 모델 사용
  python run_unreal_inference.py --mode rl \\
      --bundle-dir artifacts/models/team01/v1 \\
      --team-name team01 \\
      --server-ip <서버IP> --server-port 9999

  # BT만 사용 (모델 없이)
  python run_unreal_inference.py --mode bt \\
      --bt-dll AIP_BASE.dll \\
      --bt-rule-xml Rule_forTraining.xml \\
      --team-name team01 \\
      --server-ip <서버IP>

  # RL + BT 하이브리드
  python run_unreal_inference.py --mode hybrid \\
      --bundle-dir artifacts/models/team01/v1 \\
      --bt-dll AIP_BASE.dll \\
      --bt-rule-xml Rule_팀이름.xml \\
      --hybrid-mode residual --residual-scale 0.35 \\
      --team-name team01 \\
      --server-ip <서버IP>

이 파일에서 직접 실행하려면
----------------------------
  python student/my_submission.py

아래 설정을 수정한 뒤 실행하면 됩니다.
"""
from __future__ import annotations

import gc
import json
import os
import re
import subprocess
import sys
import time

# --- per-frame latency guard -------------------------------------------------------------
# The server runs at 60 Hz, so an AI step has one 16.67 ms frame. Our inference is small
# (median 0.25 ms) but the RAW process spikes badly: measured over 800 steps, p99 22.18 ms and
# max 70.68 ms -- an overrun the server reads as a missed update, which shows up as
# disconnection damage while the socket is perfectly healthy.
#
# The spikes are not compute, they are a generational GC pass landing inside a frame plus
# torch contending for cores. Freezing the startup heap, raising the GC thresholds and pinning
# torch to one thread removes them:
#
#     baseline           median 0.81   p95 8.77   p99 22.18   max 70.68 ms
#     with this guard    median 0.25   p95 1.55   p99  3.00   max  6.79 ms
#
# Thresholds rather than gc.disable(): cycles are still collected, just far too rarely to
# land inside a frame, so a long session cannot leak unboundedly. TEAM01_LOW_LATENCY=0 opts
# out if this ever needs to be isolated.
if os.getenv("TEAM01_LOW_LATENCY", "1") == "1":
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:  # noqa: BLE001 - never block startup on the guard
        pass
    gc.collect()
    gc.freeze()
    gc.set_threshold(200000, 1000, 1000)
# -------------------------------------------------------------------------------------------

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Native BT DLLs and Rule XML files are resolved from the release root.  Make the
# entry point behave the same whether VS Code starts it from the release folder,
# the student folder, or an arbitrary terminal working directory.
os.chdir(ROOT)

from dogfight.ai.bt_action_provider import BTActionProvider
from dogfight.ai.bt_rule_manager import activate_rule_xml
from dogfight.ai.hybrid_action_provider import HybridActionProvider
from dogfight.ai.rl_action_provider import RLActionProvider
from dogfight.ai.rllib_utils import build_algorithm_from_bundle
from dogfight.ai.student_hooks import load_observation_hook
from dogfight.ai.tactical_hybrid_action_provider import TacticalHybridActionProvider
from dogfight.unreal import AIType, ProviderCommandPolicy, UnrealAIPilotUDPClient


# =============================================================================
# TODO: 아래 설정을 팀에 맞게 수정하세요.
# =============================================================================

TEAM_NAME = "MAVERICK1"                            # official competition team name


def _windows_process_image(pid: int) -> str:
    """Return a process image path without requiring PowerShell or admin rights."""
    if os.name != "nt" or pid <= 0:
        return ""
    try:
        import ctypes
        from ctypes import wintypes

        query_limited_information = 0x1000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(query_limited_information, False, pid)
        if not handle:
            return ""
        try:
            size = wintypes.DWORD(32768)
            buffer = ctypes.create_unicode_buffer(size.value)
            if kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
                return buffer.value
        finally:
            kernel32.CloseHandle(handle)
    except Exception:  # noqa: BLE001 - discovery must never block explicit configuration
        return ""
    return ""


def _local_udp_listeners() -> list[tuple[int, int, str]]:
    """Return ``(port, pid, image)`` for local UDP listeners on Windows."""
    if os.name != "nt":
        return []
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    try:
        result = subprocess.run(
            ["netstat", "-ano", "-p", "udp"],
            check=False,
            capture_output=True,
            text=True,
            errors="ignore",
            startupinfo=startupinfo,
            timeout=3.0,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    listeners: list[tuple[int, int, str]] = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) < 4 or fields[0].upper() != "UDP" or fields[2] != "*:*":
            continue
        port_match = re.search(r":(\d+)$", fields[1])
        if not port_match:
            continue
        try:
            port = int(port_match.group(1))
            pid = int(fields[-1])
        except ValueError:
            continue
        listeners.append((port, pid, _windows_process_image(pid)))
    return listeners


def _find_local_viewer_port() -> int | None:
    """Find the active local Viewer safely, without probing unrelated UDP services."""
    listeners = _local_udp_listeners()
    viewer_ports = sorted(
        {
            port
            for port, _pid, image in listeners
            if "dogfightviewer" in image.lower() or "battleviewer" in image.lower()
        }
    )
    if viewer_ports:
        return 9999 if 9999 in viewer_ports else viewer_ports[0]

    bound_ports = {port for port, _pid, _image in listeners}
    if 9999 in bound_ports:
        return 9999

    # Development Viewer builds normally stay near the official 9999 port.  Only
    # accept an unlabelled port when exactly one candidate exists, avoiding an
    # accidental connection to an unrelated local UDP service.
    nearby = sorted(port for port in bound_ports if 9000 <= port <= 11000)
    return nearby[0] if len(nearby) == 1 else None


def _resolve_server_endpoint() -> tuple[str, int, str]:
    """Honor explicit competition settings, otherwise wait for a local Viewer."""
    explicit_ip = os.getenv("TEAM01_SERVER_IP", "").strip()
    explicit_port = os.getenv("TEAM01_SERVER_PORT", "").strip()
    if explicit_ip or explicit_port:
        return explicit_ip or "127.0.0.1", int(explicit_port or "9999"), "환경변수"

    wait_seconds = max(0.0, float(os.getenv("TEAM01_SERVER_WAIT_SEC", "3600")))
    deadline = time.monotonic() + wait_seconds
    last_notice = 0.0
    while True:
        port = _find_local_viewer_port()
        if port is not None:
            return "127.0.0.1", port, "로컬 Viewer 자동 탐색"
        now = time.monotonic()
        if now >= deadline:
            raise RuntimeError(
                "열린 로컬 DogFightViewer UDP 서버를 찾지 못했습니다. "
                "Viewer 서버를 열거나 TEAM01_SERVER_IP/TEAM01_SERVER_PORT를 지정하세요."
            )
        if now - last_notice >= 10.0:
            print("[MAVERICK1] 로컬 Viewer 서버가 열리기를 기다리는 중...", flush=True)
            last_notice = now
        time.sleep(1.0)


# 원격 대회 서버는 네트워크에서 안전하게 자동 발견할 수 없으므로 환경변수가 우선이다:
#   $env:TEAM01_SERVER_IP="10.0.0.5"; $env:TEAM01_SERVER_PORT="9999"
# 로컬 BattleViewer는 주소나 포트를 지정하지 않아도 열린 UDP listener를 자동으로 찾는다.
SERVER_IP, SERVER_PORT, SERVER_ENDPOINT_SOURCE = _resolve_server_endpoint()

# 사용할 백엔드 모드 선택: "rl" | "bt" | "hybrid"
# Final submission default: measured specialist safety overlay on champion BT.
MODE = os.getenv("TEAM01_MODE", "selector")   # 기본: 상황별 전문가 selector

# RL 모드 설정
# The submitted model, shipped inside this folder so the entry point runs with no
# environment variables set at all. The observation module is read from the bundle's
# own metadata (student.team01_phase_observation), so it does not need configuring.
# SUBMISSION BUNDLE.
#
# Chosen on the competition's OWN distances. Briefing slide 16 defines rounds 1-3 as
# 2000-3000 ft (610-914 m) and round 4+ as 10000 ft or more, and slide 17 makes the
# tournament best-of-3, so rounds 1-3 are the match. Every earlier judgement in this
# project used a grid whose close spawns sat at 1200 m -- 3937 ft, in neither band -- and
# re-measuring at the real distances changed the ranking and doubled V255's crash count
# (6/24 -> 11/24 against champion). Numbers below are rounds 1-3, 24 matches per cell,
# W-L-D / crashes:
#
# Scored as MATCH win probability, not round win rate: slide 16 makes round 4 a decider
# rather than another draw, so a round-1-3 draw routes the match into the merge instead of
# being neutral. match_odds.py folds that in (best-of-3, round 4 as tiebreak). Six
# opponents, 24 matches per cell:
#
#     model      champ  weapv7  alt_den  defense  neutral  offense   mean
#     L5a_c40    76.3%   97.3%    80.0%    65.1%    66.8%    77.0%   77.1%
#     L3f_c80    57.1%   87.2%    79.5%    74.3%    50.6%    78.5%   71.2%
#     L3h_c80    61.5%   77.3%    25.9%    67.5%       -        -    58.1%
#     L2b_c40    20.6%   98.1%     4.3%   100.0%       -        -    55.7%
#     v255        0.0%   98.0%       -        -        -        -    49.0%
#
# L5a_c40 is best or second on five of six. It was trained only on round-4 head-on merges
# at 3048-5000 m and generalised to every band, which is the opposite of what the astern
# spawn pools produced -- learning from the hardest geometry transferred, learning from
# gifted positions did not.
#
# CHECKPOINT MATTERS: the same run gives 77.1% at c40, 76.5% at c60 and 50.6% at c80.
# Judging on the final checkpoint would have thrown the best model away.
#
# v255 is no longer defensible: at competition distances it loses every match to the
# pursuer family and flies into the ground in 46% of them.
# In selector mode this bundle supplies the shared observation/runtime contract only.
# Point it at a bundle that is actually included in the minimal final package.
BUNDLE_DIR = os.getenv(
    "TEAM01_BUNDLE_DIR",
    "artifacts/models/team01/D4_def/bundle_000090",
)
CUTIN_BUNDLE_DIR = os.getenv(
    "TEAM01_CUTIN_BUNDLE_DIR",
        # V142 is explicitly rejected in CURRENT_RL_STATUS_AND_PLAN_2026-07-30.md.
    # V145 checkpoint 45 is the specialist that passed the Stage 3B-3 gate (11/12,
    # side B 4/5) once the evaluation pair was a true reflection.
    "artifacts/models/team01/v145_repeat6_residual_entry_50i",
)
OBSERVATION_MODE = "team01_tactical32"
DEFAULT_OBSERVATION_MODULE = "student.team01_phase_observation"
OBSERVATION_MODULE_OVERRIDE = os.getenv("TEAM01_OBSERVATION_MODULE", "").strip()

# OPPONENT-TYPE SELECTOR — on by default, TEAM01_SELECTOR=0 disables it.
#
# WHY THIS SHAPE, AFTER SEVEN THAT FAILED
#   Seven geometric handoff gates and a forward-default selector all lost to the best solo
#   policy, always the same way: they took control from the primary in the middle of a
#   solution it was already converting. This one switches at most once, near the start, on
#   WHO we are fighting rather than on momentary range and angle.
#
#   The statistic is the share of the first 30 s in which our own nose is within 30 deg of
#   the opponent. Measured across six opponents it orders them exactly by our win rate
#   (r = -0.969):
#       weapon_v7 72.2%   defense 71.7%   neutral 70.9%   offense 59.6%   champion 16.5%
#   Low means we are the one being pressed.
#
#   INVERTED on purpose: the merge specialist is the DEFAULT and the converter is what we
#   switch TO. Thirty seconds of the wrong policy against champion is unrecoverable -- a
#   forward default scored 4W-14L where the specialist alone scored 12W-9L -- while thirty
#   seconds of the specialist against an opponent we already beat costs almost nothing.
#
#   Match win probability (best-of-3, round-4 merge as tiebreak), four opponents both
#   configurations have been measured on:
#                          champion  weapv7  alt_denial  defense   mean
#       L5a_c40 alone        76.3%    97.3%      80.0%    65.1%    79.7%
#       L5a_c40 + L2b_c40    76.3%    98.0%      80.6%    78.4%    83.4%
#   The whole gain is the defense_specialist cell, which is the cell the detector exists to
#   catch: a passive opponent leaves our nose on it, so it reads as "weak" and control goes
#   to L2b_c40, which takes that match-up 100%. The other three cells are unchanged, i.e.
#   the detector correctly declines to switch where switching would cost.
#
# TURNED OFF 2026-08-17 — measured failure in a real competition scenario.
#   The Viewer's scenario menu is distance (2000/2500/3000 ft) x type (HABFM / OBFM_RED /
#   OBFM_BLUE). OBFM_BLUE hands US the offensive start: measured opening 556 m, our ATA
#   4.7 deg, opponent ATA 175.3 deg -- we begin on their six, inside phase-1 range, nose
#   almost on. In that match (vm_0817_1458):
#       0-30 s   L5a_c40 driving   0.60 s inside 1 deg / 152-914 m, best ATA 0.29 deg
#       30 s     selector reads "our nose on him 100.0%" and hands over to L2b_c40
#       30-200 s L2b_c40 driving   0.00 s of firing, best ATA 1.39 deg
#   The 0.60 s is the only firing this project has ever recorded in the Viewer. The selector
#   gave that position away and got nothing back for the remaining 170 s.
#
#   The defect is in the statistic, not the threshold. "Share of the first 30 s with our nose
#   within 30 deg" is high for two different reasons -- the opponent is passive, or we were
#   HANDED the offensive start -- and it cannot tell them apart. OBFM_BLUE pins it at 100%,
#   so every OBFM_BLUE round would hand control to the wrong policy. Raising the threshold
#   cannot fix this: 100% is above any threshold below 1.0.
#
#   The local gain that justified switching it on was +1.1 pp over six opponents (78.2 vs
#   77.1), inside noise, and measured on a grid that contains no OBFM geometry at all. A
#   noise-level gain on an incomplete grid does not pay for a measured loss on a real one.
#
#   Re-enable with TEAM01_SELECTOR=1 if the statistic is replaced with one that reads the
#   OPPONENT's behaviour rather than our own nose position.
SELECTOR_ENABLED = os.getenv("TEAM01_SELECTOR", "0") == "1"
SELECTOR_BUNDLE_DIR = os.getenv(
    "TEAM01_SELECTOR_BUNDLE", "artifacts/models/team01/L2b_c40"
)
SELECTOR_DECIDE_S = float(os.getenv("TEAM01_SELECTOR_DECIDE_S", "30.0"))
SELECTOR_ATA_DEG = float(os.getenv("TEAM01_SELECTOR_ATA", "30.0"))
SELECTOR_THRESHOLD = float(os.getenv("TEAM01_SELECTOR_THRESHOLD", "0.35"))

# Opening-geometry selector. See the wiring block in build_action_provider for the reasoning
# and for why it ships off. The BT is the 10-of-10-pool build; its rule XML sits beside it and
# BTActionProvider picks it up from this folder, so there is no rule-scope clash with the
# opponent's client (that runs from a different release directory).
GEO_SELECTOR_ENABLED = os.getenv("TEAM01_GEO_SELECTOR", "0") == "1"

# PHASE SELECTOR — 경기 시각으로 두 RL 정책을 나눈다.
#
# 측정 (champion 상대, 시드 9100, 칸당 6판, n=36, 지터 격자)
#   조합                      딜/판    피해/판   순딜/판   p1콘   p3콘   격추당함
#   c40 단독 (현 제출본)      0.0457   0.0966   -0.0509  0.050  0.272   1/36
#   T1_c060 -> c40, 100초     0.1040   0.1084   -0.0044  0.111  0.661   0/36
#   c40 -> T1_c060, 100초     0.0431   0.1406   -0.0976  0.050  0.275   3/36   <- 뒤집기 대조
#
# c40 은 phase 3(원뿔 3도)에서, T1_c060 은 phase 1(원뿔 1도)에서 점수를 낸다. 시간대로
# 상보적이므로 전반을 T1 이, 후반을 c40 이 맡는다. **순서를 뒤집으면 기준선보다 나빠지므로**
# 이득은 전환 자체가 아니라 배치에서 온다. 전환 시각도 100초가 옳다(150초 -0.0202).
#
# 기존 selector 3종(SEL20/TacticalHybrid/GEO)은 전부 기하를 추론했고 오분류가 실패 원인이었다.
# phase 는 시각으로 정해지므로 **오분류가 원리적으로 불가능하다.**
#
# **기본 OFF.** 로컬 격자에서 앞서지만 뷰어에서 확인되지 않았다. 이 격자는 뷰어를 두 번
# 오예측한 적이 있고(habfm_914 를 무추락 무승부로, habfm_762 를 무피해 무승부로), 무엇보다
# 100초 전환 순간의 명령 불연속이 실기에서 어떻게 나타나는지는 뷰어로만 알 수 있다.
# 뷰어에서 단독 정책을 이긴 뒤에 켠다.
PHASE_SELECTOR_ENABLED = os.getenv("TEAM01_PHASE_SELECTOR", "0") == "1"
PHASE_EARLY_BUNDLE = os.getenv(
    "TEAM01_PHASE_EARLY_BUNDLE",
    "artifacts/models/team01/T1_ladder_mild_100i/bundle_000060",
)
PHASE_SWITCH_S = float(os.getenv("TEAM01_PHASE_SWITCH_S", "100.0"))
GEO_BT_DLL = os.getenv("TEAM01_GEO_BT_DLL", "AIP_Team01_v12_stable.dll")
GEO_NEUTRAL_LO = float(os.getenv("TEAM01_GEO_LO", "45.0"))
GEO_NEUTRAL_HI = float(os.getenv("TEAM01_GEO_HI", "135.0"))

# BT 모드 설정
# - 이 제출본은 MODE="rl"(순수 강화학습)입니다. 팀 BT DLL/XML을 동봉하지 않으므로
#   BT/hybrid 경로는 대회 경기에서 사용되지 않습니다.
# - 이전 기본값은 AIP_DCS_climbdive.dll / Rule_team01_climbdive.xml 이었는데 두 파일 모두
#   이 폴더에 없습니다(개발 폴더에만 존재). TEAM01_MODE=bt 를 실수로 켜면 그 자리에서
#   CreateBehaviorTree 가 실패합니다. 그래서 실제로 동봉된 스톡 파일을 가리키도록 바꿉니다.
# - 팀별 BT DLL/XML을 제출하게 되면 두 파일을 Release 루트에 두고 아래 이름을 바꾸세요.
#   설명회 자료 57p 경고대로 DLL과 XML은 한 세트이므로 XML만 이름을 바꾸면 안 됩니다
#   (XML 이름은 CPPBehaviorTree.cpp 에 하드코딩되어 함께 빌드됩니다).
BT_DLL = os.getenv("TEAM01_BT_DLL", "AIP_Team01_v12_stable.dll")
BT_RULE_XML = os.getenv("TEAM01_BT_RULE", "Rule_codex_champion_v1.xml")

# ── 상황별 전문가 selector (MODE="selector", 기본값) ──────────────────────────
# 2026-08-21 상황별 격자(n=36, 상대 defense_specialist) 측정 결과와
# 이후 사다리의 완료 정점으로 구성했지만, Viewer 실전에서는 전문가 간 전환이
# 연속 기동을 끊었다. 2026-08-24 안전 오버레이 시험은 D4/C5b/F3-c80만 켜고
# 중립/위치/사격을 champion BT에 맡겨 체력 100 대 99.506 판정승을 기록했다.
# 따라서 기본값은 안전 오버레이이며 B7/E7/A5는 환경변수로 다시 켤 수 있다.
#   방어(상대가 우리 뒤)   champion BT -0.3123(킬1/사망4)  vs  D4 +0.1873(킬14/사망3)
#   교량은 B7 사다리 정점(iter 147, 보존 bundle 140, score 0.966)
#   교차방어는 C5b 사다리 정점(iter 86, 보존 bundle 80, score 0.938)
#   위치는 E7 사다리 정점(iter 65, 보존 bundle 60, WEZ/episode 67.50)
#   조준은 A5 사다리 정점(iter 54, 보존 bundle 50, WEZ/episode 44.10)
#   정면 merge 시험 후보는 corrected F3-c80(iter 79 score 0.959).
# -> 기본: 방어 D4 / 교차방어 C5b / 정면 F3-c80 / 나머지 champion BT.
#
# **위치 담당 교체는 TEAM01_POSITION_BUNDLE 한 줄만 바꾸면 된다.**
SELECTOR_DEFENSE_BUNDLE = os.getenv(
    "TEAM01_DEFENSE_BUNDLE", "artifacts/models/team01/D4_def/bundle_000090")
SELECTOR_CROSSING_BUNDLE = os.getenv(
    "TEAM01_CROSSING_BUNDLE", "artifacts/models/team01/C5_crossdefb/bundle_000080")
SELECTOR_BRIDGE_BUNDLE = os.getenv(
    "TEAM01_BRIDGE_BUNDLE", "")
SELECTOR_POSITION_BUNDLE = os.getenv(
    "TEAM01_POSITION_BUNDLE", "")
# A5 정점(iter 54, 저장 bundle 50). TEAM01_GUN_PROVIDER=bt면 champion GunTrack과 비교한다.
SELECTOR_GUN_PROVIDER = os.getenv("TEAM01_GUN_PROVIDER", "bt").strip().lower()
SELECTOR_GUN_BUNDLE = (
    ""
    if SELECTOR_GUN_PROVIDER == "bt"
    else os.getenv("TEAM01_GUN_BUNDLE", "artifacts/models/team01/A5_gun/bundle_000050")
)
SELECTOR_FRONTAL_BUNDLE = os.getenv(
    "TEAM01_FRONTAL_BUNDLE",
    "artifacts/models/team01/F3_transfer_candidate/bundle_000080",
)
# 비우면 그 전문가는 빠지고 champion BT 가 그 상황도 맡는다.
SELECTOR_GUN_RANGE_MIN_M = float(os.getenv("TEAM01_SEL_GUN_RANGE_MIN", "300"))
SELECTOR_GUN_RANGE_M = float(os.getenv("TEAM01_SEL_GUN_RANGE", "950"))
SELECTOR_GUN_ATA_DEG = float(os.getenv("TEAM01_SEL_GUN_ATA", "20"))
SELECTOR_GUN_TGT_ATA_MIN_DEG = float(os.getenv("TEAM01_SEL_GUN_TGT_ATA_MIN", "140"))
SELECTOR_ATTACK_BT_RANGE_MIN_M = float(os.getenv("TEAM01_SEL_ATTACK_BT_RANGE_MIN", "300"))
SELECTOR_ATTACK_BT_RANGE_MAX_M = float(os.getenv("TEAM01_SEL_ATTACK_BT_RANGE_MAX", "1000"))
SELECTOR_ATTACK_BT_OWN_ATA_DEG = float(os.getenv("TEAM01_SEL_ATTACK_BT_OWN_ATA", "50"))
SELECTOR_ATTACK_BT_TGT_ATA_MIN_DEG = float(os.getenv("TEAM01_SEL_ATTACK_BT_TGT_ATA_MIN", "100"))
SELECTOR_FRONTAL_ATA_DEG = float(os.getenv("TEAM01_SEL_FRONTAL_ATA", "25"))
SELECTOR_FRONTAL_EXIT_ATA_DEG = float(os.getenv("TEAM01_SEL_FRONTAL_EXIT_ATA", "35"))
SELECTOR_FRONTAL_RANGE_MIN_M = float(os.getenv("TEAM01_SEL_FRONTAL_RANGE_MIN", "500"))
SELECTOR_FRONTAL_RANGE_MAX_M = float(os.getenv("TEAM01_SEL_FRONTAL_RANGE_MAX", "4800"))
# 진입 확인 호출 수(연속). F만 별도의 짧은 완화 유지 구간을 쓴다.
SELECTOR_ENTER_CONFIRM = int(os.getenv("TEAM01_SEL_ENTER_CONFIRM", "8"))
SELECTOR_GUN_ENTER_CONFIRM = int(os.getenv("TEAM01_SEL_GUN_CONFIRM", "2"))
SELECTOR_ATTACK_BT_ENTER_CONFIRM = int(os.getenv("TEAM01_SEL_ATTACK_BT_CONFIRM", "2"))
SELECTOR_FRONTAL_ENTER_CONFIRM = int(os.getenv("TEAM01_SEL_FRONTAL_CONFIRM", "3"))
SELECTOR_FRONTAL_HOLD_CALLS = int(os.getenv("TEAM01_SEL_FRONTAL_HOLD", "24"))
SELECTOR_HOLD_CALLS = int(os.getenv("TEAM01_SEL_HOLD", "30"))
# 고도 하한: 이 아래면 무조건 BT. BT 의 CombatClimb(2000m 발동)에 여유를 준다.
SELECTOR_ALT_FLOOR_M = float(os.getenv("TEAM01_SEL_ALT_FLOOR", "1200"))

# Hybrid 모드 설정 (MODE="hybrid" 일 때만 사용)
HYBRID_MODE = "tactical"   # "residual" | "blend" | "switch" | "tactical"
RESIDUAL_SCALE = 0.35      # residual 모드 강도 (0~1, 클수록 RL 비중 증가)
ALPHA = 0.5                # blend 모드 비율 (alpha × RL + (1-alpha) × BT)

# 연결 설정
AI_TYPE = AIType.ReinforcementLearning
HEARTBEAT_SEC = 1.0
COMMAND_DELAY_SEC = 0.0
RECV_TIMEOUT_SEC = 0.2
ACTION_REPEAT = 6          # 학습 step_ratio=6과 맞춰 6개 PlaneInfo pair마다 새 policy 호출
DEBUG_ACTION_REPEAT = False
DECISION_LOG = "artifacts/logs/team01/battleviewer_tactical_decisions.csv"
TWO_CIRCLE_LOG = os.getenv(
    "TEAM01_TWO_CIRCLE_LOG",
    "artifacts/logs/team01/battleviewer_two_circle_detection.csv",
)
TURN_CIRCLE_LOG = os.getenv(
    "TEAM01_TURN_CIRCLE_LOG",
    "artifacts/logs/team01/battleviewer_turn_circle_shadow.csv",
)
CHORD_CUT_LOG = os.getenv(
    "TEAM01_CHORD_CUT_LOG",
    "artifacts/logs/team01/battleviewer_chord_cut.csv",
)
PROVIDER_TRACE = os.getenv(
    "TEAM01_DECISION_TRACE",
    "artifacts/logs/team01/battleviewer_provider_trace.csv",
)
OBSERVATION_LOG = os.getenv(
    "TEAM01_OBS_LOG",
    "artifacts/logs/team01/battleviewer_observations.csv",
)


def _load_bundle_env_config() -> dict:
    """Read the deployment contract stored with the selected RL bundle."""
    if MODE == "bt":
        return {}
    metadata_path = ROOT / BUNDLE_DIR / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return (metadata.get("algorithm_config") or {}).get("env_config") or {}
    except (OSError, ValueError, TypeError):
        return {}


def _resolve_observation_module(env_config: dict) -> str:
    """Use the observation module saved with the selected RL bundle."""
    if OBSERVATION_MODULE_OVERRIDE:
        return OBSERVATION_MODULE_OVERRIDE
    if MODE == "bt":
        return DEFAULT_OBSERVATION_MODULE

    metadata_path = ROOT / BUNDLE_DIR / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        bundle_meta = metadata.get("metadata") or {}
        module_name = str(bundle_meta.get("observation_module") or "").strip()
        if not module_name:
            module_name = str(env_config.get("observation_module") or "").strip()
        return module_name or DEFAULT_OBSERVATION_MODULE
    except (OSError, ValueError, TypeError):
        return DEFAULT_OBSERVATION_MODULE


def _resolve_viewer_contract(env_config: dict) -> tuple[int, float, int]:
    """Mirror training cadence and opening throttle unless explicitly overridden."""
    action_repeat = int(os.getenv(
        "TEAM01_ACTION_REPEAT",
        str(env_config.get("step_ratio", ACTION_REPEAT)),
    ))
    opening_floor = float(os.getenv(
        "TEAM01_OPENING_THROTTLE_FLOOR",
        str(env_config.get("opening_throttle_floor", 0.0)),
    ))
    opening_updates = int(os.getenv(
        "TEAM01_OPENING_THROTTLE_UPDATES",
        str(env_config.get("opening_throttle_steps", 0)),
    ))
    return max(1, action_repeat), opening_floor, max(0, opening_updates)


# =============================================================================
# 예시: 학습 결과 확인 (로컬 테스트용 백엔드)
# =============================================================================
# 경진대회 제출 전 로컬에서 결과 확인:
#   python run_local_dogfight.py \\
#       --ownship-backend rl \\
#       --ownship-bundle-dir artifacts/models/team01/v1 \\
#       --target-backend bt \\
#       --save-log


# =============================================================================
# 실행 로직 (수정 불필요)
# =============================================================================

def build_action_provider():
    if MODE == "scripted":
        # Physics probe: ignore observations entirely and fly a fixed command schedule,
        # so a Viewer run can be compared against the identical schedule replayed in the
        # local sim.  Isolates environment differences from controller differences.
        from dogfight.ai.action_provider import ActionProvider, ActionResult
        import csv as _csv

        class _ScriptedProvider(ActionProvider):
            SEGMENTS = [
                (50, [0.00, 0.00, 0.0, 0.90]),
                (50, [0.35, 0.25, 0.0, 0.90]),
                (50, [0.00, 0.30, 0.0, 0.90]),
                (50, [-0.35, 0.25, 0.0, 0.90]),
                (50, [0.00, -0.15, 0.0, 0.60]),
            ]

            def __init__(self):
                self.n = 0
                self.rows = []
                self.path = ROOT / "probe_viewer.csv"

            def reset(self, context=None):
                return None

            def _action_for(self, step):
                t = 0
                for count, act in self.SEGMENTS:
                    if step < t + count:
                        return np.array(act, dtype=np.float32)
                    t += count
                return np.array(self.SEGMENTS[-1][1], dtype=np.float32)

            def compute_action(self, context):
                own = (context.info or {}).get("my_plane_data")
                if own is not None:
                    self.rows.append([self.n, own.LocationX, own.LocationY, own.LocationZ,
                                      own.Roll, own.Pitch, own.Yaw, own.Speed])
                    if self.n % 10 == 0:
                        with self.path.open("w", newline="", encoding="utf-8") as fh:
                            w = _csv.writer(fh)
                            w.writerow(["step", "x", "y", "z", "roll", "pitch", "yaw", "speed"])
                            w.writerows(self.rows)
                act = self._action_for(self.n)
                self.n += 1
                return ActionResult(action=act, source="scripted", confidence=1.0, info={})

        print(f"[{TEAM_NAME}] SCRIPTED physics probe: fixed command schedule")
        return _ScriptedProvider()

    if MODE == "bt":
        print(f"[{TEAM_NAME}] BT 백엔드 사용: {BT_DLL}")
        return BTActionProvider(dll_name=BT_DLL)

    if MODE == "selector":
        # 기본 모드. 하나의 Ray 런타임에서 모든 경량 전문가를 함께 유지한다.
        # 이 플래그가 없으면 다음 번들을 읽을 때 앞서 만든 전문가 런타임을 종료한다.
        os.environ["DOGFIGHT_MULTI_BUNDLE_RUNTIME"] = "1"
        from student.specialist_selector import SpecialistSelector

        bt = BTActionProvider(dll_name=BT_DLL)
        print(f"[{TEAM_NAME}] selector 바탕 BT: {BT_DLL} / {BT_RULE_XML}")

        def _rl(rel: str, label: str):
            if not rel:
                return None
            path = ROOT / rel
            if not path.exists():
                print(f"[{TEAM_NAME}] {label} 번들 없음 -> BT 가 대신함: {path}")
                return None
            print(f"[{TEAM_NAME}] {label} 전문가: {rel}")
            return RLActionProvider(
                bundle_dir=str(path), algorithm_factory=build_algorithm_from_bundle)

        sel = SpecialistSelector(
            bt,
            defense_provider=_rl(SELECTOR_DEFENSE_BUNDLE, "방어(D4)"),
            crossing_provider=_rl(SELECTOR_CROSSING_BUNDLE, "교차방어(C5b)"),
            bridge_provider=_rl(SELECTOR_BRIDGE_BUNDLE, "교량(B7)"),
            position_provider=_rl(SELECTOR_POSITION_BUNDLE, "위치(E7)"),
            gun_provider=_rl(SELECTOR_GUN_BUNDLE, "조준(A5)"),
            frontal_provider=_rl(SELECTOR_FRONTAL_BUNDLE, "정면 merge(F3-c80)"),
            gun_range_min_m=SELECTOR_GUN_RANGE_MIN_M,
            gun_range_m=SELECTOR_GUN_RANGE_M,
            gun_ata_deg=SELECTOR_GUN_ATA_DEG,
            gun_tgt_ata_min_deg=SELECTOR_GUN_TGT_ATA_MIN_DEG,
            attack_bt_range_min_m=SELECTOR_ATTACK_BT_RANGE_MIN_M,
            attack_bt_range_max_m=SELECTOR_ATTACK_BT_RANGE_MAX_M,
            attack_bt_own_ata_deg=SELECTOR_ATTACK_BT_OWN_ATA_DEG,
            attack_bt_tgt_ata_min_deg=SELECTOR_ATTACK_BT_TGT_ATA_MIN_DEG,
            frontal_ata_deg=SELECTOR_FRONTAL_ATA_DEG,
            frontal_exit_ata_deg=SELECTOR_FRONTAL_EXIT_ATA_DEG,
            frontal_range_min_m=SELECTOR_FRONTAL_RANGE_MIN_M,
            frontal_range_max_m=SELECTOR_FRONTAL_RANGE_MAX_M,
            enter_confirm_calls=SELECTOR_ENTER_CONFIRM,
            gun_enter_confirm_calls=SELECTOR_GUN_ENTER_CONFIRM,
            attack_bt_enter_confirm_calls=SELECTOR_ATTACK_BT_ENTER_CONFIRM,
            frontal_enter_confirm_calls=SELECTOR_FRONTAL_ENTER_CONFIRM,
            frontal_hold_calls=SELECTOR_FRONTAL_HOLD_CALLS,
            hold_calls=SELECTOR_HOLD_CALLS,
            transition_log_path=ROOT / "artifacts/logs/team01/selector_mode_trace.csv",
            bt_altitude_floor_m=SELECTOR_ALT_FLOOR_M,
        )
        gun_label = "A5" if SELECTOR_GUN_BUNDLE else "BT"
        defense_label = "D4" if SELECTOR_DEFENSE_BUNDLE else "BT"
        crossing_label = "C5b" if SELECTOR_CROSSING_BUNDLE else "BT"
        frontal_label = "F3-c80" if SELECTOR_FRONTAL_BUNDLE else "BT"
        position_label = "E7" if SELECTOR_POSITION_BUNDLE else "BT"
        bridge_label = "B7" if SELECTOR_BRIDGE_BUNDLE else "BT"
        print(
            f"[{TEAM_NAME}] selector: 완전사격창->{gun_label} / 부분공격창->BT / "
            f"완전후방->{defense_label} / 정면merge->{frontal_label} / "
            f"교차위협->{crossing_label} / astern->{position_label} / "
            f"중립교착->{bridge_label} / 나머지->BT"
        )
        return sel

    bundle_path = ROOT / BUNDLE_DIR
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"모델 번들을 찾을 수 없습니다: {bundle_path}\n"
            f"먼저 학습을 완료하고 BUNDLE_DIR 경로를 확인하세요."
        )

    print(f"[{TEAM_NAME}] RL 모델 로드: {bundle_path}")
    if MODE == "hybrid" and HYBRID_MODE == "tactical":
        # Loading the PPO specialist must not restart Ray underneath the SAC
        # gun policy. Both lightweight models share one local runtime.
        os.environ["DOGFIGHT_MULTI_BUNDLE_RUNTIME"] = "1"

    rl_provider = RLActionProvider(
        bundle_dir=str(bundle_path),
        algorithm_factory=build_algorithm_from_bundle,
    )

    if MODE == "rl":
        # Ground-collision guard. Over a 300-match local screen c225 crashed 23 times, and
        # against the weapon_v7 BT EVERY loss was a crash (12/12) with zero losses to enemy
        # fire. Adding this took the same screen from 38%/48% win/loss to 57%/28% -- and wins
        # went UP (23 -> 33), so it does not cost the tracking solution. It intervenes on
        # about 5% of steps. See student/floor_guard.py for the measured sign conventions.
        # V255 is submitted as the exact single-model policy that passed the
        # clean local gate. Keep experimental wrappers opt-in so the default
        # submission path cannot silently alter its controls.
        if os.getenv("TEAM01_FLOOR_GUARD", "0") == "1":
            from student.floor_guard import FloorGuardProvider

            guarded = FloorGuardProvider(
                rl_provider,
                reserve_ratio=float(os.getenv("TEAM01_GUARD_RESERVE", "2.0")),
                min_arm_altitude_m=float(os.getenv("TEAM01_GUARD_MIN_ARM", "2200")),
            )
            print(f"[{TEAM_NAME}] RL 전용 모드 + 추락 가드")
            return guarded
        if SELECTOR_ENABLED:
            from student.opponent_type_selector import OpponentTypeSelector

            secondary = RLActionProvider(
                bundle_dir=str(ROOT / SELECTOR_BUNDLE_DIR),
                algorithm_factory=build_algorithm_from_bundle,
            )
            rl_provider = OpponentTypeSelector(
                rl_provider,
                secondary,
                dt=0.1,
                decide_after_s=SELECTOR_DECIDE_S,
                pursued_ata_deg=SELECTOR_ATA_DEG,
                pursued_threshold=SELECTOR_THRESHOLD,
                invert=True,
            )
            print(
                f"[{TEAM_NAME}] opponent-type selector: default={BUNDLE_DIR} "
                f"switch={SELECTOR_BUNDLE_DIR} decide@{SELECTOR_DECIDE_S:.0f}s "
                f"ownATA<{SELECTOR_ATA_DEG:.0f} thr={SELECTOR_THRESHOLD:.2f}"
            )
        # OPENING-GEOMETRY SELECTOR — off by default, TEAM01_GEO_SELECTOR=1 enables it.
        #
        # Picks the controller ONCE on frame 1 and never switches. Neutral opening (HABFM,
        # measured 91 deg) flies the RL bundle; a decided opening (OBFM, measured 4.7 deg when
        # we start behind and 175.3 deg when they do) flies the native BT.
        #
        # Why this shape: every structure that swapped control DURING a match has lost here --
        # seven RL->RL gates, the 30 s opponent-type selector above, and BT-manoeuvres/RL-shoots
        # (which dealt 0.05-0.09 where either alone dealt 0.21). Deciding before anything happens
        # removes the handoff entirely. The signal is unambiguous: the three Viewer scenarios sit
        # 90 deg apart, so a 45-135 band has 40+ deg of margin, unlike the 30 s statistic which
        # had to separate offense 43.6% from neutral 45.6%.
        #
        # NOT YET VERIFIED IN THE VIEWER. On the local BFM grid it led on single-match expected
        # value (0.675 vs RL 0.500 vs BT 0.450), but that grid has twice mispredicted the Viewer
        # today: it called habfm_914 a draw with 0 crashes where c40 actually crashed, and called
        # habfm_762 a no-damage draw where we actually took hits. It also rates the BT above the
        # RL on obfm_blue, while in the Viewer the RL won that cell outright. So this stays off
        # until it beats the shipped solo policy in the Viewer, not on the grid.
        if GEO_SELECTOR_ENABLED:
            from student.opening_geometry_selector import OpeningGeometrySelector

            bt_mover = BTActionProvider(dll_name=GEO_BT_DLL)
            rl_provider = OpeningGeometrySelector(
                rl_provider,
                bt_mover,
                neutral_lo_deg=GEO_NEUTRAL_LO,
                neutral_hi_deg=GEO_NEUTRAL_HI,
            )
            print(
                f"[{TEAM_NAME}] opening-geometry selector: neutral "
                f"{GEO_NEUTRAL_LO:.0f}-{GEO_NEUTRAL_HI:.0f}deg -> RL({BUNDLE_DIR}), "
                f"else BT({GEO_BT_DLL})"
            )
        if PHASE_SELECTOR_ENABLED:
            from student.phase_selector import PhaseSelector

            early_dir = ROOT / PHASE_EARLY_BUNDLE
            early_provider = RLActionProvider(
                bundle_dir=str(early_dir),
                algorithm_factory=build_algorithm_from_bundle,
                policy_id="default_policy",
            )
            # 전반 = 조준 특화(T1), 후반 = 현 제출 정책(c40).
            rl_provider = PhaseSelector(
                early_provider, rl_provider, switch_s=PHASE_SWITCH_S
            )
            print(
                f"[{TEAM_NAME}] phase selector: t<{PHASE_SWITCH_S:.0f}s -> "
                f"{PHASE_EARLY_BUNDLE}, t>={PHASE_SWITCH_S:.0f}s -> {BUNDLE_DIR}"
            )
        print(f"[{TEAM_NAME}] RL 전용 모드 (가드 비활성)")
        return rl_provider

    # hybrid
    bt_provider = BTActionProvider(dll_name=BT_DLL)
    if HYBRID_MODE == "tactical":
        cutin_bundle_path = ROOT / CUTIN_BUNDLE_DIR
        cutin_rl_provider = None
        if cutin_bundle_path.exists():
            print(f"[{TEAM_NAME}] PPO cut-in model: {cutin_bundle_path}")
            cutin_rl_provider = RLActionProvider(
                bundle_dir=str(cutin_bundle_path),
                algorithm_factory=build_algorithm_from_bundle,
            )
        else:
            print(
                f"[{TEAM_NAME}] PPO cut-in model missing; selector disabled: "
                f"{cutin_bundle_path}"
            )
        print(
            f"[{TEAM_NAME}] Tactical hybrid: BT maneuver + PPO cut-in + "
            "RL gun-window aim"
        )
        return TacticalHybridActionProvider(
            rl_provider=rl_provider,
            bt_provider=bt_provider,
            cutin_rl_provider=cutin_rl_provider,
            cutin_max_updates=int(os.getenv("TEAM01_CUTIN_UPDATES", "100")),
            cutin_throttle_floor=float(
                os.getenv("TEAM01_CUTIN_THROTTLE_FLOOR", "0.88")
            ),
            cutin_exit_ata_deg=float(os.getenv("TEAM01_CUTIN_EXIT_ATA", "45")),
            # V145's handoff distribution was own ATA ~50-56 deg; arm only near it.
            cutin_min_own_ata_deg=float(os.getenv("TEAM01_CUTIN_MIN_ATA", "40")),
            cutin_max_own_ata_deg=float(os.getenv("TEAM01_CUTIN_MAX_ATA", "75")),
            cutin_exit_target_ata_deg=float(
                os.getenv("TEAM01_CUTIN_EXIT_TARGET_ATA", "60")
            ),
            # The real fights descend well below 4 km, so the default 4000 m RL gate
            # kept the aimer OFF the whole match (0 hand-offs = pure BT mirror draw).
            # Lower it so RL takes over the LOW gun windows; the dive-guard still
            # protects the floor. Env-overridable.
            safe_rl_altitude_m=float(os.getenv("TEAM01_RL_ALT_GATE", "1200")),
            # The climb-dive mover WINS by taking the fight to ~13 km and attacking in a
            # dive, so the hybrid's own anti-zoom guard (5800 m) would fight its own mover,
            # and the -5 deg pitch gate would switch the aimer off during every attack.
            zoom_ceiling_m=float(os.getenv("TEAM01_ZOOM_CEIL", "14500")),
            safe_rl_pitch_deg=float(os.getenv("TEAM01_RL_PITCH_GATE", "-40")),
            # Local full-stack sweep: at the stock 0.60/55 deg the aimer OVERRODE the
            # mover and lost the beam_L kill. 0.30/25 deg keeps the BT in charge and only
            # trims the nose in the true terminal window -> kills on BOTH geometries.
            aim_residual_scale=float(os.getenv("TEAM01_AIM_SCALE", "0.30")),
            gun_min_range_m=float(os.getenv("TEAM01_GUN_MIN_RANGE", "300")),
            gun_max_range_m=float(os.getenv("TEAM01_GUN_MAX_RANGE", "1219.2")),
            gun_ata_deg=float(os.getenv("TEAM01_GUN_ATA", "18")),
            gun_exit_max_range_m=float(os.getenv("TEAM01_GUN_EXIT_RANGE", "1500")),
            gun_exit_ata_deg=float(os.getenv("TEAM01_GUN_EXIT_ATA", "35")),
            terminal_full_attitude=True,
            terminal_keep_bt_throttle=True,
            # Hand the damage band back to the mover: RL trims the nose on the approach,
            # the BT owns the terminal shot (a wide gate inside the band cancelled its kill).
            gun_handback_range_m=float(os.getenv("TEAM01_HANDBACK", "0")),
            # The dive guard (alt<5000 & pitch<-6 -> forced pull-up) cancelled every diving
            # attack our climb-dive mover makes.  The node has its own 1200 m hard deck, so
            # only keep a genuine ground guard here.
            dive_guard_altitude_m=float(os.getenv("TEAM01_DIVE_GUARD_ALT", "2200")),
            dive_guard_pitch_deg=float(os.getenv("TEAM01_DIVE_GUARD_PITCH", "-30")),
            decision_log_path=str(ROOT / DECISION_LOG),
            two_circle_log_path=str(ROOT / TWO_CIRCLE_LOG),
            turn_circle_log_path=str(ROOT / TURN_CIRCLE_LOG),
            chord_cut_log_path=str(ROOT / CHORD_CUT_LOG),
            enable_turn_circle_cut=os.getenv(
                "TEAM01_CHORD_CUT_ENABLE", "0"
            ).strip().lower() in {"1", "true", "yes", "on"},
        )
    print(f"[{TEAM_NAME}] Hybrid 모드: {HYBRID_MODE} (scale={RESIDUAL_SCALE}, alpha={ALPHA})")
    return HybridActionProvider(
        primary_provider=rl_provider,
        secondary_provider=bt_provider,
        mode=HYBRID_MODE,
        alpha=ALPHA,
        residual_scale=RESIDUAL_SCALE,
    )


def _eager_create_bt(provider):
    """Build the native BT while OUR Rule XML is still the active one.

    Both clients copy their rule onto the shared Rule_forTraining.xml, and the BT is
    otherwise created lazily on the first PlaneInfo -- by then the opponent client has
    overwritten that file, so the tree gets built from the WRONG rule and
    CreateBehaviorTree throws (Windows 0xe06d7363).  Creating it here removes the race.
    """
    from dogfight.ai.bt_action_provider import BTActionProvider, REMOTE_BT_FIGHTER_ID

    seen = []
    for attr in ("bt_provider", "secondary_provider", "rl_provider", "primary_provider"):
        seen.append(getattr(provider, attr, None))
    seen.append(provider)
    for candidate in seen:
        if isinstance(candidate, BTActionProvider):
            try:
                candidate._ensure_remote_behavior_tree(REMOTE_BT_FIGHTER_ID, 1)
                print(f"[{TEAM_NAME}] BT pre-built from {BT_RULE_XML}")
            except Exception as exc:  # noqa: BLE001
                print(f"[{TEAM_NAME}] BT pre-build failed: {exc}")
            return


def main():
    print(f"=== {TEAM_NAME} 경진대회 클라이언트 시작 ===")
    print(f"서버: {SERVER_IP}:{SERVER_PORT} ({SERVER_ENDPOINT_SOURCE})")
    print(f"모드: {MODE}")
    if MODE in {"bt", "hybrid", "selector"}:
        print(f"BT DLL/XML: {BT_DLL} / {BT_RULE_XML}")

    observation_log_path = ROOT / OBSERVATION_LOG
    observation_log_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["TEAM01_OBS_LOG"] = str(observation_log_path)
    bundle_env_config = _load_bundle_env_config()
    observation_module = _resolve_observation_module(bundle_env_config)
    action_repeat, opening_floor, opening_updates = _resolve_viewer_contract(
        bundle_env_config
    )
    print(f"[{TEAM_NAME}] observation module: {observation_module}")
    print(
        f"[{TEAM_NAME}] viewer contract: action_repeat={action_repeat}, "
        f"opening_throttle={opening_floor:.2f} for {opening_updates} updates"
    )

    with activate_rule_xml(BT_RULE_XML, ROOT):
        action_provider = build_action_provider()
        _eager_create_bt(action_provider)
        observation_hook = (
            load_observation_hook(observation_module)
            if observation_module
            else None
        )
        command_policy = ProviderCommandPolicy(
            action_provider=action_provider,
            observation_mode=observation_hook["mode"] if observation_hook else OBSERVATION_MODE,
            observation_fn=observation_hook["build_observation"]
            if observation_hook
            else None,
            ownship_force_side=1,
            target_force_side=2,
            action_repeat=action_repeat,
            debug_action_repeat=DEBUG_ACTION_REPEAT,
            decision_trace_path=str(ROOT / PROVIDER_TRACE),
            opening_throttle_floor=opening_floor,
            opening_throttle_policy_updates=opening_updates,
        )

        client = UnrealAIPilotUDPClient(
            command_policy=command_policy,
            server_ip=SERVER_IP,
            server_port=SERVER_PORT,
            team_name=TEAM_NAME,
            ai_type=AI_TYPE,
            heartbeat_interval_sec=HEARTBEAT_SEC,
            command_delay_sec=COMMAND_DELAY_SEC,
            recv_timeout_sec=RECV_TIMEOUT_SEC,
            enable_terminal_monitor=True,   # 패킷 모니터 표시
        )

        try:
            client.run()
        finally:
            action_provider.close()
            print(f"[{TEAM_NAME}] 클라이언트 종료")


if __name__ == "__main__":
    main()
