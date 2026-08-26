# -*- coding: utf-8 -*-
"""상황별 전문가 selector — 기하 관측으로 고른다.

왜 이것이 기존 selector 들과 다른가
-----------------------------------
이 프로젝트의 selector 는 전부 실패했다(SEL20, GEO, TacticalHybrid, gun_handoff).
공통 원인은 **상대 유형을 추론**해서 판단했다는 것이다. "지금 저 상대가 방어형인가"를
맞춰야 했고, 틀리면 엉뚱한 컨트롤러가 조종간을 잡았다.

**여기서는 아무것도 추론하지 않는다.** 우리 기수와 상대 기수, 그리고 거리 — 전부
`ActionContext` 에서 직접 읽는 관측값이다. 오분류라는 개념이 성립하지 않는다.

측정 근거 (2026-08-21, 상황별 격자 n=36, 상대 defense_specialist)
------------------------------------------------------------------
    상황                        champion BT              RL 전문가
    방어(상대가 우리 뒤)         -0.3123 (킬1/사망4)      D4 +0.1873 (킬14/사망3)   -> D4 +0.4996
    astern(우리가 상대 뒤)       +0.4995 (킬17/사망4)     E1 +0.6088 (킬18/사망0)   -> E1 +0.1093
    사격창(사거리내 정렬)         +0.9446 (킬27/사망0)     A4b +0.0696 (킬3/사망4)   -> BT 압도

조준은 기본적으로 champion GunTrack 이 맡지만, Viewer 비교를 위해 검증된 A5 정점 번들을
선택적으로 꽂을 수 있다. 그 외/중립은 BT 가 맡는다.

시각이 아니라 기하로 거는 이유
------------------------------
`bt_outcome_selector.py` 시험에서 200초 경기가 **316.2초**로 계산됐다.
`compute_action` 이 정책 스텝(10Hz)이 아니라 더 자주 불린다(약 3162회 관측).
그래서 **시간 기반 조건은 쓰지 않는다.** 유지 시간도 초가 아니라 **호출 횟수**로 센다.
"""
from __future__ import annotations

import math
import os
import csv
from pathlib import Path

import numpy as np

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult
from dogfight.sim.state_schema import StateIndex


def _forward(state) -> tuple[float, float, float]:
    """자세각에서 전방벡터. 좌표는 [북, 동, 하]."""
    # FighterSim과 Viewer 변환 모두 StateIndex 자세각을 도(deg)로 제공한다.
    pitch = math.radians(float(state[StateIndex.PITCH]))
    yaw = math.radians(float(state[StateIndex.YAW]))
    cp = math.cos(pitch)
    return cp * math.cos(yaw), cp * math.sin(yaw), math.sin(pitch)


def _geometry(own, tgt) -> tuple[float, float, float]:
    """(우리 ATA, 상대 ATA, 거리). ATA 는 기수에서 상대까지의 각(도)."""
    dn = float(tgt[StateIndex.N]) - float(own[StateIndex.N])
    de = float(tgt[StateIndex.E]) - float(own[StateIndex.E])
    # D 는 아래가 양수이므로 위쪽 성분은 부호를 뒤집는다.
    du = -(float(tgt[StateIndex.D]) - float(own[StateIndex.D]))
    dist = math.sqrt(dn * dn + de * de + du * du)
    if dist < 1e-6:
        return 0.0, 0.0, 0.0

    fn, fe, fu = _forward(own)
    own_ata = math.degrees(math.acos(max(-1.0, min(1.0, (dn * fn + de * fe + du * fu) / dist))))

    gn, ge, gu = _forward(tgt)
    # 상대 기수에서 우리까지: 방향을 뒤집는다.
    tgt_ata = math.degrees(math.acos(max(-1.0, min(1.0, (-dn * gn - de * ge - du * gu) / dist))))
    return own_ata, tgt_ata, dist


class SpecialistSelector(ActionProvider):
    """champion BT 를 기본으로 두고, 관측된 기하에 맞는 RL 전문가에게 넘긴다."""

    BT = "bt"
    DEF = "defense"
    CROSS = "crossing"
    BRIDGE = "bridge"
    POS = "position"
    GUN = "gun"
    FRONT = "frontal"
    ATTACK_BT = "attack_bt"

    def __init__(
        self,
        bt_provider: ActionProvider,
        defense_provider: ActionProvider | None = None,
        crossing_provider: ActionProvider | None = None,
        bridge_provider: ActionProvider | None = None,
        position_provider: ActionProvider | None = None,
        gun_provider: ActionProvider | None = None,
        frontal_provider: ActionProvider | None = None,
        *,
        # 사격창: A5는 400~800m 동방향 후방추적 상태에서만 학습됐다.
        # 정면 교차/맞대응을 넘기지 않도록 거리와 상대 ATA까지 계약에 포함한다.
        gun_range_min_m: float = 300.0,
        gun_range_m: float = 950.0,
        gun_ata_deg: float = 20.0,
        gun_tgt_ata_min_deg: float = 140.0,
        # 부분 우세 공격창: A5의 완전 후방 계약 전 단계는 사격 검증이 강한 BT가 맡는다.
        attack_bt_range_min_m: float = 300.0,
        attack_bt_range_max_m: float = 1000.0,
        attack_bt_own_ata_deg: float = 50.0,
        attack_bt_tgt_ata_min_deg: float = 100.0,
        # 방어: 상대가 우리 뒤에 있고(우리 ATA 큼) 우리를 겨눈다(상대 ATA 작음).
        def_own_ata_deg: float = 90.0,
        def_tgt_ata_deg: float = 45.0,
        def_range_m: float = 1800.0,
        # 교차 방어: C5b 최종 게이트가 실제로 검증한 30~60도 / 6도 이내 위협 띠.
        # 사다리 계승만 믿고 145도까지 넓히면 검증 밖 상태에서도 C가 조종권을 잡는다.
        cross_own_ata_min_deg: float = 30.0,
        cross_own_ata_max_deg: float = 60.0,
        cross_tgt_ata_deg: float = 6.0,
        cross_range_min_m: float = 700.0,
        cross_range_m: float = 1300.0,
        # 교량: B7까지 이어진 검증 계보의 105~135도 고각 교착과 거리 1600~2500m.
        # Viewer 추적 오차와 시나리오 지터만큼만 여유를 둔다.
        bridge_own_ata_min_deg: float = 100.0,
        bridge_own_ata_max_deg: float = 140.0,
        bridge_tgt_ata_min_deg: float = 70.0,
        bridge_tgt_ata_max_deg: float = 115.0,
        bridge_range_min_m: float = 1500.0,
        bridge_range_max_m: float = 2700.0,
        # 위치: 우리가 상대 뒤에 있고(우리 ATA 작음) 상대는 우리를 못 본다.
        pos_own_ata_deg: float = 60.0,
        pos_tgt_ata_deg: float = 90.0,
        pos_range_m: float = 1600.0,
        # 정면 merge: F1-transfer가 정식 통과한 정면 접근 띠만 맡긴다.
        # 아직 armed-BT 단계인 F2는 통과 전이므로 F1의 학습 범위를 보수적으로 사용한다.
        # 2026-08-23 Viewer trace: 실제 정면 위협 띠는 1783m/ATA 24도부터
        # 300m 부근까지 47 policy update 동안 이어졌다. 기존 1800m/20도 문턱은
        # 이 구간을 전부 놓쳐 F가 한 번도 발동하지 않았다.
        frontal_ata_deg: float = 25.0,
        frontal_exit_ata_deg: float = 35.0,
        frontal_range_min_m: float = 500.0,
        frontal_range_max_m: float = 4800.0,
        # 고도 하한: 이 아래에서는 무조건 BT. **안전망일 뿐 정상 교전을 막으면 안 된다.**
        # 2026-08-22 뷰어 실전(138.6초, 우리 승)에서 우리 고도가 60초 이후 계속
        # 2500m 아래였다. 하한을 2500 으로 두면 selector 가 사실상 BT 전용이 되어
        # RL 전문가가 아예 안 쓰인다 -- 그런데 그 판은 RL 이 쓰이면서 이겼다.
        # 그래서 추락선(304.8m)의 약 4배인 1200m 로 낮춘다. 그 판의 최저는 446m 였다.
        bt_altitude_floor_m: float = 1200.0,
        # **진입 문턱 + 짧은 유지.** 2026-08-22 두 판의 트레이스로 정했다.
        #  1차(승): 진입 4회 / 전문가 사용 5.6% / 고도 최저 446m -- 이겼지만 위험했다.
        #  2차(교착): 진입 문턱 20회로 올렸더니 **한 번도 발동 안 함**.
        #             조건 충족 구간의 최장 연속이 방어 18회 / 위치 19회여서 문턱을 못 넘었다.
        #  트레이스 시뮬레이션(2차 판 기준):
        #             문턱8/유지30 -> 진입 22회, 사용 5.7%  <- 1차(5.6%)와 같은 수준
        #             문턱8/유지 0 -> 0.5% (너무 짧아 기동 자체를 못 한다)
        #             문턱5/유지30 -> 12.4% (과다)
        enter_confirm_calls: int = 8,
        gun_enter_confirm_calls: int = 2,
        attack_bt_enter_confirm_calls: int = 2,
        frontal_enter_confirm_calls: int = 3,
        frontal_hold_calls: int = 24,
        hold_calls: int = 30,
        transition_log_path: str | os.PathLike[str] | None = None,
        verbose: bool = True,
    ):
        self.bt = bt_provider
        self.defense = defense_provider
        self.crossing = crossing_provider
        self.bridge = bridge_provider
        self.position = position_provider
        self.gun = gun_provider
        self.frontal = frontal_provider
        self.gun_range_min_m = gun_range_min_m
        self.gun_range_m = gun_range_m
        self.gun_ata_deg = gun_ata_deg
        self.gun_tgt_ata_min_deg = gun_tgt_ata_min_deg
        self.attack_bt_range_min_m = attack_bt_range_min_m
        self.attack_bt_range_max_m = attack_bt_range_max_m
        self.attack_bt_own_ata_deg = attack_bt_own_ata_deg
        self.attack_bt_tgt_ata_min_deg = attack_bt_tgt_ata_min_deg
        self.def_own_ata_deg = def_own_ata_deg
        self.def_tgt_ata_deg = def_tgt_ata_deg
        self.def_range_m = def_range_m
        self.cross_own_ata_min_deg = cross_own_ata_min_deg
        self.cross_own_ata_max_deg = cross_own_ata_max_deg
        self.cross_tgt_ata_deg = cross_tgt_ata_deg
        self.cross_range_min_m = cross_range_min_m
        self.cross_range_m = cross_range_m
        self.bridge_own_ata_min_deg = bridge_own_ata_min_deg
        self.bridge_own_ata_max_deg = bridge_own_ata_max_deg
        self.bridge_tgt_ata_min_deg = bridge_tgt_ata_min_deg
        self.bridge_tgt_ata_max_deg = bridge_tgt_ata_max_deg
        self.bridge_range_min_m = bridge_range_min_m
        self.bridge_range_max_m = bridge_range_max_m
        self.pos_own_ata_deg = pos_own_ata_deg
        self.pos_tgt_ata_deg = pos_tgt_ata_deg
        self.pos_range_m = pos_range_m
        self.frontal_ata_deg = frontal_ata_deg
        self.frontal_exit_ata_deg = max(frontal_ata_deg, frontal_exit_ata_deg)
        self.frontal_range_min_m = frontal_range_min_m
        self.frontal_range_max_m = frontal_range_max_m
        self.bt_altitude_floor_m = bt_altitude_floor_m
        self.enter_confirm_calls = enter_confirm_calls
        self.gun_enter_confirm_calls = max(1, int(gun_enter_confirm_calls))
        self.attack_bt_enter_confirm_calls = max(1, int(attack_bt_enter_confirm_calls))
        self.frontal_enter_confirm_calls = max(1, int(frontal_enter_confirm_calls))
        self.frontal_hold_calls = max(0, int(frontal_hold_calls))
        self.hold_calls = hold_calls
        self.transition_log_path = Path(transition_log_path) if transition_log_path else None
        self.verbose = verbose

        self._mode = self.BT
        self._want = self.BT
        self._streak = 0
        self._held = 0
        self._calls = 0
        self.floor_calls = 0
        self.mode_calls = {
            self.BT: 0, self.DEF: 0, self.CROSS: 0,
            self.BRIDGE: 0, self.POS: 0, self.GUN: 0, self.FRONT: 0,
            self.ATTACK_BT: 0,
        }
        # 화면 보고용
        self.engaged_steps = 0
        self.total_steps = 0

    def _decide(self, own_ata: float, tgt_ata: float, dist: float, alt_m: float) -> str:
        # 0) 고도 하한이 **최우선**. 고도를 지키는 것은 BT 의 CombatClimb 뿐이다.
        if alt_m < self.bt_altitude_floor_m:
            return self.BT
        # 1) 사격창. gun_provider 가 없으면 champion GunTrack 으로 되돌아간다.
        if (
            self.gun_range_min_m <= dist <= self.gun_range_m
            and own_ata <= self.gun_ata_deg
            and tgt_ata >= self.gun_tgt_ata_min_deg
        ):
            return self.GUN if self.gun is not None else self.BT
        # 2) 부분 우세 공격창. E7이 위치만 다듬다가 가까운 사격 기회를 놓치지 않게 한다.
        # 완전 후방 정렬은 위의 A5가 우선하며, 넓은 위치 기동은 아래 E7이 계속 맡는다.
        if (
            self.attack_bt_range_min_m <= dist <= self.attack_bt_range_max_m
            and own_ata <= self.attack_bt_own_ata_deg
            and tgt_ata >= self.attack_bt_tgt_ata_min_deg
        ):
            return self.ATTACK_BT
        # 3) 정면 merge: 두 기수가 서로를 향하는 F1의 검증 범위.
        if (
            self.frontal is not None
            and self.frontal_range_min_m <= dist <= self.frontal_range_max_m
            and own_ata <= self.frontal_ata_deg
            and tgt_ata <= self.frontal_ata_deg
        ):
            return self.FRONT
        # 4) 교차 방어: C5b 최종 시나리오가 검증한 좁은 위협 띠만 맡긴다.
        if (
            self.crossing is not None
            and self.cross_range_min_m <= dist <= self.cross_range_m
            and self.cross_own_ata_min_deg <= own_ata <= self.cross_own_ata_max_deg
            and tgt_ata <= self.cross_tgt_ata_deg
        ):
            return self.CROSS
        # 5) 방어: 상대가 완전히 뒤에서 우리를 겨눈다.
        if (
            self.defense is not None
            and dist <= self.def_range_m
            and own_ata >= self.def_own_ata_deg
            and tgt_ata <= self.def_tgt_ata_deg
        ):
            return self.DEF
        # 6) 위치: 우리가 뒤를 잡았고 상대는 우리를 못 본다.
        if (
            self.position is not None
            and dist <= self.pos_range_m
            and own_ata <= self.pos_own_ata_deg
            and tgt_ata >= self.pos_tgt_ata_deg
        ):
            return self.POS
        # 7) 교량: 사격/위치/방어 어느 쪽도 아닌 중립 교착을 다음 전문가 띠로 운반한다.
        if (
            self.bridge is not None
            and self.bridge_range_min_m <= dist <= self.bridge_range_max_m
            and self.bridge_own_ata_min_deg <= own_ata <= self.bridge_own_ata_max_deg
            and self.bridge_tgt_ata_min_deg <= tgt_ata <= self.bridge_tgt_ata_max_deg
        ):
            return self.BRIDGE
        return self.BT

    def reset(self, context: ActionContext | None = None) -> None:
        """Reset selector and every expert at a match boundary."""
        self._mode = self.BT
        self._want = self.BT
        self._streak = 0
        self._held = 0
        self._calls = 0
        self.floor_calls = 0
        self.mode_calls = {
            self.BT: 0, self.DEF: 0, self.CROSS: 0,
            self.BRIDGE: 0, self.POS: 0, self.GUN: 0, self.FRONT: 0,
            self.ATTACK_BT: 0,
        }
        self.engaged_steps = 0
        self.total_steps = 0
        for provider in (
            self.bt, self.defense, self.crossing, self.bridge, self.position, self.gun,
            self.frontal,
        ):
            if provider is not None:
                provider.reset(context)

    def _log_transition(
        self,
        old_mode: str,
        new_mode: str,
        want: str,
        own_ata: float,
        tgt_ata: float,
        dist: float,
        alt_m: float,
    ) -> None:
        if self.transition_log_path is None:
            return
        try:
            path = self.transition_log_path
            path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not path.exists() or path.stat().st_size == 0
            with path.open("a", newline="", encoding="utf-8") as fp:
                writer = csv.writer(fp)
                if write_header:
                    writer.writerow(
                        ["call", "old_mode", "new_mode", "wanted_mode", "own_ata_deg",
                         "target_ata_deg", "distance_m", "altitude_m", "hold_remaining"]
                    )
                writer.writerow(
                    [self._calls, old_mode, new_mode, want, f"{own_ata:.3f}",
                     f"{tgt_ata:.3f}", f"{dist:.3f}", f"{alt_m:.3f}", self._held]
                )
        except OSError:
            # 진단 로그 실패가 조종 경로를 중단해서는 안 된다.
            pass

    def close(self) -> None:
        seen: set[int] = set()
        for provider in (
            self.bt, self.defense, self.crossing, self.bridge, self.position, self.gun,
            self.frontal,
        ):
            if provider is not None and id(provider) not in seen:
                seen.add(id(provider))
                provider.close()

    def compute_action(self, context: ActionContext) -> ActionResult:
        self._calls += 1
        self.total_steps += 1

        own = context.ownship_state
        tgt = context.target_state
        want = self.BT
        own_ata = tgt_ata = dist = alt_m = 0.0
        if own is not None and tgt is not None:
            try:
                own_ata, tgt_ata, dist = _geometry(own, tgt)
                alt_m = -float(own[StateIndex.D])
                if dist > 0.0:
                    want = self._decide(own_ata, tgt_ata, dist, alt_m)
            except (IndexError, TypeError, ValueError):
                want = self.BT
        if alt_m and alt_m < self.bt_altitude_floor_m:
            self.floor_calls += 1

        # **비대칭 히스테리시스.**
        #  - 대부분의 전문가는 조건이 깨지면 즉시 BT 로 돌아간다.
        #  - F만 좁은 완화 범위에서 제한된 호출 수 동안 유지해 merge 경계 떨림을 막는다.
        #  - 전문가로 들어가는 것은 조건이 `enter_confirm_calls` 회 **연속** 유지될 때만.
        #    경계 떨림은 막으면서 스쳐가는 상황에 갇히지 않는다.
        if want == self._want:
            self._streak += 1
        else:
            self._want = want
            self._streak = 1

        below_floor = bool(alt_m) and alt_m < self.bt_altitude_floor_m
        relaxed_front = (
            self._mode == self.FRONT
            and self.frontal is not None
            and self.frontal_range_min_m <= dist <= self.frontal_range_max_m
            and own_ata <= self.frontal_exit_ata_deg
            and tgt_ata <= self.frontal_exit_ata_deg
        )
        if below_floor:
            # 고도 하한은 유지 중에도 **즉시** 개입한다. 1차 판에서 446m 까지
            # 떨어진 것을 막는 장치이므로 어떤 유지 카운터보다 우선한다.
            new_mode = self.BT
            self._held = 0
        elif want == self.BT and relaxed_front and self._held > 0:
            # 정면 merge 경계에서 ATA가 몇 도 흔들려도 F가 한 기동을 마칠 시간을 준다.
            # 진입 문턱은 그대로 두고, 완화 범위와 호출 수를 모두 제한한다.
            new_mode = self.FRONT
            self._held -= 1
        elif want == self.BT:
            # F의 제한된 완화 유지에도 해당하지 않으면 즉시 BT로 복귀한다.
            new_mode = self.BT
            self._held = 0
        elif want == self.GUN:
            # 짧은 확인 동안에는 BT GunTrack을 유지해 순간 정렬에도 사격 기회를 잃지 않는다.
            if self._mode == self.GUN or self._streak >= self.gun_enter_confirm_calls:
                new_mode = self.GUN
                self._held = 0
            else:
                new_mode = self.BT
        elif want == self.ATTACK_BT:
            # 실제 공격창은 짧으므로 빠르게 표시하고 BT GunTrack에 계속 맡긴다.
            if self._mode == self.ATTACK_BT or self._streak >= self.attack_bt_enter_confirm_calls:
                new_mode = self.ATTACK_BT
                self._held = 0
            else:
                new_mode = self.BT
        elif want == self.FRONT:
            if self._mode == self.FRONT or self._streak >= self.frontal_enter_confirm_calls:
                new_mode = self.FRONT
                self._held = self.frontal_hold_calls
            else:
                new_mode = self.BT
        elif (
            self._mode == self.FRONT
            and want == self.CROSS
            and self._held > 0
            and self._streak < self.enter_confirm_calls
        ):
            # want=CROSS 자체가 C의 좁은 검증 띠를 만족했다는 뜻이다. F의 일반 이탈
            # ATA와 무관하게 C 확인이 끝날 때까지 BT 공백을 만들지 않는다.
            new_mode = self.FRONT
            self._held -= 1
        elif self._mode == want:
            new_mode = want
            self._held = max(0, self._held - 1)
        elif self._streak >= self.enter_confirm_calls:
            new_mode = want                         # 연속 확인 후 진입
            self._held = self.hold_calls
        else:
            # 다른 전문가 조건으로 바뀌었을 때 이전 전문가를 붙들지 않는다.
            new_mode = self.BT

        if new_mode != self._mode:
            old_mode = self._mode
            if self.verbose:
                print(
                    f"[selector] {self._mode} -> {new_mode} "
                    f"(호출 {self._calls}, ATA 우리 {own_ata:.0f}도 / 상대 {tgt_ata:.0f}도, "
                    f"거리 {dist:.0f}m, 고도 {alt_m:.0f}m)",
                    flush=True,
                )
            self._log_transition(
                old_mode, new_mode, want, own_ata, tgt_ata, dist, alt_m,
            )
            self._mode = new_mode

        self.mode_calls[self._mode] = self.mode_calls.get(self._mode, 0) + 1
        if self._mode != self.BT:
            self.engaged_steps += 1

        if self._mode == self.DEF and self.defense is not None:
            return self.defense.compute_action(context)
        if self._mode == self.CROSS and self.crossing is not None:
            return self.crossing.compute_action(context)
        if self._mode == self.BRIDGE and self.bridge is not None:
            return self.bridge.compute_action(context)
        if self._mode == self.POS and self.position is not None:
            return self.position.compute_action(context)
        if self._mode == self.GUN and self.gun is not None:
            return self.gun.compute_action(context)
        if self._mode == self.FRONT and self.frontal is not None:
            return self.frontal.compute_action(context)
        return self.bt.compute_action(context)

    def summary(self) -> str:
        n = max(1, self._calls)
        parts = [f"{k} {100.0 * v / n:.0f}%" for k, v in self.mode_calls.items()]
        parts.append(f"고도하한발동 {100.0 * self.floor_calls / n:.0f}%")
        return "  ".join(parts)
