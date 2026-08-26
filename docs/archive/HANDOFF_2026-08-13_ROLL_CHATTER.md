# 2026-08-13 — 정자세 조준 진동의 원인과 측정

작업 폴더: `Release_260722_plz`. 제출본(`Release_submit`)은 이 세션에서 건드리지 않았다.

---

## 1. 진동의 정체 (측정 완료)

실제 Viewer 트레이스 `Release_submit/artifacts/logs/team01/battleviewer_provider_trace.csv`
(policy 결정 4,445회, 그중 건 윈도우 = 거리 <1500 m & ATA <45° 안이 2,071회):

| 자세 | roll 포화 | 평균 \|Δcmd\| | roll 부호반전 | pitch 포화 |
|---|---|---|---|---|
| 정자세 (\|roll\|<60°) | 42.0% | **0.720** | **2.12 Hz** | 89.2% |
| 배면 (\|roll\|>120°) | 36.7% | 0.614 | 1.81 Hz | 90.3% |

**세 줄 요약**

1. **pitch는 죽은 축이다.** 두 자세 모두 ~90% 포화. 건 윈도우 내내 풀 당김에 박혀 있고
   부호 반전은 0.2 Hz뿐이다. 조준은 전부 roll이 한다.
2. **roll이 bang-bang이다.** 100 ms 정책 스텝마다 스틱이 평균 0.72 움직인다
   (풀스케일 2.0 대비 36%), 초당 2회 부호를 뒤집는다.
3. **기체는 못 따라간다.** 학습 리플레이 366개에서 기체의 실제 roll rate 진동은
   정자세 0.81 Hz / 배면 0.73 Hz. 명령 2.1 Hz vs 응답 0.8 Hz —
   **1 Hz 위쪽은 물리적으로 실행 불가능한 제어 에너지**다. 이게 보이는 떨림이고,
   `action_repeat=6` ZOH 때문에 100 ms 블록당 실제 실린 충격량이 거의 무작위가 된다.

**배면이 덜한 이유**: pitch가 양쪽 다 포화라 기체는 "당기기만 되는" 물체이고 조준은
뱅크각으로만 한다. 배면 당김은 중력이 도와 뱅크각이 안정적이다. 정자세 최대 G 당김은
중력과 싸우므로 같은 조준 오차에 더 크고 빠른 뱅크 보정이 필요하다. 다른 고장이 아니라
**같은 리밋 사이클이 정자세에서 더 뜨겁게 도는 것**이다.

배포는 deterministic(`tanh(mean)`, `rl_action_provider.py:163`, `explore=False`)이므로
SAC 샘플링 노이즈가 아니다. 순수한 폐루프 리밋 사이클이다.

---

## 2. 이미 있던 구현과 그 결함

| 파일 | 상태 |
|---|---|
| `student/gunline_hold.py` | 같은 아이디어. 게이트가 `ATA≤2°`, 기본 꺼짐 |
| `student/smooth_guard.py` | **존재하지 않음.** `screen_models.py --smooth`가 import → ImportError |

**게이트 크기 문제** (같은 트레이스로 측정):

| 게이트 | 커버 스텝 | 잡히는 roll 채터 |
|---|---|---|
| 기존 ATA≤2°, 152–1219 m | 2.1% | **1.7%** |
| ATA≤4° | 6.5% | 5.4% |
| d<1500 m, ATA<45° | 55.9% | **55.1%** |
| 위 + 정자세만 | 22.4% | 23.9% |

좁은 창 안에서 채터가 제일 심하지만(평균 |Δcmd| 0.861, 4.62 flips/s — 기수가 표적에
올라탔을 때 가장 심하게 헌팅한다), 채터의 98%는 접근 구간에서 일어난다.

**alpha 붕괴 버그 (수정함)**

```python
limited = last + clip(raw - last, ±rate_limit)
action[0] = alpha*raw + (1-alpha)*limited      # rate cap이 느슨하면 limited == raw
```

rate limit이 느슨하면 `alpha*raw + (1-alpha)*raw == raw`. **alpha가 아무 일도 안 했다.**
스무딩은 전부 rate limiter 혼자 하고 있었다. EMA를 이전 *출력* 기준으로 바꾸고 rate cap을
뒤에 붙였다. `alpha=1.0, rate=2.0`이 정확한 pass-through인 것은 4,445스텝 비트 일치로 검증.

---

## 3. 관측 자립화 (완료, 비트 검증)

`student/team01_phase_observation.py`는 35채널 중 3개만 만들고 32채널을 `src/`에 위임한다.
그 `src/` 코드는 **원본 Release에 없다** (`Release_260526_ori` 대조):

| 파일 | 원본 | 우리가 추가 |
|---|---|---|
| `src/dogfight/envs/observation.py` | tactical32 없음 | `_build_tactical32` 전체 |
| `src/dogfight/sim/state_schema.py` | 13개 인덱스만 | `U,V,W,P,Q,R,AOA,AOS,THROTTLE,VERTICAL_SPEED,NZ,NY` |
| `src/dogfight/config.py` | `wez_phases` 없음 | phase 테이블 |

Release 폴더를 통째로 넘기면 문제없다(실측 확인). 주최측이 `student/`만 걷어 깨끗한
Release에 넣으면 즉사한다.

→ `student/team01_phase_observation_standalone.py` 작성. numpy와 넘겨받는 `GeometryInfo`
외 의존 없음. `scratch_obs_equivalence.py`가 랜덤 20,000 + 코너 69개(배면, yaw wrap,
phase 경계 8개, 표적 수직 상하, 영속도)를 비교 → **20,069개 전부 비트 일치**.

**아직 교체하지 않았다.** 스윕이 기존 모듈을 쓰는 중이라 끝난 뒤 스왑할 것.

---

## 3.5. `screen_models.py`는 배포 케이던스를 재현한 적이 없다 (수정함)

**이게 오늘 찾은 것 중 가장 파급이 크다.**

`DogFightEnv._step_controlled_aircraft`는 `for _ in range(step_ratio)` 프레임 루프
**안에서** 호출되고, 그 안에서 ownship provider를 매 프레임 폴링한다. 직접 계측:

```
step_ratio=1:  env steps 11999   provider calls 11999
step_ratio=6:  env steps  2000   provider calls 12000
```

`step_ratio`는 env-step 단위를 바꿀 뿐 **provider 호출 주기를 바꾸지 않는다.**
200초 경기에서 둘 다 12,000회다 = 60 Hz.

배포는 정반대다. `ProviderCommandPolicy`는 `ACTION_REPEAT=6` pair마다만 재폴링한다 = 10 Hz.
즉 이 스크린은 **정책을 실제 클라이언트보다 6배 빠르게 돌려왔다.** 이 세션 이전 실행 전부 포함.

정책 자체보다 **provider 래퍼가 직격이다.** 결정당 EMA인 α는 60 Hz에서 코너가 6배 높아,
스크린이 자신의 존재 이유인 래퍼 평가에 정확히 부적합했다. `floor_guard`를 포함해
이 하네스로 판정한 과거 래퍼 실험도 같은 문제를 안는다.

수정: `screen_models.py`에 `_ActionRepeat` 래퍼 추가 (`--action-repeat`, 기본 6).
유지 프레임에는 내부 provider가 실제로 호출되지 않으므로 EMA·rate limiter 자체 상태도
배포와 같은 속도로 진행된다.

**효과가 절대 성능에도 컸다** (같은 번들·시드, 80경기):

| | 킬 | dmg/경기 | ataMed |
|---|---|---|---|
| V255 @ 60 Hz 폴링 | 12 | 0.320 | 25.6 |
| V255 @ 10 Hz (배포와 동일) | **18** | **0.415** | **16.8** |

**부수 발견**: `step_ratio` 자체는 provider 경로 결과를 거의 바꾸지 않는다
(240경기 페어 비교에서 승패 변화 0건, 피해 96~98%가 0.01 이내). 어제 `step_ratio` 누락을
근거로 결과를 폐기한 판단은 **틀렸다** — 진짜 원인은 매 프레임 폴링이었다.
`screen_models.py`에 넣은 `step_ratio` 필수 가드는 그대로 두되(설정 일치는 여전히 옳다),
케이던스를 결정하는 것은 `--action-repeat`다.

---

## 4. V259b c150 — 재검토가 필요하다

기록된 clean gate 전수 집계:

| 모델 | 킬 | 추락 | 자기피해 | 최저고도 |
|---|---|---|---|---|
| **v259b_c150** | **18/24** | 0 | 0.00 | 613 m |
| v255export_c100 | 15/24 | 0 | 0.00 | 613 m |
| v257_c150 | 12/24 | 0 | 0.00 | 313 m |
| v215_c300 | 9/24 | 0 | 0.00 | 312 m |

기각 사유는 "정상 자세 진동이 남아서"였다. 그런데 `upright_*`/`inverted_*` 계측 컬럼은
V255 게이트가 돈 **뒤에** 추가되어서, 그 판단에는 비교할 V255 기준선이 없었다.
궤적에서 동일한 방식으로 재계산:

| | upright \|roll rate\| | 진동 |
|---|---|---|
| V255 c100 | 22.2 °/s | 0.81 Hz |
| V259b c150 | 21.2 °/s | 0.86 Hz |

**같다.** V259b에 진동이 남은 건 사실이지만 V255도 똑같이 진동한다.
기각 기준이 V255도 통과 못 하는 기준이었다.

배포 계약도 동일하다: 같은 `student.team01_phase_observation`, `team01_phase35`,
LSTM 없음, SAC, `step_ratio=6`. `BUNDLE_DIR`만 바꾸면 되는 드롭인이다.
번들: `artifacts/models/team01/v259b_c150/` (metadata.json + policy_weights.pkl.gz).

**단서**: 고정 게이트의 3 rep은 소수점까지 서로 같다. 결정론적이라 **실질 표본은 24가
아니라 8**이고, 18 vs 15는 시나리오 한 개(beamL) 차이다. 뷰어 A/B 없이 교체를 확정할
근거로는 얇다.

---

## 5. 랜덤 A/B (배포 케이던스, arm당 80경기 독립 표본)

`experiments/team01_screen_gate8_rand.yaml` — 같은 8기하에 스폰 산포. 모든 arm이 동일한
(시나리오, rep) 격자와 시드를 쓰므로 **경기 단위로 짝지어 검정**한다. 페어 검정은 여기서
훨씬 강력하다: 경기당 분산을 지배하는 스폰 산포가 arm 간 공유되어 상쇄된다.

| arm | 킬 | 승 | 패 | 추락 | 자기피해 | dmg/경기 | ataMed |
|---|---|---|---|---|---|---|---|
| v255_rand | 18 | 65 | 12 | 0 | 0.00 | 0.415 | 16.8 |
| **v259b_rand** | 18 | **75** | **3** | 0 | 0.00 | **0.444** | **16.2** |
| v255_damp045 | 15 | 69 | 6 | 0 | 0.00 | 0.376 | 20.5 |

**V259b vs V255 — 두 검정 모두 유의**

- 피해: V259b 우세 47경기, V255 우세 22, 무승부 11 → **부호검정 p = 0.004**
- 패배: 3 vs 12. V259b만 진 경기 1, V255만 진 경기 10 → **McNemar p = 0.016**
- 킬: 18 vs 18 (동일)
- 평균 피해차 +0.029, 95% CI [-0.031, +0.085] — **0을 포함**

평균차의 CI가 0을 포함하는데 부호검정이 강하게 유의한 것은 모순이 아니다.
**효과의 크기는 작지만 방향이 매우 일관된다**는 뜻이고, 그건 부호검정이 잡고 평균이 못 잡는다.
실전 의미가 가장 큰 것은 **패배 12 → 3**이다.

**댐퍼는 여기서 끝났다**

- 피해: 40 vs 40 **정확히 동률** (p = 1.000)
- 킬: 15 vs 18 (**악화**)
- ataMed: 20.5 vs 16.8 (**악화**)

60 Hz 실험에서 댐퍼가 ataMed를 25.6 → 18.9로 크게 줄이는 것처럼 보였던 건 케이던스
아티팩트였다. 배포 케이던스에서는 조준이 오히려 나빠진다. 패배는 6 vs 12로 줄지만
McNemar p = 0.211로 유의하지 않다.

기본값 `TEAM01_GUNLINE_HOLD=0` 유지가 맞다.

---

## 5b. 폐기된 측정 (참고용 보관)

`artifacts/logs/team01/sweep_*_INVALID_60hz*.jsonl` — 전부 60 Hz 폴링.
결론이 배포 케이던스 결과와 반대로 나오므로 인용하지 말 것.

## 6. (구) 댐퍼 α 스윕 — 60 Hz, 무효

`experiments/team01_screen_gate8.yaml` (8기하, provider 경로, α=1.0이 검증된 pass-through 대조군):

| arm | W | L | 추락 | 피해 | **cone_s** | ataMed |
|---|---|---|---|---|---|---|
| a100_control | 6 | 0 | 0 | 2.83 | **1.6** | 15.6 |
| a065_all | 7 | 0 | 0 | 4.42 | **1.6** | 15.9 |
| a045_all | 8 | 0 | 0 | 3.36 | **1.4** | 14.8 |
| a025_all | 8 | 0 | 0 | 2.78 | **1.5** | 18.0 |
| a045_upright_only | 8 | 0 | 0 | 3.64 | **1.6** | 19.6 |

**`cone_s`(1° 원뿔 체류시간 = 실제로 데미지를 내는 양)가 완전히 평평하다.** ataMed도 추세 없음.
시나리오별 피해는 α에 대해 단조롭지 않다 — beamL이 0.46 → 1.00 → **0.20** → 1.00.
스무딩 계수를 균등하게 낮췄는데 인접 값끼리 격추/무득점을 오간다면 개선이 아니라
**궤적 카오스가 어느 패스가 성사되는지 재배열**하는 것이다.

확정된 것은 **안전성뿐**: 어떤 arm도 추락하지 않았고 자기피해 0이었다.

---

## 6. 진행 중 / 다음

- `sweep_v259b_compare.sh` — V255 vs V259b를 같은 provider 경로에서 직접 비교
- `sweep_rand_ab.sh` — `team01_screen_gate8_rand.yaml`로 스폰 산포를 넣어 결정론을 깨고
  arm당 80경기 독립 표본. v255 / v259b / v255+댐퍼(α0.45). 랜덤화가 실제로 먹는지
  step 0에서 검사하고 안 먹으면 중단한다.

**아직 미해결**: pitch가 ~90% 포화라는 것 자체가 근본 문제일 수 있다. 조종면 하나가
상시 포화면 미세 제어가 남지 않는다. 이건 리워드/재학습 영역이라 예산이 필요하다.

## 변경 파일

| 파일 | 내용 |
|---|---|
| `student/gunline_hold.py` | alpha 붕괴 버그 수정, 게이트 확대, 필터 warm 유지, `upright_only`. 백업 `.pre_widen_bak` |
| `student/my_submission.py` | GUNLINE 기본값 + `_RANGE`/`_UPRIGHT_ONLY` 노브. **`TEAM01_GUNLINE_HOLD=0` 기본 유지 — 제출 경로 불변** |
| `screen_models.py` | `--smooth`를 존재하는 래퍼로 연결 (기존엔 ImportError) |
| `student/team01_phase_observation_standalone.py` | 신규, 자립형 관측 (미교체) |
| `scratch_obs_equivalence.py` | 비트 일치 검증 |
| `scratch_sweep_digest.py` | 스윕 집계 |
| `experiments/team01_screen_gate8{,_rand}.yaml` | provider 경로 게이트 / 랜덤화판 |

cut-in은 건드리지 않았다 — GUNLINE 블록은 `MODE=="rl"` 분기 안에만 있고 cut-in은
hybrid 분기에 있다.
