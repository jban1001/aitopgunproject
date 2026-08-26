# AI Top Gun Specialist RL Archive

AI 전투기 1:1 근접전을 대상으로 SAC 정책, 단계별 커리큘럼, 전문가 selector를 연구한 코드 모음입니다.

이 저장소는 대회 배포본 전체가 아니라 **학습과 복기에 필요한 팀 작성 코드와 대표 산출물만 선별한 보존본**입니다. Codex와 Claude가 함께 진행한 커리큘럼 설계, 최종 selector, ABCDEF 대표 번들, 최종 제출 스냅샷, 통과 단계의 핵심 로그를 포함합니다. 반복 체크포인트, Ray 세션, 전체 Viewer 로그, 중복 Release 폴더와 대회 기본 런타임은 제외했습니다.

`submission_snapshot/`은 팀 작성 부분과 실제 사용 번들을 보존한 것이며, 완전한 대회 SDK 배포본은 아닙니다. 실행에는 주최 측 `dogfight` 런타임과 시뮬레이터가 별도로 필요합니다.

## 빠른 탐색

| 목적 | 먼저 볼 곳 |
|---|---|
| 전체 설계와 최종 상태 파악 | 이 README와 [`docs/DESIGN_NOTES.md`](docs/DESIGN_NOTES.md) |
| 모델별 출처와 승급 여부 확인 | [`docs/ARTIFACT_INDEX.md`](docs/ARTIFACT_INDEX.md) |
| 시나리오가 단계별로 어떻게 변했는지 비교 | [`experiments/`](experiments/) |
| 승급 gate와 continuation 흐름 이해 | [`scripts/`](scripts/) |
| 실제 selector 우선순위와 전환 조건 확인 | [`submission_snapshot/student/specialist_selector.py`](submission_snapshot/student/specialist_selector.py) |
| 최종 제출 배선 확인 | [`submission_snapshot/student/my_submission.py`](submission_snapshot/student/my_submission.py) |
| Claude/Codex의 시행착오와 판단 근거 복기 | [`docs/archive/`](docs/archive/) |

## 보존 범위와 재현 수준

| 항목 | 포함 여부 | 재현 수준 |
|---|---:|---|
| 팀 작성 학습·scorer·selector 코드 | 포함 | 코드와 설정을 그대로 검토 가능 |
| A-F 대표 정책 가중치 | 포함 | inference 및 비교 평가 가능 |
| 승급 단계의 핵심 로그와 요약 | 포함 | peak 선택과 gate 통과 근거 확인 가능 |
| optimizer·replay buffer·전체 native checkpoint | 제외 | 모든 학습을 중간 상태부터 완전히 재개할 수는 없음 |
| 주최 측 SDK·Viewer·BattleServer | 제외 | 실제 실행에는 공식 배포 환경이 필요 |
| 모든 실패 실험과 중복 Release 폴더 | 제외 | 설계상 의미 있는 이력만 문서와 소형 로그로 보존 |

즉, 이 저장소의 목표는 대회 환경 전체를 복제하는 것이 아니라 **팀이 만든 의사결정과 대표 결과를 작고 검토 가능한 형태로 남기는 것**입니다. `docs/archive/`와 `scripts/legacy/`의 절대 경로는 당시 환경을 보여 주는 역사 자료이며, 새 환경에서 그대로 실행하는 진입점이 아닙니다.

## 10분 둘러보기

1. `docs/ARTIFACT_INDEX.md`에서 A-F 역할과 정식 승급 여부를 확인합니다.
2. `experiments/team01_F2_guard.yaml`과 `experiments/team01_F3_transfer.yaml`을 비교해 incoming-warning 수업이 어떻게 계승됐는지 봅니다.
3. 시뮬레이터 없이 보존 로그를 scorer에 넣어 승급 판정을 다시 읽습니다.

```powershell
python .\scripts\f_transfer_score.py `
  .\evidence\F3_transfer\training_log.csv `
  --threshold 0.82 --min-iteration 100
```

4. `submission_snapshot/student/specialist_selector.py`에서 안전 우선순위와 confirmation 조건을 확인합니다.
5. `submission_snapshot/student/my_submission.py`에서 최종적으로 활성화된 전문가와 champion BT fallback을 확인합니다.

## 핵심 아이디어

하나의 정책에 모든 상황을 한꺼번에 가르치기보다, 실패 상태를 역할별로 나누고 쉬운 상태에서 어려운 상태로 이동합니다.

- **A 계열, gun cone**: 이미 유리한 후방 위치에서 조준 원뿔과 실제 사격 기회를 만듭니다.
- **B 계열, bridge/handoff**: 불리하지도 유리하지도 않은 교착 상태를 다음 전문가가 처리할 수 있는 상태로 운반합니다.
- **C 계열, crossing defense**: 상대가 사격선을 만드는 교차 상황에서 노출을 끊고 생존합니다.
- **D 계열, defense**: 상대가 뒤를 잡은 상태에서 생존하고 위협 기하를 해제합니다.
- **E 계열, tail/position**: 후방 추적 우세를 만들고 사격 전문가가 받을 수 있는 위치로 이동합니다.
- **F 계열, frontal transfer**: 정면 merge에서 상대 사격을 부정하고 속도 180 m/s 이상을 유지한 채 B/C가 검증된 영역으로 인계합니다.
- **Selector**: 상대 유형을 추측하지 않고 거리, ATA, aspect, 고도처럼 직접 관측 가능한 기하만 사용합니다.

실제 최종 연결 코드는 `submission_snapshot/student/`에 보존했습니다. 기본 제출 구성은 D4, C5b, corrected F3를 안전 오버레이로 사용하고 나머지는 champion BT에 맡깁니다. A5, B7, E7은 연구용 대표 번들로 보존했지만 최종 기본 selector에서는 비활성화했습니다.

## 저장소 구조

```text
.
├─ experiments/                 # A-F 커리큘럼 YAML과 F guard 수정안
├─ scripts/
│  ├─ orchestrate_b_lane.ps1   # B 사다리 자동 실행과 승격
│  ├─ orchestrate_c_lane.ps1   # C 사다리 자동 실행과 승격
│  ├─ orchestrate_f_transfer_lane.ps1
│  ├─ b_curriculum_score.py    # 역할별 통과 점수 계산
│  ├─ c_curriculum_score.py
│  ├─ f_transfer_score.py
│  ├─ orchestrate_f2_guard_then_resume.ps1
│  └─ legacy/                   # A/D/E 초기 사다리 원본 기록
├─ student/
│  ├─ team01_phase_observation.py  # 35차원 Markov 관측
│  └─ specialist_selector.py       # 기하 기반 전문가 선택
├─ submission_snapshot/         # 최종 팀 코드, BT 파일, 실제/대표 번들
├─ evidence/                    # 최종 단계 training_log와 승급 요약
├─ docs/
│  ├─ DESIGN_NOTES.md
│  ├─ ARTIFACT_INDEX.md
│  └─ archive/                  # Claude/Codex 협업 설계와 시행착오 기록
└─ train_rllib.py              # SAC 학습, 복원, 번들/체크포인트 저장
```

## 학습 사다리의 동작

각 orchestrator는 다음 절차를 자동화합니다.

1. 이전 단계의 정점 bundle에서 다음 단계를 시작합니다.
2. 최소 iteration 전에는 승격하지 않습니다.
3. 역할별 scorer가 안전, 생존, 인계, 공격 지표를 계산합니다.
4. 점수와 필수 안전 조건을 만족한 뒤 plateau가 확인되면 정상 종료합니다.
5. 통과하지 못하면 마지막 native checkpoint에서 optimizer와 replay buffer를 포함한 **한 번의 full-state continuation**을 수행합니다.
6. continuation도 실패하면 다음 단계로 억지 승격하지 않습니다.

가장 마지막 체크포인트가 가장 좋은 모델이라는 보장은 없습니다. 승격에는 scorer가 기록한 **peak iteration 이하의 가장 가까운 bundle**을 사용합니다.

## 관측과 행동

`team01_phase_observation.py`는 다음 범주를 포함하는 35차원 관측을 만듭니다.

- 자세와 각속도
- 기체축 속도, 총속도, 고도, 수직속도
- AoA, sideslip, normal-G, throttle
- 상대 거리와 closure
- body-frame LOS
- ATA, aspect, 상대 헤딩
- 양측 속도, 고도차, 체력
- WEZ 여부와 남은 경기 시간
- 현재 phase의 사격각과 최대 사거리

행동은 roll, pitch, yaw, throttle의 연속 4차원 명령입니다. SAC가 출력하는 값이 장시간 `-1` 또는 `+1`에 붙으면 정책이 강한 기동을 배운 것이 아니라 **행동 포화로 막힌 것**일 수 있습니다. 평균 보상만 보지 말고 포화율과 반전 횟수를 함께 확인해야 합니다.

## 실행 예시

PowerShell에서 사용할 Python을 지정합니다.

```powershell
$env:AIP_PYTHON = "C:\path\to\python.exe"
```

대회 런타임과 필요한 초기 bundle이 준비되어 있다는 전제에서 사다리를 실행합니다.

```powershell
.\scripts\orchestrate_b_lane.ps1 -InitialBundle "C:\models\position_expert\bundle_000060"
.\scripts\orchestrate_c_lane.ps1 -InitialBundle "C:\models\defense_expert\bundle_000090"
.\scripts\orchestrate_f_transfer_lane.ps1 -InitialBundle "C:\models\B7_bridge\bundle_000140"
```

scorer만 독립적으로 읽을 수도 있습니다.

```powershell
python .\scripts\f_transfer_score.py .\artifacts\logs\team01\F2_transfer\training_log.csv --threshold 0.82 --min-iteration 100
```

## 추천 공부 순서

### 1. 공중전 기하

먼저 ATA, aspect angle, LOS, closure, WEZ, two-circle/one-circle, energy state, hard deck를 공부합니다. 보상 함수나 selector 조건은 이 용어를 모르면 숫자 튜닝으로만 보입니다.

학습 목표:

- 상대가 나를 조준하는 상태와 내가 상대를 조준하는 상태를 벡터로 구분하기
- 거리 변화와 closure의 부호 해석하기
- 속도와 고도를 에너지로 함께 보기
- 사격 가능 상태와 단순히 기수가 가까운 상태를 구분하기

### 2. MDP와 관측 설계

정책이 필요한 정보를 관측에서 얻을 수 있는지 확인합니다. 현재 관측만으로 다음 상태가 충분히 예측되지 않으면 reward를 아무리 고쳐도 진동과 오판이 남습니다.

실습:

- `team01_phase_observation.py`의 35개 채널을 범주별로 그리기
- 같은 기하에서 body-frame LOS가 자세에 따라 어떻게 변하는지 확인하기
- health와 phase 채널을 제거했을 때 어떤 비-Markov 문제가 생기는지 설명하기

### 3. SAC 기본기

actor, twin critic, target network, replay buffer, entropy temperature, off-policy 학습을 순서대로 공부합니다.

특히 확인할 항목:

- 작은 `alpha`가 탐색 부족과 포화에 미치는 영향
- replay buffer가 이전 단계 행동을 얼마나 오래 유지하는지
- bundle 전이와 full-state checkpoint 복원의 차이
- critic 과대평가와 정책 붕괴를 로그에서 찾는 방법

### 4. 보상 설계

결과 보상만으로는 희소하고, shaping을 과하게 주면 대리 목표를 악용합니다. 각 shaping 항목은 반드시 관측 가능한 행동 결과와 연결해야 합니다.

권장 순서:

1. 추락과 피격 같은 안전 조건
2. 목표 역할의 진입과 dwell
3. 에너지 및 인계 가능성
4. 공격 기회와 실제 damage
5. 보조 진단 지표

`F` 계열처럼 역할이 인계라면 post-cross 공격을 필수 통과 조건으로 두지 않습니다. 다른 전문가의 책임을 한 정책에 다시 요구하면 커리큘럼 경계가 흐려집니다.

### 5. 커리큘럼과 전이 학습

`experiments/`의 YAML을 B, C, F 순서대로 비교합니다. 한 단계에서 무엇이 추가되고 무엇이 그대로 유지되는지를 표로 작성해 보세요.

좋은 사다리의 조건:

- 한 단계가 한 가지 난이도만 추가함
- 좌우 mirror와 속도/고도 변형을 포함함
- 최소 학습량과 안전 gate가 있음
- 실패 시 continuation은 한 번만 허용함
- 다음 단계는 이전 단계의 정점에서 시작함

### 6. 평가와 디버깅

평균 reward만으로 모델을 고르지 않습니다. 고정 상대, 고정 seed, 여러 시작 거리에서 다음 항목을 따로 기록합니다.

- kill과 누적 per-step damage
- ownship hard-deck과 target hard-deck
- 첫 명중 시간과 strict WEZ dwell
- 최저 고도, 속도, closure
- roll/pitch/yaw/throttle 포화율
- 명령 반전 횟수와 two-circle 교착
- selector가 실제로 선택한 전문가와 체류 시간

실전 Viewer와 로컬 scorer가 충돌하면 Viewer 기록을 새로운 시나리오로 환원하되, 한 경기만 보고 gate를 넓히지는 않습니다.

### 7. 전문가 selector

마지막으로 `specialist_selector.py`를 읽습니다. 중요한 주제는 우선순위, 진입 확인 횟수, 즉시 안전 복귀, 전문가 계약 범위입니다.

추천 실습:

- 기록된 trace를 재생해 각 모드의 선택 구간을 표시하기
- confirmation call 수를 바꾸고 chatter와 늦은 진입을 비교하기
- 고도 하한이 전문가 체류율을 얼마나 줄이는지 측정하기
- 조건을 넓혔을 때 미학습 상태가 전문가에게 들어가는 사례 찾기

## 시행착오에서 얻은 원칙

- **안전과 공격은 분리 측정한다.** 상대의 지면 충돌은 우리 공격 성공이 아닙니다.
- **누적 damage를 합산한다.** 마지막 step의 damage만 보면 실제 교전을 놓칠 수 있습니다.
- **정점과 최신본을 구분한다.** 학습 후반의 정책 붕괴는 흔합니다.
- **selector 활성화를 먼저 증명한다.** 전문가 성능을 논하기 전에 실제로 조종권을 받았는지 확인합니다.
- **포화는 강한 기동과 다르다.** 끝값 고정과 빠른 반전은 별도 실패 지표입니다.
- **한 번에 한 가설만 바꾼다.** 보상, 관측, 상대 league, 시나리오를 동시에 바꾸면 원인을 알 수 없습니다.
- **holdout을 학습에 섞지 않는다.** 마지막 평가는 학습에 쓰지 않은 상대와 기하에서 수행합니다.

## 현재 스냅샷

| 계열 | 대표 산출물 | 상태 | 최종 기본 selector |
|---|---|---|---:|
| A | A5 gun bundle 50 | 연구용 대표 정점 | 비활성 |
| B | B7 bridge bundle 140 | 정식 승급 | 비활성 |
| C | C5b crossing-defense bundle 80 | 정식 승급 | 활성 |
| D | D4 defense bundle 90 | 대표 정점 | 활성 |
| E | E7 position bundle 60 | 연구용 대표 정점 | 비활성 |
| F | F3 bundle 130 | corrected F3 정식 승급 | 비교 보존 |
| F | F3 bundle 80 | Viewer에서 선택한 제출 후보 | 활성 |
| F4 | training log, iter 88 | 정상 저장 후 사용자 요청으로 종료, 미승급 | 비활성 |

최종 제출 구성은 **D4/C5b/F3-c80 안전 오버레이 + champion BT fallback**입니다. 전체 대회 SDK, 반복 체크포인트, Ray 임시 파일, 중복 Release, 원시 Viewer 영상과 대용량 통신 로그는 제외했습니다.

## 제출 스냅샷 무결성

`submission_snapshot/MANIFEST_SHA256.txt`에는 팀 코드, DLL/XML, 대표 번들의 SHA-256이 기록되어 있습니다. 파일을 복사하거나 공식 SDK와 조립한 뒤 다음처럼 변조 또는 누락 여부를 확인할 수 있습니다.

```powershell
$root = Resolve-Path .\submission_snapshot
Get-Content $root\MANIFEST_SHA256.txt | ForEach-Object {
  $hash, $relative = $_ -split '  ', 2
  $path = Join-Path $root $relative
  $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower()
  if ($actual -ne $hash) { Write-Error "hash mismatch: $relative" }
}
```

파일별 출처와 보존 이유는 [docs/ARTIFACT_INDEX.md](docs/ARTIFACT_INDEX.md), 세부 단계 표와 평가 체크리스트는 [docs/DESIGN_NOTES.md](docs/DESIGN_NOTES.md)를 참고하세요.
