# AI Top Gun RL Study Project

AI 전투기 1:1 근접전을 대상으로 SAC 정책, 단계별 커리큘럼, 전문가 selector를 연구한 코드 모음입니다.

이 저장소는 대회 배포본 전체가 아니라 **학습과 복기에 필요한 학생 작성 코드만 선별한 학습용 스냅샷**입니다. 모델 가중치, DLL/XML, 대용량 로그, 서버 연결 코드, 제출 패키지는 포함하지 않습니다. 실제 실행에는 별도의 대회 시뮬레이터와 `dogfight` 런타임이 필요합니다.

## 핵심 아이디어

하나의 정책에 모든 상황을 한꺼번에 가르치기보다, 실패 상태를 역할별로 나누고 쉬운 상태에서 어려운 상태로 이동합니다.

- **B 계열, bridge/handoff**: 불리하지도 유리하지도 않은 교착 상태를 다음 전문가가 처리할 수 있는 상태로 운반합니다.
- **C 계열, crossing defense**: 상대가 사격선을 만드는 교차 상황에서 노출을 끊고 생존합니다.
- **F 계열, frontal transfer**: 정면 merge에서 상대 사격을 부정하고 속도 180 m/s 이상을 유지한 채 B/C가 검증된 영역으로 인계합니다.
- **Selector**: 상대 유형을 추측하지 않고 거리, ATA, aspect, 고도처럼 직접 관측 가능한 기하만 사용합니다.

공격, 방어, 위치, 조준 전문가 전체를 합친 제출 시스템은 공개하지 않습니다. 이 저장소의 `specialist_selector.py`는 전문가 조합과 히스테리시스를 공부하기 위한 핵심 로직입니다.

## 저장소 구조

```text
.
├─ experiments/                 # B1-B7, C1-C5, F1-F5 커리큘럼 YAML
├─ scripts/
│  ├─ orchestrate_b_lane.ps1   # B 사다리 자동 실행과 승격
│  ├─ orchestrate_c_lane.ps1   # C 사다리 자동 실행과 승격
│  ├─ orchestrate_f_transfer_lane.ps1
│  ├─ b_curriculum_score.py    # 역할별 통과 점수 계산
│  ├─ c_curriculum_score.py
│  └─ f_transfer_score.py
├─ student/
│  ├─ team01_phase_observation.py  # 35차원 Markov 관측
│  └─ specialist_selector.py       # 기하 기반 전문가 선택
├─ docs/DESIGN_NOTES.md
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

- B 사다리: B7까지 설계
- C 사다리: C5까지 설계
- F 사다리: F1-F5와 동일 단계 continuation 설계
- 공개 제외: 모델 가중치, 실제 대회 DLL/XML, 제출 통신 코드, 비공개 전투 로그

세부 단계 표와 평가 체크리스트는 [docs/DESIGN_NOTES.md](docs/DESIGN_NOTES.md)를 참고하세요.
