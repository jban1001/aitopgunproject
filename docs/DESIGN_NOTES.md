# Curriculum Design Notes

## 역할 계약

| 계열 | 역할 | 성공 기준 | 맡지 않는 역할 |
|---|---|---|---|
| B | 교착 상태를 공격/방어 전문가가 처리 가능한 상태로 운반 | bridge 진입, dwell, 속도 유지, 생존 | 최종 조준과 kill |
| C | 교차 사격 위협에서 노출 제거 | exposure 감소, crossing 성공, 생존, hard-deck 안전 | 정면 merge와 장시간 추격 |
| F | 정면 merge 사격 부정 후 B/C 영역으로 인계 | incoming 방어, 생존, 180 m/s 이상, transfer dwell | post-cross 공격 완성 |

## B1-B7

| 단계 | 추가 난이도 |
|---|---|
| B1 | 기본 bridge shaping과 위치/사격창 연결 |
| B2 | 65-85도 비대칭 bridge 상태와 좌우 mirror |
| B3 | 90-100도 two-circle 교착 |
| B4 | 속도·고도 에너지 불균형 |
| B5 | 35-88도 혼합 상태와 짧은 인계 |
| B6 | 105-135도 고각 mutual bridge |
| B7 | Viewer에서 측정한 실제 실패 상태와 mirror |

## C1-C5

| 단계 | 추가 난이도 |
|---|---|
| C1 | 깊은 후방 위협에서 생존과 노출 해제 |
| C2 | 120-140도 crossing defense |
| C3 | 90-115도 위협으로 전진 |
| C4 | 60-85도 근접 crossing |
| C5 | 30-55도와 실제 Phase-3 측정 상태 |

## F1-F5

| 단계 | 추가 난이도 |
|---|---|
| F1 | 협조적 정면 접근에서 안전한 transfer 계약 학습 |
| F2 | 무장 BT 정면 접근, 2,000-6,000m 거리 다양화 |
| F3 | 속도·고도·좌우 offset 에너지 불균형 |
| F4 | 이전 단계 설정 계승과 일반화 확인 |
| F5 | Viewer 측정 실패 상태와 Round 4 거리 anchor |

F 계열에서 cross와 post-cross 공격은 진단값입니다. 승격 필수 조건은 incoming fire 부정, 생존, 속도, transfer입니다.

## Gate 설계

기본 orchestrator 패턴:

- 첫 run 최소 iteration: 100
- 역할 점수 threshold: 0.82
- plateau patience: 25
- 실패 시 full-state continuation: 한 번
- continuation 최소 iteration: 80
- stage별 필수 안전 조건을 score와 별도로 확인

점수 하나만 넘었다고 승격하지 않습니다. scorer의 `quality_ok`는 역할별 hard gate를 포함해야 합니다.

## 평가 체크리스트

### 실험 통제

- 고정 seed
- randomization 비활성 비교군
- 상대 bundle/DLL/XML 명시
- 거리와 고도별 균형 panel
- 평가와 학습을 병렬 실행하지 않기

### 결과

- [ ] kill 수
- [ ] 누적 target damage
- [ ] 누적 ownship damage
- [ ] ownship hard-deck
- [ ] target hard-deck
- [ ] strict WEZ dwell
- [ ] 첫 명중 시간
- [ ] 최저 고도와 최저 속도
- [ ] 상대 사격 노출 시간
- [ ] action saturation과 reversal
- [ ] two-circle deadlock
- [ ] selector mode별 체류 시간

## F 계열 디버깅 예시

학습 점수가 통과선에 가까워도 Viewer에서 pitch/yaw가 계속 `-1` 또는 `+1`에 붙는다면 다음 순서로 확인합니다.

1. selector가 F를 실제 선택했는지 확인
2. F 활성 구간만 잘라 action saturation 계산
3. 피격 시점이 F 활성 구간 안인지 확인
4. scorer의 incoming 지표가 실제 피격을 충분히 반영하는지 확인
5. reward를 바꾸기 전에 observation에 위협 구분 정보가 있는지 확인
6. 변경 후 동일 seed와 동일 기하로 다시 평가

이 순서를 지키면 selector 오작동, 정책 포화, scorer 괴리를 서로 혼동하지 않을 수 있습니다.
