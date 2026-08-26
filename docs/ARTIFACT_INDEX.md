# Artifact Index

이 문서는 2026-08-27 로컬 정리 전에 보존한 핵심 산출물의 출처와 의미를 기록합니다.

## 대표 전문가 번들

| 계열 | 보존 경로 | 역할 | 상태 |
|---|---|---|---|
| A | `submission_snapshot/artifacts/models/team01/A5_gun/bundle_000050` | 후방 조준/사격창 | 연구용 대표 정점, 최종 기본 selector에서는 비활성 |
| B | `submission_snapshot/artifacts/models/team01/B7_bridge/bundle_000140` | 고각 교착 bridge | B7 정식 승급 번들 |
| C | `submission_snapshot/artifacts/models/team01/C5_crossdefb/bundle_000080` | crossing defense | C5b 정식 승급 번들, 최종 제출에서 활성 |
| D | `submission_snapshot/artifacts/models/team01/D4_def/bundle_000090` | 후방 위협 방어 | 최종 제출에서 활성 |
| E | `submission_snapshot/artifacts/models/team01/E7_tail/bundle_000060` | 후방 위치 우세 | 연구용 대표 정점, 최종 기본 selector에서는 비활성 |
| F | `submission_snapshot/artifacts/models/team01/F3_transfer_candidate/bundle_000080` | 정면 merge 방어/인계 | Viewer 시험과 최종 제출에서 사용 |
| F | `submission_snapshot/artifacts/models/team01/F3_transfer_promoted/bundle_000130` | corrected F3 정식 정점 | scorer 기준 정식 승급 번들 |

각 bundle에는 `metadata.json`과 압축된 정책 가중치만 있습니다. optimizer, replay buffer, Ray checkpoint는 저장소 크기를 줄이기 위해 제외했습니다.

## 코드

- `train_rllib.py`: SAC 학습, bundle/checkpoint 저장, stop-file 정상 종료.
- `experiments/`: A-F 시나리오와 corrected F3 incoming-warning 설정.
- `scripts/`: B/C/F scorer와 gate 기반 orchestrator, F2 guard 복구 흐름.
- `scripts/legacy/`: A/D/E를 만들 때 실제 사용한 초기 shell 사다리. 절대 경로가 남은 역사 자료이므로 그대로 실행하기보다 설계 비교용으로 읽습니다.
- `student/`: Markov 관측과 최종 기하 selector 핵심.
- `submission_snapshot/student/`: 실제 제출 진입점과 selector 배선.

## 검증 자료

`evidence/<stage>/training_log.csv`에는 A5, B7, C5b, D4, E7, F2_guardb, F3, 중단된 F4의 마지막 학습 기록이 있습니다. B/C/F 단계에는 가능한 경우 formal stage summary도 함께 보존했습니다.

루트 `evidence/*orchestrator.log`는 승급, continuation, peak bundle 선택의 시간 순서를 복원하기 위한 작은 원본 로그입니다.

## 협업 기록

`docs/archive/`에는 Claude와 Codex가 함께 정리한 specialist 계획, 실험 이력, 위험 우선순위, roll-chatter 분석, 인수인계 문서를 보존했습니다. 현재 코드와 충돌하는 오래된 제안은 역사 자료이며, 최종 상태는 이 문서와 루트 README를 우선합니다.

## 의도적으로 제외한 항목

- 전체 Release 복제본과 주최 측 기본 SDK
- 모든 native checkpoint와 replay buffer
- Ray session, dashboard, cache, `__pycache__`
- 반복 bundle과 실패 실험의 대용량 원본
- Viewer 영상, 화면 캡처, 전체 패킷/통신 로그
- 최종 제출 ZIP 자체

최종 제출 ZIP은 로컬 보존본으로 관리하고, Git에는 검토 가능한 팀 작성 코드와 필요한 대표 가중치만 둡니다.
