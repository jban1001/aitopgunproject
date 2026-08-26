# Final Submission Snapshot

2026-08-24 최종 제출본에서 팀이 작성하거나 선택한 부분만 보존한 스냅샷입니다. 주최 측 전체 `dogfight` 런타임은 포함하지 않습니다.

## 기본 구성

- 기본 모드: `selector`
- 바탕 기동/공격: `AIP_Team01_v12_stable.dll` + `Rule_codex_champion_v1.xml`
- defense: D4 bundle 90
- crossing defense: C5b bundle 80
- frontal merge: corrected F3 candidate bundle 80
- A5/B7/E7: 비교 연구용으로 함께 보존했지만 기본값에서는 비활성

`student/my_submission.py`의 상대경로는 이 폴더를 Release 루트로 놓았을 때의 구조를 유지합니다.

## 실행 전제

1. 주최 측 Release/SDK의 `src/dogfight` 런타임과 네이티브 의존성을 준비합니다.
2. 이 폴더의 내용을 Release 루트에 겹쳐 놓습니다.
3. `requirements.txt`와 `추가필요라이브러리_MAVERICK1.txt`를 확인합니다.
4. `RUN_SUBMISSION.bat` 또는 `python student/submission.py`로 실행합니다.

환경변수로 전문가를 다시 켜거나 바꿀 수 있지만, 루트 README에 기록된 최종 기본 구성은 D4/C5b/F3-c80 안전 오버레이입니다.

## 주의

`F3_transfer_promoted/bundle_000130`은 formal scorer가 고른 정식 F3 정점이고, `F3_transfer_candidate/bundle_000080`은 실제 Viewer 시험과 제출 구성에서 사용한 후보입니다. 두 모델의 용도를 혼동하지 않습니다.
