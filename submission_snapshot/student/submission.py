# -*- coding: utf-8 -*-
"""제출 진입점 별칭 — `python student/submission.py`.

대회 릴리스가 규정한 진입점 파일명은 `my_submission.py`이고(README 595행),
실제 설정과 로직은 전부 거기에 있다. 이 파일은 이름만 다른 같은 프로그램이다.
둘 중 무엇을 실행해도 동일하게 동작한다:

    python student/my_submission.py
    python student/submission.py

설정을 바꿀 때는 반드시 `my_submission.py`를 고칠 것. 여기에 값을 복사해 두면
두 진입점이 갈라진다.
"""
from __future__ import annotations

import sys
from pathlib import Path

# 저장소 루트를 sys.path에 올린다 — 이 파일을 직접 실행하면 student/ 가 cwd 기준이
# 아니라 스크립트 위치 기준으로 잡히기 때문에, my_submission.py 와 같은 처리가 필요하다.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from student.my_submission import main  # noqa: E402

if __name__ == "__main__":
    main()
