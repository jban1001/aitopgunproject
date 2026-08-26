set -u
PY="C:/Users/JUN/miniconda3/envs/aip/python.exe"
cd "C:/Users/JUN/Desktop/airop/Release_260722_plz"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
L="artifacts/logs/team01"; mkdir -p "$L"
# 원뿔 폭 커리큘럼. **한 번에 하나만** 돈다(병렬 학습은 불안정했고 BSOD 직전이었다).
# 각 단계는 앞 단계의 마지막 번들에서 웜스타트한다. 앞 단계가 번들을 못 남기면 멈춘다.
for r in A1:50 A2:50 A3:50 A4:60; do
  tag="${r%%:*}"; it="${r#*:}"
  B="artifacts/models/team01/${tag}_cone/bundle_$(printf %06d $it)"
  [ -d "$B" ] && { echo "[$(date +%H:%M)] $tag 이미 완료 — 건너뜀"; continue; }
  # 앞 단계 번들 확인
  need=$(grep -oP 'init_bundle: \K.*' experiments/team01_${tag}_cone.yaml)
  if [ ! -d "$need" ]; then echo "[$(date +%H:%M)] $tag 중단: 선행 번들 없음 $need"; exit 1; fi
  echo "[$(date +%H:%M)] $tag 학습 시작 ($it iter)"
  "$PY" train_rllib.py --experiment experiments/team01_${tag}_cone.yaml > "$L/cone_${tag}.log" 2>&1
  if [ ! -d "$B" ]; then echo "[$(date +%H:%M)] $tag 실패 — $L/cone_${tag}.log 확인"; tail -5 "$L/cone_${tag}.log"; exit 1; fi
  echo "[$(date +%H:%M)] $tag 완료"
done
echo "[$(date +%H:%M)] 원뿔 사다리 완료"
