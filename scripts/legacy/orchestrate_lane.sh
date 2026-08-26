set -u
PY="C:/Users/JUN/miniconda3/envs/aip/python.exe"
R="C:/Users/JUN/Desktop/airop/Release_260722_plz"
cd "$R"; export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
L="artifacts/logs/team01"; mkdir -p "$L"
ORCH="$L/orchestrator.log"
LANE="$1"; WAIT_TAG="$2"; PARENT="$3"; PARENT_PEAK="$4"; shift 4
QUEUE="$@"
say(){ echo "[$(date '+%m-%d %H:%M')] [$LANE] $*" >> "$ORCH"; echo "[$LANE] $*" >&2; }

# ── 갈래별 오케스트레이터 ────────────────────────────────────────────────
# 두 갈래(E / D)를 나란히 돌린다. 각 갈래는 **자기 큐만** 순차로 처리하고,
# 시작할 때 자기 앞의 학습($WAIT_TAG) 하나만 기다린다.
# 규칙(PLAN_2026-08-20.md §9): 상승이 멈출 때 승급 / 정점 번들 인계 /
# 바닥값 미달이면 b 이어달리기(restore-checkpoint 로 옵티마이저까지 복원).

tag_running(){ powershell.exe -NoProfile -Command "(@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '--output-tag $1' })).Count" 2>/dev/null | tr -d '\r' | grep -oE '^[0-9]+' || echo 0; }
wait_tag(){ [ -z "$1" ] && return; while [ "$(tag_running "$1")" != "0" ]; do sleep 120; done; }

peak_of(){
  # 10칸 이동평균 정점. 표본 10개가 찬 뒤부터만 본다(초반 버스트 방지).
  grep -aoE "iter=\[[0-9]+\].*WEZ_ep=\[[0-9.]+\]" "$1" 2>/dev/null \
  | sed -E 's/iter=\[([0-9]+)\].*WEZ_ep=\[([0-9.]+)\]/\1 \2/' \
  | awk '{it[NR]=$1; v[NR]=$2}
         END{ if(NR==0){print "0 0"; exit}
              if(NR<10){ s=0; for(i=1;i<=NR;i++) s+=v[i]; printf "%d %.2f", it[NR], s/NR; exit }
              best=-1; bi=0;
              for(i=10;i<=NR;i++){ s=0; for(j=i-9;j<=i;j++) s+=v[j];
                 m=s/10; if(m>best){best=m; bi=it[i]} }
              printf "%d %.2f", bi, best }'
}
peak_bundle(){
  n=$(( $2 / 10 * 10 )); [ "$n" -lt 10 ] && n=10
  B="artifacts/models/team01/$1/bundle_$(printf %06d $n)"
  [ -d "$B" ] && echo "$B" || ls -d artifacts/models/team01/$1/bundle_* 2>/dev/null | sort | tail -1
}
last_ckpt(){ ls -d artifacts/checkpoints/team01/$1/checkpoint_* 2>/dev/null | sort | tail -1; }

train(){
  _tag="$1"; _exp="$2"; _init="$3"; _cap="$4"; _minit="$5"; _pat="$6"
  LOG="$L/run_${_tag}.log"
  case "$_init" in
    *checkpoint_*) IA="--restore-checkpoint $_init" ;;
    "")            IA="" ;;
    *)             IA="--init-bundle $_init" ;;
  esac
  say "  $_tag 시작 (상한 $_cap / 최소 $_minit / 정체 $_pat) <- $(basename "$_init")"
  "$PY" train_rllib.py \
    --algorithm sac --iterations "$_cap" \
    --output-name team01 --output-tag "$_tag" \
    --framework torch --lr 3e-05 --initial-alpha 1e-05 --gamma 0.99 \
    --train-batch-size 256 --minibatch-size 256 --tau 0.005 --target-entropy -1.0 \
    --replay-buffer-capacity 200000 \
    --model-fcnet-hiddens 256,256 --model-fcnet-activation relu \
    --model-head-fcnet-hiddens "" --model-head-fcnet-activation relu \
    --observation-mode custom --observation-module student.team01_phase_observation \
    --target-behavior-dll AIP_BASE_team_climb_dive_approach.dll --target-mode behavior_tree \
    --max-engage-time 200.0 --episode-step-limit 2000 \
    --num-env-runners 2 --num-envs-per-env-runner 1 \
    --rollout-fragment-length auto --batch-mode truncate_episodes \
    --lightweight-bundle-frequency 10 --native-checkpoint-frequency 25 --save-native-checkpoint \
    $IA --dashboard-logdir artifacts/dashboard \
    --experiment-yaml "$R/experiments/$_exp" > "$LOG" 2>&1 &
  pid=$!
  best=0; stall=0; seen=-1
  while kill -0 "$pid" 2>/dev/null; do
    sleep 120
    it=$(grep -aoE "iter=\[[0-9]+\]" "$LOG" 2>/dev/null | grep -oE "[0-9]+" | tail -1)
    [ -z "$it" ] && continue
    [ "$it" = "$seen" ] && continue
    seen="$it"
    m=$(grep -aoE "WEZ_ep=\[[0-9.]+\]" "$LOG" | grep -oE "[0-9.]+" | tail -10 | awk 'NF{s+=$1;n++} END{if(n>0) printf "%.2f", s/n; else print "0"}')
    [ -z "$m" ] && m=0
    if awk -v a="$m" -v b="$best" 'BEGIN{exit !(a > b*1.05 + 0.2)}'; then best="$m"; stall=0; else stall=$((stall+1)); fi
    if [ "$it" -ge "$_minit" ] && [ "$stall" -ge "$_pat" ]; then
      say "    $_tag 상승 멈춤 (iter=$it, 최고 $best)"
      kill "$pid" 2>/dev/null; sleep 25; break
    fi
  done
  wait "$pid" 2>/dev/null || true
}

stage(){
  tag="$1"; exp="$2"; init="$3"; floor="$4"
  train "$tag" "$exp" "$init" 300 80 25
  pk=$(peak_of "$L/run_${tag}.log"); pit="${pk%% *}"; pval="${pk##* }"
  say "  $tag 1차 종료: 정점 iter=$pit  WEZ_ep=$pval"
  if [ -n "$floor" ] && awk -v a="$pval" -v b="$floor" 'BEGIN{exit !(a < b)}'; then
    ck=$(last_ckpt "$tag")
    if [ -n "$ck" ]; then
      say "  $tag 정점 $pval < 바닥값 $floor -> ${tag}b 이어달리기(전체 상태 복원)"
      train "${tag}b" "$exp" "$ck" 350 80 28
      pk2=$(peak_of "$L/run_${tag}b.log"); pit2="${pk2%% *}"; pval2="${pk2##* }"
      say "  ${tag}b 종료: 정점 $pval2"
      if awk -v a="$pval2" -v b="$pval" 'BEGIN{exit !(a > b)}'; then
        echo "$(peak_bundle "${tag}b" "$pit2")|$pval2"; return
      fi
    fi
  fi
  echo "$(peak_bundle "$tag" "$pit")|$pval"
}

say "=== 갈래 시작. 대기대상=${WAIT_TAG:-없음}  큐=$QUEUE ==="
wait_tag "$WAIT_TAG"
say "대기 해제"

prev="$PARENT"; peak="$PARENT_PEAK"
# 대기대상이 끝났으면 그 정점을 부모로 삼는다(있을 때만)
if [ -n "$WAIT_TAG" ] && [ -f "$L/run_${WAIT_TAG}.log" ]; then
  pk=$(peak_of "$L/run_${WAIT_TAG}.log"); pit="${pk%% *}"; pval="${pk##* }"
  nb=$(peak_bundle "$WAIT_TAG" "$pit")
  if [ -n "$nb" ] && [ "${pit:-0}" -gt 0 ]; then
    say "$WAIT_TAG 정점 iter=$pit ($pval) -> 부모 $(basename "$nb")"
    prev="$nb"; peak="$pval"
  fi
fi

for s in $QUEUE; do
  case "$s" in
    E*) t="${s}_tail"; exp="team01_${s}_tail.yaml";;
    D*) t="${s}_def";  exp="team01_${s}_def.yaml";;
    A*) t="${s}_gun";  exp="team01_${s}_gun.yaml";;
    *)  say "알 수 없는 단계 $s -- 중단"; exit 1;;
  esac
  floor=""
  [ -n "$peak" ] && floor=$(awk -v p="$peak" 'BEGIN{printf "%.2f", p*0.30}')
  say "── $t (바닥값 ${floor:-없음}, 부모 $(basename "$prev")) ──"
  out=$(stage "$t" "$exp" "$prev" "$floor")
  nb="${out%%|*}"; np="${out##*|}"
  if [ -z "$nb" ]; then say "$t 번들 없음 -- 갈래 중단"; exit 1; fi
  say "$t 완료. 정점 $np -> 다음 부모 $(basename "$nb")"
  prev="$nb"; peak="$np"
done
say "=== 갈래 완료. 최종 $prev ==="
