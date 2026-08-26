set -u
PY="C:/Users/JUN/miniconda3/envs/aip/python.exe"
R="C:/Users/JUN/Desktop/airop/Release_260722_plz"
cd "$R"; export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
L="artifacts/logs/team01"; mkdir -p "$L"

# 원뿔 폭 커리큘럼. 상승이 멈출 때 다음 단계로 넘어간다(절대 목표치 없음).
# 지표 WEZ_ep = 에피소드당 사격창 체류 = 조준 학습 여부를 직접 재는 값.
#
# 2026-08-20 수정 2건:
#  (1) **정점 번들을 넘긴다.** 이전에는 정체 시점의 마지막 번들을 넘겨서 내리막을
#      물려줬다. A1 은 이동평균 정점이 iter 26(16.85)인데 iter 49 번들을 넘겼고,
#      A2 는 그 상태에서 시작해 13.73 -> 10.75 로 떨어지는 중이었다.
#  (2) **바닥값 게이트.** 정체했더라도 정점이 앞 단계 정점의 30% 에 못 미치면
#      승급하지 않고 상한까지 더 돌린다. 정체가 "다 뽑았다"가 아니라 "무너졌다"일 수 있다.
#
# 함정 기록: train_rllib.py 는 YAML 의 runtime:/output: 를 읽지 않는다.
# --experiment-yaml 은 env_config 만 병합한다. iterations/output-tag 를 CLI 로 안 주면
# 기본값(5 iter, f16_single_agent/latest)으로 조용히 돈다.

latest_bundle(){ ls -d artifacts/models/team01/$1_cone/bundle_* 2>/dev/null | sort | tail -1; }
mean_wez(){ grep -aoE "WEZ_ep=\[[0-9.]+\]" "$1" 2>/dev/null | grep -oE "[0-9.]+" | tail -"$2" \
    | awk 'NF{s+=$1;n++} END{if(n>0) printf "%.2f", s/n; else print "0"}'; }
last_iter(){ grep -aoE "iter=\[[0-9]+\]" "$1" 2>/dev/null | grep -oE "[0-9]+" | tail -1; }

# 10 iter 이동평균의 정점 iter 와 그 값. 번들은 10 iter 마다 저장된다.
peak_iter(){
  grep -aoE "iter=\[[0-9]+\].*WEZ_ep=\[[0-9.]+\]" "$1" 2>/dev/null \
  | sed -E 's/iter=\[([0-9]+)\].*WEZ_ep=\[([0-9.]+)\]/\1 \2/' \
  | awk '{it[NR]=$1; v[NR]=$2}
         END{ if(NR==0){print "0 0"; exit}
              best=-1; bi=0;
              for(i=1;i<=NR;i++){ s=0;c=0; for(j=(i-9>1?i-9:1);j<=i;j++){s+=v[j];c++}
                 m=s/c; if(m>best){best=m; bi=it[i]} }
              printf "%d %.2f", bi, best }'
}
peak_bundle(){
  n=$(( $2 / 10 * 10 )); [ "$n" -lt 10 ] && n=10
  B="artifacts/models/team01/$1_cone/bundle_$(printf %06d $n)"
  if [ -d "$B" ]; then echo "$B"; else latest_bundle "$1"; fi
}

run_rung(){
  tag="$1"; minit="$2"; patience="$3"; cap="$4"; init="$5"; floor="$6"
  LOG="$L/cone_${tag}.log"
  echo "[$(date +%H:%M)] $tag 시작 (최소 ${minit}iter, 정체 ${patience}회, 상한 ${cap}, 바닥값 ${floor:-없음})" >&2
  echo "            init=$init" >&2
  "$PY" train_rllib.py \
    --algorithm sac --iterations "$cap" \
    --output-name team01 --output-tag "${tag}_cone" \
    --framework torch --lr 3e-05 --initial-alpha 1e-05 --gamma 0.99 \
    --train-batch-size 256 --minibatch-size 256 --tau 0.005 --target-entropy -1.0 \
    --replay-buffer-capacity 200000 \
    --model-fcnet-hiddens 256,256 --model-fcnet-activation relu \
    --model-head-fcnet-hiddens "" --model-head-fcnet-activation relu \
    --observation-mode custom --observation-module student.team01_phase_observation \
    --target-behavior-dll AIP_BASE_team_climb_dive_approach.dll --target-mode behavior_tree \
    --max-engage-time 200.0 --episode-step-limit 2000 \
    --num-env-runners 3 --num-envs-per-env-runner 1 \
    --rollout-fragment-length auto --batch-mode truncate_episodes \
    --lightweight-bundle-frequency 10 --native-checkpoint-frequency 25 --save-native-checkpoint \
    --init-bundle "$init" \
    --dashboard-logdir artifacts/dashboard \
    --experiment-yaml "$R/experiments/team01_${tag}_cone.yaml" \
    > "$LOG" 2>&1 &
  pid=$!
  best=0; stall=0; lastseen=-1
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
    it=$(last_iter "$LOG"); [ -z "$it" ] && continue
    [ "$it" = "$lastseen" ] && continue
    lastseen="$it"
    m=$(mean_wez "$LOG" 10)
    if awk "BEGIN{exit !($m > $best*1.05 + 0.2)}"; then best="$m"; stall=0; else stall=$((stall+1)); fi
    okfloor=1
    if [ -n "$floor" ] && awk "BEGIN{exit !($best < $floor)}"; then okfloor=0; fi
    echo "[$(date +%H:%M)]   $tag iter=$it  WEZ_ep(최근10)=$m  최고=$best  정체=$stall/$patience$([ "$okfloor" = 0 ] && echo "  (바닥값 $floor 미달 -- 승급보류)")" >&2
    if [ "$it" -ge "$minit" ] && [ "$stall" -ge "$patience" ] && [ "$okfloor" = "1" ]; then
      echo "[$(date +%H:%M)]   $tag 상승 멈춤 -- 다음 단계로 (최고 $best)" >&2
      kill "$pid" 2>/dev/null; sleep 25; break
    fi
  done
  wait "$pid" 2>/dev/null || true
  pk=$(peak_iter "$LOG"); pit="${pk%% *}"; pval="${pk##* }"
  B=$(peak_bundle "$tag" "${pit:-0}")
  if [ -z "$B" ]; then echo "[$(date +%H:%M)] $tag 번들 없음 -- 중단" >&2; return 1; fi
  echo "[$(date +%H:%M)] $tag 종료. 정점 iter=$pit (WEZ_ep=$pval) -> $B" >&2
  echo "$pval" > "$L/cone_${tag}.peak"
  echo "$B"
}

# A1 은 이미 완료됐다. 정점(iter 26 부근) 번들에서 A2 를 다시 시작한다.
prev="$R/artifacts/models/team01/A1_cone/bundle_000030"
prevpeak="16.85"
for spec in A2:30:15:200 A3:30:15:200 A4:40:25:300; do
  tag="${spec%%:*}"; r1="${spec#*:}"; minit="${r1%%:*}"; r2="${r1#*:}"; pat="${r2%%:*}"; cap="${r2#*:}"
  floor=""
  if [ -n "$prevpeak" ]; then floor=$(awk "BEGIN{printf \"%.2f\", $prevpeak*0.30}"); fi
  out=$(run_rung "$tag" "$minit" "$pat" "$cap" "$prev" "$floor") || exit 1
  prev=$(echo "$out" | tail -1)
  prevpeak=$(cat "$L/cone_${tag}.peak" 2>/dev/null || echo "")
  echo "[$(date +%H:%M)] 다음 단계 웜스타트 <- $prev (정점 $prevpeak)" >&2
done
echo "[$(date +%H:%M)] 원뿔 사다리 완료. 최종 $prev" >&2
