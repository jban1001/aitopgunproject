# RL Handoff: Viewer Merge Curriculum

> **과거 기록 (2026-08-01 기준).** 현행 계획은 `SPECIALIST_PLAN.md`.
> 이 시점 이후 물리 괴리가 해소되고(f16.xml 연료) 데미지 규칙이 정정됐다.


Last updated: 2026-07-27

## Objective

Build a **pure RL** Blue policy that first survives the actual BattleViewer
close reciprocal merge, then reacquires/turns back in, then converts the
resulting gun window to a kill. Do not begin self-play until the Viewer-like
merge is locally reliable.

The current late-kill target is:

- DLL: `C:/Users/JUN/Desktop/airop/release_red/AIP_latekill_practice_bt.dll`
- XML: `C:/Users/JUN/Desktop/airop/release_red/Rule_latekill_practice_bt.xml`

## Non-Negotiable Execution Rule

Always launch YAML experiments through:

```powershell
C:\Users\JUN\miniconda3\envs\aip\python.exe scripts\run_experiment.py experiments/<file>.yaml
```

Do **not** launch `train_rllib.py --experiment-yaml ...` directly. The direct
trainer only deep-merges `env_config`; it ignores the YAML top-level `env` and
`bt_assets` sections. That accidentally loads `AIP_BASE_target.dll` with the
wrong `Rule_forTraining.xml` and fails native BT creation.

`scripts/run_experiment.py` temporarily activates the matching XML and passes
the correct target DLL, SAC settings, checkpoint, and runtime values.

## Viewer Facts Confirmed

- BattleViewer Blue starts on plane 0 only when Blue connects first.
- The observed Viewer opening is a close level reciprocal merge: about 760 m,
  altitude about 4572 m, both aircraft about 200 m/s.
- Earlier tail-chase curricula (1.45-2.2 km) do not transfer to this start.
- V095 was slow in Viewer because RL itself output low throttle, not because
  of pitch sign or an inference bundle activation mismatch.
- Existing V095 bundles use a ReLU `[256,256,128]` architecture. Do not force
  stale Tanh `[256,256]` metadata onto those bundles.

## Source Changes That Must Be Preserved

### Lightweight bundle restoration

- `src/dogfight/ai/checkpoint_io.py` saves `metadata.trained_model_config`.
- `src/dogfight/ai/rllib_utils.py` prefers that exact trained config while
  retaining legacy compatibility.

### Shared opening-throttle contract

Files:

- `src/dogfight/config.py`
- `src/dogfight/envs/single_agent_env.py`
- `src/dogfight/unreal/policies.py`
- `run_unreal_inference.py`

The optional setting below applies physical throttle in both JSBSim training
and Viewer command output. It is not a Viewer-only safety wrapper.

```yaml
env_config:
  opening_throttle_floor: 0.88
  opening_throttle_steps: 60
```

For Viewer validation, use the matching command options:

```text
--opening-throttle-floor 0.88 --opening-throttle-policy-updates 60
```

The Viewer counts policy updates, matching training RL decisions under
`--action-repeat 6`.

### Tactical reward logging

`train_rllib.py` now writes merge fields into `training_log.csv`, including:

- `ep_reward_merge_cross`
- `ep_reward_merge_handoff`
- `ep_reward_merge_turnin`
- `ep_reward_merge_stalemate`
- attack/front/aim/angle/range/lost components

`src/dogfight/envs/reward.py` has opt-in `merge_turnin_scale` and
`merge_turnin_ata_deg`. It supplies continuous reward after a detected merge
as ATA falls from the configured maximum. It is disabled unless set in YAML.

### Conservative SAC curriculum transfer

Files:

- `src/dogfight/ai/conservative_sac_learner.py`
- `src/dogfight/ai/rllib_utils.py`
- `scripts/run_experiment.py`
- `train_rllib.py`
- `src/dogfight/ai/dashboard_logger.py`

The V108 regression was not caused by old random replay. RLlib native
checkpoints restored about 389k lifetime environment steps, but replay began
empty and grew from 3,072 fresh steps because
`store_buffer_in_checkpoints=False`. The actual failure was:

1. SAC began updating after only the default 1,500 fresh samples because the
   restored lifetime counter already exceeded the learning-start threshold.
2. The restored critic had not learned the new rear/side distribution, but
   actor and critic changed together.
3. `runtime.explore: false` made environment actions deterministic, but SAC
   still re-sampled Gaussian actions inside its actor loss.
4. Restored alpha stayed near 1.0, so entropy pressure was far too high for a
   delicate curriculum transfer.

The opt-in YAML contract is:

```yaml
algo:
  replay_buffer_config:
    type: EpisodeReplayBuffer
    capacity: 60000
  transfer:
    enabled: true
    actor_freeze_fresh_steps: 3072
    actor_update_interval: 4
    actor_anchor_coeff: 10000.0
    reset_alpha_after_restore: 0.02
    reset_optimizers_after_restore: true
```

This preserves actor/critic weights while clearing stale Adam momentum,
resets entropy, lets the critic adapt first, delays actor updates, and anchors
the actor mean to a frozen post-restore snapshot. The transfer state is
written to CSV/dashboard metrics under `curriculum_*`.

## Completed Experiments

### V103: opening throttle contract

File: `experiments/team01_v103_pure_rl_viewer_merge_contract_24i.yaml`

- Correctly proved the shared training/Viewer throttle contract.
- Failed as a flight policy: WEZ and damage remained zero; mean distance grew
  back above 7 km late in the run.
- Do not use as a Viewer candidate or checkpoint source.

### V104: short, weapon-locked merge geometry

File: `experiments/team01_v104_pure_rl_viewer_merge_reentry_36i.yaml`

- Ran only through iteration 12, then deliberately stopped.
- Best checkpoint: `artifacts/checkpoints/team01/v104_pure_rl_viewer_merge_reentry_36i/checkpoint_000004`
- Best early final ATA was about 22.5 degrees; attack-band reward appeared.
- Later iterations regressed because handoff reward never activated.
- This checkpoint was the base for V108/V109. V111 is now the promoted
  post-merge reacquisition parent.

### V105/V106: reward-window diagnosis

- V105 fixed the close-pass credit and produced `merge_cross=5`, proving the
  merge detector works for the real roughly-55 m pass.
- V105/V106 still had `merge_handoff=0`: the target was not returned to the
  forward hemisphere after the pass.
- V106's continuous `merge_turnin` term recorded correctly but did not lower
  final ATA. Both were stopped early; do not promote their models.

### V107: exploratory merge lesson

File: `experiments/team01_v107_pure_rl_viewer_merge_explore_16i.yaml`

- Used SAC exploration and higher learning rates only in the 18-second
  weapon-locked lesson.
- Failed rapidly: final ATA worsened about 37 -> 68 degrees by iteration 4.
- Do not resume V107. It shows that simply adding stochastic exploration to
  the full reciprocal merge is harmful.

No V103-V107 bundle is a BattleViewer submission candidate.

### V109: conservative transfer proof

File: `experiments/team01_v109_pure_rl_conservative_reacquire_10i.yaml`

- Started from V104 checkpoint 4 with uniform replay, cleared optimizers,
  alpha reset to 0.05, and actor/alpha frozen for 12,288 fresh steps.
- Mean final ATA stayed below 60 degrees for all 10 iterations. V108 had
  degraded from 33.5 to 117.8 degrees over the same region; V109 ended at
  54.5 degrees.
- Per-scenario analysis exposed a hidden asymmetry at the final checkpoint:
  rear 27.2, rear-left 36.1, but rear-right 95.6 degrees.
- Use `checkpoint_000004` only as the critic-warmed, actor-still-frozen parent.
  Do not use V109 final as a submission or integration parent.

### V110: delayed actor only

File: `experiments/team01_v110_pure_rl_delayed_actor_right_reacquire_8i.yaml`

- Tested one actor update per four critic updates without an actor anchor.
- Stopped after iteration 2 when mean final ATA crossed 60 degrees.
- Rejected. Lower update frequency alone does not prevent destructive drift.

### V111: anchored rear-right correction

File: `experiments/team01_v111_pure_rl_anchored_right_reacquire_6i.yaml`

- Parent: V109 `checkpoint_000004`.
- Added a frozen parent-actor action-mean anchor, coefficient 10,000, while
  retaining alpha 0.02, uniform replay, one warm-up iteration, and 1:4 delayed
  actor updates.
- Completed all 6 iterations. Final randomized 12-episode breakdown:

  | Scenario | Mean final ATA | Mean minimum altitude |
  | --- | ---: | ---: |
  | rear | 27.9 deg | 4504 m |
  | rear-left | 32.7 deg | 4424 m |
  | rear-right | 53.0 deg | 4584 m |

- All three starts pass the 60-degree reacquisition gate without a crash or
  low-altitude failure. Promoted checkpoint:
  `artifacts/checkpoints/team01/v111_pure_rl_anchored_right_reacquire_6i/checkpoint_final`
- This is a curriculum parent, not a BattleViewer candidate. Weapons were
  locked and final ranges remained roughly 4.5-7.6 km.

The trajectory breakdown is reproducible with:

```powershell
python scripts/analyze_v109_transfer.py artifacts/logs/team01/v111_pure_rl_anchored_right_reacquire_6i/engagement_replays --iterations 0 1 5
```

## 2026-07-27 Update

### V112: merge/reacquire integration - PASSED

File:
`experiments/team01_v112_pure_rl_merge_reacquire_integration_8i.yaml`

Parent:
`artifacts/checkpoints/team01/v111_pure_rl_anchored_right_reacquire_6i/checkpoint_final`

Target:

- DLL: `C:/Users/JUN/Desktop/airop/release_red/AIP_latekill_practice_bt.dll`
- XML: `C:/Users/JUN/Desktop/airop/release_red/Rule_latekill_practice_bt.xml`
- Viewer PlaneInfo adapter, BT decision period 6

V112 explicitly represented the Viewer 2,493 ft / 759.8664 m reciprocal
opening inside a scenario pool, together with the three accepted V111
rear/side starts. Weapons were locked for the full 18-second lesson. The
actor was frozen for 6,144 fresh steps, then updated at 1:4 with alpha 0.02
and a 12,000 parent-action anchor.

The final iteration completed without crash or low-altitude termination:

| Scenario family | Final ATA | Minimum ownship altitude |
| --- | ---: | ---: |
| Viewer head-on | 26.0 deg | 3970 m |
| rear | 28.0 deg | 4494 m |
| rear-left | 37.0 deg | 4416 m |
| rear-right | 47.9 deg | 4567 m |

All families passed the 60-degree gate. The actor remained stable after
release; aggregate final ATA stayed in the 28-34 degree range for all eight
iterations. Promoted curriculum checkpoint:

`artifacts/checkpoints/team01/v112_pure_rl_merge_reacquire_integration_8i/checkpoint_final`

This is still a curriculum parent, not yet a Viewer submission candidate.
The head-on final range was about 4.3 km after 18 seconds, so aim/turn-in is
present but a sustained gun conversion is not.

New reproducible analyzer:

```powershell
python scripts/analyze_merge_reacquire.py artifacts/logs/team01/v112_pure_rl_merge_reacquire_integration_8i/engagement_replays --iterations 0 4 7
```

### V113: delayed gun conversion - RUNNING

File:
`experiments/team01_v113_pure_rl_delayed_gun_conversion_10i.yaml`

Parent:
`artifacts/checkpoints/team01/v112_pure_rl_merge_reacquire_integration_8i/checkpoint_final`

V113 isolates the next function. It uses 1.45-1.85 km rear/offset entries
against the same late-kill Viewer-contract BT, locks weapons for 10 seconds,
then rewards a sustained 850-1600 m / ATA <= 15 degree gun window and actual
damage. It deliberately does not train the reciprocal merge in this run.
The same conservative transfer system is active: 6,144-step actor freeze,
1:4 actor updates, alpha 0.02, optimizer reset, and anchor 12,000.

Status at the time of this update:

| Iteration | WEZ steps/episode | Damage reward | Crash | Notes |
| --- | ---: | ---: | ---: | --- |
| 0 | 0.8 | 8.0 | 0 | actor frozen |
| 1 | 10.8 | 146.4 | 0 | actor frozen |
| 2 | 6.6 | 53.4 | 0 | actor released at 1:4 |

The deterministic iteration-0 replay reduced mean target health to 0.923
across 12 episodes. Because weapons are hard-locked through 10 seconds, this
is not a spawn kill. It proves real delayed damage, but mean final range was
still about 7.1 km: the current behavior gets a burst and then loses the
target. No V113 checkpoint is promoted yet. Let the run complete, then
select among `checkpoint_000002/4/6/8/10` by deterministic replay, not by
training reward alone.

Current logs:

- `artifacts/logs/team01/v113_pure_rl_delayed_gun_conversion_10i/training_log.csv`
- `artifacts/logs/team01/v113_pure_rl_delayed_gun_conversion_10i/engagement_replays`
- launcher output:
  `artifacts/logs/team01/v113/launcher_stdout.log`

Evaluation command after completion:

```powershell
python scripts/analyze_merge_reacquire.py artifacts/logs/team01/v113_pure_rl_delayed_gun_conversion_10i/engagement_replays --iterations 0 1 2 3 4 5 6 7 8 9
```

Promotion requires repeated target-health reduction at more than one
checkpoint, increasing WEZ dwell, crash rate zero, and no catastrophic ATA
regression. Prefer the earliest checkpoint that meets the gate. Do not
blindly choose `checkpoint_final`.

## Next Steps

1. Finish and score V113. If damage disappears after actor release, use
   V113 `checkpoint_000002` as the parent; it corresponds to the strongest
   frozen-actor result seen so far. If a later deterministic replay produces
   lower target health without crashes, select that checkpoint instead.
2. Build V114 as an armed consolidation stage. Mix roughly 50% exact Viewer
   head-on, 30% V113 delayed gun entries, and 20% V111 rear/side rehearsal.
   Keep the 10-second weapon lock and conservative transfer controls. The
   gate is post-merge damage, not merely initial tail damage.
3. Only after V114 produces repeatable delayed damage, begin the league.
   Do not discard V112 or the selected V113 checkpoint; they are league
   anchors and regression tests.

## Self-Play Design

The existing in-process league supports a random pool of frozen lightweight
RL bundles through `env_config.selfplay_opponent_pool`. It cannot combine
that pool with `target_viewer_bt` in one run. Native BT DLLs also read the
process-wide `Rule_forTraining.xml`, so multiple DLL/XML pairs must not be
hot-swapped per episode.

Use an alternating league instead:

1. **RL league round:** set `target_viewer_bt: false` and provide a pool of
   compatible tactical32 bundles, including V112, selected V113, V114, and
   an older aiming anchor such as V094.
2. **Late-kill BT round:** return to
   `AIP_latekill_practice_bt.dll + Rule_latekill_practice_bt.xml`.
3. **Climb/dive BT round:** use
   `AIP_BASE_team_climb_dive_approach.dll +
   Rule_team_climb_dive_approach.xml`.
4. **Team01 fly-BT round:** use
   `AIP_team01.dll + Rule_team01_weapon_v7.xml`.
5. Add each accepted new RL bundle to the next RL-pool round. Retain older
   anchors so the policy cannot improve against one opponent by forgetting
   another.

Every round must use the SAC transfer controls and a mixed scenario pool.
Reject any checkpoint that regresses the frozen V112 geometry gate, loses
V113 delayed damage, or introduces a crash. This is genuine league-style
self-play even though BT opponents are alternated between processes instead
of loaded simultaneously.

## Current State

- V107, V108, V109-final, and V110 are rejected.
- V111 and V112 are accepted curriculum parents.
- V113 is running and has demonstrated delayed damage but no kill yet.
- The original `Rule_forTraining.xml` was restored after every completed
  scoped run. Verify it again when V113 exits.
- No external AI communication code was added or changed. Training,
  analysis, BT calls, and Viewer protocol handling remain local.

## Handoff Operating Rule

For every new experiment, add a small dated entry here containing: YAML path,
parent checkpoint, exact target DLL/XML, decision/action-repeat setting,
promotion gate, stop condition, and the observed outcome. Mark rejected
checkpoints explicitly so a future agent does not accidentally submit or
resume them. Keep source comments limited to contracts and non-obvious unit
conversions; this handoff is the canonical experiment history.

## Self-Play Gate

V112 has passed the geometry/reacquisition gate. Self-play now waits only for
an armed V114 consolidation candidate with repeatable delayed damage. Start
the alternating RL/BT league immediately after that gate; do not begin from
V113 solely because one deterministic burst happened.

## 2026-07-27 Continuation (Claude)

### V113 scoring - checkpoint_000002 PROMOTED

Deterministic replay (analyze_merge_reacquire, headon, 12 ep) confirmed the
handoff prediction: delayed damage peaks while the actor is frozen then fades
after release. tgt_hp by iter: 0=0.923, 1=0.800 (best), 2=0.804, 4=0.946,
5=1.000. Promoted `v113_.../checkpoint_000002` (strongest retained delayed
damage) as the V114 parent.

### V114: armed consolidation - REJECTED (range blowout)

File: `experiments/team01_v114_pure_rl_armed_consolidation_10i.yaml`
Parent: V113 checkpoint_000002. Target: latekill BT, decision period 6,
10 s lock. Mix 50% viewer head-on / 30% V113 delayed-gun / 20% V111 rear.
Both merge (V112) and gun/damage (V113) reward terms active.

Result: FAILED the post-merge-damage gate. Head-on angle is fine (post_best
~2.9 deg, final ATA 6-14 deg, min alt ~4.1 km, no crash) BUT final range
blows out to 8.5-9.3 km and tgt_hp stays ~1.000 (only iter 2 = 0.996). Same
"brief burst then loses the target" failure as V113 (there ~7.1 km). Not a
Viewer candidate. Do not promote V114.

### V115: anti-separation - RUNNING

File: `experiments/team01_v115_pure_rl_antiseparation_10i.yaml`
Parent: V114 checkpoint_000002. Attacks the post-merge range blowout:
far_range 0.28->0.55 (start 2800->1900), escape_opening 0.24->0.50 (start
2600->2200), reacquire_turnin 0.55->1.30 range 2600->4200, reacquire 7->10,
range_progress 0.08->0.14, engagement band tightened. Goal: turn back in and
RE-CLOSE to the 850-1600 m gun band after the reciprocal merge, then convert.
Same conservative transfer, 10 s lock, 50/30/20 pool. Gate unchanged. Score
by deterministic replay when complete; STOP + flag for Viewer if it passes.

NOTE (cross-track): the independent V060-V070 track (Claude, in
[[v020-pure-rl-plan]]) hit the SAME sustained-gun-window wall vs strong BTs
after fixing the period=6 decision-rate transfer gap; V068 there is a robust
does-not-lose model but also gets no kill. The post-merge range blowout is the
central unsolved problem on both tracks.

### V115: anti-separation - REJECTED (same range blowout)

Deterministic replay (headon): final range still 8.7-9.2 km every iter,
tgt_hp 1.000 (zero damage), final ATA 9.5-18 deg. The stronger far-range /
reacquire / range-progress rewards did NOT bring the range in. Not a Viewer
candidate. Do not promote V115.

### STRUCTURAL DIAGNOSIS + open decision (blocking the armed candidate)

Two focused reward attempts (V114, V115) both failed the SAME way: after the
759.9 m reciprocal head-on merge, the range blows out to ~9 km and no gun
window is sustained -> zero post-merge damage. Root cause looks GEOMETRIC, not
reward-tunable: reciprocal 200 m/s -> ~400 m/s opening; by the time the 10 s
weapon lock releases the jets are already ~4 km apart, a 180 deg reversal costs
another ~15-20 s / several km, and a co-speed stern chase from ~9 km cannot
re-close inside the 60 s episode. Angle is fine (post_best ~2.9 deg, final
6-14 deg, min alt ~4.1 km, no crash) - only RANGE fails.

Per-handoff design forbids the merge-pass "spawn-kill shortcut" via the 10 s
lock, but that lock may be removing the ONLY realistic damage window on a
reciprocal merge (the mutual nose-on approach through 152-1219 m). Resolving
this is a STRATEGY decision, deferred to the user:
- Option A: shorten/remove the head-on weapon lock (~2-3 s) and train to WIN
  the merge-pass gun exchange (out-nose the BT through the approach) while
  keeping post-merge reacquire in the mix. Deviates from the locked-weapon
  design on purpose.
- Option B: viewer-test the best-angle curriculum model anyway (curriculum
  angles may transfer better than the V068 track did) to see real RedFlyBT
  behavior before more training.
- Option C: accept that a reciprocal 1v1 merge vs a co-performance BT is a
  draw and optimize for the HP/damage tiebreak + opponents that err.
Stopped autonomous iteration here rather than spin a 3rd same-wall tweak.

### V116: head-on turn-around (loosened transfer) - REJECTED, but FOUND the real cause

File: `experiments/team01_v116_pure_rl_headon_turnaround_16i.yaml` (from V114 c2;
transfer loosened anchor 12000->1500 / freeze 3072 / interval 2 / lr 6e-7; turn-in
reward boosted; 65/15/20 pool). Head-on final range STILL ~8.9-9.3 km, tgt_hp ~1.0.

BUT the per-step trajectory (iter12 headon ep0) reveals the real mechanism, and it
is NOT a horizontal extension or a failure to turn: **our jet DOES reverse (yaw
355->162, final ATA ~8 deg) - the TARGET BT ZOOM-CLIMBS away.** Target altitude
post-merge: 4586 -> 6562 (t26) -> 9131 (t45) -> 10968 m (t59), climbing ~6.4 km
straight up while we stay ~4.2-5.8 km. Most of the "9 km range" is VERTICAL: the
latekill BT runs to the stratosphere to avoid the fight. Our `zoom_ceiling_m: 6000`
penalty forbids following, and chasing to 11 km is an energy trap anyway.

**CONSOLIDATED CONCLUSION (16 experiments, both tracks): the strong BTs (latekill
here; RedFlyBT climbed to 7723 m in the V068 viewer test) draw/evade by ZOOM-CLIMB.
Follow them up = energy trap; don't follow = they run the clock = draw. A gun-only
kill vs a committed zoom-climb evader is structurally near-impossible in a 60 s 1v1;
the realistic outcome is a DRAW (both survive, 0 damage) -> HP/damage tiebreak.**
Both tracks HAVE does-not-lose models (V068; V112/V116 keep min alt ~4.1 km, no
crash, good final ATA). Neither can force a kill. This is now a STRATEGY decision
for the user, not a reward-tuning problem. Stopped autonomous iteration.

### Easy-opponent proof + V117 self-play - PIPELINE CONFIRMED, V116 is a Viewer candidate

User strategy: verify kills vs an easy opponent, then self-play. Both confirm the
gun pipeline is FUNCTIONAL and the only wall was BT zoom-climb evasion:
- **Easy fixed (non-evasive) target:** V116 dealt 51% damage (tgt_hp 1.0->0.49).
- **V117 self-play gen1** (from V116 cfinal; opponent = frozen V116 RL bundle,
  target_viewer_bt false, weapons live at 5 s, 60% gun-offset / 20% head-on / 20%
  rear): deterministic replay HEAD-ON tgt_hp 0.266 at iter0 (= V116 itself deals
  **73% damage** to a fighting opponent) at ~1750 m range. Confirms: vs an opponent
  that does NOT zoom-climb, our merge-trained policy closes and nearly kills.
- BUT V117 TRAINING REGRESSED (head-on tgt_hp 0.266 -> 0.491) and the BEHIND
  scenarios stay tgt_hp 1.0: our own policy ESCAPES/extends when defending, so
  self-play degenerates to mutual escape (the same evasion the BT uses). Pure RL
  self-play does not bootstrap kills here. V117 NOT promoted.
- **`v116_pure_rl_headon_turnaround_16i` is the current best VIEWER CANDIDATE:**
  merge-trained, executes the 180 deg reversal, deals 73% damage to fighting
  opponents in the exact-Viewer head-on. Recommend a real RedFlyBT viewer test
  (blue = V116) - it should engage the reciprocal merge better than the V068 track
  did (V068 was not merge-trained and got out-angled 0.00). Viewer test needs the
  user (open server). Deploy: `TEAM01_MODE=rl TEAM01_BUNDLE_DIR=artifacts/models/
  team01/v116_pure_rl_headon_turnaround_16i python student/my_submission.py` (blue
  first) + the RedFlyBT run_unreal_inference red client.

### V116 solo + HYBRID viewer tests (user authorized leaving pure RL to WIN)

- **V116 solo vs RedFlyBT (viewer):** BEST pure-RL viewer result yet - out-angle
  **0.54** (V068 was 0.00), min ATA 7.6, closed to 243 m. BUT followed the BT into
  a low descending two-circle, bled to 143 m/s and **CRASHED at 305 m (LOSS)**. The
  user (BFM-literate) correctly named it: descending two-circle rate fight that
  grinds to the floor; needs a CUT to resolve. Authorized a hybrid / leaving pure RL.
- **HYBRID = BT mover + RL gun-window aimer** (existing `TacticalHybridActionProvider`;
  my_submission MODE=hybrid). Mover = team01_weapon_v7 BT (= RedFlyBT's own rule =
  energy-disciplined mirror, never dies). Made `BT_DLL`/`BT_RULE`/`safe_rl_altitude_m`
  env-overridable in student/my_submission.py (TEAM01_BT_DLL / TEAM01_BT_RULE /
  TEAM01_RL_ALT_GATE).
  - v1 (RL gate default 4000 m): SURVIVED (fixed the crash, energy kept) -> DRAW,
    but RL aimer activated **0 times** (fight was below 4000 m; `_rl_allowed` gated
    RL off) = pure BT mirror. out-angle 0.38, cone 0-0.
  - v2 (RL gate lowered to 1200 m): RL activated **10 times** (gate fix worked),
    out-angle **0.80** (dominant, best ever), SURVIVED (alt min 1654, spd 240). But
    STILL cone 0-0 - the 10 RL hand-offs were at 5592 m / ata ~50 deg (not terminal
    shots); min ATA only 45.7 this match. Mutual mirror still denies the tight
    <=4 deg AND 152-1219 m combo.
- **STATE: the hybrid is a solid does-not-lose + DOMINANT-position model (out-angle
  0.80, never crashes) but cannot force a kill vs the strong reference BT -> DRAW.**
  Consistent with the whole investigation: a gun-only kill vs a well-tuned mirror/
  evader BT is structurally near-impossible; the realistic outcome is a draw decided
  on position/HP tiebreak. Hybrid (team01_weapon_v7 mover + V116 aimer, RL gate 1200)
  is the current best SUBMISSION candidate for a not-lose + position-advantage result.

## 2026-07-28: WINNER FOUND -- team01 climb-dive BT kills RedFlyBT on both geometries

### Why we kept drawing (root causes, both now fixed)
1. **Mirror symmetry.** The hybrid used team01_weapon_v7 as our mover -- the SAME rule
   RedFlyBT runs. Identical logic from a symmetric start cannot win. Proven locally:
   weapon_v7 vs weapon_v7 = draw; every codex-family rule (v8/v12/v43/codex_v5) = draw.
2. **The anti-zoom doctrine was wrong.** RedFlyBT wins by ZOOM-CLIMBING (it took us to
   7.7-11.0 km). Our reward configs penalised altitude (zoom_ceiling 5300-7800) and the
   hybrid had a 5800 m zoom guard, i.e. we had BANNED the only winning tactic. The
   reference climb_dive BT beats RedFlyBT by climbing 4572 -> 13924 m, holding a median
   +971 m advantage, then diving.

### Why the training BT would not run in the Viewer (user's question -- it CAN)
Not a DLL limitation: all BT DLLs export the same C API. The failure was a RULE-FILE
RACE. Blue and Red both copy their rule onto the shared Rule_forTraining.xml, and
BTActionProvider creates the tree LAZILY on the first PlaneInfo -- by then Red had
overwritten the file, so Blue built its tree from Red's rule and CreateBehaviorTree
threw 0xe06d7363. **Fix: `_eager_create_bt()` in student/my_submission.py builds the
tree inside the `activate_rule_xml` block, before the opponent client starts.**
Verified: `[team01] BT pre-built from ...` with both the Release_red climb-dive DLL and
our own rebuilt one. Also made `REMOTE_BT_FIGHTER_ID` overridable (TEAM01_BT_ID).

### Our own implementation (submission-legal, built from AIP_LIB_work)
- NEW node `Task_ClimbDiveTargetApproach` (.h/.cpp) + registered in CPPBehaviorTree.cpp,
  TaskNodes.h and AIP_DCS.vcxproj. Behaviour: climb toward a point anchored on the
  bandit's HORIZONTAL position with the climb angle capped (a pure vertical zoom made the
  jet porpoise and separate to 12 km), hold the perch, convert to a diving PURE-pursuit
  attack only when the bandit is inside dive range; guards for hard deck and thin-air
  stall recovery (both steer TOWARD the bandit, never away).
- Tunables read from CD_* env vars for sweeping; **winning values are now the built-in
  defaults**: CLIMB_UNTIL 1500, DIVE_UNTIL 400, DIVE_RANGE 2800, MAX_TAN 0.87, LEAD 0.0
  (pure pursuit -- lead 0.25/0.4 always lost), STALL 140, PERCH_EXTRA 500, HARD_DECK 1200.
- Built Debug|x64 with VS2019 v142 (project targets v143/v145 which are not installed;
  pass -p:PlatformToolset=v142). Artefacts: `AIP_DCS_climbdive.dll` +
  `Rule_team01_climbdive.xml` in the release root.

### Result (local, 200 s, competition damage cone +-4 deg / 152.4-1219.2 m, BT decision period 6)
| Geometry | Outcome | Our HP | Target HP |
| --- | --- | ---: | ---: |
| beam_R (the real Viewer start) | **target destroyed** | 1.0 | -0.005 |
| beam_L (mirrored) | **target destroyed** | 1.0 | -0.034 |
Reproduced with NO env vars set (baked defaults), so a submission run behaves identically.

### Deploy for the Viewer test
```
TEAM01_MODE=bt TEAM01_BT_DLL=AIP_DCS_climbdive.dll TEAM01_BT_RULE=Rule_team01_climbdive.xml \
  python student/my_submission.py     # blue FIRST, wait for plane_id=0
```
then the RedFlyBT red client. For a BT+RL build use TEAM01_MODE=hybrid with
TEAM01_BUNDLE_DIR=<rl bundle> and TEAM01_RL_ALT_GATE=1200 -- but note the RL aimer only
fires inside the gun window, and the pure BT already kills, so validate the pure BT first.

### Not yet done
- Real Viewer validation of this BT (needs the user to open the server).
- V118 (out-climb RL, zoom ceiling removed, alt_advantage 0.65) finished training; not
  yet scored -- it is the RL counterpart of the same doctrine flip.

## 2026-07-28 (2): Viewer-gap re-analysis + role split

### Gap checks
1. **Damage model MATCHES the competition spec** (ruled out): `update_damage()` uses
   `half_wez = wez.angle_deg / 2` and fires when `half >= |ATA|` inside
   `min_range_m..max_range_m`. Our eval yaml sets angle_deg 8.0 / 152.4-1219.2, i.e.
   exactly |ATA| <= 4 deg in the competition band. Local "target destroyed" results are
   therefore real rule-legal kills, not a scoring artifact.
2. **Remote vs local BT data path**: both feed the same `OPlaneData`
   (position/attitude/speed/team); the DLL derives the blackboard itself. Not the gap.
3. **THE GAP IS ALTITUDE ESCALATION.** Same BT, same rule, both sides:
   | run | our alt max/end | opponent alt max/end |
   | --- | ---: | ---: |
   | LOCAL (kill) | 6578 / 5453 | 6901 / 5861 |
   | VIEWER (0 dmg) | 13414 / **13354** | 13131 / **13131** |
   Locally both jets climb to ~6.6 km and come back DOWN, where turn rate still exists and
   the kill happens. In the Viewer RedFlyBT zooms to 13 km and never descends; our node
   followed it (perch = target + 1500) so both ended stuck in thin air at 152-175 m/s where
   nobody can point (our best ATA 31.7 deg pure BT / 6.5 deg hybrid, but never <= 4).
   **Fix applied: `CD_ALT_CAP` (default 8000 m) in Task_ClimbDiveTargetApproach -- we stop
   following above 8 km. Local kills on BOTH geometries are preserved after the change
   (target destroyed, our HP 1.0, both R and L).** Needs Viewer validation.

### Viewer results so far with the new BT
| build | RL hand-offs | our ATA min | <=12 | in-band | out-angle | their <=4 | damage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| pure climb-dive BT | - | 31.7 | 0 | 54 | 0.20 | 7 | 0-0 |
| hybrid (climb-dive + V116 aimer) | **161** | **6.5** | 28 | **125** | **0.78** | **0** | 0-0 |
The hybrid is clearly the right architecture: the BT builds the window (125 in-band frames,
out-angles 0.78, denies the opponent every shot) and the RL tightens the aim (31.7 -> 6.5).
Two hybrid guards had to be opened or they fought their own mover:
`TEAM01_ZOOM_CEIL=14500` (was 5800, would cancel the climb) and `TEAM01_RL_PITCH_GATE=-40`
(was -5, switched the aimer off during every diving attack). `TEAM01_RL_ALT_GATE=1200`.

### RL role split
- V118 (out-climb RL) scored: best ATA 10.5 -> 4.6 across training but cone ~0.5 frames and
  target HP ~1.0 -- it aims better but does not kill, and it did not really out-climb.
- **V119 RUNNING** `team01_v119_gunwindow_aimer_150i` (from V118 c150): trains the RL as the
  TERMINAL AIMER only -- episodes START inside the gun window (400-800 m, small offsets, one
  diving entry), weapons live at 2 s, gun_window ata 6 deg / scale 1.2 / dwell 0.15,
  damage_scale 120, far-range penalty from 1400 m, alt_advantage 0.05 and zoom ceiling 8000
  so it never climbs away. 45 s episodes = many terminal reps.
  Gate: real cone frames + target-health reduction; then drop it into the hybrid as the aimer.

## 2026-07-28 (3): Full stack validated LOCALLY -- kills on both beam geometries

New local harness `scratch_hybrid_screen.py` runs the REAL submission stack
(TacticalHybrid = climb-dive BT mover + RL aimer) against RedFlyBT's rule without the
Viewer, using the same per-force-side rule-swap patch. Knobs via env: RL_BUNDLE,
OWN_RULE/OWN_DLL, EVAL_YAML, TEAM01_AIM_SCALE, TEAM01_GUN_ATA, TEAM01_GUN_MAX,
TEAM01_RL_ALT_GATE, TEAM01_ZOOM_CEIL, TEAM01_RL_PITCH_GATE.

### RL aimer training
- **V119** `team01_v119_gunwindow_aimer_150i` (terminal-aimer specialist): cone frames per
  episode 0.4-0.6 (V118) -> **1.9-3.2**, target HP 1.00 -> **0.914** over a 45 s episode,
  best ATA 7.7, <=12 deg 26.5 frames. This is the aimer to use.
- **V120 RUNNING** `team01_v120_aimer_randomized_120i` (from V119 c150): same lesson with a
  much wider entry distribution (150 m position, 14-20 deg heading, 15 m/s speed) to fix the
  off-nominal cases below.

### Hybrid authority sweep (the aimer was overriding the mover)
| aim_scale / gun_ata | beam_R | beam_L |
| --- | --- | --- |
| 0.60 / 55 (stock) | kill | **draw** |
| 0.30 / 55 | kill | draw |
| 0.60 / 25 | kill | tgt 0.026 |
| **0.30 / 25** | **kill (tgt -0.005)** | **kill (tgt -0.031)** |
Baked as the defaults in `student/my_submission.py` (`TEAM01_AIM_SCALE=0.30`,
`TEAM01_GUN_ATA=25`). Lesson: the RL is a TRIM on the mover, not a co-pilot -- at stock
authority it cancelled the BT's own gun solution.

### Robustness (honest)
Randomized starts (position 120 m / heading 10 deg / speed 12 m/s), 3 runs with the winning
config: **1 kill, 2 draws, 0 losses** (we never take damage). The fixed beam_R start -- which
is the geometry the Viewer actually opens with -- is a reliable kill. V120 targets the
off-nominal draws.

### Current best submission stack
```
TEAM01_MODE=hybrid \
TEAM01_BUNDLE_DIR=artifacts/models/team01/v119_gunwindow_aimer_150i \
TEAM01_BT_DLL=AIP_DCS_climbdive.dll TEAM01_BT_RULE=Rule_team01_climbdive.xml \
python student/my_submission.py        # blue FIRST
```
(aim 0.30 / gun_ata 25 / alt gate 1200 / zoom ceil 14500 / pitch gate -40 are now defaults;
the BT carries CD_ALT_CAP=8000 internally.)  Still to do: Viewer validation of the
alt-capped BT + V119/V120 aimer.

### 2026-07-28 (4): aimer A/B on randomized starts -- profile is "never loses"
5 randomized runs each (position 120 m / heading 10 deg / speed 12 m/s), full stack,
aim 0.30 / gun_ata 25:
| aimer | kills | damage-superior runs | losses |
| --- | ---: | ---: | ---: |
| V120 (wide-entry) | 0/5 | 2 (tgt 0.127, 0.958) | **0** |
| V119 | 0/5 | 2 (tgt 0.280, 0.987) | **0** |
Both still KILL both fixed beam geometries (V120: R -0.006 / L -0.001).
So the stack is: reliable kill on the geometry the Viewer actually opens with, and
win-or-draw everywhere else -- we never take damage in any local run.
`V120` is set as the default `BUNDLE_DIR` in student/my_submission.py (wider training
distribution = better prior for real Viewer variability; the two are otherwise equal).

**Blocked on the user:** the decisive test is the Viewer with the alt-capped BT + V120 aimer.
Everything else is tuned as far as offline evidence supports.

## 2026-07-28 (5): the RL aimer BREAKS the BT gun setup -- submission default is now pure BT

Viewer run with the alt-capped BT (`CD_ALT_CAP=8000`) confirmed the altitude fix works:
max alt 8517 m (was 13414) and speed 234 m/s (was 152), out-angle 0.85, in-band 136 frames,
range median 2000 m -- the mover now wins the position fight with energy intact. But damage
stayed 0-0, and the decision log showed only 31 RL hand-offs because `gun_ata_deg` had been
tightened to 25 while our in-band ATA was ~60 deg.

Re-ran with the gate opened (ata 60): **191 hand-offs, our ATA min 0.3 deg (15 frames <= 4)**
-- yet still 0 damage, because those <=4 deg moments were OUTSIDE 152-1219 m while the
in-band frames had ATA 49-103 deg. A PHASING problem, not an aiming one.

Added terminal-phase control to the node (bleed overspeed + lead pursuit inside
`CD_TERM_RANGE`, knobs CD_TERM_RANGE/LEAD/THROTTLE/DELTA) -- pure BT still kills both
geometries with it, so it is kept.

**Controlled isolation on beam_R (the real Viewer geometry), local full stack:**
| configuration | result |
| --- | --- |
| hybrid, RL + guards both disabled | kill (-0.00480) == pure BT exactly |
| RL OFF, guards ON (zoom 14500, dive 2200/-30) | **kill (-0.00480)** |
| RL ON (ata 60, handback 1250, aim 0.30), guards OFF | **draw (1.0)** |
So the wrapper and the guards are innocent: **the RL aimer intervening on the approach is
what destroys the BT's gun setup.** It pulls the nose early (ATA 0.3 deg far out) and the
mover then arrives in the damage band mis-phased. The earlier "0.30/25 kills both" result
was simply the RL being gated almost entirely OFF.

Also added (kept, harmless, useful later): `gun_handback_range_m` on
TacticalHybridActionProvider -- hand the damage band back to the mover -- plus env knobs
TEAM01_HANDBACK / TEAM01_DIVE_GUARD_ALT / TEAM01_DIVE_GUARD_PITCH / TEAM01_GUN_ATA /
TEAM01_AIM_SCALE / TEAM01_GUN_MAX.

**Submission default is now `MODE=bt`** (climb-dive BT alone), verified with NO env vars:
beam_R target destroyed (tgt -0.0048, our HP 1.0), beam_L target destroyed (tgt -0.0343,
our HP 1.0).  To re-enable the hybrid for experiments: `TEAM01_MODE=hybrid`.

**Next for RL:** the aimer must be trained to trim the nose WITHOUT changing the approach
geometry -- e.g. train it only inside 152-1219 m with the BT flying the approach (behaviour
cloning of the BT outside the band), or restrict its authority to yaw/pitch trim with a hard
cap on deviation from the BT command. Until then pure BT is strictly better.

## 2026-07-28 (6): ROOT CAUSE of the local-vs-Viewer gap (confirmed) + partial-fix failure

**Cause found in code:** the two paths feed the native BT completely different inputs.
- LOCAL: `AIPilot.Step(id, force, tgt_id, tgt_force, my_fdm, tgt_fdm)` -> internally calls
  `ChangeData(..., J_NavigationData*)`, which carries Lat/Lon/Alt, phi/theta/psi,
  **u/v/w body velocities, p/q/r body rates, Ax/Ay/Az, AOA/AOS, KCAS/KTAS, Nz/VV** and
  updates the DLL's internal aircraft state.
- VIEWER: `StepWithPlaneData(OPlaneData)` -> position, attitude, speed, team ONLY.
  `ChangeData` is never called, so the DLL's internal rate/energy state stays zero.
Measured local values the BT actually receives: p=50745 (=50.7 deg/s), q=-7622, r=287,
u=746761 (=227.6 m/s). Scale factors (probed): lat/lon deg*1e6, Alt & u/v/w FEET-based
*1000, phi/theta/psi & p/q/r deg*1000.
This explains the symptom exactly: gross manoeuvring (climb, positioning, out-angle 0.85,
136 in-band frames) transfers on position/attitude alone, but fine gun tracking -- which
needs rates for lead/damping -- does not, so in-band ATA stayed 49-103 deg and damage 0.

**Partial fix attempted and REVERTED:** added `make_navigation_data()` (native_bt.py),
wired it through policies.py -> bt_action_provider so the Viewer path uses ChangeData+Step.
Viewer result: the jet DIVERGED -- altitude 4996/3582/544/2867/854/2885/5644/309, roll
+-180, pitch +-90, crash at 309 m. Cause: the packet was only partially filled (rates and
velocities set, but **AOA/AOS/KCAS/KTAS/Nz/VV and control positions left at zero**), and the
native controller evidently gain-schedules on them. The divergence is itself PROOF that the
BT consumes this data.
Now gated behind `TEAM01_BT_NAVI=1` (default OFF = the stable position/attitude path).

**Next step to actually close the gap:** fill the remaining nav fields from Viewer telemetry
-- KCAS/KTAS from speed + altitude density, AOA/AOS from body velocities
(atan2(w,u)/atan2(v,u)), Nz from finite-differenced velocity, VV from vertical speed -- then
re-test. policies.py already computes most of these for the RL observation.
