# AI Pilot Handoff - 2026-08-03

## Read this first

Project root: `C:\Users\JUN\Desktop\airop\Release_260722_plz`

This is the current authoritative handoff. Some older Korean Markdown files are
mojibake in the current console. Do not treat their rendered text as newer than
this file. Keep new documentation UTF-8 and prefer plain ASCII headings where
console encoding is uncertain.

The competition permits project-wide changes except for any external-AI or
external-network communication behavior. Do not change the BattleViewer UDP
client/protocol or add any external communication. Do not start BattleViewer
unless the user explicitly asks.

## Competition objective

Build a reliable submission which:

1. survives and holds energy,
2. finds/reacquires the opponent,
3. breaks a two-circle merge with a cut-in,
4. hands off to a pure-RL Shooter for real weapon damage and kills,
5. recovers after overshoot without ground impact,
6. later handles the 10000 ft Round-4 head-on start.

The local simulator and current Viewer physics were reconciled previously by
matching the F-16 fuel mass. Do not restart broad physics/Viewer investigation
unless a fixed-control probe disproves parity again.

## Non-negotiable workflow

- One causal change per new YAML/model branch. Never overwrite a prior YAML.
- Promote only on fixed-seed deterministic replay, not training reward alone.
- Read actual `end_condition`, target health, ownship health, and crash data.
  Do not infer victory from a reward curve.
- Do not use spawn kills: preserve the configured weapon lock in shooter work.
- Do not switch back to broad self-play or generic reward shaping before the
  staged local gates below pass.
- Do not create a BattleViewer candidate until the local stage gate says so.

## Roadmap and current state

| Stage | Purpose | Status | Evidence / gate |
|---|---|---|---|
| M0 | Measure merge geometry | done | Preserve as baseline. |
| M1 | Easy cut-in lesson | done | Baseline retained. |
| M2 | Full left/right two-circle chord cut | passed | Left 6/6, right 5/6, zero crashes. |
| M3 | Cut-in against real RedFlyBT weapon_v7 | passed | 12/12 handoffs, zero crash/incoming damage. |
| S3 | Handoff to Shooter, damage/kill | active | S3-A training, mandatory c50 screen. |
| S4 | Overshoot/re-entry | blocked by S3 | Do not start before S3 passes. |
| S5 | Split-S safety | pending | `astern_1100` must be 0/6 crash. |
| S6 | BT league + limited self-play | pending | only after core skills pass. |
| S7 | Round-4 far head-on | pending | separate Far specialist. |
| FINAL | Submission selector | pending | Viewer validation only after local gates. |

## Promoted M2/M3 implementation

### Frozen positioner

`artifacts/models/team01/m2f_target_ata_tailhold_strong_150i/bundle_000050`

Against the exact BT:

- DLL: `AIP_DCS_codex.dll`
- rule XML: `training_assets/team01_fly_bt/Rule_forTraining.team01_weapon_v7.xml`
- decision period / action repeat: 6

The successful left-side causal rule is deliberately small:

1. Preserve frozen M2f behavior on the already strong right side.
2. On the weak left side detect the true opening-to-closing range-rate reversal.
3. Make one committed chord cut with throttle floor 0.90.
4. At 1400--2400 m, own ATA <=20 deg, target ATA >=55 deg, relax pitch pull
   to `-0.45`. This maintains lag instead of pulling through present LOS.

Do not resume reward-only M2 branches. Canonical turn-sign correction was a real
contract fix, but it was not the cause of the original left-side geometric loss.

### Reusable selector pieces

- `src/dogfight/ai/measured_cutin_action_provider.py`
- `src/dogfight/ai/control_position_handoff_provider.py`
- `src/dogfight/ai/turn_circle_cut_controller.py`
- `scratch_m2_terminal_bias_probe.py`

`ControlPositionHandoffProvider` runs the Shooter in shadow each frame so its
recurrent state is warm. It permanently latches after 0.20 seconds of qualifying
handoff state. Smoke evidence shows handoff around 24.9 seconds. This is not a
BT/RL blend: the positioner owns controls until latch, then the Shooter owns all
controls.

## S3: current exact problem

The positioner can create the Shooter entry state. The Shooter cannot finish it.

Rejected full-chain screens:

- c225 Shooter `C:\Users\JUN\Desktop\airop\BEST_MODELS\s2b_beam_dominant_bundle_000225`:
  0/12 target damage, 0 kills.
- old M1b strict-line c20 Shooter:
  0/6 target damage, 0 kills.

In both cases selector handoff worked. The c225 Shooter reaches roughly 8 deg
ATA only near the end of the episode and never holds the strict 0.5 deg phase-1
gun cone. Therefore the current hypothesis is **terminal 8 deg -> 0.5 deg aim
and dwell**, not cut-in, target selection, or selector failure.

### Active experiment (do not interrupt)

YAML: `experiments/team01_s3a_actual_handoff_strictline_100i.yaml`

- Restore checkpoint:
  `artifacts/checkpoints/team01/m1b_measured_cutin_fast_actor_80i/checkpoint_000020`
- Tag: `s3a_actual_handoff_strictline_100i`
- Current run was launched through `scripts/run_experiment.py`.
- It uses only two measured live post-M3 Shooter entry states plus their mirrors.
- Target is the real `AIP_DCS_codex.dll` / weapon_v7 BT.
- weapon lock: 1.5 s, to exclude spawn scoring.
- Mandatory c50 screen. At the time this handoff was written it had reached c20
  and was still consuming CPU normally.

First gate at c50:

1. Fixed deterministic `s3_live` evaluation: target-health reduction in >=4/12
   episodes **and** strict phase-1 WEZ appears.
2. Only then run full chain with M2f c50 positioner + S3-A c50 Shooter:
   target damage >=6/12, >=2 actual kills, zero ownship crash/loss.

If S3-A fails the first gate:

- Stop this branch.
- Use logs/replays to choose exactly one cause: no terminal-cone entry, cone
  entry without damage, actor collapse/no movement, or energy/overshoot.
- Create exactly one S3-B YAML/code change. The likely next candidate is a more
  representative pool of *actual handoff-frame* states, but do not make that
  change unless the c50 traces support it.

### Active parallel companion

The user approved use of a second training slot. `S3-B` is deliberately not a
second reward experiment:

- YAML: `experiments/team01_s3b_actual_handoff_long_dwell_100i.yaml`
- Tag: `s3b_actual_handoff_long_dwell_100i`
- Same restore checkpoint, live entry state pool, target BT, reward, actor
  constraints, and 1.5-second weapon lock as S3-A.
- The **only** change is episode horizon: 25 s / 250 steps becomes 45 s / 450
  steps. It tests whether S3-A's terminal failure is lack of gun-line dwell
  time rather than inability to obtain strict aim.

At launch both S3-A and S3-B learner processes were confirmed alive. Compare
them at matching checkpoints using the same deterministic screen; do not
promote S3-B merely because its longer episode has more timeout time.

### S3-A c50 result (rejected)

`bundle_000050` was evaluated deterministically for 12 `s3_live` episodes
(25 seconds, seed 8600, weapon lock 1.5 seconds):

- target-health reduction: 0/12
- kills: 0
- strict phase-1 WEZ: one frame in one episode only
- all episodes timed out; no ownship damage in this specialist screen

It can momentarily point into a good ATA/range geometry but cannot maintain the
phase-1 gun cone. This rejects S3-A as a Shooter. Its verified launcher tree was
stopped after the c50 evaluation to release CPU. Preserve its artifacts; do not
resume it. S3-B remains the single active comparison branch.

### S3-C terminal-entry focus (active)

After S3-A freed a slot, `experiments/team01_s3c_terminal_entry_focus_100i.yaml`
was launched alongside S3-B. It makes exactly one change relative to S3-A:

- retain only the two closest validated A entry states and their mirror;
- remove the farther B states that disperse the strict terminal correction;
- preserve all target, lock, reward, randomization magnitude, restore, and
  25-second horizon settings.

Evaluate c50 first on C's focused states, then on the original mixed A/B
`s3_live` states. It is not promotable unless it transfers to the mixed screen.

### S3-B c50 result (rejected)

The 45-second dwell hypothesis failed on the same deterministic 12-episode
screen: 0/12 target-health reduction, 0 kills, only 3 strict WEZ frames total,
and 2 ownship altitude-below-min crashes. Extra time is therefore not the
Shooter bottleneck. Its verified launcher tree was stopped; do not resume it.

### S3-D terminal-cone pressure (active)

`experiments/team01_s3d_terminal_cone_pressure_100i.yaml` is active alongside
S3-C after S3-B freed a slot. It makes exactly one reward change to S3-A:
`terminal_cone_scale: 24 -> 80`. The dense 3-to-0 degree signal is strengthened
while all state, target, weapon, gun-window, lock, and actor settings remain
identical. Its c50 promotion screen is the original mixed A/B `s3_live` set.

## Known failures not to repeat

- Generic merge/cut-in reward shaping and angle-advantage farming: repeated
  circular flight, WEZ 0, or no real handoff.
- Early reward conclusions before a fixed episode screen.
- Pure self-play before a real-BT local gate: it does not reproduce RedFlyBT and
  produced transfer failures.
- Simultaneous changes to observation, controller, reward, curriculum, and
  selector: causality was lost.
- BT as a permanent low-level action blender: it can override RL and cause
  control/Viewer mismatch. Use hard ownership handoff instead.
- Trusting target/ownship columns by name without checking force-side mapping.
- Spawn/initial kills as success evidence.
- Relaxing action repeat without rechecking Viewer parity.

## Technical cautions

- Deterministic SAC action must be `tanh(mean)` before environment mapping. Raw
  mean + clipping was previously a critical inference mismatch.
- Lightweight bundle activation must match trained model configuration; validate
  with `scratch_bundle_activation_check.py` before deployment.
- Preserve the canonical turn-sign path in `single_agent_env.py` for both
  lightweight providers and pure RL direct actions.
- The user previously identified an SAC recurrent critic-state restore hazard:
  do not claim a restored recurrent policy is good solely from training metrics.
  Use fresh deterministic fixed-seed episodes for all promotion decisions.
- The target rule XML and DLL must match. `CreateBehaviorTree` failures commonly
  come from mismatched DLL/XML paths.
- Training can be CPU-heavy and an iteration can take minutes; do not kill an
  actively consuming learner merely because the dashboard/log file is quiet.
- Existing user edits may coexist. Never reset, revert, or overwrite unrelated
  changes.

## Immediate next commands after S3-A c50

Specialist evaluation:

```powershell
$py = 'C:\Users\JUN\miniconda3\envs\aip\python.exe'
& $py scratch_m2_terminal_bias_probe.py `
  --bundle artifacts\models\team01\s3a_actual_handoff_strictline_100i\bundle_000050 `
  --experiment-yaml experiments\team01_s3a_actual_handoff_strictline_100i.yaml `
  --episodes 12 --seconds 25 --seed 8600 --scenario-prefix s3_live `
  --weapon-lock-seconds 1.5 `
  --output-dir artifacts\eval\team01\s3a_c50_specialist
```

If and only if that gate passes, full-chain evaluation:

```powershell
$py = 'C:\Users\JUN\miniconda3\envs\aip\python.exe'
& $py scratch_m2_terminal_bias_probe.py `
  --bundle artifacts\models\team01\m2f_target_ata_tailhold_strong_150i\bundle_000050 `
  --shooter-bundle artifacts\models\team01\s3a_actual_handoff_strictline_100i\bundle_000050 `
  --experiment-yaml experiments\team01_m2f_target_ata_tailhold_strong_150i.yaml `
  --episodes 12 --seconds 45 --seed 8400 --scenario-prefix comp_beam `
  --turn-circle-teacher --teacher-chord-inset 0.65 --teacher-lag-pitch-floor -0.45 `
  --weapon-lock-seconds 3.0 --continue-after-handoff `
  --output-dir artifacts\eval\team01\s3a_full_chain_c50\combined
```

Calculate damage/kills from per-episode `steps.csv` and actual end conditions,
not from the last aggregate summary row.

## What the next agent should do

1. Check whether S3-A reaches c50 and that the learner is alive.
2. Run the specialist c50 gate above; inspect policy probes and engagement logs.
3. Either promote to full chain or make one narrow S3-B correction.
4. Do not jump to S4/S5/S6/S7 or BattleViewer before a Shooter that produces
   real damage/kills through the promoted M2/M3 chain exists.
5. Append results to this document and `ROADMAP_EXECUTION_STATUS_2026-08-03.md`.

---

## CORRECTION 2026-08-03 (late): the "0/12 damage" reading was a probe bug

**Every S3 branch was scoring damage. The screens reported zero because of a defect in
`scratch_m2_terminal_bias_probe.py`, not because the Shooter failed.**

### Root cause

`env.target_damage` is the damage dealt in **one policy step**. `single_agent_env.py`
reassigns it every step from that step's sub-frames (`self.target_damage =
target_damage_total`); it does **not** accumulate over the episode.

The probe recorded the episode's damage as the **final step's** value:

```python
"target_damage": float(info.get("target_damage", rows[-1]["target_damage"]))
```

Episodes almost always end outside the gun cone, so that value is ~always `0.0`.
The per-step data in `steps.csv` was correct all along; only the summary was wrong.

Verified per episode: `summary.json` says `0.000000` where `steps.csv` shows e.g.
`0.016490`. In one branch 2 of 12 episodes happened to end mid-burst and those two agreed --
which is why the failure looked like a real, if odd, result rather than a bug.

### Corrected results (episode damage = SUM over steps)

| branch | damaged | total damage | phase-1 frames | best ATA | crashes | hit |
|---|---|---|---|---|---|---|
| S3-A | 5/12 | 0.1104 | 6 | 0.22 | 0 | 0 |
| **S3-B** | **6/12** | **0.5104** | **15** | 0.30 | 2 | 0 |
| **S3-C** | **8/12** | 0.1743 | 9 | 0.24 | 1 | 0 |
| S3-D | 1/12 | 0.0089 | 1 | 0.71 | 0 | 0 |
| S3-E | 1/12 | 0.0064 | 1 | 0.63 | 0 | 0 |

Crash counts come from `summary.json` `end_condition` (they are NOT in `steps.csv`; counting
crashes from the step log yields a false zero). The S3-B crash observation in the earlier
section was correct.

### What this changes

1. **S3-A and S3-B were rejected on bad data.** S3-B is the highest-scoring branch so far
   (0.5104 total, 15 phase-1 frames). The 45-second dwell hypothesis was not disproved.
2. **The stated S3 hypothesis is wrong.** The Shooter *does* enter and hold the phase-1 cone
   (0.22-0.30 deg, well inside the 1.0 deg requirement) and *does* damage the target. It is
   not a terminal-aim failure.
3. **S3-D and S3-E regressed ~10x** (1/12, best ATA 0.6-0.7). Both were designed to fix a
   problem that did not exist, and both measurably hurt.
4. The real remaining gaps are **kills (0 everywhere)** and **crashes (S3-B 2, S3-C 1)** --
   not the ability to score.

### Fix applied

`scratch_m2_terminal_bias_probe.py` now records:

- `target_damage` / `ownship_damage` as the **sum over steps** (the episode total)
- `target_damage_final_step` and `target_damage_best_step` for comparison
- `damage_steps` (how many steps scored)
- damage and `end_condition` in the printed per-episode line, which previously made a
  scoring branch look identical to a dead one

`scratch_s3_screen_digest.py` recomputes any past screen from `steps.csv`.

### Recommended next step

Do **not** open more branches against the terminal-aim hypothesis. Suggested order:

1. Let S3-G finish (it was at iteration 82/100) and screen it with the corrected probe.
   Its single change is `gun_window_ata_deg: 12.0 -> 1.0`; note S3-D/S3-E, which also
   tightened the terminal signal, both regressed badly.
2. Re-screen **S3-B** and **S3-C** with the corrected probe, then run the full chain
   (M2f positioner + that Shooter) which has never been measured correctly.
3. Target the two real gaps: converting 0.1-0.5 damage into kills, and the 1-2 crashes.

**Method note:** the damage columns are `target_damage` / `ownship_damage`. There are no
`*_health` columns and no `end_condition` column in `steps.csv`. Asking for a column that
does not exist returns nothing and reads as a clean zero -- this bug and my own first
analysis of it both failed that exact way.

---

## S3 ROOT CAUSE FOUND (2026-08-04): train/deploy state mismatch, not terminal aim

The corrected probe made the full chain measurable for the first time. Result for
M2f c50 positioner + S3-C c50 Shooter, 12 episodes, seed 8400, `comp_beam`:

```
shooter_latched      12/12    the Shooter DOES take control, at 17-34 s
shooter_entry_seen    1/12     but its trained entry condition almost never occurs
target damage         0/12
in-range ATA after latch   3.5 - 99 deg (mostly 12-30)
```

So the selector is fine and the latch is fine. The Shooter simply cannot work from the
state it is actually given.

### The mismatch, measured

| | distance | altitude |
|---|---|---|
| S3-C trained-on states (one mirror pair) | **879 m** | 2418 m |
| Real latch frames (12, measured) | **1369-1394 m** (median 1385) | 2774-3989 m |

The positioner hands the fight over **58% farther out** and 400-1600 m higher than anything
the Shooter trained on. That is exactly why the specialist screen scores 8/12 and the chain
scores 0/12: the specialist screen *starts* the Shooter at its training distance.

**S3-C moved the wrong way.** Its single change was to keep only the two closest entry
states and drop the farther ones — but the real handoff is farther than both. S3-D and S3-E
then tightened the terminal signal further and regressed ~10x. Four branches were spent
optimising against a hypothesis the data does not support.

The 12 latch frames are tightly clustered (all ~1385 m, own ATA 6.5-31 deg, target ATA
64-82 deg), so this is a well-defined target distribution, not noise.

### S3-H (next branch)

`experiments/team01_s3h_real_handoff_pool_100i.yaml`, generated by `make_s3h_variant.py`.

- extends **S3-B** — the genuinely best-scoring branch (6/12, total damage 0.5104, 15
  phase-1 frames), not S3-C
- **one causal change**: the scenario pool becomes the 12 measured latch frames
- horizon 45 s, reward, restore checkpoint, target BT and weapon lock all inherited
- randomisation kept small (12 m / 1 deg) so the handoff distribution is matched, not blurred

**Gate: full-chain damage in >=6/12 with zero crash.** A specialist-screen pass is explicitly
NOT sufficient — that screen is what concealed this failure for four branches.

### Tools added

- `scratch_s3_screen_digest.py` — recompute any past screen from `steps.csv`
- `scratch_extract_handoff_states.py` — dump real latch frames as scenario spawns
- `make_s3h_variant.py` — build S3-H from those frames

### Standing lesson

Two independent zero-readings in one day came from asking for data that was not there:
the probe took the final step's damage (~always 0), and my first digest asked for
`target_health` and `end_condition` columns that do not exist in `steps.csv`. **A missing
column and a genuine zero look identical.** Verify a metric can be non-zero before trusting
a zero, and screen a specialist in the configuration it will actually be deployed in.

### S3-G screened (2026-08-04) — completes the pattern

S3-G ran to 100/100 and exited on its own. Its c50 was screened with the corrected probe
(12 episodes, seed 8600, `s3_live`, lock 1.5 s), giving the full branch ranking:

| branch | its one change | damaged | total damage |
|---|---|---|---|
| **S3-C** | narrowed the entry pool to the 2 closest | **8/12** | 0.1743 |
| **S3-B** | horizon 25 s -> 45 s | **6/12** | **0.5104** |
| S3-A | baseline | 5/12 | 0.1104 |
| S3-G | `gun_window_ata_deg` 12 -> 1 | 3/12 | 0.0548 |
| S3-D | `terminal_cone_scale` 24 -> 80 | 1/12 | 0.0089 |
| S3-E | gunline desaturate | 1/12 | 0.0064 |

**Every branch that tightened or pressured the terminal signal (G, D, E) got monotonically
worse.** That direction is exhausted; do not open another branch along it. The terminal cone
was never the binding constraint — the Shooter already reaches 0.22-0.30 deg ATA.

Note S3-C's 8/12 is on the *specialist* screen only. Its full chain is 0/12 (see the root
cause above), which is precisely why the S3-H gate is the full chain.

### S3-H launched and verified

`experiments/team01_s3h_real_handoff_pool_100i.yaml` is training. Resolved config confirmed
against the live run, not just the YAML:

```
weapon_lock_seconds 1.5      target_behavior_dll AIP_DCS_codex.dll
max_engage_time 45.0         episode_step_limit 450
gun_window_ata_deg 0.5       terminal_cone_scale 24      (S3-A values, not S3-D's 80)
restore  m1b_measured_cutin_fast_actor_80i/checkpoint_000020
scenarios 12 x s3_handoff_ep*  (1369-1394 m, alt 2774-3989 m)
```

`artifacts/dashboard/.../config.json` does NOT store the fully resolved env config -- it
showed empty reward/lock fields for this run. Verify inheritance with the `[argv]` line in
`logs/s3h_stdout.log` plus `load_experiment_env_config(...)`, not with the dashboard file.

### Correction: the mismatch is ALTITUDE (and single-point training), not distance alone

The "58% distance mismatch" framing above was incomplete. Tested directly by moving the latch
window instead of retraining -- `--handoff-max-range-m 950`, exposed on the probe (defaults
reproduce the provider's own 500/1400 m window, so omitting it changes nothing):

```
latch <= 1400 m   latched 12/12   Shooter handed 1323-1370 m   damage 0/12
latch <=  950 m   latched 10/12   Shooter handed  888- 920 m   damage 0/12
```

Handing the Shooter its exact training distance produced no damage at all. Full axis
comparison (`scratch_state_gap.py`) shows why:

| axis | S3-C trained on | actually handed | |
|---|---|---|---|
| distance_m | 879 | 1323-1370 | DISJOINT |
| **altitude_m** | **2418** | **2781-3984** | **DISJOINT even after fixing distance** |
| own_ata_deg | 8 | 7-36 | shifted |
| own_speed_mps | 208 | 184-263 | shifted |
| closure_deg | 123 | 122-167 | shifted |

**S3-C trained on exactly two states -- one geometry plus its mirror -- so every axis is a
single point.** Altitude never overlaps the real handoff in either configuration. The Shooter
has never seen the altitude band it is deployed into, and has never seen variation at all.

This makes S3-H the right response and for a better-supported reason than originally stated:
it matches the handoff distribution on *every* axis, not just distance. Verified against the
true latch frames:

```
axis          S3-H pool          real latch        gap
distance_m    1369-1394          1323-1370         shifted (~34 m, within tolerance)
own_ata_deg      6-31               7-36           ok
target_ata_deg  64-82              64-83           ok
altitude_m    2774-3989          2781-3984         ok
own_speed_mps  184-262            184-263          ok
closure_deg    123-171            122-167          ok
```

### Tool trap found while doing this

`first_handoff_time_s` is when the **cut-in** handoff criterion is first met. It is NOT when
the Shooter latches, and it does not move when the latch window changes -- so extracting
states at that timestamp returned identical ~1385 m states for both the 1400 m and 950 m
runs, which is how the error surfaced. The real latch is `shooter_handoff_frame`, in 60 Hz
sim frames (`time_s = frame / 60`). `scratch_extract_handoff_states.py` now uses it.

Also note the handoff happens at ~1350 m, outside the phase-1 gun envelope (max 914 m), so
the Shooter must close before it can score at all. Any future latch-window tuning should be
judged on damage, not on how close the handoff looks.

---

## CRITICAL 2026-08-04: the S3 gate was being measured at the wrong episode length

**The frozen c225 baseline already passes the full S3 gate. The S3 screens were run at 25 s
and 45 s; the competition match is 200 s.**

Identical 12 `comp_beam` episodes, identical seed 8400, identical weapon lock, c225 alone --
the ONLY difference is episode duration:

| duration | damaged | kills | total damage |
|---|---|---|---|
| 45 s | 1/12 | 0 | 0.0004 |
| **200 s** | **12/12** | **3** | **~5.01** |

At 200 s: `target destroyed` in episodes 5, 8 and 10; every other episode timed out with
damage; no crash (min altitude 355-1617 m against the 304.8 m floor).

The S3 gate is "damage in >=6/12, >=2 kills, zero crash". **c225 alone: 12/12, 3 kills,
0 crashes.** It passes outright, with no positioner, no cut-in, no handoff.

### What this means

Eight branches (S3-A through S3-H) have been optimising a Shooter against a gate that the
existing frozen baseline already clears -- the gate was simply being evaluated over a window
in which *nothing* can score. At 45 s even c225 gets 1/12. The chain screens were not
measuring "can this Shooter finish"; they were measuring "can anything finish in 45 s",
and the answer is no.

This also re-explains the earlier findings without needing them to be wrong:

- Specialist screens at 25 s scored 5-8/12 only because those scenarios *start* at the firing
  geometry. Given 25 s from a 1350 m crossing pass, nothing scores.
- The chain's 0/12 was never evidence about the Shooter.

### Immediate consequence

Re-evaluate every promotion decision at 200 s before spending another training slot. In
particular the cut-in chain must be justified by beating c225 alone **at 200 s**, which is a
much higher bar than the one it has been measured against.

### Method note

This is the fourth false zero in two days, and the same shape as the others: a number that
cannot be non-zero under the measurement conditions was read as evidence about the model.
Before trusting a zero, verify the metric is capable of being non-zero in that configuration
-- here, by running the known-good baseline through the identical screen.

### The chain does NOT beat the frozen baseline at 200 s

Same 12 `comp_beam` episodes, same seed 8400, same weapon lock, both run for the real match
length:

| configuration | damaged | kills | crashes | total damage |
|---|---|---|---|---|
| **c225 alone** | **12/12** | 3 | **0** | ~5.01 |
| M2f positioner -> c225 (full chain) | 10/12 | 3 | **1** | ~5.18 |

Comparable damage and identical kills, but the chain leaves two episodes at zero and adds an
`ownship altitude below min` -- an outright loss in competition. **The cut-in chain currently
costs more than it adds.**

### S3-H stopped at 24/100

Its premise failed twice over: it trains a Shooter for handoff states judged over a 45 s
window in which even c225 scores 1/12, and it serves a chain that does not beat c225 alone at
200 s. Artifacts are preserved; `experiments/team01_s3h_real_handoff_pool_100i.yaml` and
`make_s3h_variant.py` remain if the premise is ever restored.

### Recommended direction

1. **Treat c225 as the submission baseline.** It passes the S3 gate alone at 200 s.
2. **Any cut-in / positioner / Shooter work must beat c225 alone at 200 s** on the same
   episodes and seed. That is the only bar that matters; specialist screens and short-horizon
   screens have both proven misleading.
3. The two known c225 defects are worth more than the chain right now:
   - `astern_1100` split-S crash (4/4 reproducible, cause diagnosed: commits to a vertical
     reversal from 1536 m; lever is altitude-aware, not spawn coverage)
   - the 1 crash seen in chain ep09 and the low minimum altitudes (285-892 m) in several
     200 s episodes suggest the same failure mode is present in the baseline
4. **BattleViewer validation of c225 at 2500 ft has still never been run.** It remains the
   largest unknown in the project.

### Evaluation rule going forward

Screen at **200 s**. If a shorter horizon is used for training-speed reasons, the promotion
screen must still be 200 s, and the known-good baseline must be run through the identical
screen to prove the metric can be non-zero.

### c225's "zero crashes" is a 50 m margin, not safety

From the same 200 s screens, minimum altitude per episode (crash floor 304.8 m):

```
c225 alone   355  372  410  478  566  577  854 1096 1174 1349 1422 1617
             -> 6 of 12 episodes go below 600 m; lowest is 355 m = 50 m of margin
             -> minimum speed 114 m/s; 3 of 12 drop below 150 m/s

full chain   301  440  507  617  663  773  802  892 1278 1315 1612 2857
             -> lowest 301 m: through the floor. That is the ep09 crash.
```

The submission model wins with 50 m to spare in half its episodes. Zero crashes in 12 (and in
the 26-match randomised screen) is therefore not evidence of a safe policy -- it is a narrow
margin that happened to hold. Combined with the reproducible `astern_1100` 4/4 split-S crash,
crash is the live risk to the submission, not a theoretical one.

This is why **S5 (`experiments/team01_s5_dive_guard.yaml`) is the priority over any chain
work**. Its single change is `dive_penalty_scale: 0 -> 0.5`; that penalty was never enabled,
so c225 has never once been punished for a steep dive. The defaults it activates
(`safety_altitude_m` 1800 m, `dive_pitch_deg` -12 deg) bracket the measured failure exactly:
the crash dive begins at 1536 m and reaches -53 deg pitch, at 254-276 m/s -- so it is a
control choice, not an energy failure.

S5 warm-starts from `s2b_beam_dominant/checkpoint_000200` (the block that produced c225,
BEAM crash 0.00) and must pass all four gates before replacing c225:

1. `astern_1100` 0/6 crash (currently 4/4)
2. astern 200/300/400/600/800 damage maintained (300 m and 600 m are kills today)
3. randomised beam: no regression from 26/26, zero crash
4. `comp_beam` **at 200 s**: no regression from 12/12 damage, 3 kills, 0 crash

If S5 does not clear all four, keep c225. It already passes the S3 gate on its own.

---

## S5 dive guard — c50 gate results (2026-08-04)

Single change from c225: `dive_penalty_scale 0 -> 0.5`, warm-started from
`s2b_beam_dominant/checkpoint_000200`. That penalty had never been enabled, so c225 had never
been punished for a steep dive.

### Gate 1 — `astern_1100` crash: PASSED, decisively

| | crashes | net damage |
|---|---|---|
| c225 | **3/3** | 0.0000 |
| S5 c50 | **0/3** | **0.2938** |

It not only stops crashing, it scores there. (Identical to four decimals across runs --
fixed scenarios are deterministic.)

### Gate 2 — astern range profile: mixed, but better by match outcome

| range | c225 | S5 c50 |
|---|---|---|
| 200 m | 0.3886 | **0.5467** |
| 300 m | **1.0143 kill** | 0.6928 |
| 400 m | 0.2991 | **0.7563** |
| 600 m | **1.0015 kill** | 0.4997 |
| 800 m | 0.3462 | **0.4135** |
| 1100 m | **0.0000 crash** | **0.2938** |

Total damage rises (3.05 -> 3.20) and becomes far more uniform, but **both kills are lost**.
Scored by match outcome rather than damage magnitude, S5 is strictly better here: two wins
stay wins with a smaller margin, and one **loss becomes a win**.

### Gate 3 — randomised beam (the competition start): FAILED

```
S5 c50   WIN 7 / draw 5 / LOSS 0   our crashes 0   kills 0   mean net 0.0552
c225     WIN 26 / draw 0 / LOSS 0  our crashes 0   kills 6   mean net 0.5814
```

Draws go from 0/26 to 5/12 and mean net damage falls ~10x. This is the failure mode the
project has hit before: **a safety term suppressing engagement**. Gate 4 (comp_beam at 200 s)
was not run because Gate 3 already disqualifies this checkpoint.

**S5 c50 does not replace c225.**

### But do not reject the design yet

c50 is early -- the run goes to 150 and the policy is still adapting to a penalty it has never
seen. Re-screen at c100 and c150 before deciding. If offence has not recovered by then, the
next lever is a **smaller** penalty (`dive_penalty_scale` 0.5 -> 0.2), not a different
mechanism: gate 1 proves the mechanism works.

### Screening bug found and fixed

`"altitude below min"` matches **both** `ownship altitude below min` (our loss) and
`target altitude below min` (our win). The gate-3 screen matched the bare substring and
scored an episode where the OPPONENT crashed as our loss. Corrected numbers are above.

`Release_viewertest/scratch_screen.py` now exists as the single screening entry point with
the classification written once: ownship-crash before the generic substring, target-crash and
`target destroyed` as wins, `end_condition` always parsed alongside health.

### S5 rejected at scale 0.5 — the guard was mis-AIMED, not too strong

S5 was stopped at 69/150. Disjoint blocks show BEAM damage falling monotonically as it
trained: 0.3416 -> 0.1880 -> 0.0180 -> 0.0958. Screens confirm the loss arrives almost
immediately:

| | astern_1100 crash | randomised beam mean net |
|---|---|---|
| c225 | 3/3 | **0.5814** (26/26 wins, 6 kills) |
| S5 **c25** | **0/2** | 0.0341 (7W/4D/1L) |
| S5 c50 | **0/3** | 0.0552 (7W/5D/0L) |

The crash fix works and works early. But offence is already gone by **c25** -- 25 iterations
after enabling the penalty. A scale that destroys the policy that fast is not merely "too
strong".

**The mechanism, from the reward code:**

```python
if altitude < safety_altitude:        # default 1800 m
    penalty -= low_altitude_penalty * low_fraction
    if pitch < dive_pitch and altitude_loss > 0:
        penalty -= dive_penalty_scale * (...)
```

c225 fights at **355-1617 m in half its episodes**, so it is below 1800 m for most of every
engagement. The guard therefore taxed *all low-altitude manoeuvring* -- exactly the behaviour
that produces c225's damage -- rather than the crash dive. The crash itself happens at 302 m.

**This is the key structural fact about c225: its strength and its risk share a root.** It
wins by fighting low and it comes within 50 m of the floor for the same reason. Penalising low
altitude removes both. What must be penalised is the *unrecoverable dive commitment*, not
altitude.

### S5b — retargeted, now training

`experiments/team01_s5b_dive_guard_targeted.yaml`: `safety_altitude_m 1800 -> 900`,
`dive_penalty_scale` left at 0.5, same warm start. The guard now covers the danger zone
instead of the battlefield.

Judge S5b on **offence retention first** (the crash fix is already proven by gate 1):
randomised beam mean net must come back near 0.5814, with `astern_1100` still 0 crashes.
If offence is still suppressed, the next lever is `dive_pitch_deg` (default -12 deg is shallow
-- the crash dive reached -53 deg, so -30 would isolate genuine dive commitments), not a
smaller scale.

### Correction: the dive guard failed on MAGNITUDE, not aim. S5b abandoned, S5c launched.

The "mis-aimed guard" explanation above is wrong, and the data that disproves it was already
on disk. From c225's own 200 s run (`scratch_scoring_altitude.py`):

```
altitude of SCORING frames : min 1194   median 2394   max 4729
altitude of ALL frames     : min  355   median 2146   max 5890
```

c225 is **above** 1800 m for most of every engagement -- my claim that it "fights below 1800 m
half the time" confused per-episode *minimum* altitude with where it actually flies. Only 20%
of its damage is earned below 1800 m. A 20% tax cannot cause a 94% collapse in offence.

The real cause is magnitude. Summing the guard's own formula over that same run against the
damage reward it competes with (`damage_scale` 60 x 5.0126 total damage = 300.8):

| safety_altitude_m | dive penalty total | penalty / damage reward |
|---|---|---|
| **1800 (S5)** | 2429.6 | **8.76x** |
| 900 (S5b) | 197.5 | **0.73x** |
| **600 (S5c)** | 31.5 | **0.12x** |
| 450 | 1.5 | 0.01x |

At 1800 m the guard is **8.8x larger than all damage the policy can earn**. Abandoning damage
entirely is the rational response -- which is exactly what the screens measured. At 900 m it
is still comparable to the whole objective, so S5b was heading for the same failure; its
monitor had already flagged suppressed offence at it=18. S5b was stopped at ~20.

**S5c** (`experiments/team01_s5c_dive_guard_600.yaml`, now training): `safety_altitude_m 600`,
`dive_penalty_scale` still 0.5. The guard is then 12% of the damage reward -- enough to shape
behaviour near the floor without replacing the objective. It still covers what matters:
c225's minimum altitude is 355 m and the `astern_1100` crash passes through 600 m on its way
to 302 m, while **0 of 440 scoring frames are below 600 m**, so there is no damage to lose.

**Method note:** before tuning a shaping term, sum it over a real episode set and compare it
with the reward it competes against. Both the original 1800 m default and my 900 m "fix" were
chosen by reasoning about geometry; one number from the existing logs would have rejected
both immediately. `scratch_scoring_altitude.py` now does the altitude half of this.

### S5c (guard 600 m) c25 — best of the three, not yet promotable

Full gate via `Release_viewertest/scratch_gate.py` (single classification path; validated
against c225, which reproduces its recorded astern total of 3.0497 exactly):

| gate | result | vs c225 |
|---|---|---|
| astern_1100 | **PASS** 0/2 crash, net +0.4123 | c225 crashes 3/3, scores 0.0000 |
| astern profile | **PASS** total **4.8578**, **4 kills** | c225 3.0497, 2 kills |
| randomised beam (8) | **FAIL** 6W/2D/0L, mean net 0.1186 | c225 26/26, mean 0.5814 |

The guard at 12% of the damage reward does what the 8.76x and 0.73x versions could not: it
removes the crash **and** improves offence where the fight is already won -- 200 m and 400 m
become kills, which c225 never achieved at those ranges.

The cost is the beam merge: two draws in eight, mean net down to 20% of baseline. No losses
and no crashes, so it is not dangerous -- just less decisive at the competition start.

**Not promotable at c25.** c225 remains the submission. This is iteration 25 of 150; re-gate
at c50 and c100 before judging the design.

Reading of the split: the dive guard appears to change fighting style rather than merely
suppress it -- better at converting an astern position, weaker in the head-on merge. If that
persists at c100, the natural follow-up is not a different guard value but a **spawn-weight
shift back toward the beam** (the lever that has worked every time in this project), letting
the policy re-learn the merge under the guard it now carries.

### The dive guard is abandoned: it converts one crash mode into another

S5c c50 collapsed -- **9 of 10 beam matches crashed**, plus astern 300 m and 800 m:

| gate | c25 | c50 |
|---|---|---|
| astern_1100 | 0/2 crash, +0.4123 | 0/2 crash, +0.4202 |
| astern profile | **4.8578, 4 kills** | 1.1714, 2 crashes |
| randomised beam | 6W/2D/0L, 0 crash | **1W/0D/9L, 9 crashes** |

All three guard scales (1800 / 900 / 600 m) got **worse with training**. Forensics on a c50
crash shows why, and it is not the failure the guard was built for:

```
   t     alt   pitch    roll  speed    vs   dist
112.2   1143    -2.8  -121.8    116     -8    930
115.2   1050   -31.5  -108.8    114    -62     14
117.6    853   -56.0  -120.2    142   -120    851
121.8    296   -29.3   -60.8              -> crash
```

**114 m/s, inverted, mushing.** c225's `astern_1100` crash was at 254-276 m/s -- a control
choice. This one is an energy failure. The causal chain is clear: the guard penalises
descending, so the policy holds altitude, bleeds speed, stalls and falls. More training means
better penalty-avoidance, hence worse crashes -- which matches the monotonic degradation.

**The dive guard traded a dive crash for a stall crash.** Do not resume it at any scale.

### S6 — energy floor instead (now training)

`experiments/team01_s6_energy_floor.yaml`, warm-started from
`s2b_beam_dominant/checkpoint_000200`. `dive_penalty_scale` stays unset -- the guard is
replaced, not stacked, so causality stays readable.

Calibrated the same way (sum the term over c225's own 200 s run vs the damage reward, 300.8):

| floor | scoring damage lost | penalty/reward at penalty=1.0 |
|---|---|---|
| 200 m/s | 15% | 2.66x |
| 180 m/s | 3% | 1.31x |
| **160 m/s** | **0%** | 0.55x -> **0.14x at penalty 0.25** |
| 140 m/s | 0% | 0.15x |

c225 never scores below **174 m/s** (scoring-frame minimum; median 248), so a 160 m/s floor
cannot cost damage. `low_speed_penalty 0.25` puts the term at 0.14x of the damage reward --
the same ratio at which the 600 m guard improved offence rather than replacing it.

**Judge S6 on the trend, not one checkpoint.** Every dive-guard variant looked acceptable
early (S5c c25 was the best result of the whole line) and then degraded. Gate at c25, c50 and
c100 with `scratch_gate.py`, and require the beam gate to hold at all three.

---

## VIEWER TRANSFER MEASURED (2026-08-04) — it is terminal aim, nothing else

`c225` was finally run against the live BattleViewer at the competition geometry
(2500 ft = 762 m head-on merge, red = `AIP_DCS_codex.dll` + `Rule_team01_weapon_v7.xml`,
byte-identical to the BT the local 26/26 was measured against).

### Three matches, no compensation

| match | steps | min alt | in band | best in-range ATA | our dmg | their dmg | handoff states |
|---|---|---|---|---|---|---|---|
| 1 | 2001 | 433 m | 20% | 7.93 deg | 0 | 0 | 1 |
| 2 | 1997 | 833 m | 31% | 3.24 deg | 0 | 0 | 7 |
| 3 | 1998 | 714 m | 19% | 3.09 deg | 0 | 0 | 7 |

Local reference, same model, 200 s: damage 12/12, 3 kills, best ATA <= 1.0 deg.

**What transfers:** approach (band occupancy 19-31% vs 22-45% locally), the control position
(the roadmap's own cut-in/handoff condition is met 7 times per match, held up to 6.8 s, angle
advantage up to 100 deg), and defence (opponent scored **0 in all three**).

**What does not:** the last three degrees. The plateau is stable at 3.09-3.24 deg, which is
the signature of a systematic bias, not a capability gap.

**Consequence for the roadmap: the cut-in track is dead.** c225 already produces the handoff
state unaided in the Viewer, seven times a match. Nothing is gained by training a positioner
to produce a state we already reach and cannot convert.

An earlier run crashed at 50 s, which prompted a "transfer is broken" call. It did not repeat
in three full matches (min altitude 433-833 m). **That call was made on n=1 and was wrong.**

### Rejected: lead compensation for the transport delay

Command-to-rate correlation, same policy, same log format, local vs Viewer:

```
lag (policy steps)      0        1        2        3
local                 +0.549   +0.613   +0.617   +0.566
viewer                -0.272   +0.129   +0.642   +0.578
```

Both peak at lag 2 -- that is airframe roll inertia, not a defect. The real difference is at
lag 0: locally the command explains the rate immediately, in the Viewer there is no response
for ~200 ms. The delay is real.

Compensating for it by feeding the policy a 200 ms-extrapolated state made things **much
worse**: best in-range ATA 3.09 -> **22.19 deg**, handoff states 7 -> **0**. Rolled back.

> **The lesson is more useful than the fix:** the policy is far more sensitive to observation
> FIDELITY than to the lag. Extrapolating both aircraft is an observation-distribution shift
> it never trained on, and that costs more than the lag it removes.

Implementation and full numbers kept, disabled, in `student/my_lead_provider.py`.

### Fixed: the Viewer's KCAS was 2% low, and the error varied

That lesson sent me back to the 1-2% speed mismatch the parity check had found and I had
dismissed. `plane_info_to_state` computed

```python
KCAS = TAS * sqrt(sigma)      # this is EQUIVALENT airspeed, not calibrated
```

Observation channels 10 (`own_speed`) and 26 (`target_speed`) feed `normalize(KCAS, 0, 450)`
straight to the actor. Against the FighterSim probe point recorded in the client's own
comment (202 m/s TAS at 4572 m -> 163.5 m/s CAS):

| formula | value | error |
|---|---|---|
| `TAS * sqrt(sigma)` (was) | 160.24 | **-2.00%** |
| compressible CAS (now) | 163.46 | **-0.02%** |

The bias is not constant -- it grows with Mach and altitude: 0.5% at 2000 m / 150 m/s, 5.1%
at 6000 m / 300 m/s. A dogfight crosses that whole range, so the actor saw a speed channel
drifting low by a varying amount it never met in training.

Fixed in `plane_info_to_state` via `_calibrated_airspeed()`. Observation parity afterwards:

```
                    before                       after
own_speed     -0.1760 vs -0.1961     -0.1760 vs -0.1759
target_speed  -0.2167 vs -0.2342     -0.2167 vs -0.2167
mismatched channels: 3 of 35     ->   1 of 35   (roll_rate, 0.0011)
```

**Verification pending**: three Viewer matches against the 7.93 / 3.24 / 3.09 deg baseline.
Judge on best in-range ATA, not damage -- damage is downstream of aim and noisier.
`roll_rate` is two orders of magnitude smaller than the speed error was; leave it until this
one change has been judged.

### The KCAS fix was correct and made things worse — rolled back, unexplained

Two Viewer matches with the compressible-CAS form, the second run under conditions identical
to the baseline (same bundle, same red BT, same geometry, `TEAM01_OBS_LOG` on in both):

| run | steps | min alt | best in-range ATA | handoff states |
|---|---|---|---|---|
| baseline 1 | 2001 | 433 m | 7.93 deg | 1 |
| baseline 2 | 1997 | 833 m | 3.24 deg | 7 |
| baseline 3 | 1998 | 714 m | 3.09 deg | 7 |
| CAS 1 | 1601 | **321 m** | 23.28 deg | 0 |
| CAS 2 | 1285 | **314 m** | 51.03 deg | 0 |

Both CAS runs ended early just above the 304.8 m floor. Meanwhile the change demonstrably
improved observation fidelity: `own_speed` went from -0.0201 off the training path to
-0.0001, `target_speed` to 0.0000, mismatched channels 3 -> 1.

**More faithful observation, worse behaviour.** That is the direct opposite of the lesson
drawn from the failed lead compensation ("the policy is more sensitive to observation
fidelity than to lag"). One of those two results is telling us something we have not
understood, so **that lesson should not be used to justify the next change.**

Rolled back in `plane_info_to_state`, with the measurement in a comment there and
`_calibrated_airspeed()` kept unused beside it. Do not re-enable without first explaining the
contradiction.

Attempted diagnosis of "did the code take effect, or was the hypothesis wrong": inconclusive.
`scratch_obs_channel_shift.py` compares live observation logs, but the two matches diverged so
completely (`cos_ata` moved -0.73, `cos_aspect` -0.47) that a distribution comparison carries
no signal. **That tool only works when the two runs fly similar fights.**

### Where the Viewer work leaves us

| | status |
|---|---|
| approach, control position, defence | transfer intact (handoff state 7x/match, opponent scored 0 in all 3) |
| terminal aim | stuck at 3.09-3.24 deg against a 1.0 deg requirement |
| crash | 1 of 4 Viewer runs, not reproduced in the 3 clean matches |
| client-side observation fixes | 2 tried, both rejected, one of them actively harmful |

**The cut-in track is dead** and should be closed in the roadmap: c225 reaches the roadmap's
own handoff condition seven times per match unaided, so building a positioner to produce that
state adds nothing. The entire remaining gap is the last three degrees.

---

## THE ACTUAL CAUSE (2026-08-04): the opponent flies differently in the Viewer

Everything above treats "terminal aim stuck at 3 deg" as the defect. It is a **symptom**.
The causal chain, measured:

```
red goes low in the Viewer (36% of steps below 1850 m; locally 4%)
  -> its rule's CombatClimb gate (Altitude="1850") keeps firing
  -> red commands full throttle          mean 0.948 vs 0.754 locally
  -> red flies 288-299 m/s               vs 198 locally; we fly 222-244 in both
  -> we cannot close                     median range 1590-2046 m vs 831 m locally
  -> best aim is 3 deg                   the 1 deg shots locally happen inside 930 m
  -> damage 0
```

**Our policy is not the problem.** Our throttle command (0.87 viewer / 0.83 local) and our
speed (222-244 / 225) are the same in both environments. The asymmetry is entirely red's:
locally we out-energise it (225 vs 198), in the Viewer it out-energises us (222 vs 288).

Range distribution, which is what selects red's BT task:

| | <430 m | 430-930 | 930-2500 | >2500 | median |
|---|---|---|---|---|---|
| local | 20% | 39% | 36% | 5% | **831 m** |
| viewer 1/2/3 | ~1% | 8-19% | 62-79% | 8-29% | **1590-2046 m** |

Locally the fight collapses to gun range and red is in `GunTrack` / `LagOutOfOvershoot`.
In the Viewer it stays in `LagPursuit` / `LeadPursuit`, which run the throttle up.

The first 50 s are nearly identical in both (both open to ~3 km, both close to ~800-1000 m at
t=25 s, our speed 254 vs 261). The divergence starts around t=30 s, when Viewer red settles
into a low, fast regime and never leaves it.

### Why this matters more than anything else found today

**Every local result was measured against a different adversary than the one in the Viewer.**
The 26/26, the 12/12 at 200 s, every gate in every experiment -- all against a red that flies
100 m/s slower and 1500 m higher than the red the submission actually meets. Local scores are
not evidence about competition performance until that gap is closed or explained.

### Two causality reversals I made along the way

Recording these because the same shape of error happened twice in one diagnosis:

1. **"red is faster, therefore red's throttle setting is the bug."** Wrong: red's throttle is
   selected by its rule from RANGE, so the throttle difference is downstream of the fight
   being at longer range.
2. **"we are slower in the Viewer."** Wrong: our speed and throttle are the same in both. Red
   is faster. I had compared our Viewer speed against red's Viewer speed instead of against
   our own local speed.

Both were caught by measuring the other side of the comparison. **When one side of a
difference looks anomalous, measure the same quantity for the other side before assigning
cause.**

### Still open: why does red go low in the Viewer

Unverified lead: `make_navigation_data` builds the BT's altitude as
`nav.Alt = alt_m * M_TO_FT * 1000`. If that conversion or its units disagree with what
FighterSim feeds the local BT, red misjudges its own altitude and the 1850 m CombatClimb gate
fires at the wrong times. **Not yet checked.** Check it before any further training.

### What this does to the plan

- Plan B (train against a 200 ms action delay) is **cancelled** -- the premise test showed the
  delay costs 0.23 deg locally (0.67 -> 0.90), nowhere near the 2.5 deg needed, and the delay
  is not the cause anyway.
- The aim work, the dive guards, the lead compensation and the KCAS change were all aimed at
  symptoms of this.
- **The next lever is the opponent**, not the policy: either fix red's Viewer-side navigation
  input so it matches the local BT, or make local training face the red that the Viewer
  actually produces.

---

## COMPETITION-CRITICAL (2026-08-04): we were overrunning the per-frame budget

The user observed the Viewer applying disconnection damage while the connection was healthy.
The server runs at 60 Hz, so an AI step has **one 16.67 ms frame** (not the 0.1667 s figure
recorded in `CLAUDE.md`, which is unsourced -- the user confirmed it is per frame).

Measured on the real deployed path, 800 `compute_action` calls with `bundle_000225`:

| | median | p95 | p99 | max |
|---|---|---|---|---|
| **as deployed** | 0.81 ms | 8.77 ms | **22.18 ms** | **70.68 ms** |
| with the guard | 0.25 ms | 1.55 ms | 3.00 ms | 6.79 ms |

Inference is not the problem -- the median is a quarter of a millisecond. The spikes are a
generational GC pass landing inside a frame plus torch contending for cores. An earlier run of
the same probe peaked at 19.71 ms, this one at 70.68 ms: **the spikes are variable and can be
far worse than a single sample suggests.**

At ~1% of steps over budget, a 200 s match (2000 policy steps) overruns roughly 20 times.
The rules record disqualification after repeated network faults, so this alone could have
lost matches regardless of how well the policy flew.

### Fix

`student/my_submission.py`, at import time, behind `TEAM01_LOW_LATENCY` (default on):

```python
torch.set_num_threads(1)
gc.collect(); gc.freeze(); gc.set_threshold(200000, 1000, 1000)
```

Thresholds rather than `gc.disable()`: cycles are still collected, just far too rarely to land
inside a frame, so a long session cannot leak unboundedly. Full disable measures slightly
better (max 3.61 ms) but removes the safety valve for no needed margin.

It lives in `student/` deliberately -- it survives a platform release update and is
unambiguously ours.

### Also worth knowing

**Bundle load takes 26.8 s.** That is a one-off before the client connects, so it does not eat
match time as things are sequenced now, but if a match ever starts while a client is still
loading, that client is silent for 27 s and looks exactly like a dead connection. Connect
first, confirm, then start the match.

### Status of the red-opponent fix, same session

Red was rewired to feed its BT full navigation data (`make_navigation_data` + the newer
`bt_action_provider` + the missing `StateIndex` import, run with `TEAM01_BT_NAVI=1`). It ran a
full match with **no divergence** -- the failure mode the code comments warned about did not
occur -- and moved toward the local BT's behaviour:

| | local BT | broken viewer red | rewired red |
|---|---|---|---|
| throttle | 0.754 | 0.948 | 0.921 |
| altitude median | 3924 m | 2360 m | **3581 m** |

Altitude has largely closed; throttle has not. **That match is not scored** -- it ran with the
frame overruns still present, so it is a contaminated measurement. Re-run the three-match
comparison with the latency guard in place before drawing any conclusion about either the
opponent fix or our own aim.

---

## SOLVED (2026-08-04): the Viewer transfer failure was ONE channel — body rates

**First damage ever scored in the Viewer.** Same bundle `c225`, same geometry, same opponent:

| | median range | in band | best ATA | our damage | damage taken |
|---|---|---|---|---|---|
| before | 1590 m | 31% | 3.24 deg | **0.0000** | 0 |
| before | 1787 m | 19% | 3.09 deg | **0.0000** | 0 |
| **after the fix** | **884 m** | **76%** | **0.08 deg** | **0.4027** | 0 |
| local reference | 831 m | — | <=1.0 | damage in 12/12 | 0 |

Every metric snapped to the local values.

### Root cause

`PlaneInfo` carries position, attitude and velocity but **no angular rates**, so the client
differences attitude to reconstruct P/Q/R. That reconstruction ran inside
`_compute_provider_action`, i.e. **once per policy step = every 6 frames = a 0.1 s interval**.

A 0.1 s attitude difference is the *average* rate over the last six frames. Measured response
to a step change from 0 to 120 deg/s:

```
   t (s)     every frame (fixed)     every 6th frame (was)
   0.55            98.6                    7.0
   0.70           119.6                   72.3
   1.00           120.0                  106.9
```

**The old estimate lagged roughly half a second.** Feeding a control policy half-second-stale
rate feedback makes it oscillate — which is exactly what the flight logs showed:

```
                 roll cmd saturated   roll flips/s   median ATA
local                   5%                1.1           20.1
viewer (before)      46-56%            2.5-3.6        56-82
viewer (after)          44%                4.0           19.5
```

### Fix

`ProviderCommandPolicy._track_body_rates()` in `src/dogfight/unreal/policies.py`: update the
body-rate estimate **every frame** (dt = 1/60 s) with a light EMA (alpha 0.35, ~45 ms), and
have `_inject_own_rates_throttle` consume it. Reset clears it so attitude cannot leak across
an episode boundary. Verified offline: recovers a constant 60 deg/s exactly, and reaches 95%
of a step change within 0.1 s instead of 0.7 s.

### This was already written down and missed

`scratch_obs_parity.py`'s own docstring says: *"The same bundle that keeps its actions smooth
in training (saturation 0.058) slams the controls in the Viewer (0.96) and never gets its nose
inside 22 deg."* The problem was documented and left open. I read that comment, saw the
observation parity come back 32/35 identical, and dismissed the one remaining mismatched
channel -- `roll_rate`, 0.0011 -- as two orders of magnitude smaller than the speed error.

**A small mean difference does not mean a usable signal.** `roll_rate` was right on average and
half a second late, and rate feedback is what damps a control loop. Compare distributions and
timing, not just means, on any channel a controller differentiates or damps on.

### What this retires

Everything chased before this was a symptom of it: the three dive guards, the lead
compensation, the KCAS correction, and the entire red-opponent investigation. The opponent
*is* different in the Viewer (that finding stands) but it was never why we scored nothing.

### Still open

- **Roll command still saturates 44% of the time** (local 5%) while the nose now tracks
  correctly. High-frequency chatter remains; the natural next try is a stronger EMA
  (alpha 0.35 -> 0.2). Not yet shown to cost anything.
- **n = 1.** One match. Repeat before treating it as settled.
- Red took the full navigation `Step()` path for the first time in this same match, so the
  opponent condition changed too. Re-run with both held fixed.

### Reproduced, and the second run was a kill

Same code, same settings, nothing changed between runs:

| | steps | median range | in band | ATA med | best ATA | roll sat | our damage | taken |
|---|---|---|---|---|---|---|---|---|
| before | 1997 | 1590 m | 31% | 56.3 | 3.24 | 54% | **0.0000** | 0 |
| before | 1998 | 1787 m | 19% | 62.4 | 3.09 | 56% | **0.0000** | 0 |
| fix run 1 | 2000 | 884 m | 76% | 19.5 | 0.08 | 44% | **0.4027** | 0 |
| fix run 2 | **1489** | 689 m | 81% | 10.3 | 0.47 | 48% | **1.0402** | 0 |

Run 2 ended at 149 s with damage 1.0402 against a starting health of 1.0 — **a kill**, which
is why it terminated early. Median ATA 10.3 deg is better than the local reference (20.1).
Damage taken is still zero in every Viewer match ever run.

n=2, consistent direction, large effect. The transfer failure is fixed.

### The remaining chatter: raising the EMA is a trade, not a free win

Roll command still saturates 44-48% against 5% locally, and the user reports watching it
nearly lose a tracking solution a few times — so the chatter is real, not just cosmetic.

**But more smoothing costs lag, and lag is precisely what was just fixed.** alpha 0.35 is
~45 ms; the broken version was effectively ~500 ms. Anything that moves back toward the latter
risks re-creating the oscillation it cured. There is no a-priori answer.

If it is tried (alpha 0.35 -> 0.2, ~90 ms):

- judge on **damage and best ATA**, not on the saturation percentage — a smoother stick that
  scores less is a loss;
- keep the same gate as this run: median range < 900 m, in band > 60%, best ATA < 1.0 deg,
  damage > 0, zero taken;
- and hold everything else fixed, including the red client's navigation path.

Note the policy is intrinsically bang-bang: on random observations it saturates 6/6. Local
saturation is only 5% because the observations there are smooth and in-distribution, so some
residual chatter in the Viewer may be irreducible without retraining.

**Default position: do not change it.** The current configuration wins and takes no damage.
Replacing a winning configuration on an unmeasured hunch is the specific mistake this session
made five times.
