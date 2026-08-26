# C-series crossing-threat defense curriculum

## Role

C is the missing defense role between D4's full-six escape and the neutral BT/A/E handoff.
It is active conceptually when the opponent is accurately nose-on while own ATA is roughly
30-120 degrees. C1 deliberately does not train the measured Phase-3 event directly.

## Ladder

| Stage | Own ATA | Target ATA | Range | Purpose |
|---|---:|---:|---:|---|
| C1 | 145-165 deg | 8-12 deg | 700-1000 m | D4-compatible bridge |
| C2 | 120-140 deg | 6-10 deg | 700-1000 m | aft-quarter crossing |
| C3 | 90-115 deg | 4-7 deg | 750-1050 m | beam/crossing bridge |
| C4 | 60-85 deg | 2-5 deg | 800-1100 m | forward-quarter threat |
| C5 | 30-55 deg | 1-4 deg | 850-1200 m | measured Viewer threat |

All stages are left/right mirrored and jittered. The only reward-contract change is moving
the existing danger-exposure gate down with the curriculum. The penalty magnitude and all
D4 safety/angle terms stay unchanged.

## Automatic promotion

- Train at least 100 iterations.
- Compute a 10-completed-episode C defense score from threat-angle clearance, selector
  handoff geometry, danger-exposure fraction, survival, and net damage.
- Require score >= 0.72, survival >= 0.95, threat clearance >= 0.70, and exposure clearance
  >= 0.75.
- Promote only after the qualified score has failed to improve for 25 iterations.
- Hand the best 10-iteration bundle to the next stage. If a stage reaches its cap without
  meeting the gate, run exactly one Cxb continuation from its latest native checkpoint,
  preserving optimizer and replay state. If that continuation also fails, stop the chain;
  never manufacture an unbounded Cxc/Cxd series.

## Final acceptance target

The C5 candidate is not selector-ready until a deterministic 36-episode grid passes:

- at least 80% of threats move from target ATA <= 5 deg to >= 20 deg within 3 seconds;
- zero ownship kills and zero hard-deck crashes;
- average strict opponent 1-degree dwell <= 0.10 seconds per episode;
- at least 70% finish in a usable handoff (own ATA <= 90 deg, target ATA >= 20 deg);
- no regression in the existing A5, E1/E7, and D4 specialist grids.

Only after this gate should C be added to the selector and tested in the full Phase-3 match.
