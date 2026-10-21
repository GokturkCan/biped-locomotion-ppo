# Experiment lineage

Main progression: **V13 → V16 → V17 → V18 → V19 → V20 → V21 → V22 → V23 → V24**.
V10–V12 and V14–V15 are side branches. Older source ancestry is internal implementation
history, not a set of advertised public experiment modules.

Each row distinguishes the intervention and physical evidence; outcomes are single-run.

| Version | Problem | Hypothesis | Modification | Physical metric | Result | Next limitation |
|---|---|---|---|---|---|---|
| V13 | Gait reward did not establish task progress | Velocity and slip feedback may improve forward behavior | Global command-velocity tracking and stance-slip cost | Forward velocity, stance slip | Selected forward branch; intermediate numeric results are not republished here | Completed-step quality |
| V16 | Contact events did not guarantee a useful step | Credit should follow expected completed outcomes | Completed-step outcome bonus | Qualified placement and support transfers | Outcome-aware branch retained | Accessibility of the quality target |
| V17 | Completion quality was difficult to reach | A one-step single-support target may be more accessible | Quality target adjustment | Single support and completed outcomes | Retained as the V18 parent; unsupported historical ratios omitted | Timing alignment |
| V18 | Timer phase could disagree with physical progress | Event-aligned timing may support the acquisition chain | Event-synchronized soft clock and concurrent timing changes | Bilateral gait and transfer sequencing | Large observed improvement, consistent with the hypothesis | Drag and long-horizon path quality |
| V19 | Loaded swing-foot motion | Penalizing loaded contact motion may reduce dragging | Expected-swing contact-motion drag | Swing-foot contact motion | Drag-aware branch retained; no isolated causal claim | Placement acceptance |
| V20 | Poor completed placement | A placement band may improve accepted steps | Completed-placement acceptance-band penalty | Placement distribution | Selected continuation; isolated effect not established | Heading-penalty saturation |
| V21 | Heading error could escape the penalty gradient | A nonsaturating tail may retain corrective pressure | Remove heading-penalty upper saturation | Heading error and path drift | Correction retained; no separate robust-performance claim | Actor observability |
| V22 | Actor lacked desired direction after heading drift | Body-frame projection may expose signed heading error | Command-frame observation | Directional response and heading | Observation made direction available | Reward/frame inconsistency |
| V23 | Observation and reward could request incompatible velocities | Matching command frames may remove that conflict | Vector command-frame tracking/progress | Forward progress and long-horizon drift | Forward branch retained; bilateral/path drift remained | Bilateral long-horizon stability |
| V24 | Bilateral imbalance and persistent lateral drift | Explicit bilateral/path memory may improve long-horizon behavior | Symmetry/growth penalties and observation extensions | Displacement, velocity, gait validity, falls, saturation | 6.9976 m/60 s, zero falls, valid alternation | Single-run evidence; robustness/generalization unproven |

V18 improvement is consistent with timing synchronization, not strict causal proof:
timing settings also changed and the evidence is single-seed.

U0–U5 examined migration, command transitions, stopping, yaw assistance and lateral/routed
experts. U6 variants examined scratch acquisition, event chains, phase-conditioned heads,
recurrence and reward changes. These are negative/comparative history, not evidence of
omnidirectional capability. U13 reuses software while initializing training from scratch.
