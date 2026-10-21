# Implementation mapping

The retained method equations are grouped by responsibility. Explicit ancestor dispatch
preserves source `super()` and direct-ancestor routing without shipping historical classes.
The manifest gives source line references, hashes and each destination function.

| Source behavior | Release location |
|---|---|
| Active V24 V2 control/reference override | `control/source_kernels.py` |
| V18 event synchronization | `gait/phase_kernels.py` |
| Contact/ground reaction and foot state | `gait/contact_kernels.py` |
| V6–V9 qualified acquisition and expected-side FSM | `rewards/acquisition_kernels.py` |
| V13/V16–V24 forward/placement terms | `rewards/forward_kernels.py` |
| Shared shaping | `rewards/common_kernels.py` |
| V24/U13 actor/critic construction | `observations/*_kernels.py` |
| State/writeback order and resets | `environments/*_lifecycle.py` |
| U13 command scheduler | `commands/process.py` |
| U13 reward gates/accounting | `rewards/unified_kernels.py` |
| Fresh U13 heads and pinned preprocessing | `training/source_initialization.py` |

Runtime adaptation: external model bindings replace all source joint/body names, geometry,
joint limits, initial qpos, gains and residual scales. Side masks use semantic runtime roles.
A disabled U13 teacher stub replaces historical assistance; teacher-only telemetry differs.
The active V24 law is fixed-scale settled-reference residual control, as explained in
`control_architecture.md`. Generic public phase-authority helpers are not injected into it.

Extracted method entries: 107. Source files are not imported at runtime.
