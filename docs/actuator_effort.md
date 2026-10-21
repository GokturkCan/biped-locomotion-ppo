# Actuator-effort methodology

Read generalized actuator effort from **`data.qfrc_actuator[dof_address]` only**.
Do not substitute `actuator_force`, inverse dynamics, motor current or PD target error.
For every mapped scalar joint, report peak absolute, p99 absolute and RMS from the sample
stream. Revolute joints use N*m; prismatic joints use N. The historical collection rate
was 500 Hz. Joint identifiers/DOF indices come from private runtime bindings.

`results/v24_effort_summary.csv` contains only anonymized aggregate channels, type, units,
sample counts and effort statistics. It excludes raw trajectories, joint names, ranges,
force limits, positions, timing extrema and model mappings. No raw 500-Hz dataset is included.

These are simulated generalized actuator efforts, not hardware motor requirements.
Transmission, efficiencies, thermal/current limits, final physical parameters and hardware
measurements are needed for motor sizing. Native CPU-versus-MJX physical effort parity
has not been established during this migration.
