# Actuator-effort methodology

Read generalized actuator effort from **`data.qfrc_actuator[dof_address]` only**.
Do not substitute `actuator_force`, inverse dynamics, motor current or PD target error.
For every mapped scalar joint, report peak absolute, p99 absolute and RMS from the sample
stream. Revolute joints use N*m; prismatic joints use N. The historical collection rate
was 500 Hz. Joint identifiers/DOF indices come from private runtime bindings.

Public documentation retains only the methodology and high-level interpretation.
Per-channel effort summaries are withheld: anonymized labels can still expose actuation
structure through channel count, ordering and joint types. No replacement per-channel
table, raw 500-Hz stream, model mapping, limit or reference-pose data is distributed.
This more conservative data boundary does not change the historical research claims.

These are simulated generalized actuator efforts, not hardware motor requirements.
Transmission, efficiencies, thermal/current limits, final physical parameters and hardware
measurements are needed for motor sizing. Native CPU-versus-MJX physical effort parity
has not been established during this migration.
