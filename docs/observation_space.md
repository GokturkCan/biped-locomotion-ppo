# Observation space

V24 actor: 57 values. Critic: 82. No public model pose is required to explain the layout.

| Actor indices | Group | Purpose |
|---|---|---|
| 0–11 | Joint position minus settled reference | Relative posture |
| 12–23 | Joint velocity | Movement feedback |
| 24–26 | Up vector | Balance/orientation |
| 27–29 | Body-frame linear velocity | Task response |
| 30–32 | Angular velocity | Rotational stability |
| 33–34 | Sine/cosine gait phase | Temporal context |
| 35–46 | Previous action | Control continuity |
| 47–48 | Foot contacts | Support context |
| 49–51 | Command-frame velocity and yaw command | Task direction |
| 52–56 | Lateral error, sine/cosine heading error, signed bilateral imbalance, pair validity | Long-horizon path and placement context |

The critic's 74-value prefix is the 52-value actor prefix plus contacts (2), global
linear/angular velocity (6), normalized actuator force (12), and knee positions (2).
V24 appends the same five task fields and three private-state placement memories.
Critic-only channels never enter the actor.


## U13: rewritten task context and reserved channels

U13 constructs its own 61-value actor prefix, then appends three zeros to produce 64.
It reuses the proprioceptive groups at 0–48, with phase sine/cosine multiplied by the
locomotion-request gate. Command channels 49–51 are fixed-scaled body-frame commands;
52–56 use the unified path/heading reference and bilateral placement state.
The four added actor context channels are:

| Actor index | Meaning |
|---|---|
| 57 | Locomotion request |
| 58 | Command planar norm divided by its fixed observation scale |
| 59 | Start swing side |
| 60 | Command transition progress |
| 61–63 | Reserved block, exactly zero in U13 |

The reserved block is not the older U6/V8R expected-side/phase/load-ratio observation.
Those intermediate values may be computed by shared code, but U13 neutralizes the block
before emitting observations. The inherited sine/cosine clock at 33–34 remains active.

U13 critic layout differs from appending seven channels to an intact V24 critic:

| Critic indices | Meaning |
|---|---|
| 0–60 | U13 actor prefix before the reserved block |
| 61–62 | Additional foot contacts |
| 63–65 | Global linear velocity |
| 66–68 | Global angular velocity |
| 69–80 | Normalized actuator force |
| 81–82 | Two knee positions |
| 83–85 | Last left/right placement and previous absolute imbalance |
| 86–88 | Reserved block, exactly zero in U13 |

The result is 89 critic values. Privileged channels are not fed into the actor.
Fixed-scaled command/gate indices 49,50,51,57,58,59,60 bypass U13 running normalization.
Public helpers validate grouping/order; extracted kernels remain the authoritative runtime
construction. Physical source-versus-release parity still requires authorized private assets.
