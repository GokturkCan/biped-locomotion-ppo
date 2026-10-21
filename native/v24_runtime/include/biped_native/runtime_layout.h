// Public observation/action contract of the V24 policy runtime.
//
// Only model-independent facts live here: tensor sizes, the 57-value actor
// observation index layout (documented in docs/observation_space.md), control
// timing and two observation scalings. Everything that identifies or
// parameterises a particular robot -- joint/sensor/geom names, residual action
// scales, reference/initial pose, root-height settings -- is NOT compiled in;
// it is supplied at runtime through RuntimeBindings (runtime_bindings.h).
#pragma once

#include <cstddef>

namespace biped {

constexpr int kNumJoints = 12;
constexpr int kObsSize = 57;
constexpr int kActionSize = 12;

// Timing (configs/v24_environment.json: sim_dt, ctrl_dt, settle_steps).
inline constexpr double kSimDt = 0.002;
inline constexpr double kCtrlDt = 0.02;
inline constexpr int kSubstepsPerTick = 10;  // kCtrlDt / kSimDt
inline constexpr int kSettleSteps = 1500;

// Actor observation index layout (57 values):
//   0-11  joint position minus settled reference
//   12-23 joint velocity
//   24-26 up vector (gravity direction in body frame)
//   27-29 body-frame linear velocity
//   30-32 angular velocity (gyro)
//   33-34 sin/cos(2*pi*gait_phase)
//   35-46 previous action
//   47-48 left/right foot contact (0/1)
//   49-50 command-frame velocity: vx*cos(heading_error), vx*sin(heading_error)
//   51    yaw-rate command (raw)
//   52    lateral error (root_y - baseline) / kLateralScale
//   53-54 sin/cos(heading_error)
//   55    signed bilateral placement imbalance
//   56    bilateral pair valid
inline constexpr int kIdxQposStart = 0;
inline constexpr int kIdxQvelStart = 12;
inline constexpr int kIdxGravity = 24;
inline constexpr int kIdxLocalLinvel = 27;
inline constexpr int kIdxGyro = 30;
inline constexpr int kIdxSinPhase = 33;
inline constexpr int kIdxCosPhase = 34;
inline constexpr int kIdxLastActStart = 35;
inline constexpr int kIdxLeftContact = 47;
inline constexpr int kIdxRightContact = 48;
inline constexpr int kIdxCmdX = 49;
inline constexpr int kIdxCmdY = 50;
inline constexpr int kIdxYawRate = 51;
inline constexpr int kIdxEyObs = 52;
inline constexpr int kIdxSinHeading = 53;
inline constexpr int kIdxCosHeading = 54;
inline constexpr int kIdxBilateralImbalance = 55;
inline constexpr int kIdxPairValid = 56;

inline constexpr float kLateralScale = 0.10f;    // e_y_obs = e_y / 0.10
inline constexpr float kBilateralEps = 1.0e-6f;  // eps in (rr-ll)/(rr+ll+eps)

}  // namespace biped
