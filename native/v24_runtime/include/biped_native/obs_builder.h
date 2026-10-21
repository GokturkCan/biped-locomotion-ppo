// 57-value actor-observation construction for the V24 policy.
//
// The physics-derived part is read from mjModel/mjData through names supplied
// by RuntimeBindings; the episodic part (gait_phase, last_act, command,
// root_y_baseline, bilateral placement memory) is passed in explicitly as
// ObsInfo and carried by the caller (the event machine, see event_machine.h).
// BuildObservation() extracts raw numbers from mjData and delegates to
// BuildObservationFromRaw(), the pure-arithmetic core that can be tested with
// plain numbers and no physics engine.
#pragma once

#include <mujoco/mujoco.h>

#include <array>

#include "biped_native/runtime_bindings.h"
#include "biped_native/runtime_layout.h"

namespace biped {

// Non-physical per-step info the V24 observation depends on (see
// REQUIRED_OBS_INFO_FIELDS in verify_v24_parity.py -- this struct is the C++
// mirror; every field is mandatory, there is no default/fallback).
struct ObsInfo {
  std::array<float, kNumJoints> bridge_q_ref_base;
  float gait_phase;
  std::array<float, kActionSize> last_act;
  std::array<float, 3> command;  // [vx_cmd, vy_cmd(unused), yaw_rate_cmd]
  float root_y_baseline;
  float v24_last_left_placement;
  float v24_last_right_placement;
  float v24_pair_valid;
};

struct SensorAddrs {
  int orientation_adr = -1;
  int gravity_adr = -1;
  int local_linvel_adr = -1;
  int gyro_adr = -1;
  int root_pos_adr = -1;
};

struct ContactGeoms {
  int floor_id = -1;
  int left_foot_id = -1;
  int right_foot_id = -1;
};

struct JointAddrs {
  std::array<int, kNumJoints> qposadr;
  std::array<int, kNumJoints> dofadr;
};

// Resolves sensor addresses / geom ids / joint addresses once from the model.
// Aborts (throws) if any required name is missing.
SensorAddrs ResolveSensorAddrs(const mjModel* m, const RuntimeBindings& b);
ContactGeoms ResolveContactGeoms(const mjModel* m, const RuntimeBindings& b);
JointAddrs ResolveJointAddrs(const mjModel* m, const RuntimeBindings& b);

// Checks the model against the bindings (qpos size, actuator count, sensor widths,
// scalar joints, named geoms/body). Throws std::runtime_error on any mismatch.
void ValidateModelAgainstBindings(const mjModel* m, const RuntimeBindings& b);

// mjData must already reflect the physics state to observe (i.e. mj_forward
// or mj_step must have been called on it beforehand -- this function only
// reads data->qpos/qvel/sensordata/contact, it does not step or forward).
std::array<float, kObsSize> BuildObservation(const mjModel* m, const mjData* d,
                                              const SensorAddrs& sensors,
                                              const ContactGeoms& geoms,
                                              const JointAddrs& joints,
                                              const ObsInfo& info);

// Pure-arithmetic core: every quantity is passed in as plain numbers, no
// mjModel/mjData involved. BuildObservation() delegates here so both paths share
// one arithmetic implementation.
struct RawObsInputs {
  std::array<float, kNumJoints> joint_qpos;   // qpos[ACTIVE_JOINTS], raw (not yet - bridge)
  std::array<float, kNumJoints> joint_qvel;   // qvel[ACTIVE_JOINTS]
  std::array<float, 3> gravity;
  std::array<float, 3> local_linvel;
  std::array<float, 3> gyro;
  std::array<float, 4> orientation_quat;      // wxyz
  float left_contact;
  float right_contact;
  float root_y;
};

std::array<float, kObsSize> BuildObservationFromRaw(const RawObsInputs& raw, const ObsInfo& info);

}  // namespace biped
