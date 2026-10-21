// Tests for obs_builder.cpp: index layout and formulas of the 57-value
// observation (pure arithmetic, no physics), plus consistency of the
// mjData-based path with the arithmetic core on the synthetic model.
//
// Usage: test_obs_builder
#include <cmath>
#include <stdexcept>

#include <mujoco/mujoco.h>

#include "biped_native/obs_builder.h"
#include "biped_native/runtime_layout.h"
#include "check.h"
#include "synthetic_model.h"

namespace {
using testing::Check;
using testing::CheckNear;
}  // namespace

int main() {
  using namespace biped;

  // ---- arithmetic core ----------------------------------------------------
  RawObsInputs raw{};
  for (int i = 0; i < kNumJoints; ++i) {
    raw.joint_qpos[i] = 0.5f + 0.01f * i;
    raw.joint_qvel[i] = -0.1f * i;
  }
  raw.gravity = {0.0f, 0.0f, 1.0f};
  raw.local_linvel = {0.1f, 0.2f, 0.3f};
  raw.gyro = {0.4f, 0.5f, 0.6f};
  raw.orientation_quat = {1.0f, 0.0f, 0.0f, 0.0f};  // zero yaw
  raw.left_contact = 1.0f;
  raw.right_contact = 0.0f;
  raw.root_y = 0.05f;

  ObsInfo info{};
  for (int i = 0; i < kNumJoints; ++i) info.bridge_q_ref_base[i] = 0.5f;
  info.gait_phase = 0.25f;
  for (int i = 0; i < kActionSize; ++i) info.last_act[i] = 0.01f * i;
  info.command = {0.1f, 0.0f, 0.0f};
  info.root_y_baseline = 0.0f;
  info.v24_last_left_placement = 0.06f;
  info.v24_last_right_placement = 0.10f;
  info.v24_pair_valid = 1.0f;

  const auto obs = BuildObservationFromRaw(raw, info);
  CheckNear(obs[kIdxQposStart + 3], 0.03, 1e-6, "qpos minus reference");
  CheckNear(obs[kIdxQvelStart + 2], -0.2, 1e-6, "qvel passthrough");
  CheckNear(obs[kIdxGravity + 2], 1.0, 1e-6, "gravity");
  CheckNear(obs[kIdxLocalLinvel + 1], 0.2, 1e-6, "local linvel");
  CheckNear(obs[kIdxGyro + 0], 0.4, 1e-6, "gyro");
  CheckNear(obs[kIdxSinPhase], 1.0, 1e-6, "sin(2*pi*0.25)");
  CheckNear(obs[kIdxCosPhase], 0.0, 1e-6, "cos(2*pi*0.25)");
  CheckNear(obs[kIdxLastActStart + 5], 0.05, 1e-6, "last action");
  CheckNear(obs[kIdxLeftContact], 1.0, 0.0, "left contact");
  CheckNear(obs[kIdxRightContact], 0.0, 0.0, "right contact");
  CheckNear(obs[kIdxCmdX], 0.1, 1e-6, "command-frame x at zero heading error");
  CheckNear(obs[kIdxCmdY], 0.0, 1e-6, "command-frame y at zero heading error");
  CheckNear(obs[kIdxYawRate], 0.0, 0.0, "yaw-rate command passthrough");
  CheckNear(obs[kIdxEyObs], 0.5, 1e-6, "lateral error scaled by 1/0.10");
  CheckNear(obs[kIdxSinHeading], 0.0, 1e-6, "sin heading");
  CheckNear(obs[kIdxCosHeading], 1.0, 1e-6, "cos heading");
  CheckNear(obs[kIdxBilateralImbalance], (0.10 - 0.06) / (0.16 + 1e-6), 1e-6, "bilateral imbalance");
  CheckNear(obs[kIdxPairValid], 1.0, 0.0, "pair valid");

  // pair_valid = 0 must zero the imbalance channel
  info.v24_pair_valid = 0.0f;
  CheckNear(BuildObservationFromRaw(raw, info)[kIdxBilateralImbalance], 0.0, 0.0,
            "imbalance gated by pair_valid");
  info.v24_pair_valid = 1.0f;

  // yaw of +90 deg -> heading error -90 deg -> command rotates into the y channel
  raw.orientation_quat = {static_cast<float>(std::cos(M_PI / 4)), 0.0f, 0.0f,
                          static_cast<float>(std::sin(M_PI / 4))};
  const auto yawed = BuildObservationFromRaw(raw, info);
  CheckNear(yawed[kIdxSinHeading], -1.0, 1e-5, "sin heading at yaw +90deg");
  CheckNear(yawed[kIdxCmdY], -0.1, 1e-5, "command y at yaw +90deg");

  // ---- mjData path on the synthetic model ----------------------------------
  mjModel* m = synthetic::MakeModel();
  const RuntimeBindings b = synthetic::MakeBindings(m);
  ValidateModelAgainstBindings(m, b);
  mjData* d = mj_makeData(m);
  for (int i = 0; i < m->nq; ++i) d->qpos[i] = b.initial_qpos[i];
  const JointAddrs joints = ResolveJointAddrs(m, b);
  d->qpos[joints.qposadr[4]] = 0.3;
  d->qvel[joints.dofadr[4]] = 0.7;
  mj_forward(m, d);
  const SensorAddrs sensors = ResolveSensorAddrs(m, b);
  const ContactGeoms geoms = ResolveContactGeoms(m, b);
  ObsInfo zero_info{};
  zero_info.command = {0.1f, 0.0f, 0.0f};
  const auto o = BuildObservation(m, d, sensors, geoms, joints, zero_info);
  CheckNear(o[kIdxQposStart + 4], 0.3, 1e-6, "mjData joint qpos read through bindings");
  CheckNear(o[kIdxQvelStart + 4], 0.7, 1e-6, "mjData joint qvel read through bindings");
  CheckNear(o[kIdxGravity + 2], 1.0, 1e-5, "up-vector sensor for upright root");
  CheckNear(o[kIdxCosHeading], 1.0, 1e-5, "identity orientation -> zero heading error");

  // validation failures
  RuntimeBindings bad = b;
  bad.initial_qpos.pop_back();
  bool threw = false;
  try { ValidateModelAgainstBindings(m, bad); } catch (const std::runtime_error&) { threw = true; }
  Check(threw, "initial_qpos size mismatch rejected");
  bad = b;
  bad.joint_names[0] = "no_such_joint";
  threw = false;
  try { ValidateModelAgainstBindings(m, bad); } catch (const std::runtime_error&) { threw = true; }
  Check(threw, "unknown joint name rejected");
  bad = b;
  bad.sensor_gyro = "synthetic_root_pos";  // wrong name is fine (3 wide) ...
  bad.sensor_orientation = "synthetic_root_pos";  // ... but 3-wide quaternion is not
  threw = false;
  try { ValidateModelAgainstBindings(m, bad); } catch (const std::runtime_error&) { threw = true; }
  Check(threw, "wrong sensor width rejected");

  mj_deleteData(d);
  mj_deleteModel(m);
  return testing::Finish("test_obs_builder");
}
