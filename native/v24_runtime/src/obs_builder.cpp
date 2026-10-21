#include "biped_native/obs_builder.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include "biped_native/contact_utils.h"

namespace biped {

namespace {

int NameToId(const mjModel* m, mjtObj type, const char* name) {
  int id = mj_name2id(m, type, name);
  if (id < 0) throw std::runtime_error(std::string("obs_builder: missing name ") + name);
  return id;
}

// Matches ResidualForwardWalkingV1._forward_yaw_from_quat.
float ForwardYawFromQuat(float qw, float qx, float qy, float qz) {
  return std::atan2(2.0f * (qw * qz + qx * qy), 1.0f - 2.0f * (qy * qy + qz * qz));
}

}  // namespace

SensorAddrs ResolveSensorAddrs(const mjModel* m, const RuntimeBindings& b) {
  SensorAddrs s;
  s.orientation_adr = m->sensor_adr[NameToId(m, mjOBJ_SENSOR, b.sensor_orientation.c_str())];
  s.gravity_adr = m->sensor_adr[NameToId(m, mjOBJ_SENSOR, b.sensor_gravity.c_str())];
  s.local_linvel_adr = m->sensor_adr[NameToId(m, mjOBJ_SENSOR, b.sensor_local_linvel.c_str())];
  s.gyro_adr = m->sensor_adr[NameToId(m, mjOBJ_SENSOR, b.sensor_gyro.c_str())];
  s.root_pos_adr = m->sensor_adr[NameToId(m, mjOBJ_SENSOR, b.sensor_root_pos.c_str())];
  return s;
}

ContactGeoms ResolveContactGeoms(const mjModel* m, const RuntimeBindings& b) {
  ContactGeoms g;
  g.floor_id = NameToId(m, mjOBJ_GEOM, b.floor_geom.c_str());
  g.left_foot_id = NameToId(m, mjOBJ_GEOM, b.left_foot_geom.c_str());
  g.right_foot_id = NameToId(m, mjOBJ_GEOM, b.right_foot_geom.c_str());
  return g;
}

JointAddrs ResolveJointAddrs(const mjModel* m, const RuntimeBindings& b) {
  JointAddrs j;
  for (int i = 0; i < kNumJoints; ++i) {
    const int jid = NameToId(m, mjOBJ_JOINT, b.joint_names[i].c_str());
    j.qposadr[i] = m->jnt_qposadr[jid];
    j.dofadr[i] = m->jnt_dofadr[jid];
  }
  return j;
}

void ValidateModelAgainstBindings(const mjModel* m, const RuntimeBindings& b) {
  if (static_cast<int>(b.initial_qpos.size()) != m->nq) {
    throw std::runtime_error("obs_builder: initial_qpos has " +
                             std::to_string(b.initial_qpos.size()) + " entries but model nq=" +
                             std::to_string(m->nq));
  }
  if (m->nu < kNumJoints) {
    throw std::runtime_error("obs_builder: model has " + std::to_string(m->nu) +
                             " actuators, need at least " + std::to_string(kNumJoints));
  }
  const struct {
    const std::string* name;
    int width;
  } sensors[] = {{&b.sensor_orientation, 4}, {&b.sensor_gravity, 3}, {&b.sensor_local_linvel, 3},
                 {&b.sensor_gyro, 3}, {&b.sensor_root_pos, 3}};
  for (const auto& s : sensors) {
    const int id = NameToId(m, mjOBJ_SENSOR, s.name->c_str());
    if (m->sensor_dim[id] != s.width) {
      throw std::runtime_error("obs_builder: sensor '" + *s.name + "' has width " +
                               std::to_string(m->sensor_dim[id]) + ", expected " +
                               std::to_string(s.width));
    }
  }
  for (int i = 0; i < kNumJoints; ++i) {
    const int jid = NameToId(m, mjOBJ_JOINT, b.joint_names[i].c_str());
    if (m->jnt_type[jid] != mjJNT_HINGE && m->jnt_type[jid] != mjJNT_SLIDE) {
      throw std::runtime_error("obs_builder: joint '" + b.joint_names[i] +
                               "' must be a scalar (hinge/slide) joint");
    }
  }
  NameToId(m, mjOBJ_GEOM, b.floor_geom.c_str());
  NameToId(m, mjOBJ_GEOM, b.left_foot_geom.c_str());
  NameToId(m, mjOBJ_GEOM, b.right_foot_geom.c_str());
  NameToId(m, mjOBJ_BODY, b.root_body.c_str());
}

std::array<float, kObsSize> BuildObservation(const mjModel* m, const mjData* d,
                                              const SensorAddrs& sensors,
                                              const ContactGeoms& geoms,
                                              const JointAddrs& joints,
                                              const ObsInfo& info) {
  RawObsInputs raw{};
  for (int i = 0; i < kNumJoints; ++i) {
    raw.joint_qpos[i] = static_cast<float>(d->qpos[joints.qposadr[i]]);
    raw.joint_qvel[i] = static_cast<float>(d->qvel[joints.dofadr[i]]);
  }
  for (int i = 0; i < 3; ++i) {
    raw.gravity[i] = static_cast<float>(d->sensordata[sensors.gravity_adr + i]);
    raw.local_linvel[i] = static_cast<float>(d->sensordata[sensors.local_linvel_adr + i]);
    raw.gyro[i] = static_cast<float>(d->sensordata[sensors.gyro_adr + i]);
  }
  for (int i = 0; i < 4; ++i) {
    raw.orientation_quat[i] = static_cast<float>(d->sensordata[sensors.orientation_adr + i]);
  }
  raw.left_contact = GeomsInContact(d, d->ncon, geoms.floor_id, geoms.left_foot_id) ? 1.0f : 0.0f;
  raw.right_contact = GeomsInContact(d, d->ncon, geoms.floor_id, geoms.right_foot_id) ? 1.0f : 0.0f;
  raw.root_y = static_cast<float>(d->sensordata[sensors.root_pos_adr + 1]);
  return BuildObservationFromRaw(raw, info);
}

std::array<float, kObsSize> BuildObservationFromRaw(const RawObsInputs& raw, const ObsInfo& info) {
  std::array<float, kObsSize> obs{};

  // 0-11: qpos[ACTIVE_JOINTS] - bridge_q_ref_base
  for (int i = 0; i < kNumJoints; ++i) {
    obs[kIdxQposStart + i] = raw.joint_qpos[i] - info.bridge_q_ref_base[i];
  }
  // 12-23: qvel[ACTIVE_JOINTS]
  for (int i = 0; i < kNumJoints; ++i) {
    obs[kIdxQvelStart + i] = raw.joint_qvel[i];
  }
  // 24-26: gravity (upvector sensor)
  for (int i = 0; i < 3; ++i) obs[kIdxGravity + i] = raw.gravity[i];
  // 27-29: local_linvel
  for (int i = 0; i < 3; ++i) obs[kIdxLocalLinvel + i] = raw.local_linvel[i];
  // 30-32: gyro
  for (int i = 0; i < 3; ++i) obs[kIdxGyro + i] = raw.gyro[i];
  // 33-34: sin/cos gait phase
  obs[kIdxSinPhase] = std::sin(2.0f * static_cast<float>(M_PI) * info.gait_phase);
  obs[kIdxCosPhase] = std::cos(2.0f * static_cast<float>(M_PI) * info.gait_phase);
  // 35-46: last_act
  for (int i = 0; i < kActionSize; ++i) obs[kIdxLastActStart + i] = info.last_act[i];
  // 47-48: contacts
  obs[kIdxLeftContact] = raw.left_contact;
  obs[kIdxRightContact] = raw.right_contact;

  // Command-frame transform + heading fields (V24).
  const float yaw = ForwardYawFromQuat(raw.orientation_quat[0], raw.orientation_quat[1],
                                        raw.orientation_quat[2], raw.orientation_quat[3]);
  const float angle = -yaw;
  const float heading_error = std::atan2(std::sin(angle), std::cos(angle));
  const float vx_cmd = info.command[0];
  const float yaw_rate_cmd = info.command[2];

  // 49-51: command-frame vx*cos, vx*sin, raw yaw_rate
  obs[kIdxCmdX] = vx_cmd * std::cos(heading_error);
  obs[kIdxCmdY] = vx_cmd * std::sin(heading_error);
  obs[kIdxYawRate] = yaw_rate_cmd;

  // 52: e_y_obs
  obs[kIdxEyObs] = (raw.root_y - info.root_y_baseline) / kLateralScale;

  // 53-54: sin/cos heading error
  obs[kIdxSinHeading] = std::sin(heading_error);
  obs[kIdxCosHeading] = std::cos(heading_error);

  // 55-56: bilateral imbalance (signed) + pair_valid
  const float ll = info.v24_last_left_placement;
  const float rr = info.v24_last_right_placement;
  const float pair_valid = info.v24_pair_valid;
  obs[kIdxBilateralImbalance] = pair_valid * ((rr - ll) / (rr + ll + kBilateralEps));
  obs[kIdxPairValid] = pair_valid;

  return obs;
}

}  // namespace biped
