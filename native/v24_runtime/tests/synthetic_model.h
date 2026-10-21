// Synthetic MuJoCo model for software-contract tests.
//
// This is NOT a robot description and does not resemble the research robot: it
// is a floating box with two feet and twelve unit-mass hinge "joints", built
// through the mjSpec API so that no model file or XML is needed. All names are
// obviously synthetic. It only provides the structure the runtime contracts
// need (12 scalar joints + free base, 12 actuators, five named sensors, floor
// and two foot geoms) so that observation assembly, contact/GRF extraction and
// the event machine can be exercised without any restricted asset.
#pragma once

#include <mujoco/mujoco.h>

#include <stdexcept>
#include <string>

#include "biped_native/runtime_bindings.h"

namespace synthetic {

inline mjModel* MakeModel() {
  mjSpec* spec = mj_makeSpec();
  mjsBody* world = mjs_findBody(spec, "world");

  mjsGeom* floor = mjs_addGeom(world, nullptr);
  mjs_setName(floor->element, "synthetic_floor");
  floor->type = mjGEOM_PLANE;
  floor->size[0] = 5; floor->size[1] = 5; floor->size[2] = 0.1;

  mjsBody* root = mjs_addBody(world, nullptr);
  mjs_setName(root->element, "synthetic_root");
  root->pos[2] = 0.2;
  mjs_addJoint(root, nullptr)->type = mjJNT_FREE;

  mjsGeom* trunk = mjs_addGeom(root, nullptr);
  trunk->type = mjGEOM_BOX;
  trunk->size[0] = 0.1; trunk->size[1] = 0.1; trunk->size[2] = 0.05;
  trunk->contype = 0; trunk->conaffinity = 0;  // trunk never collides
  trunk->mass = 8.0;

  const char* foot_names[2] = {"synthetic_foot_left", "synthetic_foot_right"};
  for (int side = 0; side < 2; ++side) {
    mjsGeom* foot = mjs_addGeom(root, nullptr);
    mjs_setName(foot->element, foot_names[side]);
    foot->type = mjGEOM_BOX;
    foot->size[0] = 0.04; foot->size[1] = 0.03; foot->size[2] = 0.02;
    foot->pos[1] = side == 0 ? 0.08 : -0.08;
    foot->pos[2] = -0.05;
    foot->mass = 1.0;
  }

  mjsSite* imu = mjs_addSite(root, nullptr);
  mjs_setName(imu->element, "synthetic_imu_site");

  for (int i = 0; i < biped::kNumJoints; ++i) {
    mjsBody* b = mjs_addBody(root, nullptr);
    b->pos[0] = 0.01 * i;
    mjsJoint* j = mjs_addJoint(b, nullptr);
    const std::string jn = "synthetic_joint_" + std::to_string(i);
    mjs_setName(j->element, jn.c_str());
    j->type = mjJNT_HINGE;
    j->axis[0] = 0; j->axis[1] = 1; j->axis[2] = 0;
    j->limited = mjLIMITED_TRUE;
    j->range[0] = -1.0; j->range[1] = 1.0;
    j->armature = 0.01;  // keeps the stiff synthetic position actuators numerically stable
    j->damping = 0.5;
    mjsGeom* g = mjs_addGeom(b, nullptr);
    g->type = mjGEOM_SPHERE;
    g->size[0] = 0.01;
    g->contype = 0; g->conaffinity = 0;
    g->mass = 0.1;

    mjsActuator* a = mjs_addActuator(spec, nullptr);
    const std::string an = "synthetic_actuator_" + std::to_string(i);
    mjs_setName(a->element, an.c_str());
    a->trntype = mjTRN_JOINT;
    mjs_setString(a->target, jn.c_str());
    a->gainprm[0] = 20.0;
    a->biasprm[1] = -20.0;
    a->biastype = mjBIAS_AFFINE;
  }

  auto add_sensor = [&](const char* name, mjtSensor type, mjtObj objtype, const char* obj) {
    mjsSensor* s = mjs_addSensor(spec);
    mjs_setName(s->element, name);
    s->type = type;
    s->objtype = objtype;
    mjs_setString(s->objname, obj);
  };
  add_sensor("synthetic_orientation", mjSENS_FRAMEQUAT, mjOBJ_XBODY, "synthetic_root");
  add_sensor("synthetic_upvector", mjSENS_FRAMEZAXIS, mjOBJ_XBODY, "synthetic_root");
  add_sensor("synthetic_local_linvel", mjSENS_VELOCIMETER, mjOBJ_SITE, "synthetic_imu_site");
  add_sensor("synthetic_gyro", mjSENS_GYRO, mjOBJ_SITE, "synthetic_imu_site");
  add_sensor("synthetic_root_pos", mjSENS_FRAMEPOS, mjOBJ_XBODY, "synthetic_root");

  mjModel* m = mj_compile(spec, nullptr);
  if (!m) {
    const std::string err = mjs_getError(spec);
    mj_deleteSpec(spec);
    throw std::runtime_error("synthetic model compile failed: " + err);
  }
  mj_deleteSpec(spec);
  return m;
}

inline biped::RuntimeBindings MakeBindings(const mjModel* m) {
  biped::RuntimeBindings b;
  for (int i = 0; i < biped::kNumJoints; ++i) {
    b.joint_names[i] = "synthetic_joint_" + std::to_string(i);
    b.action_scale[i] = 0.1f;
  }
  b.sensor_orientation = "synthetic_orientation";
  b.sensor_gravity = "synthetic_upvector";
  b.sensor_local_linvel = "synthetic_local_linvel";
  b.sensor_gyro = "synthetic_gyro";
  b.sensor_root_pos = "synthetic_root_pos";
  b.floor_geom = "synthetic_floor";
  b.left_foot_geom = "synthetic_foot_left";
  b.right_foot_geom = "synthetic_foot_right";
  b.root_body = "synthetic_root";
  b.initial_qpos.assign(m->nq, 0.0);
  b.initial_qpos[2] = 0.075;  // feet just above the floor
  b.initial_qpos[3] = 1.0;   // identity quaternion
  b.target_height = 0.5;     // synthetic, test-local
  b.min_root_height = 0.1;   // synthetic, test-local
  return b;
}

}  // namespace synthetic
