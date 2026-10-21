// Native closed-loop runtime for the exported V24 actor.
//
//   gamepad -> command -> observation assembly -> ONNX inference ->
//   residual joint-position target -> mj_step x10 -> event machine -> next obs
//
// Every event-driven observation field (gait phase, expected side, bilateral
// placement memory) is derived live each tick by event_machine.h's
// UpdateEventMachine, driven by that tick's own contact / GRF / orientation;
// no recorded fixture is read. Per-tick ordering:
//   obs_t -> action_t = OnnxActor::Run(obs_t)
//         -> target_t = MapActionToTarget(action_t, ...)
//         -> apply target_t, run kSubstepsPerTick (10) mj_step calls
//         -> EventMachineInputs from the NOW-STEPPED mjData
//         -> UpdateEventMachine (exactly once per policy tick)
//         -> obs_(t+1) = BuildObservation(..., info from the updated event state)
//
// Robot-specific information is NOT compiled in: the model file, ONNX policy and
// RuntimeBindings JSON are supplied on the command line.
//
// Usage:
//   biped_v24_native --model <model.mjb|.xml> --policy <actor.onnx> \
//       --bindings <runtime_bindings.json> --out <output_dir> \
//       [--manifest <EXPORT_MANIFEST.json>] [--duration <s>] [--render]
//
// Simulation/runtime integration only: no hardware deployment is implied.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include <mujoco/mujoco.h>

#include "biped_native/camera_controller.h"
#include "biped_native/contact_utils.h"
#include "biped_native/control_law.h"
#include "biped_native/event_machine.h"
#include "biped_native/gamepad.h"
#include "biped_native/manifest_verify.h"
#include "biped_native/obs_builder.h"
#include "biped_native/onnx_actor.h"
#include "biped_native/push_controller.h"
#include "biped_native/runtime_bindings.h"
#include "biped_native/runtime_layout.h"
#include "biped_native/telemetry.h"
#ifdef BIPED_HAVE_GLFW
#include "biped_native/glfw_viewer.h"
#endif

using Clock = std::chrono::steady_clock;

namespace {

// Fall latch: root height below the bound binding or the up-vector tilted past
// kFallMinUprightZ (z component of the body up-vector sensor).
constexpr double kFallMinUprightZ = 0.3;
bool CheckFallen(const mjData* d, const biped::SensorAddrs& sensors, double min_root_height) {
  const double root_z = d->qpos[2];
  const double grav_z = d->sensordata[sensors.gravity_adr + 2];
  return (root_z < min_root_height) || (grav_z < kFallMinUprightZ);
}

// Body roll/pitch from the orientation quaternion (wxyz); independent of
// obs_builder.cpp's yaw extraction, which feeds the command-frame transform.
void RollPitchFromQuat(float qw, float qx, float qy, float qz, double* roll, double* pitch) {
  *roll = std::atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy));
  const double sinp = 2.0 * (qw * qy - qz * qx);
  *pitch = std::asin(std::min(std::max(sinp, -1.0), 1.0));
}

// Physical joint limits (radians) are read from the model's jnt_range, not from
// actuator ctrlrange: the targets are clipped to the real joint range.
struct JointLimits {
  std::array<float, biped::kNumJoints> lower;
  std::array<float, biped::kNumJoints> upper;
};

JointLimits ResolveJointLimits(const mjModel* m, const biped::RuntimeBindings& b) {
  JointLimits jl;
  for (int i = 0; i < biped::kNumJoints; ++i) {
    const int jid = mj_name2id(m, mjOBJ_JOINT, b.joint_names[i].c_str());
    if (jid < 0) throw std::runtime_error("biped_v24_native: missing joint " + b.joint_names[i]);
    if (!m->jnt_limited[jid]) {
      throw std::runtime_error("biped_v24_native: joint " + b.joint_names[i] +
                               " has no range limit in the model");
    }
    jl.lower[i] = static_cast<float>(m->jnt_range[jid * 2 + 0]);
    jl.upper[i] = static_cast<float>(m->jnt_range[jid * 2 + 1]);
  }
  return jl;
}

void SettleFromInit(const mjModel* m, mjData* d, const biped::RuntimeBindings& b) {
  for (int i = 0; i < m->nq; ++i) d->qpos[i] = b.initial_qpos[i];
  for (int i = 0; i < m->nv; ++i) d->qvel[i] = 0.0;
  mj_forward(m, d);
  for (int i = 0; i < biped::kNumJoints; ++i) d->ctrl[i] = 0.0;
  for (int s = 0; s < biped::kSettleSteps; ++s) mj_step(m, d);
  for (int i = 0; i < m->nv; ++i) d->qvel[i] = 0.0;
  mj_forward(m, d);
}

struct CliArgs {
  std::string model, policy, bindings, out_dir, manifest;
  double duration_s = 60.0;
  bool render = false;
};

bool ParseCli(int argc, char** argv, CliArgs* a) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&](std::string* dst) {
      if (i + 1 >= argc) return false;
      *dst = argv[++i];
      return true;
    };
    std::string dur;
    if (arg == "--model") { if (!value(&a->model)) return false; }
    else if (arg == "--policy") { if (!value(&a->policy)) return false; }
    else if (arg == "--bindings") { if (!value(&a->bindings)) return false; }
    else if (arg == "--out") { if (!value(&a->out_dir)) return false; }
    else if (arg == "--manifest") { if (!value(&a->manifest)) return false; }
    else if (arg == "--duration") { if (!value(&dur)) return false; a->duration_s = std::atof(dur.c_str()); }
    else if (arg == "--render") { a->render = true; }
    else return false;
  }
  return !a->model.empty() && !a->policy.empty() && !a->bindings.empty() && !a->out_dir.empty() &&
         a->duration_s > 0.0;
}

}  // namespace

int main(int argc, char** argv) {
  CliArgs cli;
  if (!ParseCli(argc, argv, &cli)) {
    std::fprintf(stderr,
                 "usage: %s --model <model.mjb|.xml> --policy <actor.onnx> --bindings <json> "
                 "--out <dir> [--manifest <json>] [--duration <s>] [--render]\n",
                 argv[0]);
    return 2;
  }
  const std::string output_dir = cli.out_dir;
  bool render = cli.render;
  const double duration_s = cli.duration_s;
#ifndef BIPED_HAVE_GLFW
  if (render) {
    std::fprintf(stderr,
                 "[biped_v24_native] --render requested but this binary was built without GLFW "
                 "-- continuing headless.\n");
    render = false;
  }
#endif

  mjModel* m = nullptr;
  mjData* d = nullptr;
  try {
    const biped::RuntimeBindings bindings = biped::LoadRuntimeBindings(cli.bindings);
    if (!cli.manifest.empty()) {
      // Hash contract: refuse to run an artifact that does not match the manifest.
      const auto base = [](const std::string& p) { return p.substr(p.find_last_of('/') + 1); };
      biped::VerifyArtifactHashOrThrow(cli.manifest, cli.policy, base(cli.policy));
    }

    char load_error[1000] = "";
    const bool is_binary = cli.model.size() > 4 && cli.model.substr(cli.model.size() - 4) == ".mjb";
    m = is_binary ? mj_loadModel(cli.model.c_str(), nullptr)
                  : mj_loadXML(cli.model.c_str(), nullptr, load_error, sizeof(load_error));
    if (!m) throw std::runtime_error("failed to load model " + cli.model + " " + load_error);
    d = mj_makeData(m);
    if (!d) throw std::runtime_error("mj_makeData failed");
    biped::ValidateModelAgainstBindings(m, bindings);

    biped::SensorAddrs sensors = biped::ResolveSensorAddrs(m, bindings);
    biped::ContactGeoms geoms = biped::ResolveContactGeoms(m, bindings);
    biped::JointAddrs joints = biped::ResolveJointAddrs(m, bindings);
    JointLimits jlim = ResolveJointLimits(m, bindings);
    biped::OnnxActor actor(cli.policy);

    const int root_id = mj_name2id(m, mjOBJ_BODY, bindings.root_body.c_str());
    if (root_id < 0) throw std::runtime_error("biped_v24_native: missing root body " + bindings.root_body);

    biped::CameraController camera(root_id);
    biped::Gamepad gamepad;
    biped::PushController push(root_id, {0.0, 60.0, 0.0}, 0.2);  // 60N lateral, 0.2s
    biped::TelemetryLogger telemetry(output_dir + "/metrics.csv");
#ifdef BIPED_HAVE_GLFW
    std::unique_ptr<biped::GlfwViewer> viewer;
    if (render) {
      viewer = std::make_unique<biped::GlfwViewer>(m);
      std::printf("[biped_v24_native] GLFW viewer window opened.\n");
    }
#endif

    // --- Settle, capture per-episode constants (bridge_q_ref_base, ---------
    // --- root_y_baseline, event-machine foot-z-nominal), reset event state -
    SettleFromInit(m, d, bindings);
    std::printf("[biped_v24_native] settled. root_z=%.4f\n", d->qpos[2]);

    std::array<float, biped::kNumJoints> bridge{};
    for (int i = 0; i < biped::kNumJoints; ++i) {
      bridge[i] = static_cast<float>(d->qpos[joints.qposadr[i]]);
    }
    const float root_y_baseline = static_cast<float>(d->sensordata[sensors.root_pos_adr + 1]);
    const float root_x_start = static_cast<float>(d->sensordata[sensors.root_pos_adr + 0]);

    biped::EventMachineState event_state{};
    biped::ResetEventMachineState(&event_state, m, d, geoms.left_foot_id, geoms.right_foot_id,
                                  bindings.target_height);

    // --- Build obs_0 exactly as ResidualForwardWalkingV18.reset() does: -----
    // --- gait_phase=LegacyGaitPhase(step=0)=0 (v9_expected_side unknown at
    // --- reset, so the event-driven phase is never used for the very first
    // --- observation), last_act=0, v24_* all zero, command=neutral.
    biped::ObsInfo info{};
    info.bridge_q_ref_base = bridge;
    info.gait_phase = 0.0f;
    info.command = {biped::kVxNeutral, 0.0f, 0.0f};
    info.root_y_baseline = root_y_baseline;
    info.v24_last_left_placement = 0.0f;
    info.v24_last_right_placement = 0.0f;
    info.v24_pair_valid = 0.0f;
    info.last_act.fill(0.0f);
    auto obs = biped::BuildObservation(m, d, sensors, geoms, joints, info);

    const int total_ticks = static_cast<int>(duration_s / biped::kCtrlDt);
    const auto budget = std::chrono::duration<double>(biped::kCtrlDt);
    // The scripted mid-run push is OFF by default -- the primary autonomous
    // smoke test (this app's exit-code pass/fail gate) is infrastructure
    // correctness (finite/deadlines/physics-step-count/forward displacement/
    // live event-machine cycling), not push-recovery robustness, which is a
    // property of the trained policy's stance-phase-dependent recovery
    // behavior (empirically confirmed: identical push, different outcomes at
    // different phase timings) and is exercised/reported separately by
    // apps/test_push_controller.cpp + this app's BIPED_PUSH_AT_S hook.
    const char* push_at_s_env = std::getenv("BIPED_PUSH_AT_S");
    const bool push_enabled = (push_at_s_env != nullptr);
    const double push_at_s = push_enabled ? std::atof(push_at_s_env) : -1.0;
    const int push_at_tick = push_enabled ? static_cast<int>(push_at_s / biped::kCtrlDt) : -1;

    bool paused = false;
    bool fall_latched = false;
    float filt_cmd_vx = biped::kVxNeutral;
    long total_physics_steps = 0;
    int v9_expected_side_flip_count = 0;
    double prev_v9_expected_side = event_state.v9_expected_side;
    const auto smoke_start = Clock::now();

    for (int tick = 0; tick < total_ticks; ++tick) {
      const auto tick_start = Clock::now();

      auto gp = gamepad.Poll();
      if (gp.reset_pressed) {
        SettleFromInit(m, d, bindings);
        biped::ResetEventMachineState(&event_state, m, d, geoms.left_foot_id,
                                       geoms.right_foot_id, bindings.target_height);
        info.last_act.fill(0.0f);
        info.gait_phase = 0.0f;
        info.v24_last_left_placement = 0.0f;
        info.v24_last_right_placement = 0.0f;
        info.v24_pair_valid = 0.0f;
        obs = biped::BuildObservation(m, d, sensors, geoms, joints, info);
        fall_latched = false;
      }
      if (gp.pause_pressed) paused = !paused;
      if (gp.camera_cycle_pressed) camera.CycleMode();
      if (gp.push_pressed) push.Trigger();
      if (push_enabled && tick == push_at_tick) {
        push.Trigger();  // scripted exercise (opt-in via BIPED_PUSH_AT_S)
      }

      if (paused) {
        telemetry.Write(biped::TelemetryRow{});  // heartbeat row while paused
        std::this_thread::sleep_for(budget);
        continue;
      }

      // 1-pole low-pass on the gamepad-derived vx command.
      constexpr float kAlpha = 0.2f;
      filt_cmd_vx = filt_cmd_vx + kAlpha * (gp.vx - filt_cmd_vx);
      info.command = {filt_cmd_vx, 0.0f, 0.0f};

      // --- actor_t = OnnxActor::Run(obs_t) ------------------------------
      const auto infer_start = Clock::now();
      auto action = actor.Run(obs);
      const double infer_ms =
          std::chrono::duration<double, std::milli>(Clock::now() - infer_start).count();

      bool any_finite_fail = false;
      for (float v : obs) any_finite_fail |= !std::isfinite(v);
      for (float v : action) any_finite_fail |= !std::isfinite(v);
      if (any_finite_fail) {
        std::fprintf(stderr, "[biped_v24_native] non-finite value at tick %d -- aborting\n", tick);
        break;
      }

      // --- ctrl_t = control law ------------------------------------------
      auto target =
          biped::MapActionToTarget(action, bridge, bindings.action_scale, jlim.lower, jlim.upper);
      bool clipped = false;
      for (int i = 0; i < biped::kNumJoints; ++i) {
        const float raw = bridge[i] + bindings.action_scale[i] * action[i];
        if (raw < jlim.lower[i] || raw > jlim.upper[i]) clipped = true;
        d->ctrl[i] = target[i];
      }

      // --- 10 physics substeps --------------------------------------------
      push.Apply(d, biped::kSimDt * biped::kSubstepsPerTick);
      for (int s = 0; s < biped::kSubstepsPerTick; ++s) mj_step(m, d);
      total_physics_steps += biped::kSubstepsPerTick;

      bool physics_finite_fail = false;
      for (int i = 0; i < m->nq; ++i) physics_finite_fail |= !std::isfinite(d->qpos[i]);
      for (int i = 0; i < m->nv; ++i) physics_finite_fail |= !std::isfinite(d->qvel[i]);
      for (int i = 0; i < m->nu; ++i) physics_finite_fail |= !std::isfinite(d->ctrl[i]);
      if (physics_finite_fail) {
        std::fprintf(stderr,
                    "[biped_v24_native] non-finite qpos/qvel/ctrl at tick %d -- aborting\n", tick);
        break;
      }

      // --- event-machine update, exactly once per policy tick, using the ---
      // --- NOW-STEPPED mjData -----------------------------------------------
      biped::EventMachineInputs em_in;
      em_in.left_contact =
          biped::GeomsInContact(d, d->ncon, geoms.floor_id, geoms.left_foot_id);
      em_in.right_contact =
          biped::GeomsInContact(d, d->ncon, geoms.floor_id, geoms.right_foot_id);
      em_in.left_foot_pos_x = d->geom_xpos[geoms.left_foot_id * 3 + 0];
      em_in.left_foot_pos_z = d->geom_xpos[geoms.left_foot_id * 3 + 2];
      em_in.right_foot_pos_x = d->geom_xpos[geoms.right_foot_id * 3 + 0];
      em_in.right_foot_pos_z = d->geom_xpos[geoms.right_foot_id * 3 + 2];
      const biped::FootGrf grf =
          biped::ComputeFootGrf(m, d, geoms.floor_id, geoms.left_foot_id, geoms.right_foot_id);
      em_in.fz_left = grf.left;
      em_in.fz_right = grf.right;
      RollPitchFromQuat(static_cast<float>(d->sensordata[sensors.orientation_adr + 0]),
                         static_cast<float>(d->sensordata[sensors.orientation_adr + 1]),
                         static_cast<float>(d->sensordata[sensors.orientation_adr + 2]),
                         static_cast<float>(d->sensordata[sensors.orientation_adr + 3]),
                         &em_in.roll, &em_in.pitch);
      em_in.gyro_x = d->sensordata[sensors.gyro_adr + 0];
      em_in.gyro_y = d->sensordata[sensors.gyro_adr + 1];
      em_in.root_z = d->sensordata[sensors.root_pos_adr + 2];

      biped::EventMachineOutputs em_out = biped::UpdateEventMachine(&event_state, em_in, m);
      if (event_state.v9_expected_side != prev_v9_expected_side) {
        ++v9_expected_side_flip_count;
        prev_v9_expected_side = event_state.v9_expected_side;
      }

      // --- obs_(t+1), built from the JUST-updated event_state -------------
      info.last_act = action;
      info.gait_phase = static_cast<float>(em_out.gait_phase);
      info.v24_last_left_placement = static_cast<float>(event_state.v24_last_left_placement);
      info.v24_last_right_placement = static_cast<float>(event_state.v24_last_right_placement);
      info.v24_pair_valid = static_cast<float>(event_state.v24_pair_valid);
      obs = biped::BuildObservation(m, d, sensors, geoms, joints, info);

      const bool fallen = CheckFallen(d, sensors, bindings.min_root_height);
      fall_latched = fall_latched || fallen;

      float amean = 0.0f, amax = 0.0f;
      for (float v : action) {
        amean += std::fabs(v);
        amax = std::max(amax, std::fabs(v));
      }
      amean /= biped::kActionSize;
      double ctrl_min = d->ctrl[0], ctrl_max = d->ctrl[0];
      for (int i = 1; i < biped::kNumJoints; ++i) {
        ctrl_min = std::min(ctrl_min, d->ctrl[i]);
        ctrl_max = std::max(ctrl_max, d->ctrl[i]);
      }

      const double tick_ms =
          std::chrono::duration<double, std::milli>(Clock::now() - tick_start).count();
      const bool missed_deadline = tick_ms > (biped::kCtrlDt * 1000.0);

      biped::TelemetryRow row;
      row.tick = tick;
      row.sim_time_s = tick * biped::kCtrlDt;
      row.raw_cmd_vx = gp.vx;
      row.raw_cmd_vy = gp.vy;
      row.raw_cmd_yaw_rate = gp.yaw_rate;
      row.filt_cmd_vx = filt_cmd_vx;
      row.root_x = d->qpos[0];
      row.root_y = d->qpos[1];
      row.root_z = d->qpos[2];
      row.local_vx = obs[27];
      row.local_vy = obs[28];
      row.local_vz = obs[29];
      row.gyro_x = obs[30];
      row.gyro_y = obs[31];
      row.gyro_z = obs[32];
      row.action_mean_abs = amean;
      row.action_max_abs = amax;
      row.ctrl_min = ctrl_min;
      row.ctrl_max = ctrl_max;
      row.target_clipped = clipped;
      row.left_contact = obs[biped::kIdxLeftContact];
      row.right_contact = obs[biped::kIdxRightContact];
      row.bilateral_imbalance = obs[biped::kIdxBilateralImbalance];
      row.pair_valid = obs[biped::kIdxPairValid];
      row.push_active = push.IsActive();
      auto pf = push.CurrentForce();
      row.push_fx = pf[0];
      row.push_fy = pf[1];
      row.push_fz = pf[2];
      row.inference_latency_ms = infer_ms;
      row.tick_wall_ms = tick_ms;
      row.missed_deadline = missed_deadline;
      row.fallen = fallen;
      telemetry.Write(row);

      if (tick % 500 == 0) {
        std::printf("[biped_v24_native] tick=%d/%d t=%.1fs root=(%.2f,%.2f,%.2f) "
                    "gait_phase=%.4f v9_expected_side=%.0f pair_valid=%.0f camera=%d push=%d "
                    "fallen=%d\n",
                    tick, total_ticks, row.sim_time_s, row.root_x, row.root_y, row.root_z,
                    em_out.gait_phase, event_state.v9_expected_side, event_state.v24_pair_valid,
                    static_cast<int>(camera.Mode()), row.push_active ? 1 : 0, fallen ? 1 : 0);
      }

#ifdef BIPED_HAVE_GLFW
      if (viewer) {
        if (!viewer->RenderFrame(d, camera.Camera())) {
          std::printf("[biped_v24_native] viewer window closed by user -- continuing headless.\n");
          viewer.reset();
        }
      }
#endif

      // Real-time pacing: sleep off the remainder of the 20ms budget.
      const auto elapsed = Clock::now() - tick_start;
      if (elapsed < budget) {
        std::this_thread::sleep_for(budget - elapsed);
      }
    }

    const double total_wall_s = std::chrono::duration<double>(Clock::now() - smoke_start).count();
    telemetry.WriteSummaryJson(output_dir + "/summary.json", total_wall_s);

    const float root_x_end = static_cast<float>(d->sensordata[sensors.root_pos_adr + 0]);
    const double forward_displacement_m = root_x_end - root_x_start;
    const int expected_physics_steps =
        static_cast<int>(duration_s / biped::kCtrlDt) * biped::kSubstepsPerTick;

    std::printf("[biped_v24_native] done. ticks=%d wall=%.2fs missed_deadlines=%d "
                "fall_ever=%s gamepad_hardware=%s total_physics_steps=%ld (expected %d) "
                "forward_displacement_m=%.4f v9_expected_side_flips=%d\n",
                telemetry.RowsWritten(), total_wall_s, telemetry.MissedDeadlines(),
                fall_latched ? "true" : "false", gamepad.HasHardware() ? "true" : "false",
                total_physics_steps, expected_physics_steps, forward_displacement_m,
                v9_expected_side_flip_count);

    mj_deleteData(d);
    mj_deleteModel(m);

    // "Meaningful" forward displacement: at least 30% of the distance covered at
    // the minimum commanded speed (kVxMin) over this duration -- a conservative
    // floor that a stationary, twitching or fallen robot would not reach.
    const double meaningful_displacement_floor = 0.3 * duration_s * biped::kVxMin;
    const bool pass = (telemetry.MissedDeadlines() == 0) &&
                       (total_physics_steps == expected_physics_steps) &&
                       (!fall_latched) &&
                       (forward_displacement_m > meaningful_displacement_floor) &&
                       (v9_expected_side_flip_count > 0);
    std::printf("[biped_v24_native] pass_criteria: missed_deadlines==0:%s "
                "physics_steps_exact:%s no_fall:%s displacement>%.2fm:%s "
                "autonomous_gait_events:%s\n",
                (telemetry.MissedDeadlines() == 0) ? "true" : "false",
                (total_physics_steps == expected_physics_steps) ? "true" : "false",
                (!fall_latched) ? "true" : "false", meaningful_displacement_floor,
                (forward_displacement_m > meaningful_displacement_floor) ? "true" : "false",
                (v9_expected_side_flip_count > 0) ? "true" : "false");
    std::printf("[biped_v24_native] %s\n",
                pass ? "PASS (no missed deadlines, exact physics-step count, never fell, "
                       "meaningful forward displacement, autonomous gait events observed)"
                     : "FAIL");
    return pass ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[biped_v24_native] EXCEPTION: %s\n", e.what());
    if (d) mj_deleteData(d);
    if (m) mj_deleteModel(m);
    return 1;
  }
}
