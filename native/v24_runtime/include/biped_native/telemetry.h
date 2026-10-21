// CSV + JSON telemetry logger, following the metrics.csv + summary.json
// convention established by mujoco_playground/experimental/sim2sim/
// play_t1_joystick_25m.py and eval/robustness_play.py.
#pragma once

#include <array>
#include <cstdio>
#include <string>
#include <vector>

#include "biped_native/runtime_layout.h"

namespace biped {

struct TelemetryRow {
  int tick = 0;
  double sim_time_s = 0.0;
  float raw_cmd_vx = 0.0f, raw_cmd_vy = 0.0f, raw_cmd_yaw_rate = 0.0f;
  float filt_cmd_vx = 0.0f;
  double root_x = 0.0, root_y = 0.0, root_z = 0.0;
  float local_vx = 0.0f, local_vy = 0.0f, local_vz = 0.0f;
  float gyro_x = 0.0f, gyro_y = 0.0f, gyro_z = 0.0f;
  float action_mean_abs = 0.0f, action_max_abs = 0.0f;
  double ctrl_min = 0.0, ctrl_max = 0.0;
  bool target_clipped = false;
  float left_contact = 0.0f, right_contact = 0.0f;
  float bilateral_imbalance = 0.0f, pair_valid = 0.0f;
  bool push_active = false;
  double push_fx = 0.0, push_fy = 0.0, push_fz = 0.0;
  double inference_latency_ms = 0.0;
  double tick_wall_ms = 0.0;
  bool missed_deadline = false;
  bool fallen = false;
};

class TelemetryLogger {
 public:
  explicit TelemetryLogger(const std::string& csv_path) : csv_path_(csv_path) {
    f_ = std::fopen(csv_path.c_str(), "w");
    if (f_) {
      std::fprintf(f_,
          "tick,sim_time_s,raw_cmd_vx,raw_cmd_vy,raw_cmd_yaw_rate,filt_cmd_vx,"
          "root_x,root_y,root_z,local_vx,local_vy,local_vz,gyro_x,gyro_y,gyro_z,"
          "action_mean_abs,action_max_abs,ctrl_min,ctrl_max,target_clipped,"
          "left_contact,right_contact,bilateral_imbalance,pair_valid,"
          "push_active,push_fx,push_fy,push_fz,"
          "inference_latency_ms,tick_wall_ms,missed_deadline,fallen\n");
    }
  }
  ~TelemetryLogger() {
    if (f_) std::fclose(f_);
  }

  void Write(const TelemetryRow& r) {
    if (!f_) return;
    std::fprintf(f_,
        "%d,%.4f,%.6f,%.6f,%.6f,%.6f,"
        "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
        "%.6f,%.6f,%.6f,%.6f,%d,"
        "%.2f,%.2f,%.6f,%.2f,"
        "%d,%.4f,%.4f,%.4f,"
        "%.4f,%.4f,%d,%d\n",
        r.tick, r.sim_time_s, r.raw_cmd_vx, r.raw_cmd_vy, r.raw_cmd_yaw_rate, r.filt_cmd_vx,
        r.root_x, r.root_y, r.root_z, r.local_vx, r.local_vy, r.local_vz, r.gyro_x, r.gyro_y,
        r.gyro_z, r.action_mean_abs, r.action_max_abs, r.ctrl_min, r.ctrl_max,
        r.target_clipped ? 1 : 0, r.left_contact, r.right_contact, r.bilateral_imbalance,
        r.pair_valid, r.push_active ? 1 : 0, r.push_fx, r.push_fy, r.push_fz,
        r.inference_latency_ms, r.tick_wall_ms, r.missed_deadline ? 1 : 0, r.fallen ? 1 : 0);
    ++rows_written_;
    if (r.missed_deadline) ++missed_deadlines_;
    if (r.fallen) fall_detected_ = true;
    if (r.inference_latency_ms > max_inference_ms_) max_inference_ms_ = r.inference_latency_ms;
    if (r.tick_wall_ms > max_tick_ms_) max_tick_ms_ = r.tick_wall_ms;
    sum_tick_ms_ += r.tick_wall_ms;
  }

  void WriteSummaryJson(const std::string& json_path, double total_wall_s) const {
    FILE* jf = std::fopen(json_path.c_str(), "w");
    if (!jf) return;
    std::fprintf(jf,
        "{\n"
        "  \"metrics_csv_path\": \"%s\",\n"
        "  \"ticks\": %d,\n"
        "  \"total_wall_s\": %.3f,\n"
        "  \"mean_tick_wall_ms\": %.4f,\n"
        "  \"max_tick_wall_ms\": %.4f,\n"
        "  \"max_inference_latency_ms\": %.4f,\n"
        "  \"missed_deadlines\": %d,\n"
        "  \"fall_detected\": %s,\n"
        "  \"ctrl_dt_budget_ms\": %.3f\n"
        "}\n",
        csv_path_.c_str(), rows_written_, total_wall_s,
        rows_written_ > 0 ? sum_tick_ms_ / rows_written_ : 0.0, max_tick_ms_, max_inference_ms_,
        missed_deadlines_, fall_detected_ ? "true" : "false", kCtrlDt * 1000.0);
    std::fclose(jf);
  }

  int RowsWritten() const { return rows_written_; }
  int MissedDeadlines() const { return missed_deadlines_; }
  bool FallDetected() const { return fall_detected_; }

 private:
  std::string csv_path_;
  FILE* f_ = nullptr;
  int rows_written_ = 0;
  int missed_deadlines_ = 0;
  bool fall_detected_ = false;
  double max_inference_ms_ = 0.0;
  double max_tick_ms_ = 0.0;
  double sum_tick_ms_ = 0.0;
};

}  // namespace biped
