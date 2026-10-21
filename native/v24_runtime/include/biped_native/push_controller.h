// External-force pulse controller: applies a constant force to a body for a
// fixed duration through data->xfrc_applied, then clears it. Robot-agnostic;
// the body id and force vector are runtime configuration.
#pragma once

#include <mujoco/mujoco.h>

#include <array>

namespace biped {

class PushController {
 public:
  PushController(int body_id, std::array<double, 3> force_n, double duration_s)
      : body_id_(body_id), force_n_(force_n), duration_s_(duration_s) {}

  // Call once per tick with dt = the tick's elapsed sim time.
  void Trigger() {
    active_ = true;
    elapsed_s_ = 0.0;
  }

  // Applies (or clears) the force on `d` for this tick; advances internal timer.
  void Apply(mjData* d, double dt) {
    if (active_) {
      d->xfrc_applied[body_id_ * 6 + 0] = force_n_[0];
      d->xfrc_applied[body_id_ * 6 + 1] = force_n_[1];
      d->xfrc_applied[body_id_ * 6 + 2] = force_n_[2];
      elapsed_s_ += dt;
      // Epsilon-tolerant comparison: elapsed_s_ accumulates dt via repeated
      // float addition, which drifts below the exact mathematical sum (e.g.
      // 10 additions of 0.02 yields 0.19999999999999998, not 0.2) -- an
      // exact >= comparison against duration_s_ would keep the push active
      // for one extra tick beyond the configured duration.
      if (elapsed_s_ >= duration_s_ - 1.0e-9) {
        active_ = false;
        d->xfrc_applied[body_id_ * 6 + 0] = 0.0;
        d->xfrc_applied[body_id_ * 6 + 1] = 0.0;
        d->xfrc_applied[body_id_ * 6 + 2] = 0.0;
      }
    } else {
      d->xfrc_applied[body_id_ * 6 + 0] = 0.0;
      d->xfrc_applied[body_id_ * 6 + 1] = 0.0;
      d->xfrc_applied[body_id_ * 6 + 2] = 0.0;
    }
  }

  bool IsActive() const { return active_; }
  std::array<double, 3> CurrentForce() const {
    return active_ ? force_n_ : std::array<double, 3>{0.0, 0.0, 0.0};
  }

 private:
  int body_id_;
  std::array<double, 3> force_n_;
  double duration_s_;
  bool active_ = false;
  double elapsed_s_ = 0.0;
};

}  // namespace biped
