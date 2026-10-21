// Follow / top / free camera state machine. Configures mjvCamera (pure data, no
// GL context required); drawing the frames needs an OpenGL context and is done
// by the optional glfw_viewer.h. Testable without a display.
#pragma once

#include <mujoco/mujoco.h>

namespace biped {

enum class CameraMode { kFollow = 0, kTop = 1, kFree = 2 };

class CameraController {
 public:
  explicit CameraController(int track_body_id) : track_body_id_(track_body_id) {
    ApplyMode(CameraMode::kFollow);
  }

  void CycleMode() {
    mode_ = static_cast<CameraMode>((static_cast<int>(mode_) + 1) % 3);
    ApplyMode(mode_);
  }

  CameraMode Mode() const { return mode_; }
  const mjvCamera& Camera() const { return cam_; }

 private:
  void ApplyMode(CameraMode mode) {
    switch (mode) {
      case CameraMode::kFollow:
        cam_.type = mjCAMERA_TRACKING;
        cam_.trackbodyid = track_body_id_;
        cam_.distance = 2.0;
        cam_.azimuth = 90.0;
        cam_.elevation = -15.0;
        cam_.lookat[0] = cam_.lookat[1] = cam_.lookat[2] = 0.0;
        break;
      case CameraMode::kTop:
        cam_.type = mjCAMERA_TRACKING;
        cam_.trackbodyid = track_body_id_;
        cam_.distance = 4.0;
        cam_.azimuth = 90.0;
        cam_.elevation = -89.9;  // straight down (avoid exact -90 gimbal edge case)
        cam_.lookat[0] = cam_.lookat[1] = cam_.lookat[2] = 0.0;
        break;
      case CameraMode::kFree:
        cam_.type = mjCAMERA_FREE;
        cam_.trackbodyid = -1;
        cam_.distance = 3.0;
        cam_.azimuth = 90.0;
        cam_.elevation = -20.0;
        cam_.lookat[0] = cam_.lookat[1] = 0.0;
        cam_.lookat[2] = 0.5;
        break;
    }
  }

  int track_body_id_;
  CameraMode mode_ = CameraMode::kFollow;
  mjvCamera cam_{};
};

}  // namespace biped
