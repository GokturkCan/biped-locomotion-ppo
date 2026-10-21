// SDL2 gamepad reader. Output mapping is deliberately restrictive: only the
// forward-velocity command vx reaches the policy, clamped to the trained range
// (configs/v24_environment.json: vx_min/vx_max); vy and yaw_rate are always
// exactly zero because the policy was trained with zero lateral/yaw commands.
// Deadzone and left-stick vertical axis follow the reference joystick reader.
//
// Handles zero connected joysticks (neutral command, never blocks) and
// hot-plug via SDL_JOYDEVICEADDED/REMOVED events pumped once per Poll().
#pragma once

#include <SDL2/SDL.h>

#include <cmath>
#include <cstdio>

namespace biped {

constexpr float kVxMin = 0.08f;
constexpr float kVxMax = 0.12f;
constexpr float kVxNeutral = 0.10f;
constexpr float kDeadzone = 0.08f;

struct GamepadCommand {
  float vx = kVxNeutral;
  float vy = 0.0f;         // ALWAYS zero: policy never trained with nonzero vy.
  float yaw_rate = 0.0f;   // ALWAYS zero: frozen yaw-rate command channel.
  bool reset_pressed = false;
  bool pause_pressed = false;
  bool push_pressed = false;
  bool camera_cycle_pressed = false;
  bool log_toggle_pressed = false;
};

class Gamepad {
 public:
  Gamepad() {
    // SDL_INIT_EVENTS is needed to receive SDL_JOYDEVICEADDED/REMOVED via
    // SDL_PollEvent even before any joystick is opened.
    if (SDL_InitSubSystem(SDL_INIT_JOYSTICK | SDL_INIT_EVENTS) != 0) {
      std::fprintf(stderr, "[gamepad] SDL_Init(JOYSTICK|EVENTS) failed: %s\n", SDL_GetError());
      return;
    }
    initialized_ = true;
    // SDL_INIT_JOYSTICK enumerates devices present at init automatically as
    // synthetic SDL_JOYDEVICEADDED events on the first SDL_PollEvent pump
    // (SDL2 behavior) -- but also try an explicit open here so HasHardware()
    // is correct even if the caller never calls Poll().
    const int n = SDL_NumJoysticks();
    std::printf("[gamepad] SDL joystick count at startup: %d\n", n);
    if (n > 0) OpenDevice(0);
    if (!joystick_) {
      std::printf("[gamepad] no joystick connected; using neutral command (vx=%.2f)\n",
                  kVxNeutral);
    }
  }

  ~Gamepad() {
    if (joystick_) SDL_JoystickClose(joystick_);
    if (initialized_) SDL_QuitSubSystem(SDL_INIT_JOYSTICK | SDL_INIT_EVENTS);
  }

  GamepadCommand Poll() {
    GamepadCommand cmd;
    if (!initialized_) return cmd;  // neutral fallback

    PumpHotplugEvents();
    SDL_JoystickUpdate();
    ++poll_count_;
    if (!joystick_) return cmd;  // neutral fallback, no hardware

    // Left stick vertical (axis 1), matching gamepad_reader.py's convention:
    // pushed forward (negative raw axis) -> increase vx.
    constexpr int kAxisIndex = 1;
    const float raw = SDL_JoystickGetAxis(joystick_, kAxisIndex) / 32768.0f;
    const float dz = ApplyDeadzone(raw, kDeadzone);
    const float stick_forward = -dz;  // forward push -> positive
    // Map [-1,1] -> [kVxMin, kVxMax], neutral stick (0) -> kVxNeutral.
    if (stick_forward >= 0.0f) {
      cmd.vx = kVxNeutral + stick_forward * (kVxMax - kVxNeutral);
    } else {
      cmd.vx = kVxNeutral + stick_forward * (kVxNeutral - kVxMin);
    }
    if (cmd.vx < kVxMin) cmd.vx = kVxMin;
    if (cmd.vx > kVxMax) cmd.vx = kVxMax;
    cmd.vy = 0.0f;
    cmd.yaw_rate = 0.0f;

    // Diagnostics: axis index/raw/filtered/vx, throttled to ~once every 5s
    // at 50Hz (every 250 polls) so a 60s run doesn't flood stdout with 3000
    // near-identical lines.
    if (poll_count_ % 250 == 1) {
      std::printf("[gamepad] axis=%d raw=%.4f filtered=%.4f vx=%.4f\n", kAxisIndex, raw, dz,
                  cmd.vx);
    }

    const int nbuttons = SDL_JoystickNumButtons(joystick_);
    auto btn = [&](int i) { return i < nbuttons && SDL_JoystickGetButton(joystick_, i) != 0; };
    cmd.reset_pressed = btn(0);
    cmd.pause_pressed = btn(1);
    cmd.push_pressed = btn(2);
    cmd.camera_cycle_pressed = btn(3);
    cmd.log_toggle_pressed = btn(4);
    return cmd;
  }

  bool HasHardware() const { return joystick_ != nullptr; }

 private:
  static float ApplyDeadzone(float x, float dz) { return (std::abs(x) < dz) ? 0.0f : x; }

  void OpenDevice(int device_index) {
    joystick_ = SDL_JoystickOpen(device_index);
    if (joystick_) {
      std::printf("[gamepad] opened joystick (device_index=%d): %s (axes=%d buttons=%d)\n",
                  device_index, SDL_JoystickName(joystick_), SDL_JoystickNumAxes(joystick_),
                  SDL_JoystickNumButtons(joystick_));
    }
  }

  // Pumps SDL's event queue once and reacts to hot-plug device add/remove
  // events. Only the currently-open joystick's removal is handled by
  // clearing joystick_ (falling back to the neutral command); a newly-added
  // device is opened only if none is currently open, matching the
  // single-controller assumption elsewhere in this file.
  void PumpHotplugEvents() {
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
      if (e.type == SDL_JOYDEVICEADDED) {
        // e.jdevice.which is a DEVICE INDEX for this event type.
        std::printf("[gamepad] hot-plug: device added (index=%d)\n", e.jdevice.which);
        if (!joystick_) OpenDevice(e.jdevice.which);
      } else if (e.type == SDL_JOYDEVICEREMOVED) {
        // e.jdevice.which is an INSTANCE ID for this event type.
        if (joystick_ && SDL_JoystickInstanceID(joystick_) == e.jdevice.which) {
          std::printf("[gamepad] hot-plug: active joystick removed (instance_id=%d); "
                      "falling back to neutral command (vx=%.2f)\n",
                      e.jdevice.which, kVxNeutral);
          SDL_JoystickClose(joystick_);
          joystick_ = nullptr;
        } else {
          std::printf("[gamepad] hot-plug: a non-active joystick was removed (instance_id=%d)\n",
                      e.jdevice.which);
        }
      }
    }
  }

  bool initialized_ = false;
  SDL_Joystick* joystick_ = nullptr;
  int poll_count_ = 0;
};

}  // namespace biped
