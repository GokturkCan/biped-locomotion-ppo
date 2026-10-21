// Gamepad hardware-or-fallback test: rigorously exercises whichever path
// applies in the current environment. If a joystick is physically present,
// it prints the connected-hardware diagnostics (axis/raw/filtered/vx) for
// manual verification; if not (the usual CI case), it checks the NEUTRAL
// FALLBACK path: no hang, no crash, deterministic neutral command,
// HasHardware()==false, and all button-derived flags false. Only the fallback
// path has been exercised; behaviour with physical hardware is untested.
//
// Usage: test_gamepad
#include <cstdio>
#include <string>

#include "biped_native/gamepad.h"

namespace {
int g_checks = 0;
int g_failures = 0;

void Check(bool cond, const std::string& msg) {
  ++g_checks;
  if (!cond) {
    ++g_failures;
    std::printf("[test_gamepad] FAIL: %s\n", msg.c_str());
  }
}
}  // namespace

int main() {
  biped::Gamepad gamepad;
  const bool has_hardware = gamepad.HasHardware();
  std::printf("[test_gamepad] HasHardware()=%s\n", has_hardware ? "true" : "false");

  if (!has_hardware) {
    std::printf("[test_gamepad] no hardware present in this environment -- exercising the "
                "NEUTRAL FALLBACK path rigorously.\n");
    for (int i = 0; i < 10; ++i) {
      auto cmd = gamepad.Poll();
      Check(cmd.vx == biped::kVxNeutral,
            "fallback vx must be exactly kVxNeutral every poll (i=" + std::to_string(i) + ")");
      Check(cmd.vy == 0.0f, "fallback vy must always be 0");
      Check(cmd.yaw_rate == 0.0f, "fallback yaw_rate must always be 0");
      Check(!cmd.reset_pressed && !cmd.pause_pressed && !cmd.push_pressed &&
                !cmd.camera_cycle_pressed && !cmd.log_toggle_pressed,
            "fallback command must report no buttons pressed (i=" + std::to_string(i) + ")");
    }
    std::printf("[test_gamepad] 10 consecutive Poll() calls with no hardware: no hang, no "
                "crash, deterministic neutral command confirmed.\n");
  } else {
    std::printf("[test_gamepad] hardware IS present -- polling 10 times, printing live "
                "diagnostics for manual verification (see gamepad.h's throttled axis/raw/"
                "filtered/vx prints above).\n");
    for (int i = 0; i < 10; ++i) {
      auto cmd = gamepad.Poll();
      Check(cmd.vx >= biped::kVxMin && cmd.vx <= biped::kVxMax,
            "live vx must stay within the trained [kVxMin,kVxMax] range (i=" +
                std::to_string(i) + ")");
      Check(cmd.vy == 0.0f, "vy must always be 0 even with hardware connected (frozen contract)");
      Check(cmd.yaw_rate == 0.0f,
            "yaw_rate must always be 0 even with hardware connected (frozen contract)");
    }
  }

  std::printf("[test_gamepad] %d checks, %d failures\n", g_checks, g_failures);
  std::printf("[test_gamepad] %s\n", g_failures == 0 ? "PASS" : "FAIL");
  return g_failures == 0 ? 0 : 1;
}
