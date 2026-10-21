// Tiny shared check helpers for the native test executables.
#pragma once

#include <cmath>
#include <cstdio>
#include <string>

namespace testing {

struct Counter {
  int checks = 0;
  int failures = 0;
};

inline Counter& Global() {
  static Counter c;
  return c;
}

inline void Check(bool cond, const std::string& msg) {
  ++Global().checks;
  if (!cond) {
    ++Global().failures;
    std::printf("FAIL: %s\n", msg.c_str());
  }
}

inline void CheckNear(double actual, double expected, double tol, const std::string& msg) {
  Check(std::fabs(actual - expected) <= tol,
        msg + " (actual=" + std::to_string(actual) + " expected=" + std::to_string(expected) + ")");
}

// Returns the process exit code and prints a one-line summary.
inline int Finish(const char* name) {
  std::printf("[%s] %d checks, %d failures -> %s\n", name, Global().checks, Global().failures,
              Global().failures == 0 ? "PASS" : "FAIL");
  return Global().failures == 0 ? 0 : 1;
}

}  // namespace testing
