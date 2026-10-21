// Contract tests for runtime_bindings.cpp / json_lite.cpp: valid parse, every
// kind of malformed input and the unfilled-template case. No model needed.
//
// Usage: test_runtime_bindings [path/to/runtime_bindings.template.json]
#include <stdexcept>
#include <string>

#include "biped_native/json_lite.h"
#include "biped_native/runtime_bindings.h"
#include "check.h"

namespace {

using testing::Check;

std::string ValidJson() {
  std::string joints, scale;
  for (int i = 0; i < biped::kNumJoints; ++i) {
    joints += std::string(i ? "," : "") + "\"j" + std::to_string(i) + "\"";
    scale += std::string(i ? "," : "") + "0.1";
  }
  return "{\"joint_names\":[" + joints + "],"
         "\"sensors\":{\"orientation\":\"o\",\"gravity\":\"g\",\"local_linvel\":\"v\","
         "\"gyro\":\"w\",\"root_pos\":\"p\"},"
         "\"geoms\":{\"floor\":\"f\",\"left_foot\":\"l\",\"right_foot\":\"r\"},"
         "\"root_body\":\"root\",\"action_scale\":[" + scale + "],"
         "\"initial_qpos\":[0.0,0.0,0.5,1.0,0.0,0.0,0.0],"
         "\"target_height\":0.5,\"min_root_height\":0.2}";
}

bool Throws(const std::string& text) {
  try {
    biped::ParseRuntimeBindings(text);
  } catch (const std::runtime_error&) {
    return true;
  }
  return false;
}

std::string Replace(std::string s, const std::string& from, const std::string& to) {
  const size_t pos = s.find(from);
  if (pos != std::string::npos) s.replace(pos, from.size(), to);
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  const biped::RuntimeBindings b = biped::ParseRuntimeBindings(ValidJson());
  Check(b.joint_names[0] == "j0" && b.joint_names[11] == "j11", "joint order preserved");
  Check(b.sensor_gravity == "g" && b.root_body == "root", "names parsed");
  Check(b.action_scale[3] > 0.099f && b.action_scale[3] < 0.101f, "action scale parsed");
  Check(b.initial_qpos.size() == 7 && b.initial_qpos[2] == 0.5, "initial qpos parsed");
  Check(b.target_height == 0.5 && b.min_root_height == 0.2, "heights parsed");

  Check(Throws("{"), "truncated json rejected");
  Check(Throws("[]"), "non-object rejected");
  Check(Throws(Replace(ValidJson(), "\"root_body\":\"root\",", "")), "missing root_body rejected");
  Check(Throws(Replace(ValidJson(), "\"j11\"", "\"j0\"")), "duplicate joint name rejected");
  Check(Throws(Replace(ValidJson(), "\"j11\"", "")), "wrong joint count rejected");
  Check(Throws(Replace(ValidJson(), "0.1,0.1", "0.1,-0.1")), "non-positive action scale rejected");
  Check(Throws(Replace(ValidJson(), "\"min_root_height\":0.2", "\"min_root_height\":0.9")),
        "min_root_height >= target_height rejected");
  Check(Throws(Replace(ValidJson(), "\"o\"", "\"REPLACE_ME\"")), "placeholder rejected");
  Check(Throws(Replace(ValidJson(), "\"target_height\":0.5", "\"target_height\":null")),
        "null number rejected");

  if (argc > 1) {  // the shipped template must NOT validate until it is filled in
    bool threw = false;
    try {
      biped::LoadRuntimeBindings(argv[1]);
    } catch (const std::runtime_error&) {
      threw = true;
    }
    Check(threw, "unfilled template must be rejected");
  }
  bool missing = false;
  try {
    biped::LoadRuntimeBindings("does_not_exist.json");
  } catch (const std::runtime_error&) {
    missing = true;
  }
  Check(missing, "missing file reports an error");
  return testing::Finish("test_runtime_bindings");
}
