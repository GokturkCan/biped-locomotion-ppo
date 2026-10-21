#include "biped_native/runtime_bindings.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "biped_native/json_lite.h"

namespace biped {
namespace {

constexpr const char* kPlaceholder = "REPLACE_ME";

const json::Value& Need(const json::Value& obj, const std::string& key, const std::string& where) {
  if (!obj.IsObject() || !obj.Has(key)) {
    throw std::runtime_error("runtime_bindings: missing field '" + where + key + "'");
  }
  return obj.obj.at(key);
}

std::string NeedString(const json::Value& obj, const std::string& key, const std::string& where) {
  const json::Value& v = Need(obj, key, where);
  if (!v.IsString() || v.str.empty() || v.str.find(kPlaceholder) != std::string::npos) {
    throw std::runtime_error("runtime_bindings: '" + where + key +
                             "' must be a non-empty string (template placeholder not filled?)");
  }
  return v.str;
}

double NeedNumber(const json::Value& obj, const std::string& key, const std::string& where) {
  const json::Value& v = Need(obj, key, where);
  if (!v.IsNumber() || !std::isfinite(v.num)) {
    throw std::runtime_error("runtime_bindings: '" + where + key +
                             "' must be a finite number (template placeholder not filled?)");
  }
  return v.num;
}

const json::Value& NeedArray(const json::Value& obj, const std::string& key, size_t size) {
  const json::Value& v = Need(obj, key, "");
  if (!v.IsArray()) throw std::runtime_error("runtime_bindings: '" + key + "' must be an array");
  if (size != 0 && v.arr.size() != size) {
    throw std::runtime_error("runtime_bindings: '" + key + "' must have " + std::to_string(size) +
                             " entries, found " + std::to_string(v.arr.size()));
  }
  if (v.arr.empty()) throw std::runtime_error("runtime_bindings: '" + key + "' is empty");
  return v;
}

}  // namespace

RuntimeBindings ParseRuntimeBindings(const std::string& json_text) {
  const json::Value root = json::Parse(json_text);
  if (!root.IsObject()) throw std::runtime_error("runtime_bindings: top level must be an object");

  RuntimeBindings b;
  const json::Value& names = NeedArray(root, "joint_names", kNumJoints);
  for (int i = 0; i < kNumJoints; ++i) {
    const json::Value& n = names.arr[i];
    if (!n.IsString() || n.str.empty() || n.str.find(kPlaceholder) != std::string::npos) {
      throw std::runtime_error("runtime_bindings: joint_names[" + std::to_string(i) +
                               "] must be a non-empty string (placeholder not filled?)");
    }
    for (int j = 0; j < i; ++j) {
      if (b.joint_names[j] == n.str) {
        throw std::runtime_error("runtime_bindings: duplicate joint name '" + n.str + "'");
      }
    }
    b.joint_names[i] = n.str;
  }

  const json::Value& sensors = Need(root, "sensors", "");
  b.sensor_orientation = NeedString(sensors, "orientation", "sensors.");
  b.sensor_gravity = NeedString(sensors, "gravity", "sensors.");
  b.sensor_local_linvel = NeedString(sensors, "local_linvel", "sensors.");
  b.sensor_gyro = NeedString(sensors, "gyro", "sensors.");
  b.sensor_root_pos = NeedString(sensors, "root_pos", "sensors.");

  const json::Value& geoms = Need(root, "geoms", "");
  b.floor_geom = NeedString(geoms, "floor", "geoms.");
  b.left_foot_geom = NeedString(geoms, "left_foot", "geoms.");
  b.right_foot_geom = NeedString(geoms, "right_foot", "geoms.");
  b.root_body = NeedString(root, "root_body", "");

  const json::Value& scale = NeedArray(root, "action_scale", kActionSize);
  for (int i = 0; i < kActionSize; ++i) {
    if (!scale.arr[i].IsNumber() || !(scale.arr[i].num > 0.0) || !std::isfinite(scale.arr[i].num)) {
      throw std::runtime_error("runtime_bindings: action_scale[" + std::to_string(i) +
                               "] must be a finite positive number");
    }
    b.action_scale[i] = static_cast<float>(scale.arr[i].num);
  }

  const json::Value& q0 = NeedArray(root, "initial_qpos", 0);
  for (size_t i = 0; i < q0.arr.size(); ++i) {
    if (!q0.arr[i].IsNumber() || !std::isfinite(q0.arr[i].num)) {
      throw std::runtime_error("runtime_bindings: initial_qpos[" + std::to_string(i) +
                               "] must be a finite number");
    }
    b.initial_qpos.push_back(q0.arr[i].num);
  }

  b.target_height = NeedNumber(root, "target_height", "");
  b.min_root_height = NeedNumber(root, "min_root_height", "");
  if (!(b.min_root_height > 0.0) || !(b.target_height > b.min_root_height)) {
    throw std::runtime_error(
        "runtime_bindings: require 0 < min_root_height < target_height (metres)");
  }
  return b;
}

RuntimeBindings LoadRuntimeBindings(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("runtime_bindings: cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ParseRuntimeBindings(ss.str());
}

}  // namespace biped
