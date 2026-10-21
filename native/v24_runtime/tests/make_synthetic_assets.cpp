// Writes the synthetic model (as .mjb) and a matching runtime-bindings JSON so the
// closed-loop app can be smoke-tested end to end without any restricted asset.
//
// Usage: make_synthetic_assets <output_dir>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "synthetic_model.h"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <output_dir>\n", argv[0]);
    return 2;
  }
  const std::filesystem::path dir = argv[1];
  std::filesystem::create_directories(dir);
  std::filesystem::create_directories(dir / "out");
  mjModel* m = synthetic::MakeModel();
  mj_saveModel(m, (dir / "synthetic_model.mjb").string().c_str(), nullptr, 0);
  const biped::RuntimeBindings b = synthetic::MakeBindings(m);

  std::ofstream f(dir / "synthetic_bindings.json");
  f << "{\n  \"joint_names\": [";
  for (int i = 0; i < biped::kNumJoints; ++i) f << (i ? ", " : "") << "\"" << b.joint_names[i] << "\"";
  f << "],\n  \"sensors\": {\"orientation\": \"" << b.sensor_orientation << "\", \"gravity\": \""
    << b.sensor_gravity << "\", \"local_linvel\": \"" << b.sensor_local_linvel << "\", \"gyro\": \""
    << b.sensor_gyro << "\", \"root_pos\": \"" << b.sensor_root_pos << "\"},\n"
    << "  \"geoms\": {\"floor\": \"" << b.floor_geom << "\", \"left_foot\": \"" << b.left_foot_geom
    << "\", \"right_foot\": \"" << b.right_foot_geom << "\"},\n"
    << "  \"root_body\": \"" << b.root_body << "\",\n  \"action_scale\": [";
  for (int i = 0; i < biped::kActionSize; ++i) f << (i ? ", " : "") << b.action_scale[i];
  f << "],\n  \"initial_qpos\": [";
  for (size_t i = 0; i < b.initial_qpos.size(); ++i) f << (i ? ", " : "") << b.initial_qpos[i];
  f << "],\n  \"target_height\": " << b.target_height << ",\n  \"min_root_height\": "
    << b.min_root_height << "\n}\n";
  mj_deleteModel(m);
  std::printf("wrote synthetic assets to %s\n", dir.string().c_str());
  return 0;
}
