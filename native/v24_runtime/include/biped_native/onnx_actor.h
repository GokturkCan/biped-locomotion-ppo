// ONNX Runtime C++ wrapper around actor.onnx (57 -> 12, normalization
// embedded in the graph). CPUExecutionProvider only, per the approved plan.
#pragma once

#include <array>
#include <memory>
#include <string>

#include <onnxruntime_cxx_api.h>

#include "biped_native/runtime_layout.h"

namespace biped {

class OnnxActor {
 public:
  explicit OnnxActor(const std::string& onnx_path);

  // Runs a single [1,57] -> [1,12] forward pass.
  std::array<float, kActionSize> Run(const std::array<float, kObsSize>& obs);

  // Batched [N,57] -> [N,12], N contiguous rows.
  std::vector<float> RunBatch(const std::vector<float>& obs_batch, int n);

  std::string InputName() const { return input_name_; }
  std::string OutputName() const { return output_name_; }
  std::vector<std::string> Providers() const;

 private:
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> session_;
  Ort::MemoryInfo memory_info_;
  std::string input_name_;
  std::string output_name_;
};

}  // namespace biped
