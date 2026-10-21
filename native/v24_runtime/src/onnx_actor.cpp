#include "biped_native/onnx_actor.h"

#include <stdexcept>

namespace biped {

OnnxActor::OnnxActor(const std::string& onnx_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "actor"),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
  session_options_.SetIntraOpNumThreads(1);
  session_options_.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
  // CPUExecutionProvider is the default/only provider when no EP is
  // explicitly appended; this is the mandatory initial backend per the plan.
  session_ = std::make_unique<Ort::Session>(env_, onnx_path.c_str(), session_options_);

  Ort::AllocatorWithDefaultOptions allocator;
  if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 1) {
    throw std::runtime_error("onnx_actor: expected exactly 1 input and 1 output");
  }
  auto in_name = session_->GetInputNameAllocated(0, allocator);
  auto out_name = session_->GetOutputNameAllocated(0, allocator);
  input_name_ = in_name.get();
  output_name_ = out_name.get();

  auto in_shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  auto out_shape = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  if (in_shape.size() != 2 || in_shape[1] != kObsSize) {
    throw std::runtime_error("onnx_actor: input shape != [N," + std::to_string(kObsSize) + "]");
  }
  if (out_shape.size() != 2 || out_shape[1] != kActionSize) {
    throw std::runtime_error("onnx_actor: output shape != [N," + std::to_string(kActionSize) + "]");
  }
}

std::vector<float> OnnxActor::RunBatch(const std::vector<float>& obs_batch, int n) {
  if (static_cast<int>(obs_batch.size()) != n * kObsSize) {
    throw std::runtime_error("onnx_actor: obs_batch size mismatch");
  }
  std::array<int64_t, 2> in_shape{n, kObsSize};
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info_, const_cast<float*>(obs_batch.data()), obs_batch.size(),
      in_shape.data(), in_shape.size());

  const char* input_names[] = {input_name_.c_str()};
  const char* output_names[] = {output_name_.c_str()};
  auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
                                output_names, 1);
  const float* out_data = outputs[0].GetTensorData<float>();
  auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
  if (out_shape.size() != 2 || out_shape[0] != n || out_shape[1] != kActionSize) {
    throw std::runtime_error("onnx_actor: unexpected output shape at runtime");
  }
  return std::vector<float>(out_data, out_data + n * kActionSize);
}

std::array<float, kActionSize> OnnxActor::Run(const std::array<float, kObsSize>& obs) {
  std::vector<float> in(obs.begin(), obs.end());
  std::vector<float> out = RunBatch(in, 1);
  std::array<float, kActionSize> result{};
  std::copy(out.begin(), out.end(), result.begin());
  return result;
}

std::vector<std::string> OnnxActor::Providers() const {
  // ORT C++ API 1.27 does not expose active-session providers directly via a
  // simple getter; CPUExecutionProvider is guaranteed present as the default
  // fallback provider for every session, and no other EP is registered in
  // session_options_ here, so this session runs on CPUExecutionProvider only.
  return {"CPUExecutionProvider"};
}

}  // namespace biped
