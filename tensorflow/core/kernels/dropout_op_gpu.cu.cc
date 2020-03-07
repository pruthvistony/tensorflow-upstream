#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/kernels/random_op.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"
#include "tensorflow/core/util/guarded_philox_random.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/stream_executor/temporary_device_memory.h"
#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "dropout_op.h"

#if TENSORFLOW_USE_ROCM
#include "rocm/include/hip/hip_fp16.h"
#endif

namespace tensorflow {

template <typename T>
__device__ void apply_dropout(T& out, T in, float rng, float rate,
                              float scale) {
  out = in * (rng > rate ? T(scale) : T(0.0f));
}

__device__ void apply_dropout(__half2& out, __half2 in, __half2 rng,
                              __half2 rate, __half2 scale) {
  __half2 mask = __hgt2(rng, rate);
  out = in * mask * scale;
}

template <typename T, typename U>
__global__ void RNGAndApplyDropoutKernel(random::PhiloxRandom gen, int64 size,
                                         T* _out, const T* _in, U rate,
                                         U scale) {
  constexpr bool is_half = std::is_same<T, Eigen::half>::value;
  constexpr bool is_half2 = std::is_same<T, __half2>::value;

  // The RNG only knows how to generate half or float
  typedef
      typename std::conditional<is_half2, Eigen::half, float>::type DistGenType;

  // but, if we're running in half2, we'll cast its outputs pairwise to half2
  // too
  typedef
      typename std::conditional<is_half2, __half2, float>::type DistApplyType;

  // Cast inputs from Eigen::half to __half. TODO: is there a better way of
  // doing this?
  typedef typename std::conditional<is_half, half, T>::type TT;
  TT* out = reinterpret_cast<TT*>(_out);
  const TT* in = reinterpret_cast<const TT*>(_in);
  typedef random::UniformDistribution<random::PhiloxRandom, DistGenType> Dist;
  Dist dist;
  static_assert(Dist::kVariableSamplesPerOutput == false,
                "Wrong kVariableSamplesPerOutput");
  static_assert(
      Dist::kResultElementCount == 4 || Dist::kResultElementCount == 8,
      "wrong kResultElementCount");

  // in half2 mode, RNG produces 8 half per call which we convert into 4 half2
  const int kGroupSize = Dist::kResultElementCount / (is_half2 ? 2 : 1);

  const int32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int32 total_thread_count = gridDim.x * blockDim.x;
  int32 offset = thread_id * kGroupSize;
  gen.Skip(thread_id);

  while (offset + kGroupSize <= size) {
    const typename Dist::ResultType samples = dist(&gen);
    const DistApplyType* ps =
        reinterpret_cast<const DistApplyType*>(&samples[0]);
    for (int i = 0; i < kGroupSize; ++i)
      apply_dropout(out[offset + i], in[offset + i], ps[i], rate, scale);

    offset += total_thread_count * kGroupSize;
    gen.Skip(total_thread_count - 1);
  }

  typename Dist::ResultType samples = dist(&gen);
  const DistApplyType* ps = reinterpret_cast<const DistApplyType*>(&samples[0]);
  for (int i = 0; i < kGroupSize; ++i) {
    if (offset >= size) return;
    apply_dropout(out[offset], in[offset], ps[i], rate, scale);
    ++offset;
  }
}

template <typename T>
__global__ void ApplyDropoutGradKernel(T* outgrads, const T* grads,
                                       const T* ins, const T* outs, float rate,
                                       float scale, uint64 num_elements) {
  for (uint64 i = threadIdx.x + blockIdx.x * blockDim.x; i < num_elements;
       i += blockDim.x * gridDim.x)
    outgrads[i] = grads[i] * T((outs[i] == T(0)) ? 0.0f : scale);
}

template <>
__global__ void ApplyDropoutGradKernel(Eigen::half* _outgrads,
                                       const Eigen::half* _grads,
                                       const Eigen::half* _ins,
                                       const Eigen::half* _outs, float rate,
                                       float scale, uint64 num_elements) {
  __half* outgrads = reinterpret_cast<__half*>(_outgrads);
  const __half* grads = reinterpret_cast<const __half*>(_grads);
  const __half* outs = reinterpret_cast<const __half*>(_outs);
  for (uint64 i = threadIdx.x + blockIdx.x * blockDim.x; i < num_elements;
       i += blockDim.x * gridDim.x)
    outgrads[i] = __float2half(
        (outs[i] == __half(0.0f)) ? 0.0f : __half2float(grads[i]) * scale);
}

template <typename T>
void ApplyDropout<GPUDevice, T>::operator()(const GPUDevice& d, T* out,
                                            const T* in, const float* rng_data,
                                            float rate, uint64 num_elements,
                                            random::PhiloxRandom gen) {
  float scale = 1. / (1 - rate);
  bool do_half2 = std::is_same<T, Eigen::half>::value && !(num_elements & 1);
  if (do_half2) num_elements /= 2;
  int64 kThreadInBlock = 256;
  int64 kMaxBlock = do_half2 ? 1024 : 128;  // experimental best
  uint64 num_groups =
      (num_elements + random::PhiloxRandom::kResultElementCount - 1) /
      random::PhiloxRandom::kResultElementCount;

  uint64 num_blocks = (num_groups + kThreadInBlock - 1) / kThreadInBlock;
  num_blocks = min(kMaxBlock, num_blocks);
  if (do_half2) {
    TF_CHECK_OK(GpuLaunchKernel(
        RNGAndApplyDropoutKernel<__half2, __half2>, num_blocks, kThreadInBlock,
        0, d.stream(), gen, num_elements, reinterpret_cast<__half2*>(out),
        reinterpret_cast<const __half2*>(in), __floats2half2_rn(rate, rate),
        __floats2half2_rn(scale, scale)));
  } else {
    TF_CHECK_OK(GpuLaunchKernel(RNGAndApplyDropoutKernel<T, float>, num_blocks,
                                kThreadInBlock, 0, d.stream(), gen,
                                num_elements, out, in, rate, scale));
        }
}

template <typename T>
void ApplyDropoutGrad<GPUDevice, T>::operator()(const GPUDevice& d, T* outgrads,
                                                const T* grads, const T* ins,
                                                const T* outs, float rate,
                                                uint64 num_elements) {
  float scale = 1. / (1 - rate);
  constexpr int32 kThreadInBlock = 256;
  int64 kMaxBlock = 128;  // todo: benchmarking
  TF_CHECK_OK(GpuLaunchKernel(
      ApplyDropoutGradKernel<T>,
      min(kMaxBlock, (num_elements + kThreadInBlock - 1) / kThreadInBlock),
      kThreadInBlock, 0, d.stream(), outgrads, grads, ins, outs, rate, scale,
      num_elements));
}

template void ApplyDropout<GPUDevice, Eigen::half>::operator()(
    const GPUDevice& d, Eigen::half* out, const Eigen::half* in,
    const float* rng_data, float rate, uint64 num_elements,
    random::PhiloxRandom gen);
template void ApplyDropout<GPUDevice, float>::operator()(
    const GPUDevice& d, float* out, const float* in, const float* rng_data,
    float rate, uint64 num_elements, random::PhiloxRandom gen);
template void ApplyDropout<GPUDevice, double>::operator()(
    const GPUDevice& d, double* out, const double* in, const float* rng_data,
    float rate, uint64 num_elements, random::PhiloxRandom gen);

template void ApplyDropoutGrad<GPUDevice, Eigen::half>::operator()(
    const GPUDevice& d, Eigen::half* outgrads, const Eigen::half* grads,
    const Eigen::half* ins, const Eigen::half* outs, float rate,
    uint64 num_elements);
template void ApplyDropoutGrad<GPUDevice, float>::operator()(
    const GPUDevice& d, float* outgrads, const float* grads, const float* ins,
    const float* outs, float rate, uint64 num_elements);
template void ApplyDropoutGrad<GPUDevice, double>::operator()(
    const GPUDevice& d, double* outgrads, const double* grads,
    const double* ins, const double* outs, float rate, uint64 num_elements);

};  // namespace tensorflow

#endif
