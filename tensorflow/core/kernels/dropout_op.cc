/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/


#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/kernels/random_op.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/guarded_philox_random.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/stream_executor/temporary_device_memory.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "dropout_op.h"

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename T>
void ApplyDropout<CPUDevice, T>::operator()(const CPUDevice& d, T* out,
                                            const T* in, const float* rng_data,
                                            float rate, uint64 num_elements, random::PhiloxRandom gen) {
  T scale = T(1. / (1. - rate));
  for (uint64 i = 0; i < num_elements; i++) {
    out[i] = (rng_data[i] > rate) ? in[i] * scale : T(0.0);
  }
}

template <typename T>
void ApplyDropoutGrad<CPUDevice, T>::operator()(const CPUDevice& d, T* outgrads,
                                                const T* grads, const T* ins,
                                                const T* outs, float rate,
                                                uint64 num_elements) {
  T scale = T(1. / (1 - rate));
  for (uint64 i = 0; i < num_elements; i++) {
    outgrads[i] = (outs[i] == T(0)) ? T(0) : grads[i] * scale;
  }
};

template <typename Device, typename T>
class DropoutOp : public OpKernel {
 private:
  // todo: may be sufficient to use random::PhiloxRandom, since we don't
  // require Compute() to be reentrant
  GuardedPhiloxRandom generator_;

 public:
  explicit DropoutOp(OpKernelConstruction* context) : OpKernel(context) {
    generator_.Init(0, 0);
  }

  ~DropoutOp() override {}

  void Compute(OpKernelContext* ctx) override {
    // printf("Dropout %s\n",
    // std::is_same<Device,CPUDevice>::value?"CPU":"GPU");
    const Tensor& in0 = ctx->input(0);

    const Tensor& in1 = ctx->input(1);
    OP_REQUIRES(ctx, in0.dtype() == in1.dtype(),
                errors::InvalidArgument(
                    "Dropout rate must be same type as input tensor."));
    OP_REQUIRES(
        ctx, in1.dims() == 0,
        errors::InvalidArgument("Dropout rate must be a scalar tensor."));
    float rate = static_cast<float>(in1.scalar<T>()());

    const Tensor& in2 = ctx->input(2);
    OP_REQUIRES(ctx, in0.dims() == in2.shape().num_elements(),
                errors::InvalidArgument("MIOpen only supports input dimensions "
                                        "to match noise dimensions."));
    // Allocate output, and exit early if possible
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, in0.shape(), &output));
    if (output->NumElements() == 0) return;

    const Tensor& in3 = ctx->input(3);
    OP_REQUIRES(
        ctx, in3.dims() == 0,
        errors::InvalidArgument("Dropout seed must be a scalar tensor."));
    int64 seed = 0;
    if (in3.dtype() == DT_INT32)
      seed = in3.scalar<int32>()();
    else
      seed = in3.scalar<int64>()();
    // don't reset the seed for every call unless it is explicitly non-0
    if (seed != 0) generator_.ResetSeeds(seed, 0);

    typedef random::UniformDistribution<random::PhiloxRandom, float>
        Distribution;
    Distribution dist;
    random::PhiloxRandom gen =
      generator_.ReserveRandomOutputs(in0.NumElements(), 256);

    if (std::is_same<Device, CPUDevice>::value) {
      Tensor rng_data;
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_temp(DataTypeToEnum<float>::value,
                                        TensorShape(in0.shape()), &rng_data));
      auto rng_flat = rng_data.flat<float>();

      functor::FillPhiloxRandom<Device, Distribution>()(
          ctx, ctx->eigen_device<Device>(), gen, rng_flat.data(),
          rng_flat.size(), dist);
      ApplyDropout<Device, T>()(ctx->eigen_device<Device>(),
                                output->flat<T>().data(), in0.flat<T>().data(),
                                rng_flat.data(), rate, in0.NumElements(), gen);
    } else {
      ApplyDropout<Device, T>()(ctx->eigen_device<Device>(),
                                output->flat<T>().data(), in0.flat<T>().data(),
                                nullptr, rate, in0.NumElements(), gen);
    }
  }
};

#define REGISTER_DROPOUT(TYPE)                                      \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Dropout").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      DropoutOp<CPUDevice, TYPE>);

TF_CALL_double(REGISTER_DROPOUT);
TF_CALL_float(REGISTER_DROPOUT);
TF_CALL_half(REGISTER_DROPOUT);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
typedef Eigen::GpuDevice GPUDevice;

#define REGISTER_DROPOUT_GPU(TYPE)                        \
  REGISTER_KERNEL_BUILDER(Name("Dropout")                 \
                              .Device(DEVICE_GPU)         \
                              .TypeConstraint<TYPE>("T")  \
                              .HostMemory("rate")         \
                              .HostMemory("seed")         \
                              .HostMemory("noise_shape"), \
                          DropoutOp<GPUDevice, TYPE>);

TF_CALL_double(REGISTER_DROPOUT_GPU);
TF_CALL_float(REGISTER_DROPOUT_GPU);
TF_CALL_half(REGISTER_DROPOUT_GPU);
#endif

template <typename Device, typename T>
class DropoutGradOp : public OpKernel {
 public:
  explicit DropoutGradOp(OpKernelConstruction* context) : OpKernel(context) {
  }

  ~DropoutGradOp() override {}
  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);
    OP_REQUIRES(ctx, in0.dtype() == in1.dtype(),
                errors::InvalidArgument(
                    "Dropout rate must be same type as input tensor."));
    OP_REQUIRES(
        ctx, in1.dims() == 0,
        errors::InvalidArgument("Dropout rate must be a scalar tensor."));
    float rate = static_cast<float>(in1.scalar<T>()());

    const Tensor& in2 = ctx->input(2);
    OP_REQUIRES(ctx, in0.dims() == in2.shape().num_elements(),
                errors::InvalidArgument("MIOpen only supports input dimensions "
                                        "to match noise dimensions."));
    const Tensor& inputs = ctx->input(3);
    OP_REQUIRES(ctx, in0.NumElements() == inputs.NumElements(),
                errors::InvalidArgument("ROCm DropoutGrad dim mismatch"));

    const Tensor& outputs = ctx->input(4);
    OP_REQUIRES(ctx, in0.NumElements() == outputs.NumElements(),
                errors::InvalidArgument("ROCm DropoutGrad dim mismatch"));

    // Allocate output, and exit early if possible
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, in0.shape(), &output));
    if (output->NumElements() == 0) return;

    ApplyDropoutGrad<Device, T>()(
        ctx->eigen_device<Device>(), output->flat<T>().data(),
        in0.flat<T>().data(),  // gradients
        inputs.flat<T>().data(), outputs.flat<T>().data(), rate,
        in0.NumElements());
  }
};

#define REGISTER_DROPOUT_GRAD_CPU(TYPE)                                 \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("DropoutGrad").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      DropoutGradOp<CPUDevice, TYPE>);

TF_CALL_double(REGISTER_DROPOUT_GRAD_CPU);
TF_CALL_float(REGISTER_DROPOUT_GRAD_CPU);
TF_CALL_half(REGISTER_DROPOUT_GRAD_CPU);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_DROPOUT_GRAD_GPU(TYPE)                   \
  REGISTER_KERNEL_BUILDER(Name("DropoutGrad")             \
                              .Device(DEVICE_GPU)         \
                              .TypeConstraint<TYPE>("T")  \
                              .HostMemory("rate")         \
                              .HostMemory("noise_shape"), \
                          DropoutGradOp<GPUDevice, TYPE>);

TF_CALL_double(REGISTER_DROPOUT_GRAD_GPU);
TF_CALL_float(REGISTER_DROPOUT_GRAD_GPU);
TF_CALL_half(REGISTER_DROPOUT_GRAD_GPU);
#endif

}  // namespace tensorflow
