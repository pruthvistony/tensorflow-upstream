/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_DROPOUT_OP_H_
#define TENSORFLOW_CORE_KERNELS_DROPOUT_OP_H_

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
struct ApplyDropout {
  void operator()(const Device& d, T* out, const T* in, const float* rng_data,
                  float rate, uint64 num_elements, random::PhiloxRandom gen) {}
};

template <typename T>
struct ApplyDropout<CPUDevice, T> {
  void operator()(const CPUDevice& d, T* out, const T* in,
                  const float* rng_data, float rate, uint64 num_elements,
                  random::PhiloxRandom gen);
};

template <typename Device, typename T>
struct ApplyDropoutGrad {
  void operator()(const Device& d, T* outgrads, const T* grads, const T* ins,
                  const T* outs, float rate, uint64 num_elements) {}
};

template <typename T>
struct ApplyDropoutGrad<CPUDevice, T> {
  void operator()(const CPUDevice& d, T* outgrads, const T* grads, const T* ins,
                  const T* outs, float rate, uint64 num_elements);
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
typedef Eigen::GpuDevice GPUDevice;

template <typename T>
struct ApplyDropout<GPUDevice, T> {
  void operator()(const GPUDevice& d, T* out, const T* in,
                  const float* rng_data, float rate, uint64 num_elements,
                  random::PhiloxRandom gen);
};

template <typename T>
struct ApplyDropoutGrad<GPUDevice, T> {
  void operator()(const GPUDevice& d, T* outgrads, const T* grads, const T* ins,
                  const T* outs, float rate, uint64 num_elements);
};
};
#endif

#endif