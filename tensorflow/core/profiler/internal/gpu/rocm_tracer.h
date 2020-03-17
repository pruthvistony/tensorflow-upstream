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

#ifndef TENSORFLOW_CORE_PROFILER_INTERNAL_GPU_ROCM_TRACER_H_
#define TENSORFLOW_CORE_PROFILER_INTERNAL_GPU_ROCM_TRACER_H_

#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_set.h"
#include "absl/types/optional.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace profiler {

struct MemcpyDetails {
  // The amount of data copied for memcpy events.
  size_t num_bytes;
  // The destination device for peer-2-peer communication (memcpy). The source
  // device is implicit: its the current device.
  uint32 destination;
  // Whether or not the memcpy is asynchronous.
  bool async;
};

struct MemAllocDetails {
  // The amount of data requested for cudaMalloc events.
  uint64 num_bytes;
};

struct KernelDetails {
  // The amount of dynamic memory space used by a thread block.
  uint64 dynamic_shared_memory_usage;
  // X-dimension of a thread block.
  uint64 block_x;
  // Y-dimension of a thread block.
  uint64 block_y;
  // Z-dimension of a thread block.
  uint64 block_z;
  // X-dimension of a grid.
  uint64 grid_x;
  // Y-dimension of a grid.
  uint64 grid_y;
  // Z-dimension of a grid.
  uint64 grid_z;
};

enum class RocmTracerEventType {
  Unsupported = 0,
  Kernel = 1,
  MemcpyH2D = 2,
  MemcpyD2H = 3,
  MemcpyD2D = 4,
  MemcpyP2P = 5,
  MemcpyOther = 6,
  MemoryAlloc = 7,
  Generic = 100,
};

const char* GetTraceEventTypeName(const RocmTracerEventType& type);

enum class RocmTracerEventSource {
  ApiCallback = 0,
  Activity = 1,
};

enum class RocmTracerEventDomain {
  HIP = 0,
  HCC = 1,
};

struct RocmTracerEvent {
  static constexpr uint32 kInvalidDeviceId =
      std::numeric_limits<uint32_t>::max();
  static constexpr uint32 kInvalidThreadId =
      std::numeric_limits<uint32_t>::max();
  static constexpr uint32 kInvalidCorrelationId =
      std::numeric_limits<uint32_t>::max();
  static constexpr uint64 kInvalidStreamId =
      std::numeric_limits<uint64_t>::max();
  RocmTracerEventType type;
  RocmTracerEventSource source;
  RocmTracerEventDomain domain;
  std::string name;
  // This points to strings in AnnotationMap, which should outlive the point
  // where serialization happens.
  absl::string_view annotation;
  uint64 start_time_ns;
  uint64 end_time_ns;
  uint32 device_id = kInvalidDeviceId;
  uint32 correlation_id = kInvalidCorrelationId;
  uint32 thread_id = kInvalidThreadId;
  int64 stream_id = kInvalidStreamId;
  union {
    MemcpyDetails memcpy_info;      // If type == Memcpy*
    MemAllocDetails memalloc_info;  // If type == MemoryAlloc
    KernelDetails kernel_info;      // If type == Kernel
  };
};

struct RocmTracerOptions {
  bool enable_activity_api;

  bool required_callback_api_events;
  // The callback ids that will be enabled and monitored, if empty, all
  // Callback ids to be enabled using Callback API.
  std::vector<uint32_t> cbids_selected;
  // Activity kinds to be collected using Activity API. If empty, the Activity
  // API is disable.
  std::vector<activity_domain_t> activities_selected;
};

struct RocmTracerCollectorOptions {
  // Maximum number of events to collect from callback API; if -1, no limit.
  // if 0, the callback API is enabled to build a correlation map, but no
  // events are collected.
  uint64 max_callback_api_events;
  // Maximum number of events to collect from activity API; if -1, no limit.
  uint64 max_activity_api_events;
  // Maximum number of annotation strings that we can accommodate.
  uint64 max_annotation_strings;
  // Number of GPUs involved.
  uint32 num_gpus;
};

class AnnotationMap {
 public:
  explicit AnnotationMap(uint64 max_size, uint32 num_gpus)
      : max_size_(max_size), per_device_map_(num_gpus) {}
  void Add(uint32 device_id, uint32 correlation_id,
           const std::string& annotation);
  absl::string_view LookUp(uint32 device_id, uint32 correlation_id);

 private:
  struct PerDeviceAnnotationMap {
    // The population/consumption of annotations might happen from multiple
    // callback/activity api related threads.
    absl::Mutex mutex;
    // Annotation tends to be repetitive, use a hash_set to store the strings,
    // an use the reference to the string in the map.
    absl::node_hash_set<std::string> annotations;
    absl::flat_hash_map<uint32, absl::string_view> correlation_map;
  };
  const uint64 max_size_;
  absl::FixedArray<PerDeviceAnnotationMap> per_device_map_;

  TF_DISALLOW_COPY_AND_ASSIGN(AnnotationMap);
};

class RocmTraceCollector {
 public:
  explicit RocmTraceCollector(const RocmTracerCollectorOptions& options)
      : options_(options),
        annotation_map_(options.max_annotation_strings, options.num_gpus) {}
  virtual ~RocmTraceCollector() {}

  virtual void AddEvent(RocmTracerEvent&& event) = 0;
  virtual void OnEventsDropped(const std::string& reason,
                               uint32 num_events) = 0;
  virtual void Flush() = 0;

 protected:
  RocmTracerCollectorOptions options_;

 private:
  AnnotationMap annotation_map_;

  TF_DISALLOW_COPY_AND_ASSIGN(RocmTraceCollector);
};

class RocmApiHook {
 public:
  virtual ~RocmApiHook() {}

  virtual Status OnApiEnter(uint32_t domain, uint32_t cbid,
                            const void* callback_info) = 0;
  virtual Status OnApiExit(uint32_t domain, uint32_t cbid,
                           const void* callback_info) = 0;
  virtual Status Flush() = 0;

 protected:
  static Status AddApiCallbackEvent(RocmTraceCollector* collector,
                                    uint32_t domain, uint32_t cbid,
                                    const void* callback_info);
};

// The class use to enable cupti callback/activity API and forward the collected
// trace events to RocmTraceCollector. There should be only one RocmTracer
// per process.
class RocmTracer {
 public:
  // Returns a pointer to singleton RocmTracer.
  static RocmTracer* GetRocmTracerSingleton();

  // Only one profile session can be live in the same time.
  bool IsAvailable() const;

  void Enable(const RocmTracerOptions& option, RocmTraceCollector* collector);
  void Disable();

  Status HandleCallback(uint32_t domain, uint32_t cbid,
                        const void* callback_info);

  // This function is public because called from registered callback.
  Status ProcessActivityRecord(const char* begin, const char* end);

  static uint64 GetTimestamp();
  static int NumGpus();

 protected:
  // protected constructor for injecting mock cupti interface for testing.
  explicit RocmTracer() : num_gpus_(NumGpus()) {}

 private:
  Status EnableApiTracing();
  Status DisableApiTracing();

  Status EnableActivityTracing();
  Status DisableActivityTracing();

  Status Finalize();

  int num_gpus_;
  absl::optional<RocmTracerOptions> option_;
  RocmTraceCollector* collector_ = nullptr;

  bool api_tracing_enabled_ = false;
  bool activity_tracing_enabled_ = false;

  std::unique_ptr<RocmApiHook> roctracer_api_hook_;

  TF_DISALLOW_COPY_AND_ASSIGN(RocmTracer);
};

}  // namespace profiler
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_PROFILER_INTERNAL_GPU_ROCM_TRACER_H_
