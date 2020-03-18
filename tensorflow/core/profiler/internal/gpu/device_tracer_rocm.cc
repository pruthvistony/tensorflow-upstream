/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#if TENSORFLOW_USE_ROCM

#include <stdlib.h>

#include <memory>
#include <utility>

#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/abi.h"
#include "tensorflow/core/platform/env_time.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/profiler/internal/annotation_stack.h"
#include "tensorflow/core/profiler/internal/gpu/rocm_tracer.h"
#include "tensorflow/core/profiler/internal/parse_annotation.h"
#include "tensorflow/core/profiler/internal/profiler_factory.h"
#include "tensorflow/core/profiler/internal/profiler_interface.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace profiler {

class RocmTraceCollectorImpl : public profiler::RocmTraceCollector {
 public:
  RocmTraceCollectorImpl(const RocmTraceCollectorOptions& options,
                         uint64 start_walltime_ns, uint64 start_gputime_ns)
      : RocmTraceCollector(options),
        num_callback_events_(0),
        num_activity_events_(0),
        start_walltime_ns_(start_walltime_ns),
        start_gputime_ns_(start_gputime_ns) {}
  // todoo per_device_collector_(options.num_gpus) {}

  void AddEvent(RocmTracerEvent&& event) override {
    mutex_lock lock(aggregated_events_mutex_);
    if (event.device_id > options_.num_gpus) return;
    if (event.source == RocmTracerEventSource::ApiCallback) {
      if (num_callback_events_ > options_.max_callback_api_events) {
        OnEventsDropped(
            "RocmTraceCollector - max callback event capacity reached", 1);
        return;
      }
      num_callback_events_++;
    } else {
      // event.source == RocmTracerEventSource::Activity
      if (num_activity_events_ > options_.max_activity_api_events) {
        OnEventsDropped(
            "RocmTraceCollector - max activity event capacity reached", 1);
        return;
      }
      num_activity_events_++;
    }

    auto iter = aggregated_events_.find(event.correlation_id);
    if (iter != aggregated_events_.end()) {
      // // event with this correlation id already present
      // // agrregate this event with the existing one
      // switch (event.domain) {
      //   case RocmTracerEventDomain::HIP:
      //     switch (event.source) {
      //       case RocmEventSource::ApiCallback:
      //         break;
      //       case RocmEventSource::Activity:
      // 	      iter->second.annotation = event.annotation;
      //         break;
      //     }
      //     break;
      //   case RocmTracerEventDomain::HCC:
      //     switch (event.source) {
      //       case RocmEventSource::ApiCallback:
      //         break;
      //       case RocmEventSource::Activity:
      // 	      iter->second.device_id = event.device_id;
      // 	      iter->second.stream_id = event.stream_id;
      // 	      iter->second.start_time_ns = event.start_time_ns;
      // 	      iter->second.end_time_ns = event.end_time_ns;
      //         break;
      //     }
      //     break;
      // }
    } else {
      aggregated_events_.emplace(event.correlation_id, std::move(event));
    }
  }

  void OnEventsDropped(const std::string& reason, uint32 num_events) override {
    VLOG(-1) << "RocmTracerEvent(s) dropped (" << num_events << ") : " << reason
             << ".";
  }

  void Flush() override {}

  void Export(StepStats* step_stats) {}

  void Export(XSpace* space) {}

 private:
  std::atomic<int> num_callback_events_;
  std::atomic<int> num_activity_events_;
  uint64 start_walltime_ns_;
  uint64 start_gputime_ns_;

  mutex aggregated_events_mutex_;
  absl::flat_hash_map<uint32, RocmTracerEvent> aggregated_events_
      GUARDED_BY(aggregated_events_mutex_);

  // todoo
  // struct PerDeviceCollector {
  //   void AddEvent(RocmTracerEvent&& event) {
  //     mutex_lock lock(events_mutex);
  //     if (event.source == RocmTracerEventSource::ApiCallback) {
  //     }
  //     events.emplace_back(std::move(event));
  //   }

  //   mutex events_mutex;
  //   std::vector<RocmTracerEvent> events GUARDED_BY(events_mutex);
  // };

  // absl::FixedArray<PerDeviceCollector> per_device_collector_;
};

// GpuTracer for ROCm GPU.
class GpuTracer : public profiler::ProfilerInterface {
 public:
  GpuTracer(RocmTracer* rocm_tracer) : rocm_tracer_(rocm_tracer) {
    VLOG(-1) << "GpuTracer created.";
  }
  ~GpuTracer() override {}

  // GpuTracer interface:
  Status Start() override;
  Status Stop() override;
  Status CollectData(RunMetadata* run_metadata) override;
  Status CollectData(XSpace* space) override;
  profiler::DeviceType GetDeviceType() override {
    return profiler::DeviceType::kGpu;
  }

 private:
  Status DoStart();
  Status DoStop();
  Status DoCollectData(StepStats* step_stats);
  Status DoCollectData(XSpace* space);

  RocmTracerOptions GetRocmTracerOptions();

  RocmTraceCollectorOptions GetRocmTraceCollectorOptions(uint32 num_gpus);

  enum State {
    kNotStarted,
    kStartedOk,
    kStartedError,
    kStoppedOk,
    kStoppedError
  };
  State profiling_state_ = State::kNotStarted;

  RocmTracer* rocm_tracer_;
  std::unique_ptr<RocmTraceCollectorImpl> rocm_trace_collector_;
};

RocmTracerOptions GpuTracer::GetRocmTracerOptions() {
  RocmTracerOptions options;
  options.enable_activity_api = true;
  options.required_callback_api_events = true;
  options.cbids_selected = {
      // KERNEL
      HIP_API_ID_hipModuleLaunchKernel,
      // MEMCPY
      HIP_API_ID_hipMemcpyDtoH,
      HIP_API_ID_hipMemcpyDtoHAsync,
      HIP_API_ID_hipMemcpyHtoD,
      HIP_API_ID_hipMemcpyHtoDAsync,
      HIP_API_ID_hipMemcpyDtoD,
      HIP_API_ID_hipMemcpyDtoDAsync,
      // MALLOC / FREE
      HIP_API_ID_hipMalloc,
      HIP_API_ID_hipFree,
      // GENERIC
      HIP_API_ID_hipStreamSynchronize,
  };
  options.activities_selected = {
      ACTIVITY_DOMAIN_HIP_API,
      ACTIVITY_DOMAIN_HCC_OPS,
  };

  return options;
}

RocmTraceCollectorOptions GpuTracer::GetRocmTraceCollectorOptions(
    uint32 num_gpus) {
  RocmTraceCollectorOptions options;
  options.max_callback_api_events = 2 * 1024 * 1024;
  options.max_activity_api_events = 2 * 1024 * 1024;
  options.max_annotation_strings = 1024 * 1024;
  options.num_gpus = num_gpus;
  return options;
}

Status GpuTracer::DoStart() {
  if (!rocm_tracer_->IsAvailable()) {
    return errors::Unavailable("Another profile session running.");
  }

  AnnotationStack::Enable(true);

  RocmTraceCollectorOptions trace_collector_options =
      GetRocmTraceCollectorOptions(rocm_tracer_->NumGpus());
  uint64 start_walltime_ns = tensorflow::EnvTime::NowNanos();
  uint64 start_gputime_ns = RocmTracer::GetTimestamp();
  rocm_trace_collector_ = std::make_unique<RocmTraceCollectorImpl>(
      trace_collector_options, start_walltime_ns, start_gputime_ns);

  RocmTracerOptions tracer_options = GetRocmTracerOptions();
  rocm_tracer_->Enable(tracer_options, rocm_trace_collector_.get());

  return Status::OK();
}

Status GpuTracer::Start() {
  Status status = DoStart();
  if (status.ok()) {
    profiling_state_ = State::kStartedOk;
    return Status::OK();
  } else {
    profiling_state_ = State::kStartedError;
    return status;
  }
}

Status GpuTracer::DoStop() {
  rocm_tracer_->Disable();
  AnnotationStack::Enable(false);
  return Status::OK();
}

Status GpuTracer::Stop() {
  if (profiling_state_ == State::kStartedOk) {
    Status status = DoStop();
    profiling_state_ = status.ok() ? State::kStoppedOk : State::kStoppedError;
  }
  return Status::OK();
}

Status GpuTracer::DoCollectData(StepStats* step_stats) {
  if (rocm_trace_collector_) rocm_trace_collector_->Export(step_stats);
  return Status::OK();
}

Status GpuTracer::CollectData(RunMetadata* run_metadata) {
  switch (profiling_state_) {
    case State::kNotStarted:
      VLOG(-1) << "No trace data collected, session wasn't started";
      return Status::OK();
    case State::kStartedOk:
      return errors::FailedPrecondition("Cannot collect trace before stopping");
    case State::kStartedError:
      LOG(ERROR) << "Cannot collect, roctracer failed to start";
      return Status::OK();
    case State::kStoppedError:
      VLOG(-1) << "No trace data collected";
      return Status::OK();
    case State::kStoppedOk: {
      // Input run_metadata is shared by profiler interfaces, we need append.
      StepStats step_stats;
      DoCollectData(&step_stats);
      for (auto& dev_stats : *step_stats.mutable_dev_stats()) {
        run_metadata->mutable_step_stats()->add_dev_stats()->Swap(&dev_stats);
      }
      return Status::OK();
    }
  }
  return errors::Internal("Invalid profiling state: ", profiling_state_);
}

Status GpuTracer::DoCollectData(XSpace* space) {
  if (rocm_trace_collector_) rocm_trace_collector_->Export(space);
  return Status::OK();
}

Status GpuTracer::CollectData(XSpace* space) {
  switch (profiling_state_) {
    case State::kNotStarted:
      VLOG(-1) << "No trace data collected, session wasn't started";
      return Status::OK();
    case State::kStartedOk:
      return errors::FailedPrecondition("Cannot collect trace before stopping");
    case State::kStartedError:
      LOG(ERROR) << "Cannot collect, roctracer failed to start";
      return Status::OK();
    case State::kStoppedError:
      VLOG(-1) << "No trace data collected";
      return Status::OK();
    case State::kStoppedOk: {
      DoCollectData(space);
      return Status::OK();
    }
  }
  return errors::Internal("Invalid profiling state: ", profiling_state_);
}

// Not in anonymous namespace for testing purposes.
std::unique_ptr<profiler::ProfilerInterface> CreateGpuTracer(
    const profiler::ProfilerOptions& options) {
  if (options.device_type != profiler::DeviceType::kGpu &&
      options.device_type != profiler::DeviceType::kUnspecified)
    return nullptr;

  profiler::RocmTracer* rocm_tracer =
      profiler::RocmTracer::GetRocmTracerSingleton();
  if (!rocm_tracer->IsAvailable()) return nullptr;

  return absl::make_unique<profiler::GpuTracer>(rocm_tracer);
}

auto register_rocm_gpu_tracer_factory = [] {
  RegisterProfilerFactory(&CreateGpuTracer);
  return 0;
}();

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_USE_ROCM
