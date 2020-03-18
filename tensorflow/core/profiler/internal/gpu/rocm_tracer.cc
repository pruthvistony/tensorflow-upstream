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

#include "tensorflow/core/profiler/internal/gpu/rocm_tracer.h"

#include <iostream>
#include <sstream>

#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/profiler/internal/annotation_stack.h"

namespace tensorflow {
namespace profiler {

#define RETURN_IF_ROCTRACER_ERROR(expr)                                      \
  do {                                                                       \
    roctracer_status_t status = expr;                                        \
    if (status != ROCTRACER_STATUS_SUCCESS) {                                \
      const char* errstr = roctracer_error_string();                         \
      LOG(ERROR) << "function " << #expr << "failed with error " << errstr;  \
      return errors::Internal(absl::StrCat("roctracer call error", errstr)); \
    }                                                                        \
  } while (false)

namespace {

// GetCachedTID() caches the thread ID in thread-local storage (which is a
// userspace construct) to avoid unnecessary system calls. Without this caching,
// it can take roughly 98ns, while it takes roughly 1ns with this caching.
int32 GetCachedTID() {
  static thread_local int32 current_thread_id =
      Env::Default()->GetCurrentThreadId();
  return current_thread_id;
}

const char* GetTraceEventTypeName(const RocmTracerEventType& type) {
  switch (type) {
    case RocmTracerEventType::MemcpyH2D:
      return "MemcpyH2D";
    case RocmTracerEventType::MemcpyD2H:
      return "MemcpyD2H";
    case RocmTracerEventType::MemcpyD2D:
      return "MemcpyD2D";
    case RocmTracerEventType::MemcpyP2P:
      return "MemcpyP2P";
    case RocmTracerEventType::MemcpyOther:
      return "MemcpyOther";
    case RocmTracerEventType::Kernel:
      return "Compute";
    case RocmTracerEventType::MemoryAlloc:
      return "MemoryAlloc";
    case RocmTracerEventType::Generic:
      return "Generic";
    default:
      DCHECK(false);
      return "";
  }
  return "";
}

const char* GetActivityDomainName(uint32_t domain) {
  switch (domain) {
    case ACTIVITY_DOMAIN_HSA_API:
      return "HSA API";
    case ACTIVITY_DOMAIN_HSA_OPS:
      return "HSA OPS";
    case ACTIVITY_DOMAIN_HIP_OPS:
      return "HIP OPS";
    // case ACTIVITY_DOMAIN_HCC_OPS:
    //   return "HCC OPS";
    // case ACTIVITY_DOMAIN_HIP_VDI:
    //   return "HIP VDI";
    case ACTIVITY_DOMAIN_HIP_API:
      return "HIP API";
    case ACTIVITY_DOMAIN_KFD_API:
      return "KFD API";
    case ACTIVITY_DOMAIN_EXT_API:
      return "EXT API";
    case ACTIVITY_DOMAIN_ROCTX:
      return "ROCTX";
    default:
      DCHECK(false);
      return "";
  }
  return "";
}

const char* GetActivityPhaseName(uint32_t phase) {
  switch (phase) {
    case ACTIVITY_API_PHASE_ENTER:
      return "ENTER";
    case ACTIVITY_API_PHASE_EXIT:
      return "EXIT";
    default:
      DCHECK(false);
      return "";
  }
  return "";
}

inline void DumpApiCallbackData(uint32_t domain, uint32_t cbid,
                                const void* cbdata) {
  std::ostringstream oss;
  oss << "API callback for " << GetActivityDomainName(domain);
  if (domain == ACTIVITY_DOMAIN_HIP_API) {
    const hip_api_data_t* data =
        reinterpret_cast<const hip_api_data_t*>(cbdata);
    oss << " - " << hip_api_name(cbid);
    oss << ", correlation_id=" << data->correlation_id;
    oss << ", phase=" << GetActivityPhaseName(data->phase);
    switch (cbid) {
      case HIP_API_ID_hipModuleLaunchKernel:
        break;
      case HIP_API_ID_hipMemcpyDtoH:
        oss << ", sizeBytes=" << data->args.hipMemcpyDtoH.sizeBytes;
        break;
      case HIP_API_ID_hipMemcpyDtoHAsync:
        oss << ", sizeBytes=" << data->args.hipMemcpyDtoHAsync.sizeBytes;
        break;
      case HIP_API_ID_hipMemcpyHtoD:
        oss << ", sizeBytes=" << data->args.hipMemcpyHtoD.sizeBytes;
        break;
      case HIP_API_ID_hipMemcpyHtoDAsync:
        oss << ", sizeBytes=" << data->args.hipMemcpyHtoDAsync.sizeBytes;
        break;
      case HIP_API_ID_hipMemcpyDtoD:
        oss << ", sizeBytes=" << data->args.hipMemcpyDtoD.sizeBytes;
        break;
      case HIP_API_ID_hipMemcpyDtoDAsync:
        oss << ", sizeBytes=" << data->args.hipMemcpyDtoDAsync.sizeBytes;
        break;
      case HIP_API_ID_hipMalloc:
        oss << ", size=" << data->args.hipMalloc.size;
        break;
      case HIP_API_ID_hipFree:
        oss << ", ptr=" << data->args.hipFree.ptr;
        break;
      case HIP_API_ID_hipStreamSynchronize:
        break;
      default:
        DCHECK(false);
        break;
    }
  } else {
    oss << ": " << cbid;
  }
  VLOG(kRocmTracerVlog) << oss.str();
}

void DumpActivityRecord(const roctracer_record_t* record) {
  std::ostringstream oss;
  oss << "Activity callback for " << GetActivityDomainName(record->domain);
  oss << ", kind=" << record->kind;
  oss << ", op=" << record->op;
  oss << ", correlation_id=" << record->correlation_id;
  oss << ", begin_ns=" << record->begin_ns;
  oss << ", end_ns=" << record->end_ns;
  oss << ", bytes=" << record->bytes;
  VLOG(kRocmTracerVlog) << oss.str();
}

}  // namespace

class RocmApiCallbackImpl {
 public:
  RocmApiCallbackImpl(const RocmTracerOptions& options,
                      RocmTraceCollector* collector)
      : options_(options), collector_(collector) {}

  Status operator()(uint32_t domain, uint32_t cbid, const void* cbdata) {
    DumpApiCallbackData(domain, cbid, cbdata);

    if (domain != ACTIVITY_DOMAIN_HIP_API) return Status::OK();

    const hip_api_data_t* data =
        reinterpret_cast<const hip_api_data_t*>(cbdata);

    if (data->phase == ACTIVITY_API_PHASE_ENTER) {
      // Nothing to do here
    } else if (data->phase == ACTIVITY_API_PHASE_EXIT) {
      // todoo
      // // Set up the map from correlation id to annotation string.
      // const std::string& annotation =
      //     tensorflow::Annotation::CurrentAnnotation();
      // if (!annotation.empty()) {
      //   annotation_map_->Add(data->correlation_id, annotation);
      // }

      switch (cbid) {
        case HIP_API_ID_hipModuleLaunchKernel:
          AddKernelEventUponApiExit(cbid, data);
          break;
        case HIP_API_ID_hipMemcpyDtoH:
        case HIP_API_ID_hipMemcpyDtoHAsync:
        case HIP_API_ID_hipMemcpyHtoD:
        case HIP_API_ID_hipMemcpyHtoDAsync:
        case HIP_API_ID_hipMemcpyDtoD:
        case HIP_API_ID_hipMemcpyDtoDAsync:
          AddMemcpyEventUponApiExit(cbid, data);
          break;
        case HIP_API_ID_hipMalloc:
        case HIP_API_ID_hipFree:
          AddMallocEventUponApiExit(cbid, data);
          break;
        default:
          AddGenericEventUponApiExit(cbid, data);
          break;
      }
    }
    return Status::OK();
  }

 private:
  void AddKernelEventUponApiExit(uint32_t cbid, const hip_api_data_t* data) {
    RocmTracerEvent event;
    event.domain = RocmTracerEventDomain::HIP;
    event.type = RocmTracerEventType::Kernel;
    event.source = RocmTracerEventSource::ApiCallback;
    const hipFunction_t kernelFunc = data->args.hipModuleLaunchKernel.f;
    if (kernelFunc != nullptr) event.name = hipKernelNameRef(kernelFunc);
    event.thread_id = GetCachedTID();
    event.correlation_id = data->correlation_id;

    event.kernel_info.dynamic_shared_memory_usage =
        data->args.hipModuleLaunchKernel.sharedMemBytes;
    event.kernel_info.block_x = data->args.hipModuleLaunchKernel.blockDimX;
    event.kernel_info.block_y = data->args.hipModuleLaunchKernel.blockDimY;
    event.kernel_info.block_z = data->args.hipModuleLaunchKernel.blockDimZ;
    event.kernel_info.grid_x = data->args.hipModuleLaunchKernel.gridDimX;
    event.kernel_info.grid_y = data->args.hipModuleLaunchKernel.gridDimY;
    event.kernel_info.grid_z = data->args.hipModuleLaunchKernel.gridDimZ;

    collector_->AddEvent(std::move(event));
  }

  void AddMemcpyEventUponApiExit(uint32_t cbid, const hip_api_data_t* data) {
    RocmTracerEvent event;
    event.domain = RocmTracerEventDomain::HIP;
    event.name = roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cbid, 0);
    event.source = RocmTracerEventSource::ApiCallback;
    event.thread_id = GetCachedTID();
    event.correlation_id = data->correlation_id;

    // ROCM TODO: figure out a way to properly populate this field.
    event.memcpy_info.destination = 0;
    switch (cbid) {
      case HIP_API_ID_hipMemcpyDtoH:
        event.type = RocmTracerEventType::MemcpyD2H;
        event.memcpy_info.num_bytes = data->args.hipMemcpyDtoH.sizeBytes;
        event.memcpy_info.async = false;
        break;
      case HIP_API_ID_hipMemcpyDtoHAsync:
        event.type = RocmTracerEventType::MemcpyD2H;
        event.memcpy_info.num_bytes = data->args.hipMemcpyDtoHAsync.sizeBytes;
        event.memcpy_info.async = true;
        break;
      case HIP_API_ID_hipMemcpyHtoD:
        event.type = RocmTracerEventType::MemcpyH2D;
        event.memcpy_info.num_bytes = data->args.hipMemcpyHtoD.sizeBytes;
        event.memcpy_info.async = false;
        break;
      case HIP_API_ID_hipMemcpyHtoDAsync:
        event.type = RocmTracerEventType::MemcpyH2D;
        event.memcpy_info.num_bytes = data->args.hipMemcpyHtoDAsync.sizeBytes;
        event.memcpy_info.async = true;
        break;
      case HIP_API_ID_hipMemcpyDtoD:
        event.type = RocmTracerEventType::MemcpyD2D;
        event.memcpy_info.num_bytes = data->args.hipMemcpyDtoD.sizeBytes;
        event.memcpy_info.async = false;
        break;
      case HIP_API_ID_hipMemcpyDtoDAsync:
        event.type = RocmTracerEventType::MemcpyD2D;
        event.memcpy_info.num_bytes = data->args.hipMemcpyDtoDAsync.sizeBytes;
        event.memcpy_info.async = true;
        break;
      default:
        LOG(ERROR) << "Unsupported memcpy activity observed: " << cbid;
        break;
    }
    collector_->AddEvent(std::move(event));
  }

  void AddMallocEventUponApiExit(uint32_t cbid, const hip_api_data_t* data) {
    RocmTracerEvent event;
    event.domain = RocmTracerEventDomain::HIP;
    event.type = RocmTracerEventType::MemoryAlloc;
    event.source = RocmTracerEventSource::ApiCallback;
    event.name = roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cbid, 0);
    event.thread_id = GetCachedTID();
    event.correlation_id = data->correlation_id;

    switch (cbid) {
      case HIP_API_ID_hipMalloc:
        event.memalloc_info.num_bytes = data->args.hipMalloc.size;
        break;
      case HIP_API_ID_hipFree:
        event.memalloc_info.num_bytes = 0;
        break;
    }
    collector_->AddEvent(std::move(event));
  }

  void AddGenericEventUponApiExit(uint32_t cbid, const hip_api_data_t* data) {
    RocmTracerEvent event;
    event.domain = RocmTracerEventDomain::HIP;
    event.type = RocmTracerEventType::Generic;
    event.source = RocmTracerEventSource::ApiCallback;
    event.name = roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cbid, 0);
    event.thread_id = GetCachedTID();
    event.correlation_id = data->correlation_id;

    collector_->AddEvent(std::move(event));
  }

  RocmTracerOptions options_;
  RocmTraceCollector* collector_ = nullptr;
};

class RocmActivityCallbackImpl {
 public:
  RocmActivityCallbackImpl(const RocmTracerOptions& options,
                           RocmTraceCollector* collector)
      : options_(options), collector_(collector) {}

  Status operator()(const char* begin, const char* end) {
    const roctracer_record_t* record =
        reinterpret_cast<const roctracer_record_t*>(begin);
    const roctracer_record_t* end_record =
        reinterpret_cast<const roctracer_record_t*>(end);

    while (record < end_record) {
      DumpActivityRecord(record);

      switch (record->domain) {
        // HIP API activities.
        case ACTIVITY_DOMAIN_HIP_API:
          switch (record->op) {
            case HIP_API_ID_hipModuleLaunchKernel:
              AddHipKernelActivityEvent(record);
              break;

            case HIP_API_ID_hipMemcpyDtoH:
            case HIP_API_ID_hipMemcpyHtoD:
            case HIP_API_ID_hipMemcpyDtoD:
            case HIP_API_ID_hipMemcpyDtoHAsync:
            case HIP_API_ID_hipMemcpyHtoDAsync:
            case HIP_API_ID_hipMemcpyDtoDAsync:
              AddHipMemcpyActivityEvent(record);
              break;
            default:
              break;
          }  // switch (record->op).
          break;

        // HCC ops activities.
        case ACTIVITY_DOMAIN_HCC_OPS:
          switch (record->op) {
            case 0:  // dispatch
              AddHccKernelActivityEvent(record);
              break;
            case 1:  // copy
              AddHccMemcpyActivityEvent(record);
              break;
            // ROCM TODO: decide whther to log these activities later.
            // case 2: // barrier / marker
            //  break;
            default:
              break;
          }  // switch (record->op).
          break;
      }

      RETURN_IF_ROCTRACER_ERROR(static_cast<roctracer_status_t>(
          roctracer_next_record(record, &record)));
    }
    return Status::OK();
  }

 private:
  void AddHipKernelActivityEvent(const roctracer_record_t* record) {
    RocmTracerEvent event;
    event.domain = RocmTracerEventDomain::HIP;
    event.type = RocmTracerEventType::Kernel;
    event.source = RocmTracerEventSource::Activity;
    event.name = roctracer_op_string(record->domain, record->op, record->kind);
    event.correlation_id = record->correlation_id;
    // todoo
    // event.annotation = annotation_map->LookUp(event.correlation_id);

    event.start_time_ns = record->begin_ns;
    event.end_time_ns = record->end_ns;

    collector_->AddEvent(std::move(event));
  }

  void AddHipMemcpyActivityEvent(const roctracer_record_t* record) {
    RocmTracerEvent event;
    event.domain = RocmTracerEventDomain::HIP;
    event.source = RocmTracerEventSource::Activity;
    event.name = roctracer_op_string(record->domain, record->op, record->kind);
    event.correlation_id = record->correlation_id;
    // todoo
    // event.annotation = annotation_map->LookUp(event.correlation_id);

    event.memcpy_info.num_bytes = record->bytes;
    event.memcpy_info.destination = record->device_id;

    switch (record->op) {
      case HIP_API_ID_hipMemcpyDtoH:
        event.type = RocmTracerEventType::MemcpyD2H;
        event.memcpy_info.async = false;
        break;
      case HIP_API_ID_hipMemcpyDtoHAsync:
        event.type = RocmTracerEventType::MemcpyD2H;
        event.memcpy_info.async = true;
        break;
      case HIP_API_ID_hipMemcpyHtoD:
        event.type = RocmTracerEventType::MemcpyH2D;
        event.memcpy_info.async = false;
        break;
      case HIP_API_ID_hipMemcpyHtoDAsync:
        event.type = RocmTracerEventType::MemcpyH2D;
        event.memcpy_info.async = true;
        break;
      case HIP_API_ID_hipMemcpyDtoD:
        event.type = RocmTracerEventType::MemcpyD2D;
        event.memcpy_info.async = false;
        // ROCM TODO: figure out a way to properly populate this field.
        event.memcpy_info.destination = record->device_id;
        break;
      case HIP_API_ID_hipMemcpyDtoDAsync:
        event.type = RocmTracerEventType::MemcpyD2D;
        event.memcpy_info.async = true;
        // ROCM TODO: figure out a way to properly populate this field.
        event.memcpy_info.destination = record->device_id;
        break;
      default:
        event.type = RocmTracerEventType::MemcpyOther;
        event.memcpy_info.async = false;
        event.memcpy_info.destination = record->device_id;
        break;
    }

    event.start_time_ns = record->begin_ns;
    event.end_time_ns = record->end_ns;

    collector_->AddEvent(std::move(event));
  }
  void AddHccKernelActivityEvent(const roctracer_record_t* record) {
    RocmTracerEvent event;
    event.domain = RocmTracerEventDomain::HCC;
    event.type = RocmTracerEventType::Kernel;
    event.source = RocmTracerEventSource::Activity;
    event.name = roctracer_op_string(record->domain, record->op, record->kind);
    event.correlation_id = record->correlation_id;
    // todoo
    // event.annotation = annotation_map->LookUp(event.correlation_id);

    event.start_time_ns = record->begin_ns;
    event.end_time_ns = record->end_ns;
    event.device_id = record->device_id;
    event.stream_id = record->queue_id;

    collector_->AddEvent(std::move(event));
  }

  void AddHccMemcpyActivityEvent(const roctracer_record_t* record) {
    RocmTracerEvent event;
    event.domain = RocmTracerEventDomain::HCC;
    // Set MemcpyOther here. The field won't really be used when we aggregate
    // with other RocmTracerEvent instances coming from API callbacks.
    event.type = RocmTracerEventType::MemcpyOther;
    event.source = RocmTracerEventSource::Activity;
    event.name = roctracer_op_string(record->domain, record->op, record->kind);
    event.correlation_id = record->correlation_id;
    // todoo
    // event.annotation = annotation_map->LookUp(event.correlation_id);

    event.start_time_ns = record->begin_ns;
    event.end_time_ns = record->end_ns;
    event.device_id = record->device_id;
    event.stream_id = record->queue_id;

    collector_->AddEvent(std::move(event));
  }

  RocmTracerOptions options_;
  RocmTraceCollector* collector_ = nullptr;
};

// void AnnotationMap::Add(uint32 correlation_id, const std::string& annotation)
// {
//   if (annotation.empty()) return;
//   VLOG(kRocmTracerVlog) << "Add annotation: "
//           << " correlation_id: " << correlation_id
//           << " annotation: " << annotation;
//   absl::MutexLock lock(&map_.mutex);
//   if (map_.annotations.size() < max_size_) {
//     absl::string_view annotation_str =
//         *map_.annotations.insert(annotation).first;
//     map_.correlation_map.emplace(correlation_id, annotation_str);
//   }
// }

// absl::string_view AnnotationMap::LookUp(uint32 correlation_id) {
//   absl::MutexLock lock(&map_.mutex);
//   auto it = map_.correlation_map.find(correlation_id);
//   return it != map_.correlation_map.end() ? it->second : absl::string_view();
// }

/* static */ RocmTracer* RocmTracer::GetRocmTracerSingleton() {
  static auto* singleton = new RocmTracer();
  return singleton;
}

bool RocmTracer::IsAvailable() const {
  return !activity_tracing_enabled_ && !api_tracing_enabled_;
}

int RocmTracer::NumGpus() {
  static int num_gpus = []() -> int {
    if (hipInit(0) != hipSuccess) {
      return 0;
    }
    int gpu_count;
    if (hipGetDeviceCount(&gpu_count) != hipSuccess) {
      return 0;
    }
    LOG(INFO) << "Profiler found " << gpu_count << " GPUs";
    return gpu_count;
  }();
  return num_gpus;
}

void RocmTracer::Enable(const RocmTracerOptions& options,
                        RocmTraceCollector* collector) {
  options_ = options;
  collector_ = collector;
  api_cb_impl_ = new RocmApiCallbackImpl(options, collector);
  activity_cb_impl_ = new RocmActivityCallbackImpl(options, collector);
  EnableApiTracing().IgnoreError();
  if (options_->enable_activity_api) {
    EnableActivityTracing().IgnoreError();
  }
}

void RocmTracer::Disable() {
  DisableApiTracing().IgnoreError();
  if (options_->enable_activity_api) {
    DisableActivityTracing().IgnoreError();
  }
  delete api_cb_impl_;
  delete activity_cb_impl_;
  collector_->Flush();
  collector_ = nullptr;
  options_.reset();
}

void ApiCallback(uint32_t domain, uint32_t cbid, const void* cbdata,
                 void* user_data) {
  RocmTracer* tracer = reinterpret_cast<RocmTracer*>(user_data);
  tracer->ApiCallbackHandler(domain, cbid, cbdata);
}

void RocmTracer::ApiCallbackHandler(uint32_t domain, uint32_t cbid,
                                    const void* cbdata) {
  if (api_tracing_enabled_) (*api_cb_impl_)(domain, cbid, cbdata);
}

Status RocmTracer::EnableApiTracing() {
  if (api_tracing_enabled_) return Status::OK();
  api_tracing_enabled_ = true;

  if (!options_->cbids_selected.empty()) {
    VLOG(kRocmTracerVlog) << "Enabling API tracing for "
                          << options_->cbids_selected.size() << " APIs";

    for (auto cbid : options_->cbids_selected) {
      VLOG(kRocmTracerVlog)
          << "Enabling API tracing for: " << hip_api_name(cbid);
      RETURN_IF_ROCTRACER_ERROR(roctracer_enable_op_callback(
          ACTIVITY_DOMAIN_HIP_API, cbid, ApiCallback, this));
    }
  } else {  // Register callback for all CBIDs
    VLOG(kRocmTracerVlog) << "Enabling API tracing for all APIs";
    RETURN_IF_ROCTRACER_ERROR(roctracer_enable_callback(ApiCallback, this));
  }
  return Status::OK();
}

Status RocmTracer::DisableApiTracing() {
  if (!api_tracing_enabled_) return Status::OK();
  api_tracing_enabled_ = false;

  if (!options_->cbids_selected.empty()) {
    VLOG(kRocmTracerVlog) << "Disabling API tracing for "
                          << options_->cbids_selected.size() << " APIs";

    for (auto cbid : options_->cbids_selected) {
      VLOG(kRocmTracerVlog)
          << "Disabling API tracing for: " << hip_api_name(cbid);
      RETURN_IF_ROCTRACER_ERROR(
          roctracer_disable_op_callback(ACTIVITY_DOMAIN_HIP_API, cbid));
    }
  } else {
    VLOG(kRocmTracerVlog) << "Disabling API tracing for all APIs";
    RETURN_IF_ROCTRACER_ERROR(roctracer_disable_callback());
  }
  return Status::OK();
}

void ActivityCallback(const char* begin, const char* end, void* user_data) {
  RocmTracer* tracer = reinterpret_cast<RocmTracer*>(user_data);
  tracer->ActivityCallbackHandler(begin, end);
}

void RocmTracer::ActivityCallbackHandler(const char* begin, const char* end) {
  if (activity_tracing_enabled_) (*activity_cb_impl_)(begin, end);
}

Status RocmTracer::EnableActivityTracing() {
  if (activity_tracing_enabled_) return Status::OK();
  activity_tracing_enabled_ = true;

  if (!options_->activities_selected.empty()) {
    // Creating tracer pool.
    if (roctracer_default_pool() == NULL) {
      roctracer_properties_t properties{};
      properties.buffer_size = 0x1000;
      properties.buffer_callback_fun = ActivityCallback;
      properties.buffer_callback_arg = this;
      VLOG(kRocmTracerVlog) << "Creating roctracer activity buffer";
      RETURN_IF_ROCTRACER_ERROR(roctracer_open_pool(&properties));
    }

    VLOG(kRocmTracerVlog) << "Enabling activity tracing for "
                          << options_->activities_selected.size()
                          << " activity domains";

    for (auto activity : options_->activities_selected) {
      VLOG(kRocmTracerVlog) << "Enabling activity tracing for domain: "
                            << GetActivityDomainName(activity);
      RETURN_IF_ROCTRACER_ERROR(roctracer_enable_domain_activity(activity));
    }
  }
  return Status::OK();
}

Status RocmTracer::DisableActivityTracing() {
  if (!activity_tracing_enabled_) return Status::OK();
  activity_tracing_enabled_ = false;

  if (!options_->activities_selected.empty()) {
    VLOG(kRocmTracerVlog) << "Disabling activity tracing for "
                          << options_->activities_selected.size()
                          << " activity domains";

    for (auto activity : options_->activities_selected) {
      VLOG(kRocmTracerVlog) << "Disabling activity tracing for domain: "
                            << GetActivityDomainName(activity);
      RETURN_IF_ROCTRACER_ERROR(roctracer_disable_domain_activity(activity));
    }
    options_->activities_selected.clear();

    VLOG(kRocmTracerVlog) << "Flushing roctracer activity buffer";
    RETURN_IF_ROCTRACER_ERROR(roctracer_flush_activity());
  }
  return Status::OK();
}

/*static*/ uint64 RocmTracer::GetTimestamp() {
  uint64_t ts;
  if (roctracer_get_timestamp(&ts) == ROCTRACER_STATUS_SUCCESS) {
    return ts;
  }
  // Return 0 on error.
  return 0;
}

}  // namespace profiler
}  // namespace tensorflow
