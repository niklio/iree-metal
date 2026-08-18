// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_pjrt/common/api_impl.h"

#include <algorithm>
#include <charconv>
#include <cinttypes>
#include <iomanip>
#include <iterator>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "iree/hal/api.h"
#include "iree_pjrt/common/iree_helpers.h"
#include "iree_pjrt/common/tensor_utils.h"
// TODO: Excise. Uses deep XLA internals.
// #include "xla/pjrt/transpose.h"

using iree::vm::retain_ref;

namespace iree::pjrt {

const std::string_view kMlirFormat = "mlir";

// JAX uses "device" as the canonical memory kind for ordinary device-backed
// arrays. The HAL device name (for example, "Apple M4") is a hardware
// description, not a PJRT memory kind, and advertising it here prevents JAX
// from selecting an addressable memory for compiled inputs and outputs.
constexpr std::string_view kDeviceMemoryKind = "device";

// We hardcode the maximum number of dimensions to avoid mallocs.
constexpr int64_t kMaxDims = 9;

namespace {

// Compiler output used when restoring a serialized VMFB. The ordinary compile
// path memory maps an IREE compiler output; deserialization instead owns a copy
// of the bytes supplied by PJRT's caller.
class InMemoryCompilerOutput final : public CompilerOutput {
 public:
  InMemoryCompilerOutput(const char* data, size_t size)
      : bytes_(data, data + size) {}

  void* GetData() override { return bytes_.data(); }
  size_t GetDataSize() override { return bytes_.size(); }

 private:
  std::vector<char> bytes_;
};

struct SerializedExecutableStorage {
  explicit SerializedExecutableStorage(const CompilerOutput& output)
      : bytes(static_cast<const char*>(
                  const_cast<CompilerOutput&>(output).GetData()),
              static_cast<const char*>(
                  const_cast<CompilerOutput&>(output).GetData()) +
                  const_cast<CompilerOutput&>(output).GetDataSize()) {}
  std::vector<char> bytes;
};

std::string FingerprintCompilerOutput(CompilerOutput& output) {
  // FNV-1a is sufficient here: this is a deterministic PJRT identity, not a
  // security boundary. Prefixing the algorithm leaves room to change it in a
  // future serialization generation without ambiguity.
  constexpr uint64_t kFnvOffset = UINT64_C(14695981039346656037);
  constexpr uint64_t kFnvPrime = UINT64_C(1099511628211);
  uint64_t hash = kFnvOffset;
  const auto* data = static_cast<const uint8_t*>(output.GetData());
  for (size_t i = 0; i < output.GetDataSize(); ++i) {
    hash ^= data[i];
    hash *= kFnvPrime;
  }
  std::ostringstream stream;
  stream << "iree-vmfb-fnv1a64-" << std::hex << std::setfill('0')
         << std::setw(16) << hash;
  return stream.str();
}

struct TensorMetadata {
  PJRT_Buffer_Type element_type = PJRT_Buffer_Type_INVALID;
  std::vector<int64_t> dims;
  int64_t size_in_bytes = 0;
};

std::optional<std::pair<PJRT_Buffer_Type, int64_t>> ParseElementType(
    std::string_view type) {
  if (type == "i1") return std::pair(PJRT_Buffer_Type_PRED, 1);
  if (type == "i8") return std::pair(PJRT_Buffer_Type_S8, 1);
  if (type == "i16") return std::pair(PJRT_Buffer_Type_S16, 2);
  if (type == "i32") return std::pair(PJRT_Buffer_Type_S32, 4);
  if (type == "i64") return std::pair(PJRT_Buffer_Type_S64, 8);
  if (type == "ui8") return std::pair(PJRT_Buffer_Type_U8, 1);
  if (type == "ui16") return std::pair(PJRT_Buffer_Type_U16, 2);
  if (type == "ui32") return std::pair(PJRT_Buffer_Type_U32, 4);
  if (type == "ui64") return std::pair(PJRT_Buffer_Type_U64, 8);
  if (type == "f16") return std::pair(PJRT_Buffer_Type_F16, 2);
  if (type == "bf16") return std::pair(PJRT_Buffer_Type_BF16, 2);
  if (type == "f32") return std::pair(PJRT_Buffer_Type_F32, 4);
  if (type == "f64") return std::pair(PJRT_Buffer_Type_F64, 8);
  if (type == "complex<f32>") return std::pair(PJRT_Buffer_Type_C64, 8);
  if (type == "complex<f64>") return std::pair(PJRT_Buffer_Type_C128, 16);
  return std::nullopt;
}

std::optional<TensorMetadata> ParseTensorMetadata(std::string_view value) {
  size_t tensor_start = value.find("tensor<");
  if (tensor_start == std::string_view::npos) return std::nullopt;
  size_t content_start = tensor_start + strlen("tensor<");
  int angle_depth = 1;
  size_t content_end = content_start;
  for (; content_end < value.size(); ++content_end) {
    if (value[content_end] == '<') ++angle_depth;
    if (value[content_end] == '>' && --angle_depth == 0) break;
  }
  if (angle_depth != 0) return std::nullopt;
  std::string_view content = value.substr(content_start,
                                          content_end - content_start);

  size_t element_start = 0;
  angle_depth = 0;
  for (size_t i = 0; i < content.size(); ++i) {
    if (content[i] == '<') ++angle_depth;
    if (content[i] == '>') --angle_depth;
    if (content[i] == 'x' && angle_depth == 0) element_start = i + 1;
  }
  auto parsed_type = ParseElementType(content.substr(element_start));
  if (!parsed_type) return std::nullopt;

  TensorMetadata metadata;
  metadata.element_type = parsed_type->first;
  int64_t element_count = 1;
  std::string_view dimensions = content.substr(0, element_start);
  if (!dimensions.empty()) dimensions.remove_suffix(1);  // trailing 'x'
  size_t position = 0;
  while (position < dimensions.size()) {
    size_t separator = dimensions.find('x', position);
    std::string_view dimension = dimensions.substr(
        position, separator == std::string_view::npos
                      ? std::string_view::npos
                      : separator - position);
    int64_t extent = -1;
    if (dimension != "?") {
      auto result = std::from_chars(dimension.data(),
                                    dimension.data() + dimension.size(), extent);
      if (result.ec != std::errc() || result.ptr != dimension.data() + dimension.size() ||
          extent < 0) {
        return std::nullopt;
      }
      element_count *= extent;
    } else {
      element_count = -1;
    }
    metadata.dims.push_back(extent);
    if (separator == std::string_view::npos) break;
    position = separator + 1;
  }
  metadata.size_in_bytes =
      element_count < 0 ? 0 : element_count * parsed_type->second;
  return metadata;
}

std::vector<std::string_view> SplitFunctionValues(std::string_view values) {
  std::vector<std::string_view> result;
  int depth = 0;
  size_t start = 0;
  for (size_t i = 0; i < values.size(); ++i) {
    switch (values[i]) {
      case '<':
      case '(':
      case '{': ++depth; break;
      case '>':
      case ')':
      case '}': --depth; break;
      case ',':
        if (depth == 0) {
          result.push_back(values.substr(start, i - start));
          start = i + 1;
        }
        break;
    }
  }
  if (start < values.size()) result.push_back(values.substr(start));
  return result;
}

std::optional<std::string_view> FunctionValueList(std::string_view declaration,
                                                  bool results) {
  size_t start = results ? declaration.find("->") : declaration.find("func @");
  if (start == std::string_view::npos) return std::nullopt;
  start = declaration.find('(', start);
  if (start == std::string_view::npos) return std::nullopt;
  int depth = 1;
  size_t end = start + 1;
  for (; end < declaration.size(); ++end) {
    if (declaration[end] == '(') ++depth;
    if (declaration[end] == ')' && --depth == 0) break;
  }
  if (depth != 0) return std::nullopt;
  return declaration.substr(start + 1, end - start - 1);
}

PJRT_NamedValue MakeInt64Property(const char* name, int64_t value) {
  PJRT_NamedValue property{};
  property.struct_size = PJRT_NamedValue_STRUCT_SIZE;
  property.name = name;
  property.name_size = strlen(name);
  property.type = PJRT_NamedValue_kInt64;
  property.int64_value = value;
  property.value_size = 1;
  return property;
}

}  // namespace

struct ExternalDeviceBufferRelease {
  void* device_buffer_ptr;
  void (*callback)(void* device_buffer_ptr, void* user_arg);
  void* user_arg;
};

void ReleaseExternalDeviceBuffer(void* user_data, iree_hal_buffer_t*) {
  auto release = std::unique_ptr<ExternalDeviceBufferRelease>(
      static_cast<ExternalDeviceBufferRelease*>(user_data));
  release->callback(release->device_buffer_ptr, release->user_arg);
}

// Some general conversion functions for managing around some API layering
// that is in flight. It is expected that most of this goes away over time.
namespace PJRTApiConverter {
namespace {

iree_status_t MapBufferTypeToElementType(
    PJRT_Buffer_Type buffer_type, iree_hal_element_type_t* element_type) {
  switch (buffer_type) {
    case PJRT_Buffer_Type_INVALID:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
    case PJRT_Buffer_Type_PRED:
      *element_type = IREE_HAL_ELEMENT_TYPE_BOOL_8;
      return iree_ok_status();
    case PJRT_Buffer_Type_S4:
      *element_type = IREE_HAL_ELEMENT_TYPE_SINT_4;
      return iree_ok_status();
    case PJRT_Buffer_Type_S8:
      *element_type = IREE_HAL_ELEMENT_TYPE_SINT_8;
      return iree_ok_status();
    case PJRT_Buffer_Type_S16:
      *element_type = IREE_HAL_ELEMENT_TYPE_SINT_16;
      return iree_ok_status();
    case PJRT_Buffer_Type_S32:
      *element_type = IREE_HAL_ELEMENT_TYPE_SINT_32;
      return iree_ok_status();
    case PJRT_Buffer_Type_S64:
      *element_type = IREE_HAL_ELEMENT_TYPE_SINT_64;
      return iree_ok_status();
    case PJRT_Buffer_Type_U4:
      *element_type = IREE_HAL_ELEMENT_TYPE_UINT_4;
      return iree_ok_status();
    case PJRT_Buffer_Type_U8:
      *element_type = IREE_HAL_ELEMENT_TYPE_UINT_8;
      return iree_ok_status();
    case PJRT_Buffer_Type_U16:
      *element_type = IREE_HAL_ELEMENT_TYPE_UINT_16;
      return iree_ok_status();
    case PJRT_Buffer_Type_U32:
      *element_type = IREE_HAL_ELEMENT_TYPE_UINT_32;
      return iree_ok_status();
    case PJRT_Buffer_Type_U64:
      *element_type = IREE_HAL_ELEMENT_TYPE_UINT_64;
      return iree_ok_status();
    case PJRT_Buffer_Type_F16:
      *element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_16;
      return iree_ok_status();
    case PJRT_Buffer_Type_F32:
      *element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_32;
      return iree_ok_status();
    case PJRT_Buffer_Type_F64:
      *element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_64;
      return iree_ok_status();
    case PJRT_Buffer_Type_BF16:
      *element_type = IREE_HAL_ELEMENT_TYPE_BFLOAT_16;
      return iree_ok_status();
    case PJRT_Buffer_Type_C64:
      *element_type = IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_64;
      return iree_ok_status();
    case PJRT_Buffer_Type_C128:
      *element_type = IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_128;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "conversion from unknown buffer type %d",
                              (int)buffer_type);
  }
}

iree_status_t MapElementTypeToMlirType(iree_hal_element_type_t element_type,
                                       char const** ty) {
  switch (element_type) {
    case PJRT_Buffer_Type_INVALID:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
    case IREE_HAL_ELEMENT_TYPE_BOOL_8:
      *ty = "i1";
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_SINT_4:
      *ty = "i4";
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_SINT_8:
      *ty = "i8";
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_SINT_16:
      *ty = "i16";
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_SINT_32:
      *ty = "i32";
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_SINT_64:
      *ty = "i64";
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_UINT_4:
      *ty = "ui4";
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_UINT_8:
      *ty = "ui8";
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_UINT_16:
      *ty = "ui16";
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_UINT_32:
      *ty = "ui32";
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_UINT_64:
      *ty = "ui64";
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_FLOAT_16:
      *ty = "f16";
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_FLOAT_32:
      *ty = "f32";
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_FLOAT_64:
      *ty = "f64";
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_BFLOAT_16:
      *ty = "bf16";
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_64:
      *ty = "complex<f32>";
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_128:
      *ty = "complex<f64>";
      return iree_ok_status();
    default:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "conversion from unknown iree hal element type %d",
          (int)element_type);
  }
}

}  // namespace
}  // namespace PJRTApiConverter

//===----------------------------------------------------------------------===//
// Error
//===----------------------------------------------------------------------===//

void ErrorInstance::BindApi(PJRT_Api* api) {
  api->PJRT_Error_Destroy = +[](PJRT_Error_Destroy_Args* args) {
    if (!args->error) return;
    delete ErrorInstance::FromError(args->error);
  };
  api->PJRT_Error_Message = +[](PJRT_Error_Message_Args* args) {
    auto* error = ErrorInstance::FromError(args->error);
    if (!error) {
      args->message = "OK";
      args->message_size = 2;
      return;
    }

    const std::string& message = error->message();
    args->message = message.data();
    args->message_size = message.size();
  };
  api->PJRT_Error_GetCode = +[](PJRT_Error_GetCode_Args* args) -> PJRT_Error* {
    auto* error = ErrorInstance::FromError(args->error);
    iree_status_code_t status_code = iree_status_code(error->status());
    switch (status_code) {
      case IREE_STATUS_CANCELLED:
        args->code = PJRT_Error_Code_CANCELLED;
        break;
      case IREE_STATUS_UNKNOWN:
        args->code = PJRT_Error_Code_UNKNOWN;
        break;
      case IREE_STATUS_INVALID_ARGUMENT:
        args->code = PJRT_Error_Code_INVALID_ARGUMENT;
        break;
      case IREE_STATUS_DEADLINE_EXCEEDED:
        args->code = PJRT_Error_Code_DEADLINE_EXCEEDED;
        break;
      case IREE_STATUS_NOT_FOUND:
        args->code = PJRT_Error_Code_NOT_FOUND;
        break;
      case IREE_STATUS_ALREADY_EXISTS:
        args->code = PJRT_Error_Code_ALREADY_EXISTS;
        break;
      case IREE_STATUS_PERMISSION_DENIED:
        args->code = PJRT_Error_Code_PERMISSION_DENIED;
        break;
      case IREE_STATUS_RESOURCE_EXHAUSTED:
        args->code = PJRT_Error_Code_RESOURCE_EXHAUSTED;
        break;
      case IREE_STATUS_FAILED_PRECONDITION:
        args->code = PJRT_Error_Code_FAILED_PRECONDITION;
        break;
      case IREE_STATUS_ABORTED:
        args->code = PJRT_Error_Code_ABORTED;
        break;
      case IREE_STATUS_OUT_OF_RANGE:
        args->code = PJRT_Error_Code_OUT_OF_RANGE;
        break;
      case IREE_STATUS_UNIMPLEMENTED:
        args->code = PJRT_Error_Code_UNIMPLEMENTED;
        break;
      case IREE_STATUS_INTERNAL:
        args->code = PJRT_Error_Code_INTERNAL;
        break;
      case IREE_STATUS_UNAVAILABLE:
        args->code = PJRT_Error_Code_UNAVAILABLE;
        break;
      case IREE_STATUS_DATA_LOSS:
        args->code = PJRT_Error_Code_DATA_LOSS;
        break;
      case IREE_STATUS_UNAUTHENTICATED:
        args->code = PJRT_Error_Code_UNAUTHENTICATED;
        break;
      case IREE_STATUS_DEFERRED:
        args->code = PJRT_Error_Code_UNKNOWN;  // No mapping
        break;
      case IREE_STATUS_INCOMPATIBLE:
        args->code = PJRT_Error_Code_NOT_FOUND;
        break;
      default:
        // Should not happen.
        args->code = PJRT_Error_Code_UNKNOWN;
    }
    return nullptr;
  };
}

const std::string& ErrorInstance::message() const {
  if (cached_message_.empty()) {
    std::string buffer;
    iree_host_size_t actual_len;
    buffer.resize(1024);  // TODO: Actually reallocate to full size on trunc.
    if (!iree_status_format(status_, buffer.size(), buffer.data(),
                            &actual_len)) {
      buffer.resize(actual_len);
      if (!iree_status_format(status_, buffer.size(), buffer.data(),
                              &actual_len)) {
        actual_len = 0;
      }
    }
    buffer.resize(actual_len);
    cached_message_ = std::move(buffer);
  }
  return cached_message_;
}

//===----------------------------------------------------------------------===//
// BufferInstance
//===----------------------------------------------------------------------===//

BufferInstance::~BufferInstance() = default;

BufferInstance::BufferInstance(
    DeviceInstance& device, iree::vm::ref<iree_hal_buffer_view_t> buffer_view)
    : device_(device), buffer_view_(std::move(buffer_view)) {
  IREE_CHECK_OK(device.CreateFence(&ready_fence_));
  IREE_CHECK_OK(device.CreateFence(&done_fence_));

  // Cache the dims.
  size_t rank = iree_hal_buffer_view_shape_rank(buffer_view_.get());
  const iree_hal_dim_t* dims =
      iree_hal_buffer_view_shape_dims(buffer_view_.get());
  dims_.resize(rank);
  for (size_t i = 0; i < rank; ++i) {
    dims_[i] = dims[i];
  }
}

void BufferInstance::ComputeLayout() {
  iree_hal_encoding_type_t encoding =
      iree_hal_buffer_view_encoding_type(buffer_view_.get());
  iree_hal_element_type_t element_type =
      iree_hal_buffer_view_element_type(buffer_view_.get());

  layout_.Reset();
  if (encoding == IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR &&
      iree_hal_element_is_byte_aligned(element_type)) {
    // It is not documented, but PJRT only supports device buffers with a tiled
    // layout.
    layout_.InitializeDenseRowMajorTiled(dims_.size());
  }
}

void BufferInstance::BindApi(PJRT_Api* api) {
  api->PJRT_Buffer_Destroy =
      +[](PJRT_Buffer_Destroy_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE_NAMED("PJRT_Buffer_Destroy");
    BufferInstance* buffer = BufferInstance::Unwrap(args->buffer);
    delete buffer;
    return nullptr;
  };
  api->PJRT_Buffer_ElementType =
      +[](PJRT_Buffer_ElementType_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE_NAMED("PJRT_Buffer_ElementType");
    BufferInstance* buffer = BufferInstance::Unwrap(args->buffer);
    auto element_type = buffer->element_type();
    if (!element_type) {
      return MakeError(iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                        "Unsupported PJRT buffer type"));
    }
    args->type = *element_type;
    return nullptr;
  };
  api->PJRT_Buffer_Dimensions =
      +[](PJRT_Buffer_Dimensions_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE_NAMED("PJRT_Buffer_Dimensions");
    BufferInstance* buffer = BufferInstance::Unwrap(args->buffer);
    args->dims = buffer->dims();
    args->num_dims = buffer->num_dims();
    return nullptr;
  };
  api->PJRT_Buffer_UnpaddedDimensions =
      +[](PJRT_Buffer_UnpaddedDimensions_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE_NAMED("PJRT_Buffer_UnpaddedDimensions");
    BufferInstance* buffer = BufferInstance::Unwrap(args->buffer);
    args->unpadded_dims = buffer->dims();
    args->num_dims = buffer->num_dims();
    return nullptr;
  };
  api->PJRT_Buffer_DynamicDimensionIndices =
      +[](PJRT_Buffer_DynamicDimensionIndices_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE_NAMED("PJRT_Buffer_DynamicDimensionIndices");
    // IREE's PJRT buffers always have fully resolved runtime dimensions.
    args->dynamic_dim_indices = nullptr;
    args->num_dynamic_dims = 0;
    return nullptr;
  };
  api->PJRT_Buffer_GetMemoryLayout =
      +[](PJRT_Buffer_GetMemoryLayout_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE_NAMED("PJRT_Buffer_GetMemoryLayout");
    BufferInstance* buffer = BufferInstance::Unwrap(args->buffer);
    const PJRT_Buffer_MemoryLayout* layout = buffer->layout();
    if (!layout) {
      return MakeError(
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "Unsupported PJRT layout for buffer view"));
    }
    args->layout = *layout;
    return nullptr;
  };
  api->PJRT_Buffer_ToHostBuffer =
      +[](PJRT_Buffer_ToHostBuffer_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE_NAMED("PJRT_Buffer_ToHostBuffer");
    BufferInstance* buffer = BufferInstance::Unwrap(args->src);
    if (!args->dst) {
      // Size query.
      return MakeError(buffer->GetHostSizeInBytes(&args->dst_size));
    } else {
      // Initiate transfer.
      return MakeError(
          buffer->CopyToHost(args->dst, args->dst_size,
                             reinterpret_cast<EventInstance**>(&args->event)));
    }
  };
  api->PJRT_Buffer_OnDeviceSizeInBytes =
      +[](PJRT_Buffer_OnDeviceSizeInBytes_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE_NAMED("PJRT_Buffer_OnDeviceSizeInBytes");
    BufferInstance* buffer = BufferInstance::Unwrap(args->buffer);
    iree_device_size_t size =
        iree_hal_buffer_view_byte_length(buffer->buffer_view());
    args->on_device_size_in_bytes = size;
    return nullptr;
  };
  api->PJRT_Buffer_Delete = +[](PJRT_Buffer_Delete_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE_NAMED("PJRT_Buffer_Delete");
    BufferInstance* buffer = BufferInstance::Unwrap(args->buffer);
    buffer->Delete();
    return nullptr;
  };
  api->PJRT_Buffer_IsDeleted =
      +[](PJRT_Buffer_IsDeleted_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE_NAMED("PJRT_Buffer_IsDeleted");
    BufferInstance* buffer = BufferInstance::Unwrap(args->buffer);
    args->is_deleted = buffer->is_deleted();
    return nullptr;
  };
  api->PJRT_Buffer_CopyToDevice =
      +[](PJRT_Buffer_CopyToDevice_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE_NAMED("PJRT_Buffer_CopyToDevice");
    auto* buffer = BufferInstance::Unwrap(args->buffer);
    if (DeviceInstance::Unwrap(args->dst_device) == &buffer->device()) {
      return MakeError(iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT, "buffer is already on destination device"));
    }
    return MakeError(iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "destination device is not in this single-device Metal client"));
  };
  api->PJRT_Buffer_CopyToMemory =
      +[](PJRT_Buffer_CopyToMemory_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE_NAMED("PJRT_Buffer_CopyToMemory");
    auto* buffer = BufferInstance::Unwrap(args->buffer);
    if (MemoryInstance::Unwrap(args->dst_memory) == buffer->device().memory()) {
      return MakeError(iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT, "buffer is already in destination memory"));
    }
    return MakeError(iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "destination memory is not in this single-memory Metal client"));
  };
  api->PJRT_Buffer_CopyRawToHost =
      +[](PJRT_Buffer_CopyRawToHost_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE_NAMED("PJRT_Buffer_CopyRawToHost");
    if (args->offset < 0 || args->transfer_size < 0) {
      return MakeError(iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT, "copy offset and size must be non-negative"));
    }
    return MakeError(BufferInstance::Unwrap(args->buffer)->CopyRawToHost(
        args->dst, static_cast<iree_device_size_t>(args->offset),
        static_cast<iree_host_size_t>(args->transfer_size),
        reinterpret_cast<EventInstance**>(&args->event)));
  };
  api->PJRT_Buffer_IsOnCpu =
      +[](PJRT_Buffer_IsOnCpu_Args* args) -> PJRT_Error* {
    args->is_on_cpu = BufferInstance::Unwrap(args->buffer)->is_on_cpu();
    return nullptr;
  };
  api->PJRT_Buffer_Device = +[](PJRT_Buffer_Device_Args* args) -> PJRT_Error* {
    args->device = BufferInstance::Unwrap(args->buffer)->device();
    return nullptr;
  };
  api->PJRT_Buffer_Memory = +[](PJRT_Buffer_Memory_Args* args) -> PJRT_Error* {
    args->memory = reinterpret_cast<PJRT_Memory*>(
        BufferInstance::Unwrap(args->buffer)->device().memory());
    return nullptr;
  };
  api->PJRT_Buffer_ReadyEvent =
      +[](PJRT_Buffer_ReadyEvent_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE_NAMED("PJRT_Buffer_ReadyEvent");
    BufferInstance* buffer = BufferInstance::Unwrap(args->buffer);
    args->event = reinterpret_cast<PJRT_Event*>(
        new EventInstance(retain_ref(buffer->ready_fence())));
    return nullptr;
  };
  // TODO: Rework the API to be Aliases(b1, b2) to let the plugin explicitly
  // check for aliases.
  api->PJRT_Buffer_UnsafePointer =
      +[](PJRT_Buffer_UnsafePointer_Args* args) -> PJRT_Error* {
    BufferInstance* buffer = BufferInstance::Unwrap(args->buffer);
    return MakeError(buffer->UnsafePointer(&args->buffer_pointer));
  };
  api->PJRT_Buffer_IncreaseExternalReferenceCount =
      +[](PJRT_Buffer_IncreaseExternalReferenceCount_Args* args)
      -> PJRT_Error* {
    return MakeError(BufferInstance::Unwrap(args->buffer)
                         ->IncreaseExternalReferenceCount());
  };
  api->PJRT_Buffer_DecreaseExternalReferenceCount =
      +[](PJRT_Buffer_DecreaseExternalReferenceCount_Args* args)
      -> PJRT_Error* {
    return MakeError(BufferInstance::Unwrap(args->buffer)
                         ->DecreaseExternalReferenceCount());
  };
  api->PJRT_Buffer_OpaqueDeviceMemoryDataPointer =
      +[](PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args* args)
      -> PJRT_Error* {
    return MakeError(BufferInstance::Unwrap(args->buffer)
                         ->OpaqueDeviceMemoryDataPointer(
                             &args->device_memory_ptr));
  };
}

iree_status_t BufferInstance::GetHostSizeInBytes(iree_host_size_t* host_size) {
  *host_size = iree_hal_buffer_view_byte_length(buffer_view());
  return iree_ok_status();
}

iree_status_t BufferInstance::AsyncDeallocate() {
  IREE_TRACE_SCOPE();
  if (is_deleted_) {
    return iree_ok_status();
  }
  is_deleted_ = true;
  return IreeApi::hal_device_queue_dealloca(
      device().device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/iree_hal_fence_semaphore_list(done_fence()),
      /*signal_semaphore_list=*/iree_hal_semaphore_list_empty(),
      iree_hal_buffer_view_buffer(buffer_view_.get()),
      IREE_HAL_DEALLOCA_FLAG_NONE);
}

iree_status_t BufferInstance::Delete() {
  IREE_TRACE_SCOPE();
  is_deleted_ = true;
  if (external_ref_count_.load() == 0) {
    buffer_view_.reset();
  }
  return iree_ok_status();
}

iree_status_t BufferInstance::IncreaseExternalReferenceCount() {
  IREE_TRACE_SCOPE();
  if (is_deleted_ || !buffer_view_) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot externally reference a deleted buffer");
  }
  external_ref_count_.fetch_add(1);
  return iree_ok_status();
}

iree_status_t BufferInstance::DecreaseExternalReferenceCount() {
  IREE_TRACE_SCOPE();
  int previous_count = external_ref_count_.fetch_sub(1);
  if (previous_count <= 0) {
    external_ref_count_.fetch_add(1);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "buffer external reference count is already zero");
  }
  if (previous_count == 1 && is_deleted_) {
    buffer_view_.reset();
  }
  return iree_ok_status();
}

iree_status_t BufferInstance::UnsafePointer(uintptr_t* out_ptr) {
  IREE_TRACE_SCOPE();
  if (!buffer_view_) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot access a deleted buffer");
  }
  iree_hal_external_buffer_t external_buffer;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_export_buffer(
      device_.device_allocator(),
      iree_hal_buffer_view_buffer(buffer_view_.get()),
      IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION,
      IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &external_buffer));
  *out_ptr = static_cast<uintptr_t>(
      external_buffer.handle.device_allocation.ptr);
  return iree_ok_status();
}

iree_status_t BufferInstance::OpaqueDeviceMemoryDataPointer(void** out_ptr) {
  IREE_TRACE_SCOPE();
  if (external_ref_count_.load() <= 0 || !buffer_view_) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "opaque device pointer requires an external buffer reference");
  }
  iree_hal_external_buffer_t external_buffer;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_export_buffer(
      device_.device_allocator(),
      iree_hal_buffer_view_buffer(buffer_view_.get()),
      IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION,
      IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &external_buffer));
  *out_ptr = reinterpret_cast<void*>(
      static_cast<uintptr_t>(external_buffer.handle.device_allocation.ptr));
  return iree_ok_status();
}

iree_status_t BufferInstance::CopyToHost(void* dst, iree_host_size_t dst_size,
                                         EventInstance** out_done_event) {
  return CopyRawToHost(dst, /*source_offset=*/0, dst_size, out_done_event);
}

iree_status_t BufferInstance::CopyRawToHost(
    void* dst, iree_device_size_t source_offset, iree_host_size_t dst_size,
    EventInstance** out_done_event) {
  if (!buffer_view_) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot copy a deleted buffer");
  }
  iree_device_size_t buffer_size =
      iree_hal_buffer_view_byte_length(buffer_view_.get());
  if (source_offset > buffer_size || dst_size > buffer_size - source_offset) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "copy range [%" PRIu64 ", %" PRIu64 ") exceeds buffer size %" PRIu64,
        static_cast<uint64_t>(source_offset),
        static_cast<uint64_t>(source_offset + dst_size),
        static_cast<uint64_t>(buffer_size));
  }
  if (dst_size != 0 && !dst) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "copy destination must not be null");
  }
  // Use a data structure to handle intermediary buffer when necessary. This
  // needs to include the destination and aligned buffer, along with the size
  // so the destination can be mem-copied if necessary.
  struct CopyToHostData {
    void* alloc;
    void* aligned;
    void* dst;
    size_t size;
    // Fence will be signaled when copy to host is complete.
    iree::vm::ref<iree_hal_fence_t> copy_done_fence;
  };

  //  Configure a default structure that writes directly to dst.
  const size_t alignment = 64;
  struct CopyToHostData* copy_to_host_data = new CopyToHostData;
  copy_to_host_data->alloc = nullptr;
  copy_to_host_data->aligned = dst;
  copy_to_host_data->dst = dst;
  copy_to_host_data->size = dst_size;

  // If the destination is unaligned we need to write to an intermediary buffer.
  if (((uintptr_t)dst) & (alignment - 1)) {
    const size_t alignment_size = alignment + dst_size + sizeof(uintptr_t);
    char* alloc = new char[alignment_size];
    copy_to_host_data->alloc = alloc;
    copy_to_host_data->aligned =
        (void*)((((uintptr_t)alloc + alignment) & ~(uintptr_t)(alignment - 1)));
  }

  // Import the destination (host) buffer as an iree_hal_buffer_t so that we
  // can issue copy commands.
  iree::vm::ref<iree_hal_buffer_t> dst_buffer;
  iree_hal_buffer_params_t dst_buffer_params = {
      /*usage=*/IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET,
      // TODO: We should be able to use WRITE access here since the buffer
      // is never actually mapped to read back out (just accessed through the
      // void* later). However, that seems to cause the memory to never be
      // committed and the interaction aborted.
      /*access=*/IREE_HAL_MEMORY_ACCESS_ALL,
      /*type=*/IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
          IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
  };
  iree_hal_external_buffer_t dst_external_buffer;
  memset(&dst_external_buffer, 0, sizeof(dst_external_buffer));
  dst_external_buffer.type = IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION;
  dst_external_buffer.flags = IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE;
  dst_external_buffer.size = dst_size;
  dst_external_buffer.handle.host_allocation.ptr = copy_to_host_data->aligned;
  IREE_RETURN_IF_ERROR(IreeApi::hal_allocator_import_buffer(
      device_.device_allocator(), dst_buffer_params, &dst_external_buffer,
      /*release_callback=*/iree_hal_buffer_release_callback_null(),
      &dst_buffer));

  // Create the transfer command buffer.
  iree::vm::ref<iree_hal_command_buffer_t> transfer_cb;
  iree_hal_transfer_command_t transfer_command;
  memset(&transfer_command, 0, sizeof(transfer_command));
  transfer_command.type = IREE_HAL_TRANSFER_COMMAND_TYPE_COPY;
  transfer_command.copy.source_buffer =
      iree_hal_buffer_view_buffer(buffer_view());
  transfer_command.copy.source_offset = source_offset;
  transfer_command.copy.target_buffer = dst_buffer.get();
  transfer_command.copy.target_offset = 0;
  transfer_command.copy.length = dst_size;
  IREE_RETURN_IF_ERROR(iree_hal_create_transfer_command_buffer(
      device_.device(), IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_QUEUE_AFFINITY_ANY,
      /*transfer_count=*/1, &transfer_command, &transfer_cb));
  dst_buffer.reset();

  iree::vm::ref<iree_hal_semaphore_t> semaphore;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      device_.device(), IREE_HAL_QUEUE_AFFINITY_ANY, 0ull,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &semaphore));

  // Signaled when `dst_buffer` is ready to be consumed.
  iree::vm::ref<iree_hal_fence_t> dst_buffer_ready_fence;
  IREE_RETURN_IF_ERROR(IreeApi::hal_fence_create_at(
      semaphore.get(), 1ull, device_.client().host_allocator(),
      &dst_buffer_ready_fence));

  // Signaled when copy to host is complete.
  IREE_RETURN_IF_ERROR(IreeApi::hal_fence_create_at(
      semaphore.get(), 2ull, device_.client().host_allocator(),
      &(copy_to_host_data->copy_done_fence)));

  auto dst_buffer_callback = [](PJRT_Error* error, void* user_data) {
    const ErrorInstance* error_instance = ErrorInstance::FromError(error);
    auto* copy_data = static_cast<CopyToHostData*>(user_data);

    if (!error) {
      // If there is an allocated buffer we need to copy to the destination.
      if (copy_data->alloc) {
        std::memcpy(copy_data->dst, copy_data->aligned, copy_data->size);
      }
      iree_hal_fence_signal(copy_data->copy_done_fence.get());
    } else {
      iree_hal_fence_fail(copy_data->copy_done_fence.get(),
                          error_instance->status());
    }

    if (copy_data->alloc) {
      delete[] static_cast<char*>(copy_data->alloc);
    }
    delete copy_data;
    delete error_instance;
  };

  // This callback simply deletes the `dst_buffer_ready_event`. We could perform
  // this deletion in the `dst_buffer_callback`, but this would result in the
  // callback thread of `dst_buffer_ready_event` detaching from the main thread,
  // potentially resulting in the callback thread outliving the main thread.
  auto copy_done_callback = [](PJRT_Error* error, void* user_data) {
    EventInstance* dst_buffer_ready_event =
        static_cast<EventInstance*>(user_data);
    delete dst_buffer_ready_event;
    delete ErrorInstance::FromError(error);
  };

  auto dst_buffer_ready_event =
      new EventInstance(retain_ref(dst_buffer_ready_fence));
  dst_buffer_ready_event->OnReady(dst_buffer_callback, copy_to_host_data);

  auto copy_done_event =
      new EventInstance(retain_ref(copy_to_host_data->copy_done_fence));
  copy_done_event->OnReady(copy_done_callback, dst_buffer_ready_event);

  IREE_RETURN_IF_ERROR(IreeApi::hal_device_queue_execute(
      device_.device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/iree_hal_fence_semaphore_list(ready_fence_.get()),
      /*signal_semaphore_list=*/
      iree_hal_fence_semaphore_list(dst_buffer_ready_fence.get()),
      transfer_cb.get()));

  *out_done_event = copy_done_event;
  return iree_ok_status();
}

iree_status_t BufferInstance::AdvanceReadyFence(iree_hal_semaphore_t* semaphore,
                                                uint64_t timepoint) {
  return IreeApi::hal_fence_insert(ready_fence_.get(), semaphore, timepoint);
}

iree_status_t BufferInstance::AdvanceDoneFence(iree_hal_semaphore_t* semaphore,
                                               uint64_t timepoint) {
  return IreeApi::hal_fence_insert(done_fence_.get(), semaphore, timepoint);
}

std::optional<PJRT_Buffer_Type> BufferInstance::element_type() {
  iree_hal_element_type_t hal_element_type =
      iree_hal_buffer_view_element_type(buffer_view());

  // TODO: Cascade on bit-field sub-types to avoid large linear scan.
  switch (hal_element_type) {
    // TODO: How do I interpret signless?
    case IREE_HAL_ELEMENT_TYPE_BOOL_8:
      return PJRT_Buffer_Type_PRED;
    case IREE_HAL_ELEMENT_TYPE_INT_4:
      return PJRT_Buffer_Type_S4;
    case IREE_HAL_ELEMENT_TYPE_INT_8:
      return PJRT_Buffer_Type_S8;
    case IREE_HAL_ELEMENT_TYPE_INT_16:
      return PJRT_Buffer_Type_S16;
    case IREE_HAL_ELEMENT_TYPE_INT_32:
      return PJRT_Buffer_Type_S32;
    case IREE_HAL_ELEMENT_TYPE_INT_64:
      return PJRT_Buffer_Type_S64;
    case IREE_HAL_ELEMENT_TYPE_SINT_4:
      return PJRT_Buffer_Type_S4;
    case IREE_HAL_ELEMENT_TYPE_SINT_8:
      return PJRT_Buffer_Type_S8;
    case IREE_HAL_ELEMENT_TYPE_SINT_16:
      return PJRT_Buffer_Type_S16;
    case IREE_HAL_ELEMENT_TYPE_SINT_32:
      return PJRT_Buffer_Type_S32;
    case IREE_HAL_ELEMENT_TYPE_SINT_64:
      return PJRT_Buffer_Type_S64;
    case IREE_HAL_ELEMENT_TYPE_UINT_4:
      return PJRT_Buffer_Type_U4;
    case IREE_HAL_ELEMENT_TYPE_UINT_8:
      return PJRT_Buffer_Type_U8;
    case IREE_HAL_ELEMENT_TYPE_UINT_16:
      return PJRT_Buffer_Type_U16;
    case IREE_HAL_ELEMENT_TYPE_UINT_32:
      return PJRT_Buffer_Type_U32;
    case IREE_HAL_ELEMENT_TYPE_UINT_64:
      return PJRT_Buffer_Type_U64;
    case IREE_HAL_ELEMENT_TYPE_FLOAT_16:
      return PJRT_Buffer_Type_F16;
    case IREE_HAL_ELEMENT_TYPE_FLOAT_32:
      return PJRT_Buffer_Type_F32;
    case IREE_HAL_ELEMENT_TYPE_FLOAT_64:
      return PJRT_Buffer_Type_F64;
    case IREE_HAL_ELEMENT_TYPE_BFLOAT_16:
      return PJRT_Buffer_Type_BF16;
    case IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_64:
      return PJRT_Buffer_Type_C64;
    case IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_128:
      return PJRT_Buffer_Type_C128;
    default:
      return {};
  }
}

//===----------------------------------------------------------------------===//
// DeviceDescription
//===----------------------------------------------------------------------===//

DeviceDescription::~DeviceDescription() = default;

void DeviceDescription::BindApi(PJRT_Api* api) {
  api->PJRT_DeviceDescription_Id =
      +[](PJRT_DeviceDescription_Id_Args* args) -> PJRT_Error* {
    args->id = DeviceDescription::Unwrap(args->device_description)->client_id();
    return nullptr;
  };
  api->PJRT_DeviceDescription_ProcessIndex =
      +[](PJRT_DeviceDescription_ProcessIndex_Args* args) -> PJRT_Error* {
    args->process_index =
        DeviceDescription::Unwrap(args->device_description)->process_index();
    return nullptr;
  };
  api->PJRT_DeviceDescription_Attributes =
      +[](PJRT_DeviceDescription_Attributes_Args* args) -> PJRT_Error* {
    // TODO: Implement something.
    args->num_attributes = 0;
    args->attributes = nullptr;
    return nullptr;
  };
  api->PJRT_DeviceDescription_Kind =
      +[](PJRT_DeviceDescription_Kind_Args* args) -> PJRT_Error* {
    auto sv =
        DeviceDescription::Unwrap(args->device_description)->kind_string();
    args->device_kind = sv.data();
    args->device_kind_size = sv.size();
    return nullptr;
  };
  api->PJRT_DeviceDescription_DebugString =
      +[](PJRT_DeviceDescription_DebugString_Args* args) -> PJRT_Error* {
    auto sv =
        DeviceDescription::Unwrap(args->device_description)->debug_string();
    args->debug_string = sv.data();
    args->debug_string_size = sv.size();
    return nullptr;
  };
  api->PJRT_DeviceDescription_ToString =
      +[](PJRT_DeviceDescription_ToString_Args* args) -> PJRT_Error* {
    auto sv =
        DeviceDescription::Unwrap(args->device_description)->user_string();
    args->to_string = sv.data();
    args->to_string_size = sv.size();
    return nullptr;
  };
}

//===----------------------------------------------------------------------===//
// MemoryInstance
//===----------------------------------------------------------------------===//

void MemoryInstance::BindApi(PJRT_Api* api) {
  api->PJRT_Memory_Id = +[](PJRT_Memory_Id_Args* args) -> PJRT_Error* {
    args->id = MemoryInstance::Unwrap(args->memory)->Id();
    return nullptr;
  };
  api->PJRT_Memory_Kind = +[](PJRT_Memory_Kind_Args* args) -> PJRT_Error* {
    auto kind = MemoryInstance::Unwrap(args->memory)->Kind();
    args->kind = kind.data();
    args->kind_size = kind.size();
    return nullptr;
  };
  api->PJRT_Memory_Kind_Id =
      +[](PJRT_Memory_Kind_Id_Args* args) -> PJRT_Error* {
    // There is one memory kind in the Metal client.
    args->kind_id = 0;
    return nullptr;
  };
  api->PJRT_Memory_ToString =
      +[](PJRT_Memory_ToString_Args* args) -> PJRT_Error* {
    auto to_string = MemoryInstance::Unwrap(args->memory)->ToString();
    args->to_string = to_string.data();
    args->to_string_size = to_string.size();
    return nullptr;
  };
  api->PJRT_Memory_DebugString =
      +[](PJRT_Memory_DebugString_Args* args) -> PJRT_Error* {
    auto debug_string = MemoryInstance::Unwrap(args->memory)->DebugString();
    args->debug_string = debug_string.data();
    args->debug_string_size = debug_string.size();
    return nullptr;
  };
  api->PJRT_Memory_AddressableByDevices =
      +[](PJRT_Memory_AddressableByDevices_Args* args) -> PJRT_Error* {
    args->num_devices = 1;
    args->devices = reinterpret_cast<PJRT_Device* const*>(
        &MemoryInstance::Unwrap(args->memory)->device_);
    return nullptr;
  };
}

int MemoryInstance::Id() { return device_->device_description()->device_id(); }
std::string_view MemoryInstance::Kind() { return kDeviceMemoryKind; }
std::string_view MemoryInstance::DebugString() {
  return device_->device_description()->debug_string();
}
std::string_view MemoryInstance::ToString() {
  return device_->device_description()->user_string();
}

//===----------------------------------------------------------------------===//
// DeviceInstance
//===----------------------------------------------------------------------===//

DeviceInstance::~DeviceInstance() = default;

void DeviceInstance::BindApi(PJRT_Api* api) {
  api->PJRT_Device_IsAddressable =
      +[](PJRT_Device_IsAddressable_Args* args) -> PJRT_Error* {
    args->is_addressable =
        DeviceInstance::Unwrap(args->device)->is_addressable();
    return nullptr;
  };
  api->PJRT_Device_LocalHardwareId =
      +[](PJRT_Device_LocalHardwareId_Args* args) -> PJRT_Error* {
    args->local_hardware_id =
        DeviceInstance::Unwrap(args->device)->local_hardware_id();
    return nullptr;
  };
  api->PJRT_Device_AddressableMemories =
      +[](PJRT_Device_AddressableMemories_Args* args) -> PJRT_Error* {
    const auto& memories = DeviceInstance::Unwrap(args->device)->memories_;
    args->memories = reinterpret_cast<PJRT_Memory* const*>(memories.data());
    args->num_memories = memories.size();
    return nullptr;
  };
  api->PJRT_Device_DefaultMemory =
      +[](PJRT_Device_DefaultMemory_Args* args) -> PJRT_Error* {
    args->memory = reinterpret_cast<PJRT_Memory*>(
        DeviceInstance::Unwrap(args->device)->memories_[0]);
    return nullptr;
  };
  api->PJRT_Device_GetDescription =
      +[](PJRT_Device_GetDescription_Args* args) -> PJRT_Error* {
    args->device_description = reinterpret_cast<PJRT_DeviceDescription*>(
        DeviceInstance::Unwrap(args->device)->device_description());
    return nullptr;
  };
  api->PJRT_Device_MemoryStats =
      +[](PJRT_Device_MemoryStats_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE_NAMED("PJRT_Device_MemoryStats");
    return MakeError(DeviceInstance::Unwrap(args->device)->MemoryStats(args));
  };
}

iree_status_t DeviceInstance::MemoryStats(PJRT_Device_MemoryStats_Args* args) {
  IREE_RETURN_IF_ERROR(OpenDevice());
  iree_hal_allocator_statistics_t statistics;
  iree_hal_allocator_query_statistics(device_allocator(), &statistics);
  args->bytes_in_use = 0;
  args->peak_bytes_in_use = 0;
  args->peak_bytes_in_use_is_set = false;
  args->num_allocs = 0;
  args->num_allocs_is_set = false;
  args->largest_alloc_size = 0;
  args->largest_alloc_size_is_set = false;
  args->bytes_limit = 0;
  args->bytes_limit_is_set = false;
  args->bytes_reserved = 0;
  args->bytes_reserved_is_set = false;
  args->peak_bytes_reserved = 0;
  args->peak_bytes_reserved_is_set = false;
  args->bytes_reservable_limit = 0;
  args->bytes_reservable_limit_is_set = false;
  args->largest_free_block_bytes = 0;
  args->largest_free_block_bytes_is_set = false;
  args->pool_bytes = 0;
  args->pool_bytes_is_set = false;
  args->peak_pool_bytes = 0;
  args->peak_pool_bytes_is_set = false;
#if IREE_STATISTICS_ENABLE
  args->bytes_in_use = static_cast<int64_t>(
      statistics.host_bytes_allocated - statistics.host_bytes_freed +
      statistics.device_bytes_allocated - statistics.device_bytes_freed);
  args->peak_bytes_in_use = static_cast<int64_t>(
      statistics.host_bytes_peak + statistics.device_bytes_peak);
  args->peak_bytes_in_use_is_set = true;
#endif
  return iree_ok_status();
}

iree_status_t DeviceInstance::CreateFence(iree_hal_fence_t** out_fence) {
  return IreeApi::hal_fence_create(/*capacity=*/2, client_.host_allocator(),
                                   out_fence);
}

iree_status_t DeviceInstance::OpenDevice() {
  if (device_) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_hal_driver_create_device_by_id(
      driver_, /*device_id=*/info_.device_id(),
      /*param_count=*/0, /*params=*/nullptr, client_.host_allocator(),
      &device_));
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      device(), IREE_HAL_QUEUE_AFFINITY_ANY, 0ull,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &main_timeline_));
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      device(), IREE_HAL_QUEUE_AFFINITY_ANY, 0ull,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &transfer_timeline_));

  return iree_ok_status();
}

iree_status_t DeviceInstance::HostBufferToDeviceSplat(
    const void* data, PJRT_Buffer_Type type, const int64_t* dims,
    size_t num_dims, EventInstance** out_done_with_host_buffer_event,
    BufferInstance** out_buffer) {
  // Map element type:
  iree_hal_element_type_t element_type;
  IREE_RETURN_IF_ERROR(
      PJRTApiConverter::MapBufferTypeToElementType(type, &element_type));
  // TODO: Do something sensible with sub-byte aligned types.
  if (IREE_UNLIKELY(iree_hal_element_bit_count(element_type) == 0) ||
      IREE_UNLIKELY(!iree_hal_element_is_byte_aligned(element_type))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "opaque and sub-byte aligned element types cannot be indexed");
  }
  iree_device_size_t element_type_byte_size =
      iree_hal_element_dense_byte_count(element_type);

  // Handle strided layouts and shape.
  std::array<iree_hal_dim_t, kMaxDims> shape;
  if (num_dims > shape.size()) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "only supports up to %d dims but got %d",
                            (int)shape.size(), (int)num_dims);
  }

  iree_device_size_t byte_length = element_type_byte_size;
  for (int i = 0, s = num_dims; i < s; ++i) {
    byte_length *= dims[i];
    shape[i] = dims[i];
  }

  iree::vm::ref<iree_hal_buffer_t> buffer;

  // Allocate on stream. We serialize across 3 timepoints:
  //   0. Last transfer complete
  //   1. Allocation
  //   2. Fill is complete
  // There are various ways to be smarter about this but without more
  // information from the caller, this is ok. If we wanted to favor smaller
  // allocation scopes, it may be desirable to join with the main execution
  // timeline, but that would obviously serialize more.
  uint64_t wait_transfer_start = last_transfer_timepoint_;
  uint64_t signal_alloca_complete = ++last_transfer_timepoint_;
  uint64_t signal_copy_complete = ++last_transfer_timepoint_;
  iree_hal_buffer_params_t params;
  memset(&params, 0, sizeof(params));
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage =
      IREE_HAL_BUFFER_USAGE_DEFAULT | IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET;
  IREE_RETURN_IF_ERROR(IreeApi::hal_device_queue_alloca(
      device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/
      {1, &transfer_timeline_, &wait_transfer_start},
      /*signal_semaphore_list=*/
      {1, &transfer_timeline_, &signal_alloca_complete},
      IREE_HAL_ALLOCATOR_POOL_DEFAULT, params, byte_length,
      IREE_HAL_ALLOCA_FLAG_NONE, &buffer));

  // Queue up the buffer fill for splatting:
  iree::vm::ref<iree_hal_command_buffer_t> transfer_cb;
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_create(
      device(), IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_ANY, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &transfer_cb));
  IREE_CHECK_OK(iree_hal_command_buffer_begin(transfer_cb.get()));
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_fill_buffer(
      transfer_cb.get(), iree_hal_make_buffer_ref(buffer.get(), 0, byte_length),
      data, element_type_byte_size, IREE_HAL_FILL_FLAG_NONE));
  IREE_CHECK_OK(iree_hal_command_buffer_end(transfer_cb.get()));

  // Execute the enqueued splat:
  IREE_RETURN_IF_ERROR(IreeApi::hal_device_queue_execute(
      device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/
      {1, &transfer_timeline_, &signal_alloca_complete},
      /*signal_semaphore_list=*/
      {1, &transfer_timeline_, &signal_copy_complete}, transfer_cb.get()));

  // Wrap in a buffer view and return:
  iree::vm::ref<iree_hal_buffer_view_t> result_buffer_view;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_create(
      buffer.get(), num_dims, &shape[0], element_type,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, client_.host_allocator(),
      &result_buffer_view));

  auto instance = new BufferInstance(*this, std::move(result_buffer_view));
  instance->AdvanceReadyFence(transfer_timeline_.get(), signal_copy_complete);
  instance->AdvanceDoneFence(transfer_timeline_.get(), signal_copy_complete);
  *out_buffer = instance;

  // Splat so the data is no longer required:
  *out_done_with_host_buffer_event = new EventInstance(/*fence=*/nullptr);

  return iree_ok_status();
}

iree_status_t DeviceInstance::HostBufferToDeviceZeroDim(
    PJRT_Buffer_Type type, const int64_t* dims, size_t num_dims,
    EventInstance** out_done_with_host_buffer_event,
    BufferInstance** out_buffer) {
  // Map element type:
  iree_hal_element_type_t element_type;
  IREE_RETURN_IF_ERROR(
      PJRTApiConverter::MapBufferTypeToElementType(type, &element_type));

  std::array<iree_hal_dim_t, kMaxDims> shape;
  if (num_dims > shape.size()) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "only supports up to %d dims but got %d",
                            (int)shape.size(), (int)num_dims);
  }

  for (int i = 0, s = num_dims; i < s; ++i) {
    shape[i] = dims[i];
  }

  // We only need to wait for previous transfer and allocate data:
  uint64_t wait_transfer_start = last_transfer_timepoint_;
  uint64_t signal_alloca_complete = ++last_transfer_timepoint_;

  iree_hal_buffer_params_t params;
  iree::vm::ref<iree_hal_buffer_t> buffer;
  memset(&params, 0, sizeof(params));
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage =
      IREE_HAL_BUFFER_USAGE_DEFAULT | IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET;
  IREE_RETURN_IF_ERROR(IreeApi::hal_device_queue_alloca(
      device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/
      {1, &transfer_timeline_, &wait_transfer_start},
      /*signal_semaphore_list=*/
      {1, &transfer_timeline_, &signal_alloca_complete},
      IREE_HAL_ALLOCATOR_POOL_DEFAULT, params,
      iree_hal_element_dense_byte_count(element_type),
      IREE_HAL_ALLOCA_FLAG_NONE, &buffer));

  // Wrap in a buffer view and return.
  iree::vm::ref<iree_hal_buffer_view_t> result_buffer_view;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_create(
      buffer.get(), num_dims, &shape[0], element_type,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, client_.host_allocator(),
      &result_buffer_view));

  auto instance = new BufferInstance(*this, std::move(result_buffer_view));
  instance->AdvanceReadyFence(transfer_timeline_.get(), signal_alloca_complete);
  instance->AdvanceDoneFence(transfer_timeline_.get(), signal_alloca_complete);
  *out_buffer = instance;

  // Degenerate case ignores the data so we can just return:
  *out_done_with_host_buffer_event = new EventInstance(/*fence=*/nullptr);

  return iree_ok_status();
}

iree_status_t DeviceInstance::TransposeBroadcastDeviceBuffer(
    BufferInstance* buffer, iree_hal_element_type_t element_type,
    const iree_hal_dim_t* input_dims, const iree_hal_dim_t* output_dims,
    const int64_t* perms, size_t num_dims,
    PJRT_HostBufferSemantics host_buffer_semantics,
    EventInstance** out_done_with_host_buffer_event,
    BufferInstance** out_buffer) {
  if (num_dims > kMaxDims) {
    auto ret = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "number of dimensions exceeded max supported");
  }

  std::array<iree_hal_dim_t, kMaxDims> transpose_dims;
  for (int i = 0; i < num_dims; ++i) {
    transpose_dims[i] = input_dims[perms[i]];
  }

  auto typeBuilder = [](const iree_hal_dim_t* dims, int64_t num_dims,
                        const char* ty) {
    std::stringstream ss;
    ss << "tensor<";
    for (int i = 0; i < num_dims; ++i) {
      ss << dims[i] << "x";
    }

    ss << ty << ">";
    return ss.str();
  };

  auto arrayBuilder = [](const int64_t* vals, int64_t sz) {
    std::stringstream ss;
    ss << " {permutation = array<i64: " << vals[0];
    for (int i = 1; i < sz; ++i) ss << ", " << vals[i];
    ss << ">}";
    return ss.str();
  };

  auto broadcastBuilder = [](int64_t sz) {
    std::stringstream ss;
    ss << "{broadcast_dimensions = array<i64: 0";
    for (int i = 1; i < sz; ++i) ss << ", " << i;
    ss << ">}";
    return ss.str();
  };

  const char* mlir_ty;
  IREE_RETURN_IF_ERROR(
      PJRTApiConverter::MapElementTypeToMlirType(element_type, &mlir_ty));

  auto input_ty = typeBuilder(input_dims, num_dims, mlir_ty);
  auto transpose_ty = typeBuilder(transpose_dims.data(), num_dims, mlir_ty);
  auto output_ty = typeBuilder(output_dims, num_dims, mlir_ty);
  auto perms_str = arrayBuilder(perms, num_dims);
  auto broadcast_str = broadcastBuilder(num_dims);

  const char* program_literal = R"(func.func @main(%%arg0 : %1$s) -> (%3$s) {
   %%0 = "stablehlo.transpose"(%%arg0) %4$s : (%1$s) -> %2$s
   %%1 = "stablehlo.broadcast_in_dim"(%%0) %5$s : (%2$s) -> %3$s
   return %%1 : %3$s
  })";
  char transpose_program[512];
  size_t program_len = std::snprintf(
      transpose_program, sizeof(transpose_program), program_literal,
      input_ty.c_str(), transpose_ty.c_str(), output_ty.c_str(),
      perms_str.c_str(), broadcast_str.c_str());
  if (program_len > sizeof(transpose_program)) {
    auto ret = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "program size exceeded limit");
  }

  // Create an on stack program:
  PJRT_Program program;
  program.code = transpose_program;
  program.code_size = program_len;
  program.format = kMlirFormat.data();
  program.format_size = kMlirFormat.size();

  // Compile program and check for errors:
  LoadedExecutableInstance* executable;
  auto* error = this->client().Compile(&program, {}, &executable);
  if (error) {
    auto errinst = ErrorInstance::FromError(error);
    auto ret = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "transposition program failed to build");
    delete errinst;
    return ret;
  }

  PJRT_Buffer* input = *buffer;
  PJRT_Buffer** input_list = &input;

  PJRT_Buffer* output;
  PJRT_Buffer** output_list = &output;
  PJRT_Event* event;

  // Build the execution arguments for transposing the loaded memory:
  PJRT_LoadedExecutable_Execute_Args execute_args;
  memset(&execute_args, 0, sizeof(execute_args));

  PJRT_ExecuteOptions execute_options;
  memset(&execute_options, 0, sizeof(execute_options));
  execute_args.executable = *executable;
  execute_args.options = &execute_options;
  execute_args.argument_lists = &input_list;
  execute_args.output_lists = &output_list;
  execute_args.num_devices = 1;
  execute_args.num_args = 1;
  execute_args.device_complete_events = &event;

  // We do no support specifying the device yet.
  execute_args.execute_device = nullptr;

  auto err = executable->BatchExecute(&execute_args);
  delete executable;

  if (err) {
    return err;
  }

  *out_buffer = BufferInstance::Unwrap(output);
  *out_done_with_host_buffer_event = EventInstance::Unwrap(event);

  return iree_ok_status();
}

iree_status_t DeviceInstance::HostBufferToDevice(
    const void* data, PJRT_Buffer_Type type, const int64_t* dims,
    size_t num_dims, const int64_t* byte_strides, size_t num_byte_strides,
    PJRT_HostBufferSemantics host_buffer_semantics,
    EventInstance** out_done_with_host_buffer_event,
    BufferInstance** out_buffer) {
  IREE_RETURN_IF_ERROR(OpenDevice());

  // Map element type.
  iree_hal_element_type_t element_type;
  IREE_RETURN_IF_ERROR(
      PJRTApiConverter::MapBufferTypeToElementType(type, &element_type));
  // TODO: Do something sensible with sub-byte aligned types.
  if (IREE_UNLIKELY(iree_hal_element_bit_count(element_type) == 0) ||
      IREE_UNLIKELY(!iree_hal_element_is_byte_aligned(element_type))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "opaque and sub-byte aligned element types cannot be indexed");
  }
  iree_device_size_t element_type_byte_size =
      iree_hal_element_dense_byte_count(element_type);

  // We need to check for special cases (splatting, zerodim):
  bool is_splat = element_type_byte_size == 1 || element_type_byte_size == 2 ||
                  element_type_byte_size == 4;
  bool has_zero_dim = false;
  iree_device_size_t byte_length = element_type_byte_size;

  for (int i = 0; i < num_byte_strides; ++i) {
    is_splat &= (dims[i] == 1 || byte_strides[i] == 0);
    has_zero_dim |= (dims[i] == 0);
    byte_length *= dims[i];
  }

  byte_length = std::max(element_type_byte_size, byte_length);

  // If we encounter the zero dim case no transfer is required:
  if (has_zero_dim) {
    return HostBufferToDeviceZeroDim(
        type, dims, num_dims, out_done_with_host_buffer_event, out_buffer);
  }

  // If we encounter the splat case we can perform a fill instead:
  if (is_splat) {
    return HostBufferToDeviceSplat(data, type, dims, num_dims,
                                   out_done_with_host_buffer_event, out_buffer);
  }

  // Handle strided layouts and shape:
  std::vector<int64_t> perms(num_dims);
  std::array<iree_hal_dim_t, kMaxDims> input_shape;
  std::array<iree_hal_dim_t, kMaxDims> transpose_shape;
  std::array<iree_hal_dim_t, kMaxDims> output_shape;
  if (num_dims > input_shape.size()) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "only supports up to %d dims but got %d",
                            (int)input_shape.size(), (int)num_dims);
  }

  // Compute the input shape and permutations for the broadcast.
  iree::pjrt::computeBroadcastArgs(
      num_dims, element_type_byte_size, byte_strides, dims,
      reinterpret_cast<int64_t*>(input_shape.data()), perms.data());

  for (int i = 0, s = num_dims; i < s; ++i) {
    transpose_shape[i] = input_shape[perms[i]];
    output_shape[i] = dims[i];
  }

  bool is_dense_row_major = true;
  for (int i = 0, s = num_dims; i < s; ++i) {
    is_dense_row_major &= (input_shape[i] == dims[i]) && (perms[i] == i);
  }

  iree::vm::ref<iree_hal_buffer_t> buffer;
  // There are multiple ways to implement zero-copy/staged transfers and each
  // implementation will have different performance cliffs associated with
  // directly operating on imported host buffers. In many actual
  // host/device situations, such unified memory is a productivity (not a
  // performance) feature and best avoided. As such, we always need to be
  // able to decide to do a staged transfer and implement that here. Using
  // an imported buffer on the device is left as an optimization for
  // implementations on which we believe it will be beneficial.
  bool require_snapshot_now = host_buffer_semantics ==
                              PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
  bool caller_data_done = false;

  iree::vm::ref<iree_hal_buffer_t> host_staging_buffer;
  IREE_RETURN_IF_ERROR(AcquireHostStagingBuffer(
      iree_make_const_byte_span(data, byte_length), require_snapshot_now,
      &caller_data_done, &host_staging_buffer));
  if (!caller_data_done) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "deferred snapshot of host data not yet implemented");
  }

  // Allocate on stream. We serialize across 3 timepoints:
  //   0. Last transfer complete
  //   1. Allocation
  //   2. This transfer complete
  // There are various ways to be smarter about this but without more
  // information from the caller, this is ok. If we wanted to favor smaller
  // allocation scopes, it may be desirable to join with the main execution
  // timeline, but that would obviously serialize more.
  uint64_t wait_transfer_start = last_transfer_timepoint_;
  uint64_t signal_alloca_complete = ++last_transfer_timepoint_;
  uint64_t signal_copy_complete = ++last_transfer_timepoint_;
  iree_hal_buffer_params_t params;
  memset(&params, 0, sizeof(params));
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage =
      IREE_HAL_BUFFER_USAGE_DEFAULT | IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET;
  IREE_RETURN_IF_ERROR(IreeApi::hal_device_queue_alloca(
      device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/
      {1, &transfer_timeline_, &wait_transfer_start},
      /*signal_semaphore_list=*/
      {1, &transfer_timeline_, &signal_alloca_complete},
      IREE_HAL_ALLOCATOR_POOL_DEFAULT, params, byte_length,
      IREE_HAL_ALLOCA_FLAG_NONE, &buffer));

  // Queue up the transfer command.
  iree::vm::ref<iree_hal_command_buffer_t> transfer_cb;
  iree_hal_transfer_command_t transfer_command;
  memset(&transfer_command, 0, sizeof(transfer_command));
  transfer_command.type = IREE_HAL_TRANSFER_COMMAND_TYPE_COPY;
  transfer_command.copy.source_buffer = host_staging_buffer.get(),
  transfer_command.copy.source_offset = 0;
  transfer_command.copy.target_buffer = buffer.get();
  transfer_command.copy.target_offset = 0;
  transfer_command.copy.length = byte_length;
  IREE_RETURN_IF_ERROR(iree_hal_create_transfer_command_buffer(
      device(), IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_QUEUE_AFFINITY_ANY,
      /*transfer_count=*/1, &transfer_command, &transfer_cb));

  IREE_RETURN_IF_ERROR(IreeApi::hal_device_queue_execute(
      device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/
      {1, &transfer_timeline_, &signal_alloca_complete},
      /*signal_semaphore_list=*/
      {1, &transfer_timeline_, &signal_copy_complete}, transfer_cb.get()));

  // Wrap in a buffer view and return.
  iree::vm::ref<iree_hal_buffer_view_t> result_buffer_view;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_create(
      buffer.get(), num_dims, &input_shape[0], element_type,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, client_.host_allocator(),
      &result_buffer_view));

  auto instance = new BufferInstance(*this, std::move(result_buffer_view));
  instance->AdvanceReadyFence(transfer_timeline_.get(), signal_copy_complete);
  instance->AdvanceDoneFence(transfer_timeline_.get(), signal_copy_complete);

  if (is_dense_row_major) {
    *out_buffer = instance;

    // We snapshotted the caller data when acquiring the host staging buffer,
    // so we won't be touching it again.
    *out_done_with_host_buffer_event = new EventInstance(/*fence=*/nullptr);

    return iree_ok_status();
  }

  auto err = TransposeBroadcastDeviceBuffer(
      instance, element_type, input_shape.data(), output_shape.data(),
      perms.data(), num_dims, host_buffer_semantics,
      out_done_with_host_buffer_event, out_buffer);
  delete instance;
  return err;
}

iree_status_t DeviceInstance::CreateViewOfDeviceBuffer(
    void* device_buffer_ptr, PJRT_Buffer_Type type, const int64_t* dims,
    size_t num_dims, const PJRT_Buffer_MemoryLayout* layout, intptr_t stream,
    void (*on_delete_callback)(void* device_buffer_ptr, void* user_arg),
    void* on_delete_callback_arg, BufferInstance** out_buffer) {
  IREE_RETURN_IF_ERROR(OpenDevice());
  if (!device_buffer_ptr) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "device buffer pointer must not be null");
  }
  if (num_dims > kMaxDims) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "only supports up to %d dims but got %d",
                            (int)kMaxDims, (int)num_dims);
  }
  if (stream != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "importing a Metal device buffer with a stream is not implemented");
  }

  iree_hal_element_type_t element_type;
  IREE_RETURN_IF_ERROR(
      PJRTApiConverter::MapBufferTypeToElementType(type, &element_type));
  if (!iree_hal_element_is_byte_aligned(element_type)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "sub-byte device buffer element types cannot be imported");
  }
  iree_device_size_t element_byte_size =
      iree_hal_element_dense_byte_count(element_type);

  // IREE buffer views currently represent dense row-major tensors. Reject
  // layouts that would silently reinterpret the imported allocation.
  if (layout && layout->type == PJRT_Buffer_MemoryLayout_Type_Tiled) {
    if (layout->tiled.minor_to_major_size != num_dims ||
        layout->tiled.num_tiles != 0) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "tiled device buffer layout is not supported");
    }
    for (size_t i = 0; i < num_dims; ++i) {
      if (layout->tiled.minor_to_major[i] !=
          static_cast<int64_t>(num_dims - i - 1)) {
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "non-row-major device buffer layout is not supported");
      }
    }
  } else if (layout &&
             layout->type == PJRT_Buffer_MemoryLayout_Type_Strides) {
    if (layout->strides.num_byte_strides != num_dims) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "device buffer strides must match its rank");
    }
    int64_t expected_stride = static_cast<int64_t>(element_byte_size);
    for (size_t i = num_dims; i > 0; --i) {
      size_t dim_index = i - 1;
      if (layout->strides.byte_strides[dim_index] != expected_stride) {
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "non-row-major device buffer strides are not supported");
      }
      if (dims[dim_index] < 0 ||
          (dims[dim_index] != 0 &&
           expected_stride > std::numeric_limits<int64_t>::max() /
                                 dims[dim_index])) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "device buffer shape is too large");
      }
      expected_stride *= dims[dim_index];
    }
  } else if (layout &&
             layout->type != PJRT_Buffer_MemoryLayout_Type_Tiled) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown device buffer layout type");
  }

  std::array<iree_hal_dim_t, kMaxDims> shape;
  iree_device_size_t byte_length = element_byte_size;
  for (size_t i = 0; i < num_dims; ++i) {
    if (dims[i] < 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "device buffer dimensions must be nonnegative");
    }
    if (dims[i] != 0 &&
        byte_length > std::numeric_limits<iree_device_size_t>::max() /
                          static_cast<iree_device_size_t>(dims[i])) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "device buffer shape is too large");
    }
    byte_length *= static_cast<iree_device_size_t>(dims[i]);
    shape[i] = static_cast<iree_hal_dim_t>(dims[i]);
  }

  iree_hal_external_buffer_t external_buffer;
  memset(&external_buffer, 0, sizeof(external_buffer));
  external_buffer.type = IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION;
  external_buffer.flags = IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE;
  external_buffer.size = byte_length;
  external_buffer.handle.device_allocation.ptr =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(device_buffer_ptr));

  std::unique_ptr<ExternalDeviceBufferRelease> release;
  iree_hal_buffer_release_callback_t release_callback =
      iree_hal_buffer_release_callback_null();
  if (on_delete_callback) {
    release = std::make_unique<ExternalDeviceBufferRelease>(
        ExternalDeviceBufferRelease{device_buffer_ptr, on_delete_callback,
                                    on_delete_callback_arg});
    release_callback.fn = ReleaseExternalDeviceBuffer;
    release_callback.user_data = release.get();
  }

  iree_hal_buffer_params_t params;
  memset(&params, 0, sizeof(params));
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
  iree::vm::ref<iree_hal_buffer_t> buffer;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_import_buffer(
      device_allocator(), params, &external_buffer, release_callback,
      &buffer));
  release.release();

  iree::vm::ref<iree_hal_buffer_view_t> buffer_view;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_create(
      buffer.get(), num_dims, shape.data(), element_type,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, client_.host_allocator(),
      &buffer_view));
  *out_buffer = new BufferInstance(*this, std::move(buffer_view));
  return iree_ok_status();
}

iree_status_t DeviceInstance::AcquireHostStagingBuffer(
    iree_const_byte_span_t initial_contents, bool snapshot_initial_contents_now,
    bool* initial_contents_snapshotted, iree_hal_buffer_t** out_buffer) {
  IREE_TRACE_SCOPE();
  // There are multiple ways to do this that have different cost/benefits.
  // Here we do the simplest thing and snapshot into a new host allocation.
  // This could be replaced with either some form of staging ring buffer
  // or importing from a raw pointer (on implementations where the cost of
  // unified addressing is zero).
  iree_hal_buffer_params_t params;
  memset(&params, 0, sizeof(params));
  params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  IREE_RETURN_IF_ERROR(IreeApi::hal_allocator_allocate_buffer(
      device_allocator(), params, initial_contents.data_length, out_buffer));
  IREE_RETURN_IF_ERROR(iree_hal_device_transfer_h2d(
      device(), initial_contents.data, *out_buffer, 0,
      initial_contents.data_length, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));
  // We did a synchronous snapshot (memcpy).
  *initial_contents_snapshotted = true;
  return iree_ok_status();
}

iree_status_t DeviceInstance::GetHalDevice(iree_hal_device_t** out_device) {
  IREE_RETURN_IF_ERROR(OpenDevice());
  *out_device = device_.get();
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// ClientInstance
//===----------------------------------------------------------------------===//

ClientInstance::ClientInstance(std::unique_ptr<Platform> platform)
    : platform_(std::move(platform)) {
  host_allocator_ = iree_allocator_system();
  IREE_CHECK_OK(
      iree_hal_driver_registry_allocate(host_allocator_, &driver_registry_));
  cached_platform_version_ = "git";  // TODO: Plumb through version info.
}

ClientInstance::~ClientInstance() {
  for (auto* device : devices_) {
    delete device;
  }
  for (auto* memory : memories_) {
    delete memory;
  }
  if (device_infos_) {
    iree_allocator_free(host_allocator_, device_infos_);
  }
  // Explicitly releasing vs using a ref so as to better control shut-down
  // ordering (bad shutdown ordering of the driver is a frequent cause of
  // bugs).
  iree_hal_driver_release(driver_);
  iree_hal_driver_registry_free(driver_registry_);
}

void ClientInstance::BindApi(PJRT_Api* api) {
  // PJRT_Client_Create is polymorphic
  api->PJRT_Client_Destroy =
      +[](PJRT_Client_Destroy_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE_NAMED("PJRT_Client_Destroy");
    delete ClientInstance::Unwrap(args->client);
    return nullptr;
  };
  api->PJRT_Client_PlatformName =
      +[](PJRT_Client_PlatformName_Args* args) -> PJRT_Error* {
    auto* client = ClientInstance::Unwrap(args->client);
    args->platform_name = client->cached_platform_name().data();
    args->platform_name_size = client->cached_platform_name().size();
    return nullptr;
  };
  api->PJRT_Client_ProcessIndex =
      +[](PJRT_Client_ProcessIndex_Args* args) -> PJRT_Error* {
    args->process_index = 0;
    return nullptr;
  };
  api->PJRT_Client_PlatformVersion =
      +[](PJRT_Client_PlatformVersion_Args* args) -> PJRT_Error* {
    auto* client = ClientInstance::Unwrap(args->client);
    args->platform_version = client->cached_platform_version().data();
    args->platform_version_size = client->cached_platform_version().size();
    return nullptr;
  };
  api->PJRT_Client_Devices =
      +[](PJRT_Client_Devices_Args* args) -> PJRT_Error* {
    auto& devices = ClientInstance::Unwrap(args->client)->devices();
    args->devices = const_cast<PJRT_Device**>(
        reinterpret_cast<PJRT_Device* const*>(devices.data()));
    args->num_devices = devices.size();
    return nullptr;
  };
  api->PJRT_Client_AddressableDevices =
      +[](PJRT_Client_AddressableDevices_Args* args) -> PJRT_Error* {
    auto& devices = ClientInstance::Unwrap(args->client)->addressable_devices();
    args->addressable_devices = const_cast<PJRT_Device**>(
        reinterpret_cast<PJRT_Device* const*>(devices.data()));
    args->num_addressable_devices = devices.size();
    return nullptr;
  };
  api->PJRT_Client_LookupDevice =
      +[](PJRT_Client_LookupDevice_Args* args) -> PJRT_Error* {
    auto& devices = ClientInstance::Unwrap(args->client)->devices();
    size_t id_as_size = args->id;
    if (id_as_size >= devices.size()) {
      return MakeError(
          iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                           "because device id %d is invalid (%d devices known)",
                           (int)id_as_size, (int)devices.size()));
    }
    args->device = *devices[id_as_size];
    return nullptr;
  };
  api->PJRT_Client_LookupAddressableDevice =
      +[](PJRT_Client_LookupAddressableDevice_Args* args) -> PJRT_Error* {
    auto& devices = ClientInstance::Unwrap(args->client)->addressable_devices();
    for (auto* device : devices) {
      if (device->local_hardware_id() == args->local_hardware_id) {
        args->addressable_device = *device;
        return nullptr;
      }
    }
    return MakeError(iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "local hardware id %d does not name an addressable device",
        args->local_hardware_id));
  };
  api->PJRT_Client_AddressableMemories =
      +[](PJRT_Client_AddressableMemories_Args* args) -> PJRT_Error* {
    const auto& memories = ClientInstance::Unwrap(args->client)->memories();
    args->addressable_memories =
        reinterpret_cast<PJRT_Memory* const*>(memories.data());
    args->num_addressable_memories = memories.size();
    return nullptr;
  };
  api->PJRT_Client_Compile =
      +[](PJRT_Client_Compile_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE_NAMED("PJRT_Client_Compile");
    // TODO: It is not great that we only get a client here vs a list of
    // devices to consider (or something). The issue is that systems often
    // have unrelated devices that will not actually be scheduled and those
    // will very naturally have different tuning flags. We therefore have to
    // guess... which is an accident waiting to happen.
    // Looks like what I need is buried in the compile options... need to
    // work on that.
    auto* client = ClientInstance::Unwrap(args->client);
    LoadedExecutableInstance* executable;

    // Read compilation options.
    xla::CompileOptionsProto options_proto;
    if (!options_proto.ParseFromArray(args->compile_options,
                                      args->compile_options_size)) {
      return MakeError(iree_make_status(IREE_STATUS_INTERNAL,
                                        "could not parse compilation options"));
    }

    auto* error = client->Compile(args->program, options_proto, &executable);
    if (error) return error;
    args->executable = *executable;
    return nullptr;
  };
  api->PJRT_Client_DefaultDeviceAssignment =
      +[](PJRT_Client_DefaultDeviceAssignment_Args* args) -> PJRT_Error* {
    // TODO: Something sensible.
    for (size_t i = 0; i < args->default_assignment_size; ++i) {
      args->default_assignment[i] = 0;
    }
    return nullptr;
  };
  api->PJRT_Client_DmaMap =
      +[](PJRT_Client_DmaMap_Args* args) -> PJRT_Error* {
    // Metal's unified memory model does not require explicit host DMA
    // registration, but validate the range so the no-op is deterministic.
    if (args->size != 0 && !args->data) {
      return MakeError(iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT, "DMA mapping data must not be null"));
    }
    return nullptr;
  };
  api->PJRT_Client_DmaUnmap =
      +[](PJRT_Client_DmaUnmap_Args* args) -> PJRT_Error* {
    if (!args->data) {
      return MakeError(iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT, "DMA unmapping data must not be null"));
    }
    return nullptr;
  };
  api->PJRT_Client_BufferFromHostBuffer =
      +[](PJRT_Client_BufferFromHostBuffer_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE_NAMED("PJRT_Client_BufferFromHostBuffer");
    auto device = args->device ? DeviceInstance::Unwrap(args->device)
                               : MemoryInstance::Unwrap(args->memory)->device();
    auto status = device->HostBufferToDevice(
        args->data, args->type, args->dims, args->num_dims, args->byte_strides,
        args->num_byte_strides, args->host_buffer_semantics,
        reinterpret_cast<EventInstance**>(&args->done_with_host_buffer),
        reinterpret_cast<BufferInstance**>(&args->buffer));
    return MakeError(status);
  };
  api->PJRT_Client_CreateViewOfDeviceBuffer =
      +[](PJRT_Client_CreateViewOfDeviceBuffer_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE_NAMED("PJRT_Client_CreateViewOfDeviceBuffer");
    DeviceInstance* device = nullptr;
    if (args->memory) {
      device = MemoryInstance::Unwrap(args->memory)->device();
    } else if (args->device) {
      device = DeviceInstance::Unwrap(args->device);
    } else {
      return MakeError(iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "device or memory is required for a device buffer view"));
    }
    return MakeError(device->CreateViewOfDeviceBuffer(
        args->device_buffer_ptr, args->element_type, args->dims,
        args->num_dims, args->layout, args->stream,
        args->on_delete_callback, args->on_delete_callback_arg,
        reinterpret_cast<BufferInstance**>(&args->buffer)));
  };
}

PJRT_Error* ClientInstance::Initialize() {
  // TODO: Remove calls to iree_status_fprint once JAX properly reports
  // initialization errors: https://github.com/google/jax/issues/13763
  auto status = CreateDriver(&driver_);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    return MakeError(status);
  }

  status = InitializeVM();
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    return MakeError(status);
  }

  status = PopulateDevices();
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    return MakeError(status);
  }

  // More initialization.
  return nullptr;
}

iree_status_t ClientInstance::InitializeVM() {
  IREE_RETURN_IF_ERROR(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                               host_allocator_, &vm_instance_));
  IREE_RETURN_IF_ERROR(iree_hal_module_register_all_types(vm_instance_.get()));
  return iree_ok_status();
}

iree_status_t ClientInstance::PopulateDevices() {
  IREE_RETURN_IF_ERROR(iree_hal_driver_query_available_devices(
      driver_, host_allocator_, &device_info_count_, &device_infos_));
  devices_.resize(device_info_count_);
  for (iree_host_size_t i = 0; i < device_info_count_; ++i) {
    // Note that we assume one driver per client here.
    // But device is modeled with a driver in case if it ever becomes
    // more heterogenous.
    devices_[i] = new DeviceInstance(i, *this, driver_, &device_infos_[i]);
  }

  // For now, just make all devices addressable.
  addressable_devices_.reserve(devices_.size());
  for (auto* device : devices_) {
    addressable_devices_.push_back(device);
  }

  // initialize the one-to-one mapping of devices to memory instances
  memories_.reserve(devices_.size());
  for (auto* device : devices_) {
    auto memory = new MemoryInstance(device);
    memories_.push_back(memory);
    device->set_memory(memory);
  }

  return iree_ok_status();
}

PJRT_Error* ClientInstance::Compile(const PJRT_Program* program,
                                    xla::CompileOptionsProto options,
                                    LoadedExecutableInstance** out_executable) {
  std::unique_ptr<ArtifactDumper::Transaction> artifact_tx;
  if (platform().artifact_dumper().enabled()) {
    artifact_tx = platform().artifact_dumper().CreateTransaction();
  }

  iree_status_t status;
  std::string_view format(program->format, program->format_size);
  std::string_view code(program->code, program->code_size);
  if (artifact_tx) {
    artifact_tx->WriteArtifact(/*label=*/"program", /*extension=*/"mlirbc",
                               /*index=*/-1, code);
  }

  if (format != "mlir") {
    // See: https://github.com/google/jax/issues/13722
    return MakeError(iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "because IREE only supports MLIR input but got something else"));
  }

  auto MakeCompilerError = [&](CompilerJob& job) {
    std::string message = job.GetErrorMessage();
    return MakeError(iree_make_status(IREE_STATUS_INVALID_ARGUMENT, ": %s",
                                      message.c_str()));
  };

  std::vector<std::unique_ptr<CompilerOutput>> retained_outputs;

  // Partition.
  if (platform().partitioner()) {
    std::unique_ptr<CompilerJob> job = platform().partitioner()->StartJob();
    if (!job) {
      std::string message = platform().partitioner()->GetErrorMessage();
      return MakeError(
          iree_make_status(IREE_STATUS_CANCELLED, ": %s", message.c_str()));
    }
    if (artifact_tx) {
      job->EnableCrashDumps(artifact_tx.get());
    }

    // Set flags.
    if (!job->SetFlags(options)) return MakeCompilerError(*job);

    if (artifact_tx) {
      artifact_tx->WriteArtifact(
          /*label=*/"partitioner_flags", /*extension=*/"txt", /*index=*/-1,
          job->GetFlags());
    }

    // Parse the source.
    if (!job->ParseSourceBuffer(code.data(), code.size())) {
      return MakeCompilerError(*job);
    }

    // Partition.
    std::unique_ptr<CompilerOutput> output = job->CompileStandardPipeline();
    if (!output) {
      return MakeCompilerError(*job);
    }
    if (artifact_tx) {
      artifact_tx->WriteArtifact(
          /*label=*/"partitioned", /*extension=*/"mlir", /*index=*/-1,
          std::string_view(static_cast<const char*>(output->GetData()),
                           output->GetDataSize()));
    }

    // Update the code alias and retain the backing output for the next
    // compilation step.
    code = std::string_view(static_cast<const char*>(output->GetData()),
                            output->GetDataSize());
    retained_outputs.push_back(std::move(output));
  }

  // Main compilation.
  {
    std::unique_ptr<CompilerJob> job = platform().compiler().StartJob();
    if (!job) {
      std::string message = platform().compiler().GetErrorMessage();
      return MakeError(
          iree_make_status(IREE_STATUS_CANCELLED, ": %s", message.c_str()));
    }
    if (artifact_tx) {
      job->EnableCrashDumps(artifact_tx.get());
    }

    // Set flags.
    // TODO: This should be done as part of session setup from a named pool.
    // TODO: The HAL backends and other flags should come from the assigned
    // devices.
    if (!SetDefaultCompilerFlags(job.get())) {
      return MakeCompilerError(*job);
    }
    if (!job->SetFlags(options)) return MakeCompilerError(*job);

    if (artifact_tx) {
      artifact_tx->WriteArtifact(
          /*label=*/"flags", /*extension=*/"txt", /*index=*/-1,
          job->GetFlags());
    }

    // Parse the source.
    if (!job->ParseSourceBuffer(code.data(), code.size())) {
      return MakeCompilerError(*job);
    }

    // Perform main compilation.
    std::unique_ptr<CompilerOutput> output = job->CompileStandardPipeline();
    if (!output) {
      return MakeCompilerError(*job);
    }
    if (artifact_tx) {
      artifact_tx->WriteArtifact(
          /*label=*/"program", /*extension=*/"vmfb", /*index=*/-1,
          std::string_view(static_cast<const char*>(output->GetData()),
                           output->GetDataSize()));
    }

    // calculate devices for this computation from device assignment
    std::vector<DeviceInstance*> devices;

    const auto& build_options = options.executable_build_options();
    if (build_options.has_device_assignment()) {
      const auto& device_assignment = build_options.device_assignment();
      for (auto id :
           device_assignment.computation_devices(0).replica_device_ids()) {
        if (id < addressable_devices_.size())
          devices.push_back(addressable_devices_[id]);
      }
    }

    if (devices.empty()) {
      devices = addressable_devices_;
    }

    auto executable = std::make_unique<LoadedExecutableInstance>(
        *this,
        new ExecutableImage(std::move(output),
                            std::string(program->code, program->code_size)),
        devices);
    status = executable->LoadAll();
    if (!iree_status_is_ok(status)) {
      return MakeError(status);
    }

    *out_executable = executable.release();
  }

  // Success? Cancel the artifact so we don't persist successful runs
  // (unless if so configured).
  if (artifact_tx) {
    artifact_tx->Cancel();
  }
  return nullptr;
}

iree_status_t ClientInstance::PopulateVMModules(
    std::vector<iree::vm::ref<iree_vm_module_t>>& modules,
    iree_hal_device_t* hal_device,
    iree::vm::ref<iree_vm_module_t>& main_module) {
  // HAL module.
  iree_hal_device_group_t* device_group = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_device_group_create_from_device(
      hal_device, host_allocator(), &device_group));
  modules.push_back({});
  iree_status_t status = iree_hal_module_create(
      vm_instance(), iree_hal_module_device_policy_default(), device_group,
      IREE_HAL_MODULE_FLAG_NONE, iree_hal_module_debug_sink_stdio(stderr),
      host_allocator(), &modules.back());
  iree_hal_device_group_release(device_group);
  IREE_RETURN_IF_ERROR(status);

  // Main module.
  modules.push_back(main_module);
  return iree_ok_status();
}

std::tuple<uint64_t, uint64_t> ClientInstance::AdvanceTimeline() {
  uint64_t current = execution_timeline_;
  uint64_t next = current + 1;
  execution_timeline_ = next;
  return std::make_tuple(current, next);
}

//===----------------------------------------------------------------------===//
// EventInstance
//===----------------------------------------------------------------------===//

EventInstance::EventInstance(iree::vm::ref<iree_hal_fence_t> fence)
    : is_ready_(false) {
  if (!fence) {
    is_ready_ = true;
    return;
  }

  {
    std::lock_guard<std::mutex> guard(lock_);
    // Create a thread that waits on the fence and executes the callbacks when
    // the fence is ready.
    signal_thread_ = std::make_unique<std::thread>(
        [](EventInstance* event_instance,
           iree::vm::ref<iree_hal_fence_t> fence) {
          iree_status_t wait_status = iree_hal_fence_wait(
              fence.get(), iree_infinite_timeout(), IREE_HAL_WAIT_FLAG_DEFAULT);
          event_instance->SignalReady(wait_status);
        },
        this, std::move(fence));
  }
}

EventInstance::~EventInstance() {
  std::lock_guard<std::mutex> guard(lock_);
  if (signal_thread_) {
    if (std::this_thread::get_id() != signal_thread_->get_id()) {
      signal_thread_->join();
    } else {
      // An `EventInstance` is allowed to delete itself in one of its callbacks,
      // resulting in `signal_thread_` being the thread calling the destructor.
      // In such cases, we must let the thread continue running independent of
      // the destructor to avoid a deadlock.
      signal_thread_->detach();
      signal_thread_.release();
    }
  }
  iree_status_ignore(status_);
}

void EventInstance::BindApi(PJRT_Api* api) {
  api->PJRT_Event_Destroy = +[](PJRT_Event_Destroy_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE_NAMED("PJRT_Event_Destroy");
    auto instance = EventInstance::Unwrap(args->event);
    auto delete_event = [](PJRT_Error* error, void* user_data) {
      EventInstance* event = static_cast<EventInstance*>(user_data);
      delete event;
      if (error) {
        delete ErrorInstance::FromError(error);
      }
    };

    instance->OnReady(delete_event, args->event);
    return nullptr;
  };
  api->PJRT_Event_IsReady = +[](PJRT_Event_IsReady_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE_NAMED("PJRT_Event_IsReady");
    args->is_ready = EventInstance::Unwrap(args->event)->is_ready();
    return nullptr;
  };
  api->PJRT_Event_Error = +[](PJRT_Event_Error_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE_NAMED("PJRT_Event_Error");
    return (PJRT_Error*)EventInstance::Unwrap(args->event)->error();
  };
  api->PJRT_Event_Await = +[](PJRT_Event_Await_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE_NAMED("PJRT_Event_Await");
    return MakeError(EventInstance::Unwrap(args->event)->Await());
  };
  api->PJRT_Event_OnReady = +[](PJRT_Event_OnReady_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE_NAMED("PJRT_Event_OnReady");
    return MakeError(EventInstance::Unwrap(args->event)
                         ->OnReady(args->callback, args->user_arg));
  };
}

ErrorInstance* EventInstance::error() {
  std::lock_guard<std::mutex> guard(lock_);
  if (!iree_status_is_ok(status_)) {
    return new ErrorInstance(iree_status_clone(status_));
  }
  return nullptr;
}
bool EventInstance::is_ready() {
  std::lock_guard<std::mutex> guard(lock_);
  return is_ready_;
}

iree_status_t EventInstance::Await() {
  std::unique_lock<std::mutex> lock(lock_);
  ready_condition_.wait(lock, [this] { return is_ready_; });
  return iree_status_clone(status_);
}

iree_status_t EventInstance::OnReady(PJRT_Event_OnReadyCallback callback,
                                     void* user_arg) {
  iree_status_t local_status;
  {
    std::lock_guard<std::mutex> guard(lock_);
    if (!is_ready_) {
      pending_callbacks_.push_back({callback, user_arg});
      return iree_ok_status();
    }
    local_status = status_;
  }

  // Already signalled. Callback out of lock scope.
  // Note that the callback may destroy the event - so must only operate on
  // locals.
  callback(
      iree_status_is_ok(local_status)
          ? nullptr
          : (PJRT_Error*)new ErrorInstance(iree_status_clone(local_status)),
      user_arg);
  return iree_ok_status();
}

void EventInstance::SignalReady(iree_status_t status) {
  IREE_TRACE_SCOPE();
  iree_status_t local_status;
  std::vector<std::pair<PJRT_Event_OnReadyCallback, void*>> local_callbacks;
  {
    std::lock_guard<std::mutex> guard(lock_);
    if (is_ready_) {
      return;
    }
    local_callbacks.swap(pending_callbacks_);
    is_ready_ = true;
    status_ = status;
    local_status = status_;
  }
  ready_condition_.notify_all();

  // Trigger callbacks outside of the lock.
  // Note that the callback may destroy the event - so must only operate on
  // locals.
  for (auto& cb : local_callbacks) {
    IREE_TRACE_SCOPE_NAMED("PJRT_User_Callback_Invoke");
    cb.first(
        iree_status_is_ok(local_status)
            ? nullptr
            : (PJRT_Error*)new ErrorInstance(iree_status_clone(local_status)),
        cb.second);
  }
}

//===----------------------------------------------------------------------===//
// LoadedExecutableInstance
//===----------------------------------------------------------------------===//

void ExecutableImage::BindApi(PJRT_Api* api) {
  api->PJRT_Executable_Destroy =
      +[](PJRT_Executable_Destroy_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE_NAMED("PJRT_Executable_Destroy");
    ExecutableImage::Unwrap(args->executable)->DecRef();
    return nullptr;
  };
  api->PJRT_Executable_Name =
      +[](PJRT_Executable_Name_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE_NAMED("PJRT_Executable_Name");
    const char* dummy_name = "iree_vmfb";
    args->executable_name = dummy_name;
    args->executable_name_size = strlen(dummy_name);
    return nullptr;
  };
  api->PJRT_Executable_SizeOfGeneratedCodeInBytes =
      +[](PJRT_Executable_SizeOfGeneratedCodeInBytes_Args* args)
      -> PJRT_Error* {
    IREE_TRACE_SCOPE_NAMED("PJRT_Executable_SizeOfGeneratedCodeInBytes");
    args->size_in_bytes =
        ExecutableImage::Unwrap(args->executable)->binary->GetDataSize();
    return nullptr;
  };
  api->PJRT_Executable_Fingerprint =
      +[](PJRT_Executable_Fingerprint_Args* args) -> PJRT_Error* {
    auto* executable = ExecutableImage::Unwrap(args->executable);
    if (executable->fingerprint_.empty()) {
      executable->fingerprint_ =
          FingerprintCompilerOutput(*executable->binary);
    }
    args->executable_fingerprint = executable->fingerprint_.data();
    args->executable_fingerprint_size = executable->fingerprint_.size();
    return nullptr;
  };
  api->PJRT_Executable_NumOutputs =
      +[](PJRT_Executable_NumOutputs_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE_NAMED("PJRT_Executable_NumOutputs");
    auto* exec = ExecutableImage::Unwrap(args->executable);
    assert(exec->metadata_initialized);
    args->num_outputs = exec->result_count;
    return nullptr;
  };
  api->PJRT_Executable_NumPartitions =
      +[](PJRT_Executable_NumPartitions_Args* args) -> PJRT_Error* {
    // This should be updated once iree supports partitioning.
    args->num_partitions = 1;
    return nullptr;
  };
  api->PJRT_Executable_NumReplicas =
      +[](PJRT_Executable_NumReplicas_Args* args) -> PJRT_Error* {
    // This should be updated once iree supports replicas.
    args->num_replicas = 1;
    return nullptr;
  };
  api->PJRT_Executable_Serialize =
      +[](PJRT_Executable_Serialize_Args* args) -> PJRT_Error* {
    auto* executable = ExecutableImage::Unwrap(
        const_cast<PJRT_Executable*>(args->executable));
    auto* storage = new SerializedExecutableStorage(*executable->binary);
    args->serialized_bytes = storage->bytes.data();
    args->serialized_bytes_size = storage->bytes.size();
    args->serialized_executable =
        reinterpret_cast<PJRT_SerializedExecutable*>(storage);
    args->serialized_executable_deleter =
        +[](PJRT_SerializedExecutable* serialized_executable) {
      delete reinterpret_cast<SerializedExecutableStorage*>(
          serialized_executable);
    };
    return nullptr;
  };
  api->PJRT_Executable_DeserializeAndLoad =
      +[](PJRT_Executable_DeserializeAndLoad_Args* args) -> PJRT_Error* {
    if (!args->serialized_executable ||
        args->serialized_executable_size == 0) {
      return MakeError(iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT, "serialized executable is empty"));
    }
    auto* client = ClientInstance::Unwrap(args->client);
    auto output = std::make_unique<InMemoryCompilerOutput>(
        args->serialized_executable, args->serialized_executable_size);
    auto executable = std::make_unique<LoadedExecutableInstance>(
        *client, new ExecutableImage(std::move(output), /*code=*/""),
        client->addressable_devices());
    iree_status_t status = executable->LoadAll();
    if (!iree_status_is_ok(status)) return MakeError(status);
    args->loaded_executable = *executable.release();
    return nullptr;
  };
  api->PJRT_Executable_OptimizedProgram =
      +[](PJRT_Executable_OptimizedProgram_Args* args) -> PJRT_Error* {
    ExecutableImage* executable = ExecutableImage::Unwrap(args->executable);
    PJRT_Program* program = args->program;
    program->format = kMlirFormat.data();
    program->format_size = kMlirFormat.size();
    size_t code_size = executable->code.size();
    if (program->code == nullptr) {
      program->code_size = code_size;
    } else {
      if (program->code_size < code_size) {
        return MakeError(
            iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "expected code_size >= %lu, got code_size = %lu",
                             code_size, program->code_size));
      }
      std::memcpy(program->code, executable->code.c_str(),
                  executable->code.size());
    }
    return nullptr;
  };
  api->PJRT_Executable_GetCostAnalysis =
      +[](PJRT_Executable_GetCostAnalysis_Args* args) -> PJRT_Error* {
    auto* executable = ExecutableImage::Unwrap(args->executable);
    args->num_properties = executable->cost_properties.size();
    args->properties = executable->cost_properties.data();
    return nullptr;
  };
  api->PJRT_Executable_GetCompiledMemoryStats =
      +[](PJRT_Executable_GetCompiledMemoryStats_Args* args) -> PJRT_Error* {
    auto* executable = ExecutableImage::Unwrap(args->executable);
    args->generated_code_size_in_bytes =
        static_cast<int64_t>(executable->binary->GetDataSize());
    args->argument_size_in_bytes = executable->argument_size_in_bytes;
    args->output_size_in_bytes = executable->output_size_in_bytes;
    args->alias_size_in_bytes = executable->alias_size_in_bytes;
    args->temp_size_in_bytes = 0;
    args->host_generated_code_size_in_bytes = 0;
    args->host_argument_size_in_bytes = 0;
    args->host_output_size_in_bytes = 0;
    args->host_alias_size_in_bytes = 0;
    args->host_temp_size_in_bytes = 0;
    return nullptr;
  };
  api->PJRT_Executable_OutputElementTypes =
      +[](PJRT_Executable_OutputElementTypes_Args* args) -> PJRT_Error* {
    auto* executable = ExecutableImage::Unwrap(args->executable);
    args->output_types = executable->output_types.data();
    args->num_output_types = executable->output_types.size();
    return nullptr;
  };
  api->PJRT_Executable_OutputDimensions =
      +[](PJRT_Executable_OutputDimensions_Args* args) -> PJRT_Error* {
    auto* executable = ExecutableImage::Unwrap(args->executable);
    args->num_outputs = executable->output_dim_sizes.size();
    args->dims = executable->output_dims.data();
    args->dim_sizes = executable->output_dim_sizes.data();
    return nullptr;
  };
  api->PJRT_Executable_OutputMemoryKinds =
      +[](PJRT_Executable_OutputMemoryKinds_Args* args) -> PJRT_Error* {
    auto* executable = ExecutableImage::Unwrap(args->executable);
    args->num_outputs = executable->output_memory_kinds.size();
    args->memory_kinds = executable->output_memory_kinds.data();
    args->memory_kind_sizes = executable->output_memory_kind_sizes.data();
    return nullptr;
  };
}

void LoadedExecutableInstance::BindApi(PJRT_Api* api) {
  api->PJRT_LoadedExecutable_Destroy =
      +[](PJRT_LoadedExecutable_Destroy_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE_NAMED("PJRT_LoadedExecutable_Destroy");
    delete LoadedExecutableInstance::Unwrap(args->executable);
    return nullptr;
  };
  api->PJRT_LoadedExecutable_AddressableDevices =
      +[](PJRT_LoadedExecutable_AddressableDevices_Args* args) -> PJRT_Error* {
    auto& devices = LoadedExecutableInstance::Unwrap(args->executable)
                        ->addressable_devices();
    args->addressable_devices = const_cast<PJRT_Device**>(
        reinterpret_cast<PJRT_Device* const*>(devices.data()));
    args->num_addressable_devices = devices.size();
    return nullptr;
  };
  api->PJRT_LoadedExecutable_Delete =
      +[](PJRT_LoadedExecutable_Delete_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE_NAMED("PJRT_LoadedExecutable_Delete");
    return MakeError(
        LoadedExecutableInstance::Unwrap(args->executable)->Delete());
  };
  api->PJRT_LoadedExecutable_IsDeleted =
      +[](PJRT_LoadedExecutable_IsDeleted_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE_NAMED("PJRT_LoadedExecutable_IsDeleted");
    args->is_deleted =
        LoadedExecutableInstance::Unwrap(args->executable)->is_deleted();
    return nullptr;
  };
  api->PJRT_LoadedExecutable_Execute =
      +[](PJRT_LoadedExecutable_Execute_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE_NAMED("PJRT_LoadedExecutable_Execute");
    return MakeError(
        LoadedExecutableInstance::Unwrap(args->executable)->BatchExecute(args));
  };
  api->PJRT_LoadedExecutable_GetExecutable =
      +[](PJRT_LoadedExecutable_GetExecutable_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE_NAMED("PJRT_LoadedExecutable_GetExecutable");
    auto* loaded_exe =
        LoadedExecutableInstance::Unwrap(args->loaded_executable);
    ExecutableImage* image = loaded_exe->image_;
    if (!image->metadata_initialized) {
      auto status = loaded_exe->GetArgResultCount(&image->arg_count,
                                                  &image->result_count);
      if (!iree_status_is_ok(status)) {
        return MakeError(status);
      }
      image->metadata_initialized = true;
    }

    image->AddRef();
    args->executable = *image;
    return nullptr;
  };
  api->PJRT_LoadedExecutable_Fingerprint =
      +[](PJRT_LoadedExecutable_Fingerprint_Args* args) -> PJRT_Error* {
    auto* loaded = LoadedExecutableInstance::Unwrap(args->executable);
    if (loaded->image_->fingerprint_.empty()) {
      loaded->image_->fingerprint_ =
          FingerprintCompilerOutput(*loaded->image_->binary);
    }
    args->executable_fingerprint = loaded->image_->fingerprint_.data();
    args->executable_fingerprint_size =
        loaded->image_->fingerprint_.size();
    return nullptr;
  };
}

iree_status_t LoadedExecutableInstance::LoadAll() {
  IREE_TRACE_SCOPE();
  if (is_deleted()) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "loaded executable has been deleted");
  }
  if (!resident_executables_.empty()) return iree_ok_status();

  std::vector<ResidentExecutable> new_list;
  for (DeviceInstance* device_instance : addressable_devices_) {
    iree_hal_device_t* hal_device;
    IREE_RETURN_IF_ERROR(device_instance->GetHalDevice(&hal_device));
    new_list.push_back({});
    ResidentExecutable& loaded = new_list.back();
    loaded.device_instance = device_instance;

    // Only de-reference through the image_ shared_ptr once to get the
    // binary CompilerOutput (mmap).
    auto* binary = image_->binary.get();
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_create(
        client_.vm_instance(), IREE_VM_BYTECODE_MODULE_FLAG_NONE,
        iree_make_const_byte_span(binary->GetData(), binary->GetDataSize()),
        /*archive_allocator=*/iree_allocator_null(), client_.host_allocator(),
        &loaded.main_module));

    // Lookup main function.
    const char kNameMain[] = "main";
    IREE_RETURN_IF_ERROR(iree_vm_module_lookup_function_by_name(
        loaded.main_module.get(), IREE_VM_FUNCTION_LINKAGE_EXPORT,
        iree_string_view_t{kNameMain, sizeof(kNameMain) - 1},
        &loaded.main_function));

    // Record number of args/results.
    iree_vm_function_signature_t sig =
        iree_vm_function_signature(&loaded.main_function);
    IREE_RETURN_IF_ERROR(iree_vm_function_call_count_arguments_and_results(
        &sig, &loaded.arg_count, &loaded.result_count));

    // The compiler records source-level input/output aliases in a compact
    // reflection attribute. PJRT uses this to retire donated input wrappers;
    // the physical result allocation is allowed to differ from the input.
    loaded.output_aliases.assign(loaded.result_count, -1);
    iree_string_view_t aliases = iree_vm_function_lookup_attr_by_name(
        &loaded.main_function, IREE_SV("iree.abi.output_aliases"));
    while (!iree_string_view_is_empty(aliases)) {
      iree_string_view_t pair;
      iree_string_view_t remaining;
      iree_string_view_split(aliases, ',', &pair, &remaining);
      aliases = remaining;

      iree_string_view_t result_text;
      iree_string_view_t argument_text;
      if (iree_string_view_split(pair, ':', &result_text, &argument_text) < 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "malformed iree.abi.output_aliases metadata");
      }
      int64_t result_index = -1;
      int64_t argument_index = -1;
      if (!iree_string_view_atoi_int64(result_text, &result_index) ||
          !iree_string_view_atoi_int64(argument_text, &argument_index) ||
          result_index < 0 ||
          result_index >= static_cast<int64_t>(loaded.result_count) ||
          argument_index < 0 ||
          argument_index >= static_cast<int64_t>(loaded.arg_count)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid iree.abi.output_aliases metadata");
      }
      loaded.output_aliases[result_index] = argument_index;
    }

    if (!image_->metadata_initialized) {
      image_->arg_count = loaded.arg_count;
      image_->result_count = loaded.result_count;
      image_->argument_size_in_bytes = 0;
      image_->output_size_in_bytes = 0;
      image_->alias_size_in_bytes = 0;

      iree_string_view_t declaration_attr = iree_vm_function_lookup_attr_by_name(
          &loaded.main_function, IREE_SV("iree.abi.declaration"));
      std::string_view declaration(declaration_attr.data,
                                   declaration_attr.size);
      auto arguments = FunctionValueList(declaration, /*results=*/false);
      if (arguments) {
        for (std::string_view argument : SplitFunctionValues(*arguments)) {
          auto metadata = ParseTensorMetadata(argument);
          if (metadata) {
            image_->argument_size_in_bytes += metadata->size_in_bytes;
          }
        }
      }
      std::vector<int64_t> result_sizes;
      auto results = FunctionValueList(declaration, /*results=*/true);
      if (results) {
        for (std::string_view result : SplitFunctionValues(*results)) {
          auto metadata = ParseTensorMetadata(result);
          if (!metadata) continue;
          image_->output_types.push_back(metadata->element_type);
          image_->output_dim_sizes.push_back(metadata->dims.size());
          image_->output_dims.insert(image_->output_dims.end(),
                                     metadata->dims.begin(),
                                     metadata->dims.end());
          image_->output_memory_kinds.push_back(kDeviceMemoryKind.data());
          image_->output_memory_kind_sizes.push_back(kDeviceMemoryKind.size());
          image_->output_size_in_bytes += metadata->size_in_bytes;
          result_sizes.push_back(metadata->size_in_bytes);
        }
      }
      for (size_t i = 0; i < loaded.output_aliases.size() &&
                         i < result_sizes.size(); ++i) {
        if (loaded.output_aliases[i] >= 0) {
          image_->alias_size_in_bytes += result_sizes[i];
        }
      }
      image_->cost_properties = {
          MakeInt64Property("generated_code_size_in_bytes",
                            static_cast<int64_t>(image_->binary->GetDataSize())),
          MakeInt64Property("argument_size_in_bytes",
                            image_->argument_size_in_bytes),
          MakeInt64Property("output_size_in_bytes",
                            image_->output_size_in_bytes),
          MakeInt64Property("alias_size_in_bytes",
                            image_->alias_size_in_bytes),
      };
      image_->metadata_initialized = true;
    }

    // Defer to the client to populate the stack of modules.
    std::vector<iree::vm::ref<iree_vm_module_t>> modules;
    IREE_RETURN_IF_ERROR(
        client_.PopulateVMModules(modules, hal_device, loaded.main_module));
    std::vector<iree_vm_module_t*> module_ptrs;
    module_ptrs.resize(modules.size());
    for (size_t i = 0; i < modules.size(); ++i) {
      module_ptrs[i] = modules[i].get();
    }

    IREE_CHECK_OK(iree_vm_context_create_with_modules(
        client_.vm_instance(), IREE_VM_CONTEXT_FLAG_NONE, module_ptrs.size(),
        module_ptrs.data(), iree_allocator_system(), &loaded.vm_context));
  }

  new_list.swap(resident_executables_);
  return iree_ok_status();
}

iree_status_t LoadedExecutableInstance::GetDefaultResidentExecutable(
    ResidentExecutable** out_loaded) {
  IREE_RETURN_IF_ERROR(LoadAll());
  if (resident_executables_.empty()) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "no executables could be loaded");
  }
  *out_loaded = &resident_executables_.front();
  return iree_ok_status();
}

iree_status_t LoadedExecutableInstance::GetArgResultCount(
    iree_host_size_t* out_arg_count, iree_host_size_t* out_result_count) {
  ResidentExecutable* loaded;
  IREE_RETURN_IF_ERROR(GetDefaultResidentExecutable(&loaded));
  *out_arg_count = loaded->arg_count;
  *out_result_count = loaded->result_count;
  return iree_ok_status();
}

iree_status_t LoadedExecutableInstance::BatchExecute(
    PJRT_LoadedExecutable_Execute_Args* args) {
  if (is_deleted()) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "loaded executable has been deleted");
  }
  // Make sure loaded before resolving a requested device.
  IREE_RETURN_IF_ERROR(LoadAll());

  std::vector<ResidentExecutable*> selected_executables;
  if (args->execute_device) {
    if (args->num_devices != 1) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "specific-device execution requires exactly one device");
    }
    auto* requested_device = DeviceInstance::Unwrap(args->execute_device);
    for (auto& resident : resident_executables_) {
      if (resident.device_instance == requested_device) {
        selected_executables.push_back(&resident);
        break;
      }
    }
    if (selected_executables.empty()) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "requested device is not addressable by this executable");
    }
  } else if (args->num_devices != addressable_devices_.size()) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "incorrect number of devices to execute on (%d vs %d)",
        (int)args->num_devices, (int)addressable_devices_.size());
  } else {
    for (auto& resident : resident_executables_) {
      selected_executables.push_back(&resident);
    }
  }

  // Timeline setup. There are two timelines that we synchronize to:
  // the main execution timeline, which preserves as-called ordering to
  // execution, and the transfer timeline of each device.
  auto [wait_timepoint, signal_timepoint] = client_.AdvanceTimeline();

  // Initialize invocations.
  auto allocator = client_.host_allocator();
  struct Invocation {
    ResidentExecutable* res_exe;
    iree::vm::ref<iree_vm_list_t> inputs;
    iree::vm::ref<iree_vm_list_t> outputs;
    iree::vm::ref<iree_hal_fence_t> wait_fence;
    iree::vm::ref<iree_hal_fence_t> signal_fence;
  };
  std::vector<Invocation> invs;
  invs.resize(args->num_devices);
  for (size_t dev_index = 0; dev_index < args->num_devices; ++dev_index) {
    auto& inv = invs[dev_index];
    inv.res_exe = selected_executables[dev_index];

    // Wait fence initial value.
    // We allocate it to be able to hold two semaphores (main timeline and
    // transfer timeline) and initialize it with the global invocation order
    // of the main timeline. As we process inputs, we will also insert their
    // transfer ready semaphore value so that execution can only begin once
    // all dependencies are ready. This at most represents two unique
    // semaphores.
    IREE_RETURN_IF_ERROR(
        inv.res_exe->device_instance->CreateFence(&inv.wait_fence));
    IREE_RETURN_IF_ERROR(IreeApi::hal_fence_insert(
        inv.wait_fence.get(), inv.res_exe->device_instance->main_timeline(),
        wait_timepoint));

    // Signal fence. This signals the next tick on the main execution
    // timeline.
    IREE_RETURN_IF_ERROR(IreeApi::hal_fence_create_at(
        inv.res_exe->device_instance->main_timeline(), signal_timepoint,
        client_.host_allocator(), &inv.signal_fence));

    IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                             args->num_args, allocator,
                                             &inv.inputs));
    IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                             inv.res_exe->result_count,
                                             allocator, &inv.outputs));

    // Populate inputs.
    for (size_t i = 0; i < args->num_args; ++i) {
      auto* buffer = BufferInstance::Unwrap(args->argument_lists[dev_index][i]);
      iree_vm_ref_t bv_ref =
          iree_hal_buffer_view_retain_ref(buffer->buffer_view());
      IREE_RETURN_IF_ERROR(
          iree_vm_list_push_ref_move(inv.inputs.get(), &bv_ref));

      // Extend the execute wait to include the input's ready signal.
      IREE_RETURN_IF_ERROR(IreeApi::hal_fence_extend(inv.wait_fence.get(),
                                                     buffer->ready_fence()));

      // And extend the buffer's done fence to close over this execution.
      buffer->AdvanceDoneFence(inv.res_exe->device_instance->main_timeline(),
                               signal_timepoint);
    }

    // Add (wait, signal) fences as required by the async-external execution
    // model.
    iree_vm_list_push_ref_retain(inv.inputs.get(), inv.wait_fence);
    iree_vm_list_push_ref_retain(inv.inputs.get(), inv.signal_fence);
  }

  // Issue invocations.
  // TODO: Switch to using the async API. I've tried to structure this
  // so that we can move to that. Obviously important before we have more
  // than one device.
  iree_status_t status = iree_ok_status();
  for (size_t dev_index = 0; dev_index < args->num_devices; ++dev_index) {
    auto& inv = invs[dev_index];
    if (IreeApi::LOGGING_ENABLED) {
      IreeApi::LogInvoke(
          "vm_invoke[async]",
          "context=%p, f=%d, wait_fence=%p {%s}, signal_fence=%p {%s}",
          inv.res_exe->vm_context.get(),
          (int)inv.res_exe->main_function.ordinal, inv.wait_fence.get(),
          IreeApi::FenceToString(inv.wait_fence.get()).c_str(),
          inv.signal_fence.get(),
          IreeApi::FenceToString(inv.signal_fence.get()).c_str());
    }
    auto new_status = IreeApi::HandleStatus(
        "vm_invoke[async]",
        iree_vm_invoke(inv.res_exe->vm_context.get(),
                       inv.res_exe->main_function, IREE_VM_INVOCATION_FLAG_NONE,
                       /*policy=*/nullptr, inv.inputs.get(), inv.outputs.get(),
                       allocator));
    // Any invocation that fails needs a barrier so that signal fence is
    // incremented otherwise future waits will fail. We do this instead of
    // incrementing as only a subset of devices may fail.
    if (!iree_status_is_ok(new_status)) {
      status = new_status;
      // We can ignore the error as we are already erroring out earlier.
      IREE_IGNORE_ERROR(IreeApi::hal_device_queue_barrier(
          inv.res_exe->device_instance->device(), IREE_HAL_QUEUE_AFFINITY_ANY,
          iree_hal_fence_semaphore_list(inv.wait_fence.get()),
          iree_hal_fence_semaphore_list(inv.signal_fence.get())));
    }
  }

  // Process results.
  // Early exit before committing things to the client if anything failed.
  if (!iree_status_is_ok(status)) return status;
  for (size_t dev_index = 0; dev_index < args->num_devices; ++dev_index) {
    auto& inv = invs[dev_index];
    std::vector<BufferInstance*> donated_buffers;
    for (size_t i = 0; i < inv.res_exe->result_count; ++i) {
      iree::vm::ref<iree_hal_buffer_view_t> ret_buffer_view =
          retain_ref((iree_hal_buffer_view_t*)iree_vm_list_get_ref_deref(
              inv.outputs.get(), i, iree_hal_buffer_view_type()));
      // This should not be possible so just hard-assert.
      IREE_ASSERT_ARGUMENT(ret_buffer_view);

      // JAX represents donation by compiling an input/output alias and then
      // omitting that input from non_donatable_input_indices at execution.
      // The compiler reflects that alias independently of physical allocation
      // reuse, which is optional and cannot be inferred from opaque pointers.
      auto is_non_donatable = [&](size_t arg_index) {
        if (!args->options) return false;
        for (size_t j = 0;
             j < args->options->num_non_donatable_input_indices; ++j) {
          if (args->options->non_donatable_input_indices[j] ==
              static_cast<int64_t>(arg_index)) {
            return true;
          }
        }
        return false;
      };
      int64_t alias_arg_index = inv.res_exe->output_aliases[i];
      if (alias_arg_index >= 0) {
        size_t arg_index = static_cast<size_t>(alias_arg_index);
        auto* input_buffer =
            BufferInstance::Unwrap(args->argument_lists[dev_index][arg_index]);
        if (!is_non_donatable(arg_index)) {
          // If the same PJRT buffer occurs at a non-donatable argument index,
          // preserve it regardless of which occurrence supplied the alias.
          bool preserve_input = false;
          for (size_t other_index = 0; other_index < args->num_args;
               ++other_index) {
            preserve_input |=
                is_non_donatable(other_index) &&
                BufferInstance::Unwrap(
                    args->argument_lists[dev_index][other_index]) ==
                    input_buffer;
          }
          if (!preserve_input &&
              std::find(donated_buffers.begin(), donated_buffers.end(),
                        input_buffer) == donated_buffers.end()) {
            donated_buffers.push_back(input_buffer);
          }
        }
      }

      auto result_buffer = std::make_unique<BufferInstance>(
          *inv.res_exe->device_instance, std::move(ret_buffer_view));
      IREE_RETURN_IF_ERROR(result_buffer->AdvanceReadyFence(
          inv.res_exe->device_instance->main_timeline(), signal_timepoint));
      IREE_RETURN_IF_ERROR(result_buffer->AdvanceDoneFence(
          inv.res_exe->device_instance->main_timeline(), signal_timepoint));
      args->output_lists[dev_index][i] = *(result_buffer.release());
    }

    for (auto* donated_buffer : donated_buffers) {
      IREE_RETURN_IF_ERROR(donated_buffer->Delete());
    }

    if (args->device_complete_events) {
      args->device_complete_events[dev_index] =
          *(new EventInstance(retain_ref(inv.signal_fence)));
    }
  }

  return status;
}

iree_status_t LoadedExecutableInstance::Delete() {
  if (is_deleted_.exchange(true)) return iree_ok_status();
  resident_executables_.clear();
  return iree_ok_status();
}

static void BindUndefineds(PJRT_Api* api) {
#define _STUB(API)                                               \
  api->API = +[](API##_Args* args) -> decltype(api->API(args)) { \
    return (decltype(api->API(args)))MakeError(                  \
        iree_make_status(IREE_STATUS_UNIMPLEMENTED, #API));      \
  }

#include "stubs.inc"
}

//===----------------------------------------------------------------------===//
// Top-level API binding.
//===----------------------------------------------------------------------===//

void BindMonomorphicApi(PJRT_Api* api) {
  api->struct_size = PJRT_Api_STRUCT_SIZE;
  api->extension_start = nullptr;
  api->pjrt_api_version.major_version = PJRT_API_MAJOR;
  api->pjrt_api_version.minor_version = PJRT_API_MINOR;

  // This is a bare implementation throwing UNDEFINED errors. This way new
  // functions will not segmentation fault on invocation.
  BindUndefineds(api);
  ErrorInstance::BindApi(api);

  // PJRT_Plugin_Attributes should be implemented since it will always be
  // called from the PJRT client in the initial phase.
  // here we provide a blank implementation to avoid crash due to unimplemented.
  api->PJRT_Plugin_Attributes =
      +[](PJRT_Plugin_Attributes_Args* args) -> PJRT_Error* {
    args->num_attributes = 0;
    args->attributes = nullptr;
    return nullptr;
  };
  api->PJRT_Plugin_Initialize =
      +[](PJRT_Plugin_Initialize_Args* args) -> PJRT_Error* { return nullptr; };
  api->PJRT_ExecuteContext_Create =
      +[](PJRT_ExecuteContext_Create_Args* args) -> PJRT_Error* {
    // The base IREE plugin has no execute-context payloads yet, but returning
    // a distinct opaque token lets callers use the standard lifecycle.
    args->context = reinterpret_cast<PJRT_ExecuteContext*>(new uint8_t(0));
    return nullptr;
  };
  api->PJRT_ExecuteContext_Destroy =
      +[](PJRT_ExecuteContext_Destroy_Args* args) -> PJRT_Error* {
    delete reinterpret_cast<uint8_t*>(args->context);
    return nullptr;
  };

  // Bind by object types.
  BufferInstance::BindApi(api);
  ClientInstance::BindApi(api);
  DeviceDescription::BindApi(api);
  DeviceInstance::BindApi(api);
  EventInstance::BindApi(api);
  ExecutableImage::BindApi(api);
  LoadedExecutableInstance::BindApi(api);
  MemoryInstance::BindApi(api);
}

}  // namespace iree::pjrt
