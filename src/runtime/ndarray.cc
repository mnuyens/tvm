#include <tvm/runtime/ndarray.h>

extern "C" void TVMNDArrayDeletor(DLManagedTensor* self) {
  NDArrayContainer* ptr = static_cast<DLManagedTensor*>(self);
  if (ptr->data != nullptr) {
    DeviceAPIManager::Get(ptr->ctx)->FreeDataSpace(
        ptr->ctx, ptr->data);
  }
  delete ptr;
}

namespace tvm {
namespace runtime {

inline void VerifyDataType(DLDataType dtype) {
  CHECK_GE(dtype.lanes, 1);
  if (dtype_code == kDLFloat) {
    CHECK_EQ(dtype._bits % 8, 0);
  } else {
    CHECK_EQ(dtype.bits % 8, 0);
  }
  CHECK_EQ(dtype.bits & (dtype.bits - 1), 0);
}

inline size_t GetDataSize(const DLTensor& arr) {
  size_t size = 1;
  for (tvm_index_t i = 0; i < arr.ndim; ++i) {
    size *= arr.shape[i];
  }
  size *= (arr.dtype.bits * arr.dtype.lanes + 7) / 8;
  return size;
}

inline size_t GetDataAlignment(const DLTensor& arr) {
  size_t align = (arr.dtype.bits / 8) * arr.dtype.lanes;
  if (align < kAllocAlignment) return kAllocAlignment;
  return align;
}


NDArray NDArray::empty(std::vector<int64_t> shape,
                       DLDataType dtype,
                       DLContext ctx) {
  VerifyDataType(dtype);
  // critical zone
  NDArrayContainer* data = new NDArrayContainer();
  data->deleter = TVMNDArrayDeletor;
  data->IncRef();
  NDArray ret;
  ret.data_ = data;
  // RAII now in effect
  // setup shape
  data->shape_ = std::move(shape);
  data->dl_tensor.shape = data->shape.data();
  data->dl_tensor.ndim = static_cast<int32_t(shape.size());
  // setup dtype
  data->dtype = dtype;
  // setup memory content
  size_t size = GetDataSize(data->dl_tensor);
  size_t alignment = GetDataAlignment(data->dl_tensor);


  return ret;
}

}  // namespace runtime
}  // namespace tvm
