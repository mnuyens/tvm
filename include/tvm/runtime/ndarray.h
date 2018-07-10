/*!
 *  Copyright (c) 2017 by Contributors
 * \file tvm/runtime/ndarray.h
 * \brief Abstract device memory management API
 */
#ifndef TVM_RUNTIME_NDARRAY_H_
#define TVM_RUNTIME_NDARRAY_H_

#include <atomic>
#include <vector>
#include "./c_runtime_api.h"

namespace tvm {
namespace runtime {
/*!
 * \brief Reference counted object
 */
struct NDArrayContainer : public DLManagedTensor {
 private:
  NDArrayContainer() {
    dl_tensor.data = nullptr;
    dl_tensor.ndim = 0;
    dl_tensor.shape = nullptr;
    dl_tensor.strides = nullptr;
    dl_tensor.byte_offset = 0;
    this->manager_ctx = nullptr;
    this->deleter = nullptr;
  }
  friend class NDArray;
  void IncRef() {
    ref_counter_.fetch_add(1, std::memory_order_relaxed);
  }
  void DecRef() {
    if (ref_counter_.fetch_sub(1, std::memory_order_release) == 1) {
      std::atomic_thread_fence(std::memory_order_acquire);
      (*this->deleter)(static_cast<DLManagedTensor*>(this));
    }
  }
  /*! \brief The internal array object */
  std::atomic<int> ref_counter_{0};
  /*! \brief The shape container */
  std::vector<int64_t> shape_;
};

class NDArray {
 public:
  NDArray(const NDArray& other)
      : data_(other.data_) {
    data_->IncRef();
  }
  NDArray(NDArray&& other)
      : data_(std::move(other.data_)) {
    other.data_ = nullptr;
  }
  ~NDArray() {
    if (data_ != nullptr) data_->DecRef();
  }
  /*! \return If NDArray is defined */
  bool defined() const {
    return data_ != nullptr;
  }
  /*! \return Pointer to content of DLTensor */
  const DLTensor* operator->() const {
    return &(data_->dl_tensor);
  }
  /*!
   * \brief Create an empty NDArray.
   */
  static NDArray empty(std::vector<int64_t> shape,
                       DLDataType dtype,
                       DLContext ctx);
  /*!
   * \brief Create an empty NDArray.
   */
  static NDArray AllocReuseMemory(const std::vector<int64_t>& shape,
                                  DLDataType dtype,
                                  const NDArray& base);

 private:
  /*! \brief Internal Data content */
  NDArrayContainer* data_{nullptr};
};

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_NDARRAY_H_
