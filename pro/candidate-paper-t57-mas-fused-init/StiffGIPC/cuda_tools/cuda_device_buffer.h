#pragma once
#include "cuda_tools/cuda_tools.h"
namespace cudatool
{

template <typename T>
class CudaDeviceBuffer
{
  private:
    size_t m_size     = 0;
    size_t m_capacity = 0;
    T*     m_data     = nullptr;

  public:
    using value_type = T;

    CudaDeviceBuffer(size_t n);
    CudaDeviceBuffer();

    CudaDeviceBuffer(const CudaDeviceBuffer<T>& other);
    CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept;
    CudaDeviceBuffer& operator=(const CudaDeviceBuffer<T>& other);
    CudaDeviceBuffer& operator=(CudaDeviceBuffer<T>&& other);
    CudaDeviceBuffer& operator=(const std::vector<T>& other);

    CudaDeviceBuffer(const std::vector<T>& host);


    ~CudaDeviceBuffer();

    void copy_to_host(std::vector<T>& host) const;
    void copy_from_host(const std::vector<T>& host);

    void resize(size_t new_size);
    void reserve(size_t new_capacity);
    void clear();
    void reset_zero();
 
    size_t   size() const noexcept { return m_size; }
    size_t   capacity() const noexcept { return m_capacity; }
    T*       data() noexcept { return m_data; }
    const T* data() const noexcept { return m_data; }
};


}  // namespace cudatool


#include "cuda_tools/cuda_device_buffer.inl"