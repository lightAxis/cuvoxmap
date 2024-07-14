#pragma once

#include <cstdlib>
#include <vector>

#include <cstdint>
#include <stdexcept>

#ifdef __CUDACC__
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#endif

namespace cuvoxmap
{
        enum class eArrayAllocatorType
        {
                HOST = 1 << 0,
                DEVICE = 1 << 1,
                HOST_AND_DEVICE = HOST | DEVICE,
                NONE = 0,
        };

        template <typename T>
        class ArrayAllocator
        {
        public:
                ArrayAllocator() = default;
                ArrayAllocator(size_t size, eArrayAllocatorType alloc_type) : alloc_type_(alloc_type)
                {
                        resize(size, alloc_type);
                }
                ~ArrayAllocator() = default;

                void resize(size_t size) { resize(size, alloc_type_); }
                void resize(size_t size, eArrayAllocatorType alloc_type)
                {
                        alloc_type_ = alloc_type;
#ifdef __CUDACC__
                        if (static_cast<uint8_t>(alloc_type) & static_cast<uint8_t>(eArrayAllocatorType::HOST))
                        {
                                host_.clear();
                                host_.resize(size);
                        }
                        if (static_cast<uint8_t>(alloc_type) & static_cast<uint8_t>(eArrayAllocatorType::DEVICE))
                        {
                                device_.clear();
                                device_.resize(size);
                        }
                        return;

#else
                        if (static_cast<uint8_t>(alloc_type) & static_cast<uint8_t>(eArrayAllocatorType::HOST))
                        {
                                host_vector_.clear();
                                host_vector_.resize(size);
                        }
                        else
                        {
                                throw std::runtime_error("Need to build with CUDA support");
                        }
                        return;

#endif
                }

                T *get_host_ptr()
                {
#ifdef __CUDACC__
                        return thrust::raw_pointer_cast(host_.data());
#else
                        return host_vector_.data();
#endif
                }

                T *get_device_ptr()
                {
#ifdef __CUDACC__
                        return thrust::raw_pointer_cast(device_.data());
#else
                        return nullptr;
#endif
                }

                bool host_to_device()
                {
#ifdef __CUDACC__
                        if (alloc_type_ == eArrayAllocatorType::HOST_AND_DEVICE)
                        {
                                device_ = host_;
                                return true;
                        }
#endif
                        return false;
                }
                bool device_to_host()
                {
#ifdef __CUDACC__
                        if (alloc_type_ == eArrayAllocatorType::HOST_AND_DEVICE)
                        {
                                host_ = device_;
                                return true;
                        }
#endif
                        return false;
                }

                bool fill_host(T value)
                {
#ifdef __CUDACC__
                        if (static_cast<uint8_t>(alloc_type_) & static_cast<uint8_t>(eArrayAllocatorType::HOST))
                        {
                                thrust::fill(host_.begin(), host_.end(), value);
                                return true;
                        }
                        return false;

#else
                        std::fill(host_vector_.begin(), host_vector_.end(), value);
                        return true;
#endif
                }

                bool fill_device(T value)
                {
#ifdef __CUDACC__
                        if (static_cast<uint8_t>(alloc_type_) & static_cast<uint8_t>(eArrayAllocatorType::DEVICE))
                        {
                                thrust::fill(device_.begin(), device_.end(), value);
                                return true;
                        }
                        return false;
#else
                        return false;
#endif
                }

                T get_host(size_t idx)
                {
#ifdef __CUDACC__
                        if (static_cast<uint8_t>(alloc_type_) & static_cast<uint8_t>(eArrayAllocatorType::HOST))
                        {
                                return host_[idx];
                        }
                        return static_cast<T>(0);

#else
                        return host_vector_[idx];
#endif
                }

                T get_device(size_t idx)
                {
#ifdef __CUDACC__
                        if (static_cast<uint8_t>(alloc_type_) & static_cast<uint8_t>(eArrayAllocatorType::DEVICE))
                        {
                                return device_[idx];
                        }
                        return static_cast<T>(0);
#else
                        return static_cast<T>(0);
#endif
                }

        private:
#ifdef __CUDACC__
                thrust::host_vector<T> host_;
                thrust::device_vector<T> device_;
#else
                std::vector<T> host_vector_;
#endif
                eArrayAllocatorType alloc_type_{eArrayAllocatorType::NONE};
        };
}