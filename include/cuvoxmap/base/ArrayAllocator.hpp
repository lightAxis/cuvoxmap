#pragma once

#include <cstdlib>
#include <vector>

#ifdef __CUDACC__
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#endif

namespace cuvoxmap
{
        template <typename T>
        class ArrayAllocator
        {
        public:
                ArrayAllocator() = default;
                ~ArrayAllocator() = default;

                void resize(size_t size)
                {
#ifdef __CUDACC__
                        host_.clear();
                        host_.resize(size);
                        device_.clear();
                        device_.resize(size);
#else
                        host_vector_.clear();
                        host_vector_.resize(size);
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

                void host_to_device()
                {
#ifdef __CUDACC__
                        device_ = host_;
#endif
                }
                void device_to_host()
                {
#ifdef __CUDACC__
                        host_ = device_;
#endif
                }

                void fill(T value)
                {
#ifdef __CUDACC__
                        thrust::fill(host_.begin(), host_.end(), value);

#else
                        std::fill(host_vector_.begin(), host_vector_.end(), value);
#endif
                }

        private:
#ifdef __CUDACC__
                thrust::host_vector<T> host_;
                thrust::device_vector<T> device_;
#else
                std::vector<T> host_vector_;
#endif
        };
}