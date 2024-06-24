#pragma once
#include <cstdint>

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

namespace cuvoxmap
{
    /**
     * @brief A simple vector class, designed to deal with fixed-size vectors include device kernels
     */
    template <typename T, uint8_t Dim>
    struct Vector
    {
        T data[Dim];
        Vector() = default;
        template <typename... Args, typename = typename std::enable_if<sizeof...(Args) == Dim>::type>
        __host__ __device__ explicit Vector(Args... args) : data{static_cast<uint32_t>(args)...}
        {
            static_assert(sizeof...(Args) == Dim, "Number of arguments must match the dimension of the Vector object");
        }

        __host__ __device__ T &operator[](uint8_t idx) { return data[idx]; }
        __host__ __device__ const T &operator[](uint8_t idx) const { return data[idx]; }
        uint8_t size() const { return Dim; }
        T sum() const
        {
            T sum = 0;
            for (uint8_t i = 0; i < Dim; i++)
            {
                sum += data[i];
            }
            return sum;
        }
        T mul_sum() const
        {
            T sum = 1;
            for (uint8_t i = 0; i < Dim; i++)
            {
                sum *= data[i];
            }
            return sum;
        }
    };
    using Idx1D = Vector<uint32_t, 1>;
    using Idx2D = Vector<uint32_t, 2>;
    using Idx3D = Vector<uint32_t, 3>;
}