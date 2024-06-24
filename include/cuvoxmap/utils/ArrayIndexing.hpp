#pragma once

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

#include <array>
#include <type_traits>

#include "Vector.hpp"

namespace cuvoxmap
{

    template <uint8_t Dim>
    class Indexing
    {
    public:
        __host__ __device__ Indexing() {}

        __host__ __device__ explicit Indexing(const Vector<uint32_t, Dim> &indices)
        {
            for (uint8_t i = 0; i < Dim; i++)
            {
                indices_[i] = indices[i];
            }
        }

        template <typename... Args, typename = typename std::enable_if<sizeof...(Args) == Dim>::type>
        __host__ __device__ explicit Indexing(Args... args) : indices_{static_cast<uint32_t>(args)...}
        {
            static_assert(sizeof...(Args) == Dim, "Number of arguments must match the dimension of the indexing object");
        }

        __host__ __device__ constexpr uint8_t DIM() const { return Dim; }

        __host__ __device__ uint8_t getIdxSize(uint8_t dim) const { return indices_[dim]; }
        __host__ uint32_t merge(const std::array<uint32_t, Dim> &idx) const
        {
            uint32_t index = 0;
            uint32_t multiplier = 1;
            for (int i = 0; i < Dim; i++)
            {
                index += idx[i] * multiplier;
                multiplier *= indices_[i];
            }
            return index;
        }
        __device__ uint32_t merge_device(const Vector<uint32_t, Dim> &idx) const
        {
            uint32_t index = 0;
            uint32_t multiplier = 1;
            for (int i = 0; i < Dim; i++)
            {
                index += idx[i] * multiplier;
                multiplier *= indices_[i];
            }
            return index;
        }

        __host__ std::array<uint32_t, Dim> split(uint32_t idx) const
        {
            std::array<uint32_t, Dim> indices;
            for (int i = 0; i < Dim; ++i)
            {
                indices[i] = idx % indices_[i];
                idx /= indices_[i];
            }
            return indices;
        }

        __device__ Vector<uint32_t, Dim> split_device(uint32_t idx) const
        {
            Vector<uint32_t, Dim> indices;
            for (int i = 0; i < Dim; ++i)
            {
                indices[i] = idx % indices_[i];
                idx /= indices_[i];
            }
            return indices;
        }

    private:
        uint32_t indices_[Dim];
    };
}