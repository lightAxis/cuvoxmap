#pragma once

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

        __host__ __device__ constexpr uint8_t DIM() const { return Dim; }

        __host__ __device__ uint8_t getIdxSize(uint8_t dim) const { return indices_[dim]; }

        __host__ __device__ uint32_t merge(const Vector<uint32_t, Dim> &idx) const
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

        __host__ __device__ Vector<uint32_t, Dim> split(uint32_t idx) const
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