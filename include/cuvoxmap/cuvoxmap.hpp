#pragma once

#include "base/MapAllocator.hpp"
#include "base/MapAccessorHost.hpp"

namespace cuvoxmap
{
    class cuvoxmap2D
    {
    public:
        struct init_s
        {
            uint32_t x_axis_len;
            uint32_t y_axis_len;
            float resolution;
        };

        struct param_s
        {
            uint32_t x_axis_len;
            uint32_t y_axis_len;
            float resolution;
        };

        cuvoxmap2D() = default;
        explicit cuvoxmap2D(const init_s &init);
        ~cuvoxmap2D() = default;

    private:
        // allocate memory for the maps
        MapAllocator<float, 2> p_map_alloc_;
        MapAllocator<uint8_t, 2> o_map_alloc_;
        MapAllocator<float, 2> d_map_alloc_;

        // host accessor for the maps
        MapAccesssorHost<float, 2> p_map_accessor_;
        MapAccesssorHost<uint8_t, 2> o_map_accessor_;
        MapAccesssorHost<float, 2> d_map_accessor_;

        param_s param_;
    };
}