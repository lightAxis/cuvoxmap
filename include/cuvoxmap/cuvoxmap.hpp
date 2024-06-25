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

        enum class VoxelType
        {
            UNKNOWN = 0x0,
            OCCUPIED = 0x01,
            FREE = 0x02,
            UNOBSERVED = 0x04,
        };

        void set_pb_map(const Idx3D &idx, float value);
        void set_st_map(const Idx3D &idx, VoxelType value);
        void set_dst_map(const Idx3D &idx, float value);

        float get_pb_map(const Idx3D &Idx) const;
        VoxelType get_st_map(const Idx3D &Idx) const;
        float get_dst_map(const Idx3D &Idx) const;

        // TODO
        // probability log odd probability
        // probability raycasing
        // probability state map
        // state map get set
        // state map check various collision
        // distance map update
    private:
        // allocate memory for the maps
        MapAllocator<float, 2> pb_map_alloc_;   // probability map
        MapAllocator<uint8_t, 2> st_map_alloc_; // state map
        MapAllocator<float, 2> dst_map_alloc_;  // euclidean distance map

        // host accessor for the maps
        MapAccesssorHost<float, 2> pb_map_accessor_;
        MapAccesssorHost<uint8_t, 2> st_map_accessor_;
        MapAccesssorHost<float, 2> dst_map_accessor_;

        param_s param_;
    };
}