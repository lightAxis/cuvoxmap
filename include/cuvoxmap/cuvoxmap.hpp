#pragma once

#include "base/MapAllocator.hpp"
#include "base/MapAccessorHost.hpp"

namespace cuvoxmap
{
    enum class eVoxel
    {
        UNKNOWN = 0x0,
        OCCUPIED = 0x01,
        FREE = 0x02,
        UNOBSERVED = 0x04,
    };

    enum class eCheck
    {
        NONE = 0x00,
        OUTSIDE = 0x01,
    };

    enum class eMap
    {
        PROBABILITY = 0x01,
        STATE = 0x02,
        DISTANCE = 0x04,
    };

    template <eMap mapT>
    struct MapType
    {
        using Type = void;
    };
    // MapType specialization
    template <>
    struct MapType<eMap::PROBABILITY>
    {
        using Type = float;
    };
    template <>
    struct MapType<eMap::STATE>
    {
        using Type = uint8_t;
    };
    template <>
    struct MapType<eMap::DISTANCE>
    {
        using Type = float;
    };

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

    public:
        template <eMap mapT, eCheck checkT = eCheck::OUTSIDE>
        void set_map_withGlobIdx(const Idx2D &idx, typename MapType<mapT>::Type value)
        {
            // static_assert((mapT == eMap::PROBABILITY && std::is_same_v<T, float>) ||
            //                   (mapT == eMap::STATE && std::is_same_v<T, uint8_t>) ||
            //                   (mapT == eMap::DISTANCE && std::is_same_v<T, float>),
            //               "Invalid type for map set operation");

            if constexpr (checkT == eCheck::OUTSIDE)
            {
            }

            if constexpr (mapT == eMap::PROBABILITY)
            {
                pb_map_accessor_.set_value(idx.cast<uint32_t>(), value);
            }
            else if constexpr (mapT == eMap::STATE)
            {
            }
            else if constexpr (mapT == eMap::DISTANCE)
            {
            }
        }

        // float get_pb_map(const Idx2D &Idx) const;
        // eVoxel get_st_map(const Idx2D &Idx) const;
        // float get_dst_map(const Idx2D &Idx) const;

        // TODO
        // probability log odd probability
        // probability raycasing
        // probability state map
        // state map get set
        // state map check various collision
        // distance map update
    private:
        // allocate memory for the maps
        MapAllocator<MapType<eMap::PROBABILITY>::Type, 2> pb_map_alloc_; // probability map
        MapAllocator<MapType<eMap::STATE>::Type, 2> st_map_alloc_;       // state map
        MapAllocator<MapType<eMap::DISTANCE>::Type, 2> dst_map_alloc_;   // euclidean distance map

        // host accessor for the maps
        MapAccesssorHost<MapType<eMap::PROBABILITY>::Type, 2> pb_map_accessor_;
        MapAccesssorHost<MapType<eMap::STATE>::Type, 2> st_map_accessor_;
        MapAccesssorHost<MapType<eMap::DISTANCE>::Type, 2> dst_map_accessor_;

        param_s param_;
    };

}