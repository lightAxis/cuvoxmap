#pragma once

#include "base/MapAllocator.hpp"
#include "base/MapAccessorHost.hpp"

#include "utils/GlobLocalIdxCvt.hpp"

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

    enum class eFrame
    {
        GLOBAL = 0x01,
        LOCAL = 0x02,
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
            uIdx2D axis_sizes;
            float resolution;
        };

        cuvoxmap2D() = default;
        explicit cuvoxmap2D(const init_s &init);
        ~cuvoxmap2D() = default;

    public:
        template <eMap mapT, eCheck checkT = eCheck::OUTSIDE, eFrame frameT = eFrame::GLOBAL>
        void set_map_withIdx(const Idx2D &idx, typename MapType<mapT>::Type value)
        {
            // static_assert((mapT == eMap::PROBABILITY && std::is_same_v<T, float>) ||
            //                   (mapT == eMap::STATE && std::is_same_v<T, uint8_t>) ||
            //                   (mapT == eMap::DISTANCE && std::is_same_v<T, float>),
            //               "Invalid type for map set operation");

            Idx2D lidx;
            if constexpr (frameT == eFrame::GLOBAL)
            {
                lidx = glc_.gidx_2_lidx(idx);
            }
            else if constexpr (frameT == eFrame::LOCAL)
            {
                lidx = idx;
            }

            if constexpr (checkT == eCheck::OUTSIDE)
            {
                if (!glc_.lidx_available(lidx))
                    return;
            }

            if constexpr (mapT == eMap::PROBABILITY)
            {
                pb_map_accessor_.set_value(lidx.cast<uint32_t>(), value);
            }
            else if constexpr (mapT == eMap::STATE)
            {
                st_map_accessor_.set_value(lidx.cast<uint32_t>(), value);
            }
            else if constexpr (mapT == eMap::DISTANCE)
            {
                dst_map_accessor_.set_value(lidx.cast<uint32_t>(), value);
            }
        }

        template <eMap mapT, eCheck checkT = eCheck::OUTSIDE, eFrame frameT = eFrame::GLOBAL>
        typename MapType<mapT>::Type get_map_withIdx(const Idx2D &idx)
        {
            // static_assert((mapT == eMap::PROBABILITY && std::is_same_v<T, float>) ||
            //                   (mapT == eMap::STATE && std::is_same_v<T, uint8_t>) ||
            //                   (mapT == eMap::DISTANCE && std::is_same_v<T, float>),
            //               "Invalid type for map get operation");

            Idx2D lidx;
            if constexpr (frameT == eFrame::GLOBAL)
            {
                lidx = glc_.gidx_2_lidx(idx);
            }
            else if constexpr (frameT == eFrame::LOCAL)
            {
                lidx = idx;
            }

            if constexpr (checkT == eCheck::OUTSIDE)
            {
                if (!glc_.lidx_available(lidx))
                    return;
            }

            if constexpr (mapT == eMap::PROBABILITY)
            {
                return pb_map_accessor_.get_value(lidx.cast<uint32_t>());
            }
            else if constexpr (mapT == eMap::STATE)
            {
                return st_map_accessor_.get_value(lidx.cast<uint32_t>());
            }
            else if constexpr (mapT == eMap::DISTANCE)
            {
                return dst_map_accessor_.get_value(lidx.cast<uint32_t>());
            }
        }

        template <eMap mapT>
        void fill(const typename MapType<mapT>::Type &value)
        {
            if constexpr (mapT == eMap::PROBABILITY)
            {
                pb_map_alloc_.fill(value);
            }
            else if constexpr (mapT == eMap::STATE)
            {
                st_map_alloc_.fill(value);
            }
            else if constexpr (mapT == eMap::DISTANCE)
            {
                dst_map_alloc_.fill(value);
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
        GlobLocalCvt<float, 2> glc_;
        Indexing<2> idx2d_;
    };
}