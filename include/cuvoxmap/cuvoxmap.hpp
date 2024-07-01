#pragma once

#include "base/MapAllocator.hpp"
#include "base/MapAccessorHost.hpp"

#include "utils/GlobLocalIdxCvt.hpp"

namespace cuvoxmap
{
    enum class eVoxel
    {
        UNKNOWN = 1 << 0,
        OCCUPIED = 1 << 1,
        FREE = 1 << 2,
        UNOBSERVED = 1 << 3,
    };

    enum class eCheck
    {
        NONE = 0x00,
        BOUNDARY = 0x01,
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

    template <eMap mapT, eCheck checkT = eCheck::BOUNDARY, eFrame frameT = eFrame::GLOBAL>
    struct map_getset_s
    {
        constexpr static eMap mapT_v = mapT;
        constexpr static eCheck checkT_v = checkT;
        constexpr static eFrame frameT_v = frameT;
        using map_getset_s_type = std::true_type;
    };

    namespace getset
    {
        using PRB_FAST_GLB = map_getset_s<eMap::PROBABILITY, eCheck::NONE, eFrame::GLOBAL>;
        using PRB_FAST_LOC = map_getset_s<eMap::PROBABILITY, eCheck::NONE, eFrame::LOCAL>;
        using PRB_CHK_GLB = map_getset_s<eMap::PROBABILITY, eCheck::BOUNDARY, eFrame::GLOBAL>;
        using PRB_CHK_LOC = map_getset_s<eMap::PROBABILITY, eCheck::BOUNDARY, eFrame::LOCAL>;
        using ST_FAST_GLB = map_getset_s<eMap::STATE, eCheck::NONE, eFrame::GLOBAL>;
        using ST_FAST_LOC = map_getset_s<eMap::STATE, eCheck::NONE, eFrame::LOCAL>;
        using ST_CHK_GLB = map_getset_s<eMap::STATE, eCheck::BOUNDARY, eFrame::GLOBAL>;
        using ST_CHK_LOC = map_getset_s<eMap::STATE, eCheck::BOUNDARY, eFrame::LOCAL>;
        using DST_FAST_GLB = map_getset_s<eMap::DISTANCE, eCheck::NONE, eFrame::GLOBAL>;
        using DST_FAST_LOC = map_getset_s<eMap::DISTANCE, eCheck::NONE, eFrame::LOCAL>;
        using DST_CHK_GLB = map_getset_s<eMap::DISTANCE, eCheck::BOUNDARY, eFrame::GLOBAL>;
        using DST_CHK_LOC = map_getset_s<eMap::DISTANCE, eCheck::BOUNDARY, eFrame::LOCAL>;
    }

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
        template <typename map_getsetT>
        void set_map_withIdx(const Idx2D &idx, typename MapType<map_getsetT::mapT_v>::Type value)
        {
            static_assert(map_getsetT::map_getset_s_type::value, "Invalid type for map set operation. use map_getset_s type to set map value");

            Idx2D lidx;
            if constexpr (map_getsetT::frameT_v == eFrame::GLOBAL)
            {
                lidx = glc_.gidx_2_lidx(idx);
            }
            else if constexpr (map_getsetT::frameT_v == eFrame::LOCAL)
            {
                lidx = idx;
            }

            if constexpr (map_getsetT::checkT_v == eCheck::BOUNDARY)
            {
                if (!glc_.lidx_available(lidx))
                    return;
            }

            assert(lidx[0] >= 0 && lidx[1] >= 0); // local index should be positive

            if constexpr (map_getsetT::mapT_v == eMap::PROBABILITY)
            {
                pb_map_accessor_.set_value(lidx.cast<uint32_t>(), value);
            }
            else if constexpr (map_getsetT::mapT_v == eMap::STATE)
            {
                st_map_accessor_.set_value(lidx.cast<uint32_t>(), value);
            }
            else if constexpr (map_getsetT::mapT_v == eMap::DISTANCE)
            {
                dst_map_accessor_.set_value(lidx.cast<uint32_t>(), value);
            }
        }

        template <typename map_getsetT>
        typename MapType<map_getsetT::mapT_v>::Type get_map_withIdx(const Idx2D &idx)
        {
            static_assert(map_getsetT::map_getset_s_type::value, "Invalid type for map set operation. use map_getset_s type to get map value");

            Idx2D lidx;
            if constexpr (map_getsetT::frameT_v == eFrame::GLOBAL)
            {
                lidx = glc_.gidx_2_lidx(idx);
            }
            else if constexpr (map_getsetT::frameT_v == eFrame::LOCAL)
            {
                lidx = idx;
            }

            if constexpr (map_getsetT::checkT_v == eCheck::BOUNDARY)
            {
                if (!glc_.lidx_available(lidx))
                    return typename MapType<map_getsetT::mapT_v>::Type{};
            }

            assert(lidx[0] >= 0 && lidx[1] >= 0); // local index should be positive

            if constexpr (map_getsetT::mapT_v == eMap::PROBABILITY)
            {
                return pb_map_accessor_.get_value(lidx.cast<uint32_t>());
            }
            else if constexpr (map_getsetT::mapT_v == eMap::STATE)
            {
                return st_map_accessor_.get_value(lidx.cast<uint32_t>());
            }
            else if constexpr (map_getsetT::mapT_v == eMap::DISTANCE)
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

        inline void set_origin(const Vector<float, 2> &originPos) { glc_.set_map_origin(originPos); }
        inline GlobLocalCvt<float, 2> &get_glob_loc_cvt() { return glc_; }

        // TODO
        // probability log odd probability
        // probability raycasing
        // probability state map
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