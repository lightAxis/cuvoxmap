#pragma once

#include "cuvoxmap_param.hpp"

#include "base/MapAllocator.hpp"
#include "base/MapAccessorHost.hpp"

#include "utils/GlobLocalIdxCvt.hpp"
#include "utils/Box.hpp"
#include "utils/RayCaster.hpp"
#include "utils/SuperCover2D.hpp"

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

    // map date type specialization
    template <eMap mapT>
    struct MapType
    {
        using Type = std::conditional_t<mapT == eMap::PROBABILITY, float,
                                        std::conditional_t<mapT == eMap::STATE, uint8_t,
                                                           std::conditional_t<mapT == eMap::DISTANCE, float, void>>>;
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

    enum class eLine
    {
        RAYCAST = 0x01,
        SUPERCOVER = 0x02,
    };

    template <eCheck checkT, eFrame frameT, eLine lineT>
    struct line_check_s
    {
        constexpr static eCheck checkT_v = checkT;
        constexpr static eFrame frameT_v = frameT;
        constexpr static eLine lineT_v = lineT;
        using line_check_s_type = std::true_type;
    };

    namespace linecheck
    {
        using NON_GLB_RAY = line_check_s<eCheck::NONE, eFrame::GLOBAL, eLine::RAYCAST>;
        using NON_GLB_SUP = line_check_s<eCheck::NONE, eFrame::GLOBAL, eLine::SUPERCOVER>;
        using NON_LOC_RAY = line_check_s<eCheck::NONE, eFrame::LOCAL, eLine::RAYCAST>;
        using NON_LOC_SUP = line_check_s<eCheck::NONE, eFrame::LOCAL, eLine::SUPERCOVER>;
        using CHK_GLB_RAY = line_check_s<eCheck::BOUNDARY, eFrame::GLOBAL, eLine::RAYCAST>;
        using CHK_GLB_SUP = line_check_s<eCheck::BOUNDARY, eFrame::GLOBAL, eLine::SUPERCOVER>;
        using CHK_LOC_RAY = line_check_s<eCheck::BOUNDARY, eFrame::LOCAL, eLine::RAYCAST>;
        using CHK_LOC_SUP = line_check_s<eCheck::BOUNDARY, eFrame::LOCAL, eLine::SUPERCOVER>;
    }

    class cuvoxmap2D
    {
    public:
        struct init_s
        {
            uint32_t x_axis_len;
            uint32_t y_axis_len;
            float resolution;
            bool use_gpu;
        };

        struct param_s
        {
            uIdx2D axis_sizes;
            float resolution;
            bool use_gpu;
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

        template <eMap mapT, eMemAllocType memT>
        void fill_all(const typename MapType<mapT>::Type &value)
        {
            if constexpr (mapT == eMap::PROBABILITY)
            {
                if constexpr (static_cast<uint8_t>(memT) & static_cast<uint8_t>(eMemAllocType::HOST))
                {
                    pb_map_alloc_.fill_host(value);
                }
                if constexpr (static_cast<uint8_t>(memT) & static_cast<uint8_t>(eMemAllocType::DEVICE))
                {
                    pb_map_alloc_.fill_device(value);
                }
            }
            else if constexpr (mapT == eMap::STATE)
            {
                if constexpr (static_cast<uint8_t>(memT) & static_cast<uint8_t>(eMemAllocType::HOST))
                {
                    st_map_alloc_.fill_host(value);
                }
                if constexpr (static_cast<uint8_t>(memT) & static_cast<uint8_t>(eMemAllocType::DEVICE))
                {
                    st_map_alloc_.fill_device(value);
                }
            }
            else if constexpr (mapT == eMap::DISTANCE)
            {
                if constexpr (static_cast<uint8_t>(memT) & static_cast<uint8_t>(eMemAllocType::HOST))
                {
                    dst_map_alloc_.fill_host(value);
                }
                if constexpr (static_cast<uint8_t>(memT) & static_cast<uint8_t>(eMemAllocType::DEVICE))
                {
                    dst_map_alloc_.fill_device(value);
                }
            }
        }

        inline void set_origin(const Vector<float, 2> &originPos) { glc_.set_map_origin(originPos); }
        inline GlobLocalCvt<float, 2> &get_glob_loc_cvt() { return glc_; }

        template <typename line_checkT>
        inline bool check_line_state_map(const Vector<float, 2> &start, const Vector<float, 2> &end, uint8_t voxelBitflags)
        {
            static_assert(line_checkT::line_check_s_type::value, "Invalid type for line check operation. use line_check_s type to check line");

            Vector<float, 2> start_l = start;
            Vector<float, 2> end_l = end;

            if constexpr (line_checkT::frameT_v == eFrame::GLOBAL)
            {
                start_l = glc_.gpos_2_lpos(start);
                end_l = glc_.gpos_2_lpos(end);
            }

            if constexpr (line_checkT::checkT_v == eCheck::BOUNDARY)
            {
                Vector<float, 2> p1_out;
                Vector<float, 2> p2_out;
                if (box_.cutLine(start_l, end_l, p1_out, p2_out))
                {
                    start_l = p1_out;
                    end_l = p2_out;
                }
                else
                {
                    if (voxelBitflags & static_cast<uint8_t>(eVoxel::UNKNOWN) ||
                        voxelBitflags & static_cast<uint8_t>(eVoxel::UNOBSERVED))
                        return true;

                    return false;
                }
            }

            if constexpr (line_checkT::lineT_v == eLine::RAYCAST)
            {
                RayCaster<float, 2> ray{start_l, end_l, param_.resolution};
                Float2D pt;
                while (ray.get_next_pt(pt))
                {
                    MapType<eMap::STATE>::Type value = get_map_withIdx<getset::ST_FAST_LOC>(glc_.lpos_2_lidx(pt));
                    if (value & voxelBitflags)
                        return true;
                }
                return false;
            }
            else if constexpr (line_checkT::lineT_v == eLine::SUPERCOVER)
            {
                SuperCoverLine2D<float, int32_t> line{start_l, end_l, param_.resolution};
                Idx2D lidx;
                while (line.get_next_idx(lidx))
                {
                    MapType<eMap::STATE>::Type value = get_map_withIdx<getset::ST_FAST_LOC>(lidx);
                    if (value & voxelBitflags)
                        return true;
                }
                return false;
            }

            return false;
        }

        // TODO
        void distance_map_update_withCPU();
        void distance_map_update_withGPU();

        // TODO
        // probability log odd probability
        // probability raycasing
        // probability state map update

        /**
         * memcpy
         */
        template <eMap mapT>
        void host_to_device()
        {
            if constexpr (mapT == eMap::PROBABILITY)
            {
                pb_map_alloc_.host_to_device();
            }
            else if constexpr (mapT == eMap::STATE)
            {
                st_map_alloc_.host_to_device();
            }
            else if constexpr (mapT == eMap::DISTANCE)
            {
                dst_map_alloc_.host_to_device();
            }
        }

        template <eMap mapT>
        void device_to_host()
        {
            if constexpr (mapT == eMap::PROBABILITY)
            {
                pb_map_alloc_.device_to_host();
            }
            else if constexpr (mapT == eMap::STATE)
            {
                st_map_alloc_.device_to_host();
            }
            else if constexpr (mapT == eMap::DISTANCE)
            {
                dst_map_alloc_.device_to_host();
            }
        }

    private:
        // allocate memory for the maps
        MapAllocator<MapType<eMap::PROBABILITY>::Type, 2> pb_map_alloc_; // probability map
        MapAllocator<MapType<eMap::STATE>::Type, 2> st_map_alloc_;       // state map
        MapAllocator<MapType<eMap::DISTANCE>::Type, 2> dst_map_alloc_;   // euclidean distance map

        // host accessor for the maps
        MapAccessorHost<MapType<eMap::PROBABILITY>::Type, 2> pb_map_accessor_;
        MapAccessorHost<MapType<eMap::STATE>::Type, 2> st_map_accessor_;
        MapAccessorHost<MapType<eMap::DISTANCE>::Type, 2> dst_map_accessor_;

        // extra array for euclidean distance mapping
        MapAllocator<MapType<eMap::DISTANCE>::Type, 2> temp1_dstmap_alloc_;
        MapAllocator<MapType<eMap::DISTANCE>::Type, 1> z_buffer_alloc_;
        MapAllocator<int, 1> v_buffer_alloc_;

        param_s param_;
        GlobLocalCvt<MapType<eMap::PROBABILITY>::Type, 2> glc_;
        Indexing<2> idx2d_;
        Box<MapType<eMap::PROBABILITY>::Type, 2> box_;
    };
}