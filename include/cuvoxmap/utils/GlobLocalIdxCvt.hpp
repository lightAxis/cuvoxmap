#pragma once

#include "Vector.hpp"
#include <type_traits>

namespace cuvoxmap
{
    template <typename T, uint8_t Dim>
    class GlobLocalCvt
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Only float and double are supported");
        static_assert(Dim == 1 || Dim == 2 || Dim == 3, "Only 1D, 2D and 3D are supported");

    public:
        GlobLocalCvt() = default;
        ~GlobLocalCvt() = default;

        inline Vector<T, Dim> get_map_origin() const { return map_origin_; }
        inline void set_map_origin(const Vector<T, Dim> &origin)
        {
            map_origin_ = origin;
            update_grid_snap_diff();
        }

        inline T get_resolution() const { return resolution_; }
        inline void set_resolution(T resolution) { resolution_ = resolution; }

        inline Vector<T, Dim> gpos_2_lpos(const Vector<T, Dim> &gpos) const
        {
            return gpos - map_origin_;
        }

        inline Vector<T, Dim> lpos_2_gpos(const Vector<T, Dim> &lpos) const
        {
            return lpos + map_origin_;
        }

        inline Vector<uint32_t, Dim> lpos_2_lidx(const Vector<T, Dim> &lpos) const
        {
            return Pos2Idx(lpos + map_origin_grid_snap_diff_);
        }

        inline Vector<T, Dim> lidx_2_lpos(const Vector<uint32_t, Dim> &lidx) const
        {
            return Idx2Pos(lidx) - map_origin_grid_snap_diff_;
        }

        inline Vector<uint32_t, Dim> gpos_2_lidx(const Vector<T, Dim> &gpos) const
        {
            return lpos_2_lidx(gpos_2_lpos(gpos));
        }

        inline Vector<T, Dim> lidx_2_gpos(const Vector<uint32_t, Dim> &lidx) const
        {
            return lpos_2_gpos(lidx_2_lpos(lidx));
        }

    private:
        void update_grid_snap_diff()
        {
            map_origin_grid_snap_diff_ = map_origin_ - (Pos2Idx(map_origin_) * resolution_);
        }

        Vector<uint32_t, Dim> Pos2Idx(const Vector<T, Dim> &pos) const
        {
            if constexpr (Dim == 1)
            {
                return Vector<uint32_t, 1>{static_cast<uint32_t>(floor(pos[0] / resolution_))};
            }
            else if constexpr (Dim == 2)
            {
                return Vector<uint32_t, 2>{static_cast<uint32_t>(floor(pos[0] / resolution_)),
                                           static_cast<uint32_t>(floor(pos[1] / resolution_))};
            }
            else if constexpr (Dim == 3)
            {
                return Vector<uint32_t, 3>{static_cast<uint32_t>(floor(pos[0] / resolution_)),
                                           static_cast<uint32_t>(floor(pos[1] / resolution_)),
                                           static_cast<uint32_t>(floor(pos[2] / resolution_))};
            }
        }

        Vector<T, Dim> Idx2Pos(const Vector<uint32_t, Dim> &idx) const
        {
            if constexpr (Dim == 1)
            {
                return Vector<T, 1>{idx[0] * resolution_ + static_cast<T>(0.5) * resolution_};
            }
            else if constexpr (Dim == 2)
            {
                return Vector<T, 1>{idx[0] * resolution_ + static_cast<T>(0.5) * resolution_,
                                    idx[1] * resolution_ + static_cast<T>(0.5) * resolution_};
            }
            else if constexpr (Dim == 3)
            {
                return Vector<T, 1>{idx[0] * resolution_ + static_cast<T>(0.5) * resolution_,
                                    idx[1] * resolution_ + static_cast<T>(0.5) * resolution_,
                                    idx[2] * resolution_ + static_cast<T>(0.5) * resolution_};
            }
        }

        Vector<T, Dim> map_origin_;
        Vector<T, Dim> map_origin_grid_snap_diff_;
        T resolution_;
    };
}