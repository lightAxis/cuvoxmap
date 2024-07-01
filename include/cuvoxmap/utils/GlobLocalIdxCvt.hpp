#pragma once

#include "Vector.hpp"
#include <type_traits>
// #include <iostream>
namespace cuvoxmap
{
    template <typename T, uint8_t Dim>
    class GlobLocalCvt
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "Only float and double are supported");
        static_assert(Dim == 1 || Dim == 2 || Dim == 3, "Only 1D, 2D and 3D are supported");

    public:
        GlobLocalCvt() = default;
        GlobLocalCvt(const Vector<T, Dim> &map_origin, T resolution, const Vector<uint32_t, Dim> &local_size)
            : map_origin_(map_origin), resolution_(resolution), local_size_(local_size)
        {
            update_grid_snap_diff();
            update_origin_idx();
        }
        ~GlobLocalCvt() = default;

        inline Vector<T, Dim> get_map_origin() const { return map_origin_; }
        inline void set_map_origin(const Vector<T, Dim> &origin)
        {
            map_origin_ = origin;
            update_grid_snap_diff();
            update_origin_idx();
        }

        inline T get_resolution() const { return resolution_; }
        inline void set_resolution(T resolution) { resolution_ = resolution; }

        inline Vector<uint32_t, Dim> get_local_size() const { return local_size_; }
        inline void set_local_size(const Vector<uint32_t, Dim> &local_size) { local_size_ = local_size; }

        inline Vector<T, Dim> gpos_2_lpos(const Vector<T, Dim> &gpos) const { return gpos - map_origin_; }
        inline Vector<int, Dim> gpos_2_gidx(const Vector<T, Dim> &gpos) const { return Pos2GridIdx(gpos); }
        inline Vector<int, Dim> gpos_2_lidx(const Vector<T, Dim> &gpos) const { return lpos_2_lidx(gpos_2_lpos(gpos)); }

        inline Vector<T, Dim> gidx_2_gpos(const Vector<int, Dim> &gidx) const { return GridIdx2Pos(gidx); }
        inline Vector<T, Dim> gidx_2_lpos(const Vector<int, Dim> &gidx) const { return gpos_2_lpos(gidx_2_gpos(gidx)); }
        inline Vector<int, Dim> gidx_2_lidx(const Vector<int, Dim> &gidx) const { return gidx - map_origin_idx_; }

        inline Vector<T, Dim> lpos_2_gpos(const Vector<T, Dim> &lpos) const { return lpos + map_origin_; }
        inline Vector<int, Dim> lpos_2_gidx(const Vector<T, Dim> &lpos) const { return gpos_2_gidx(lpos_2_gpos(lpos)); }
        inline Vector<int, Dim> lpos_2_lidx(const Vector<T, Dim> &lpos) const { return Pos2GridIdx(lpos + map_origin_grid_snap_diff_); }

        inline Vector<T, Dim> lidx_2_lpos(const Vector<int, Dim> &lidx) const { return GridIdx2Pos(lidx) - map_origin_grid_snap_diff_; }
        inline Vector<T, Dim> lidx_2_gpos(const Vector<int, Dim> &lidx) const { return lpos_2_gpos(lidx_2_lpos(lidx)); }
        inline Vector<int, Dim> lidx_2_gidx(const Vector<int, Dim> &lidx) const { return lidx + map_origin_idx_; }

        inline static bool lidx_available(const Vector<uint32_t, Dim> &axises, const Vector<int, Dim> &lidx)
        {
            if constexpr (Dim == 1)
            {
                return lidx[0] >= 0 && axises[0] > lidx[0];
            }
            else if constexpr (Dim == 2)
            {
                return lidx[0] >= 0 && axises[0] > lidx[0] &&
                       lidx[1] >= 0 && axises[1] > lidx[1];
            }
            else if constexpr (Dim == 3)
            {
                return lidx[0] >= 0 && axises[0] > lidx[0] &&
                       lidx[1] >= 0 && axises[1] > lidx[1] &&
                       lidx[2] >= 0 && axises[2] > lidx[2];
            }
            return false;
        }

        inline bool lidx_available(const Vector<int, Dim> &lidx) const
        {
            return lidx_available(local_size_, lidx);
        }

    private:
        inline void update_grid_snap_diff()
        {
            map_origin_grid_snap_diff_ = map_origin_ - (Pos2GridIdx(map_origin_).template cast<T>()) * resolution_;
        }

        inline void update_origin_idx()
        {
            map_origin_idx_ = Pos2GridIdx(map_origin_);
        }

        Vector<int, Dim> Pos2GridIdx(const Vector<T, Dim> &pos) const
        {
            if constexpr (Dim == 1)
            {
                return Vector<int, 1>{static_cast<int>(floor(pos[0] / resolution_))};
            }
            else if constexpr (Dim == 2)
            {
                return Vector<int, 2>{static_cast<int>(floor(pos[0] / resolution_)),
                                      static_cast<int>(floor(pos[1] / resolution_))};
            }
            else if constexpr (Dim == 3)
            {
                return Vector<int, 3>{static_cast<int>(floor(pos[0] / resolution_)),
                                      static_cast<int>(floor(pos[1] / resolution_)),
                                      static_cast<int>(floor(pos[2] / resolution_))};
            }
        }

        Vector<T, Dim> GridIdx2Pos(const Vector<int, Dim> &idx) const
        {
            if constexpr (Dim == 1)
            {
                return Vector<T, 1>{idx[0] * resolution_ + static_cast<T>(0.5) * resolution_};
            }
            else if constexpr (Dim == 2)
            {
                return Vector<T, 2>{idx[0] * resolution_ + static_cast<T>(0.5) * resolution_,
                                    idx[1] * resolution_ + static_cast<T>(0.5) * resolution_};
            }
            else if constexpr (Dim == 3)
            {
                return Vector<T, 3>{idx[0] * resolution_ + static_cast<T>(0.5) * resolution_,
                                    idx[1] * resolution_ + static_cast<T>(0.5) * resolution_,
                                    idx[2] * resolution_ + static_cast<T>(0.5) * resolution_};
            }
        }

        Vector<T, Dim> map_origin_;
        Vector<T, Dim> map_origin_grid_snap_diff_;
        Vector<int, Dim> map_origin_idx_;
        T resolution_;
        Vector<uint32_t, Dim> local_size_;
    };
}