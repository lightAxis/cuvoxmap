#pragma once

#include "Vector.hpp"
#include <vector>

namespace cuvoxmap
{
    template <typename T, uint8_t Dim>
    class RayCaster
    {
    public:
        __host__ __device__ RayCaster() = default;
        __host__ __device__ RayCaster(const Vector<T, Dim> &startPos, const Vector<T, Dim> &endPos, T grid_res)
            : startPos_(startPos), endPos_(endPos), res_(grid_res)
        {
            assert(grid_res > static_cast<T>(1e-5f));

            const Vector<T, Dim> line = endPos_ - startPos_;
            lineLen_ = line.norm();
            if (lineLen_ > 0)
                dir_ = line.normalized() * res_;
            else
                dir_ = Vector<T, Dim>::Zeros();

            const uint32_t count = static_cast<uint32_t>(lineLen_ / res_);
            max_count_ = lineLen_ - static_cast<T>(count) * res_ > static_cast<T>(1e-5f) ? count + 2 : count + 1;
        }
        __host__ __device__ ~RayCaster() = default;

        __host__ __device__ Vector<T, Dim> get_StartPos() const { return startPos_; }
        __host__ __device__ Vector<T, Dim> get_EndPos() const { return endPos_; }
        __host__ __device__ T get_Res() const { return res_; }

        __host__ __device__ get_linePts_count() const { return max_count_; }

        __host__ __device__ bool get_next_pt()(Vector<T, Dim> &pt)
        {
            if (isFinished())
                return false;

            if (res_ * static_cast<T>(idx_) > lineLen_)
                pt = endPos_;
            else
                pt = startPos_ + dir_ * static_cast<T>(idx_);
            idx_++;
            return true;
        }

        __host__ __device__ bool isFinished() { return idx_ >= max_count_ - 1; }
        __host__ __device__ void reset_to_begin() { idx_ = 0; }

        __host__ std::vector<Vector<T, Dim>> get_all_pts()
        {
            std::vector<Vector<T, Dim>> pts;
            pts.reserve(max_count_);

            Vector<T, Dim> pt;
            while (get_next_pt(pt))
            {
                pts.push_back(pt);
            }
            return pts;
        }

    private:
        Vector<T, Dim> startPos_;
        Vector<T, Dim> endPos_;
        T res_;

        T lineLen_;
        Vector<T, Dim> dir_;
        uint32_t max_count_;
        uint32_t idx_{0};
    };
}