#pragma once

#include "Vector.hpp"

namespace cuvoxmap
{
    template <typename T, unsigned char Dim>
    class Box
    {
    public:
        Box() = default;
        ~Box() = default;

        __host__ __device__ Box(const Vector<T, Dim> &lower_bound, const Vector<T, Dim> &upper_bound) : lower_b_(lower_bound), upper_b_(upper_bound) {}
        __host__ __device__ bool cutLine(const Vector<T, Dim> &P1, const Vector<T, Dim> &P2, Vector<T, Dim> &P1_out, Vector<T, Dim> &P2_out)
        {
            const bool isInsides[2] = {isInside(P1), isInside(P2)};
            if (isInsides[0] && isInsides[1])
            {
                P1_out = P1;
                P2_out = P2;
                return true;
            }

            Vector<T, Dim> dir = P2 - P1;
            if (dir.normSquared() < static_cast<T>(1e-6))
            {
                if (isInsides[0])
                {
                    P1_out = P1;
                    P2_out = P1;
                    return true;
                }
                else if (isInsides[1])
                {
                    P1_out = P2;
                    P2_out = P2;
                    return true;
                }
                return false;
            }

            dir = dir.normalized();

            // one point is inside
            if (isInsides[0])
            {
                Vector<T, Dim> new_P2;
                for (uint8_t axis = 0; axis < 3; axis++)
                {
                    if (static_cast<T>(-1e-6) < dir[axis] && dir[axis] < static_cast<T>(1e-6))
                        continue;
                    if (isInsidePlane(dir[axis] > 0 ? upper_b_[axis] : lower_b_[axis], P1, P2, axis, new_P2))
                    {
                        P1_out = P1;
                        P2_out = new_P2;
                        return true;
                    }
                }
            }
            else if (isInsides[1])
            {
                Vector<T, Dim> new_P1;
                for (uint8_t axis = 0; axis < 3; axis++)
                {
                    if (static_cast<T>(-1e-6) < dir[axis] && dir[axis] < static_cast<T>(1e-6))
                        continue;

                    if (isInsidePlane(dir[axis] > 0 ? lower_b_[axis] : upper_b_[axis], P1, P2, axis, new_P1))
                    {
                        P1_out = new_P1;
                        P2_out = P2;
                        return true;
                    }
                }
            }

            // both points are outside
            bool found_one = false;
            Vector<T, Dim> new_P;
            for (uint8_t axis = 0; axis < 3; axis++)
            {
                if (static_cast<T>(-1e-6) < dir[axis] && dir[axis] < static_cast<T>(1e-6))
                    continue;

                if (isInsidePlane(lower_b_[axis], P1, P2, axis, new_P))
                {
                    if (found_one && (P1_out - new_P).normSquared() > static_cast<T>(1e-6))
                    {
                        P2_out = new_P;
                        return true;
                    }
                    else
                    {
                        P1_out = new_P;
                        found_one = true;
                    }
                }
            }

            for (uint8_t axis = 0; axis < 3; axis++)
            {
                if (static_cast<T>(-1e-6) < dir[axis] && dir[axis] < static_cast<T>(1e-6))
                    continue;

                if (isInsidePlane(upper_b_[axis], P1, P2, axis, new_P))
                {
                    if (found_one && (P1_out - new_P).normSquared() > static_cast<T>(1e-6))
                    {
                        P2_out = new_P;
                        return true;
                    }
                    else
                    {
                        P1_out = new_P;
                        found_one = true;
                    }
                }
            }

            if (found_one)
            {
                P2_out = P1_out;
                return true;
            }
            return false;
        }

        inline __host__ __device__ bool isInside(const Vector<T, Dim> &vec)
        {
            for (uint8_t i = 0; i < Dim; i++)
            {
                if (vec[i] < lower_b_[i] || vec[i] > upper_b_[i])
                {
                    return false;
                }
            }
            return true;
        }

    private:
        inline __host__ __device__ bool isInsidePlane(T k, const Vector<T, Dim> &P1, const Vector<T, Dim> &P2, uint8_t axis, Vector<T, Dim> &out)
        {
            k = getTofLineSeg(k, P1[axis], P2[axis]);
            out = P1 + (P2 - P1) * k;

            if (axis >= 2)
                axis = 0;
            else
                axis++;

            if (out[axis] < lower_b_[axis] || out[axis] > upper_b_[axis])
                return false;

            if (axis >= 2)
                axis = 0;
            else
                axis++;

            if (out[axis] < lower_b_[axis] || out[axis] > upper_b_[axis])
                return false;

            return true;
        }

        inline __host__ __device__ T getTofLineSeg(T k, T p1, T p2)
        {
            return (k - p1) / (p2 - p1);
        }

        inline __host__ __device__ T getLineSegWithT(T t, T p1, T p2)
        {
            return p1 + (p2 - p1) * t;
        }
        Vector<T, Dim> lower_b_;
        Vector<T, Dim> upper_b_;
    };

    using Box1f = Box<float, 1>;
    using Box2f = Box<float, 2>;
    using Box3f = Box<float, 3>;
    using Box1d = Box<double, 1>;
    using Box2d = Box<double, 2>;
    using Box3d = Box<double, 3>;

    template <>
    inline __host__ __device__ bool Box<float, 2>::isInside(const Vector<float, 2> &vec)
    {
        return vec[0] >= lower_b_[0] && vec[0] <= upper_b_[0] &&
               vec[1] >= lower_b_[1] && vec[1] <= upper_b_[1];
    }

    template <>
    inline __host__ __device__ bool Box<float, 3>::isInside(const Vector<float, 3> &vec)
    {
        return vec[0] >= lower_b_[0] && vec[0] <= upper_b_[0] &&
               vec[1] >= lower_b_[1] && vec[1] <= upper_b_[1] &&
               vec[2] >= lower_b_[2] && vec[2] <= upper_b_[2];
    }

    template <>
    inline __host__ __device__ bool Box<double, 2>::isInside(const Vector<double, 2> &vec)
    {
        return vec[0] >= lower_b_[0] && vec[0] <= upper_b_[0] &&
               vec[1] >= lower_b_[1] && vec[1] <= upper_b_[1];
    }

    template <>
    inline __host__ __device__ bool Box<double, 3>::isInside(const Vector<double, 3> &vec)
    {
        return vec[0] >= lower_b_[0] && vec[0] <= upper_b_[0] &&
               vec[1] >= lower_b_[1] && vec[1] <= upper_b_[1] &&
               vec[2] >= lower_b_[2] && vec[2] <= upper_b_[2];
    }
}