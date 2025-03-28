#pragma once

#ifdef __CUDACC__
// print error in kernel,
#include <cstdio>
#else
// remove __host__, __device__ for host cpu compilation
#define __host__
#define __device__
// print error in host cpu
#include <stdexcept>
#endif

#include <cstdint>
#include <initializer_list>
#include <cmath>
#include <cassert>

namespace cuvoxmap
{
    namespace VectorImpl
    {
        struct Impl
        {
            static __host__ __device__ float sqrt_fd(float x) { return sqrtf(x); }
            static __host__ __device__ double sqrt_fd(double x) { return sqrt(x); }

            static __host__ __device__ float abs_fd(float x) { return fabsf(x); }
            static __host__ __device__ double abs_fd(double x) { return fabs(x); }
        };
    }
    /**
     * @brief A simple vector class, designed to deal with fixed-size vectors include device kernels
     */
    template <typename T, uint8_t Dim>
    struct Vector
    {
        T data[Dim];
        Vector() = default;

        __host__ __device__ explicit Vector(T val)
        {
            for (uint8_t i = 0; i < Dim; i++)
            {
                data[i] = val;
            }
        }

        // After CUDA 7.5, nvcc support c++11 features, initializer_list is supported
        __host__ __device__ explicit Vector(std::initializer_list<T> list)
        {
            size_t num = 0;
            for (auto &i : list)
            {
                if (num >= Dim)
                {
#ifdef __CUDACC__
                    // cuda kernel function can't throw std::error
                    printf("Number of arguments must match the dimension of the Vector object\n");
                    break;
#else
                    // host cpu function can throw std::error
                    throw std::invalid_argument("Number of arguments must match the dimension of the Vector object");
#endif
                }

                data[num] = i;
                num++;
            }
        }

        __host__ __device__ static Vector<T, Dim> Ones()
        {
            Vector<T, Dim> result;
            for (uint8_t i = 0; i < Dim; i++)
            {
                result.data[i] = 1;
            }
            return result;
        }

        __host__ __device__ static Vector<T, Dim> Zeros()
        {
            Vector<T, Dim> result;
            for (uint8_t i = 0; i < Dim; i++)
            {
                result.data[i] = 0;
            }
            return result;
        }

        __host__ __device__ T &
        operator[](uint8_t idx)
        {
            return data[idx];
        }
        __host__ __device__ const T &operator[](uint8_t idx) const { return data[idx]; }
        uint8_t size() const { return Dim; }
        T sum() const
        {
            T sum = 0;
            for (uint8_t i = 0; i < Dim; i++)
            {
                sum += data[i];
            }
            return sum;
        }
        T mul_sum() const
        {
            T sum = 1;
            for (uint8_t i = 0; i < Dim; i++)
            {
                sum *= data[i];
            }
            return sum;
        }

        __host__ __device__ bool operator==(const Vector<T, Dim> &other) const
        {
            for (uint8_t i = 0; i < Dim; i++)
            {
                if (data[i] != other.data[i])
                {
                    return false;
                }
            }
            return true;
        }

        __host__ __device__ bool operator!=(const Vector<T, Dim> &other) const
        {
            return !(*this == other);
        }

        __host__ __device__ Vector<T, Dim> operator+(const Vector<T, Dim> &other) const
        {
            Vector<T, Dim> result;
            for (uint8_t i = 0; i < Dim; i++)
            {
                result.data[i] = data[i] + other.data[i];
            }
            return result;
        }

        __host__ __device__ Vector<T, Dim> operator-(const Vector<T, Dim> &other) const
        {
            Vector<T, Dim> result;
            for (uint8_t i = 0; i < Dim; i++)
            {
                result.data[i] = data[i] - other.data[i];
            }
            return result;
        }

        __host__ __device__ Vector<T, Dim> &operator+=(const Vector<T, Dim> &other)
        {
            for (uint8_t i = 0; i < Dim; i++)
            {
                data[i] += other.data[i];
            }
            return *this;
        }

        __host__ __device__ Vector<T, Dim> &operator-=(const Vector<T, Dim> &other)
        {
            for (uint8_t i = 0; i < Dim; i++)
            {
                data[i] -= other.data[i];
            }
            return *this;
        }

        __host__ __device__ Vector<T, Dim> operator*(const T &scalar) const
        {
            Vector<T, Dim> result;
            for (uint8_t i = 0; i < Dim; i++)
            {
                result.data[i] = data[i] * scalar;
            }
            return result;
        }

        __host__ __device__ Vector<T, Dim> operator/(const T &scalar) const
        {
            Vector<T, Dim> result;
            for (uint8_t i = 0; i < Dim; i++)
            {
                result.data[i] = data[i] / scalar;
            }
            return result;
        }

        __host__ __device__ Vector<T, Dim> &operator*=(const T &scalar)
        {
            for (uint8_t i = 0; i < Dim; i++)
            {
                data[i] *= scalar;
            }
            return *this;
        }

        __host__ __device__ Vector<T, Dim> &operator/=(const T &scalar)
        {
            for (uint8_t i = 0; i < Dim; i++)
            {
                data[i] /= scalar;
            }
            return *this;
        }

        __host__ __device__ Vector<T, Dim> elementWiseMul(const Vector<T, Dim> &other) const
        {
            Vector<T, Dim> result;
            for (uint8_t i = 0; i < Dim; i++)
            {
                result.data[i] = data[i] * other.data[i];
            }
            return result;
        }

        __host__ __device__ Vector<T, Dim> elementWiseDiv(const Vector<T, Dim> &other) const
        {
            Vector<T, Dim> result;
            for (uint8_t i = 0; i < Dim; i++)
            {
                result.data[i] = data[i] / other.data[i];
            }
            return result;
        }

        __host__ __device__ T dot(const Vector<T, Dim> &other) const
        {
            T result = 0;
            for (uint8_t i = 0; i < Dim; i++)
            {
                result += data[i] * other.data[i];
            }
            return result;
        }

        __host__ __device__ Vector<T, Dim> cross(const Vector<T, Dim> &other) const
        {
            static_assert(Dim == 3, "Cross product is only defined for 3D vectors");
            return Vector<T, Dim>({data[1] * other.data[2] - data[2] * other.data[1],
                                   data[2] * other.data[0] - data[0] * other.data[2],
                                   data[0] * other.data[1] - data[1] * other.data[0]});
        }
        __host__ __device__ T normSquared() const { return dot(*this); }
        __host__ __device__ T norm() const { return VectorImpl::Impl::sqrt_fd(normSquared()); }

        __host__ __device__ Vector<T, Dim> normalized() const { return *this / norm(); }
        __host__ __device__ Vector<T, Dim> &normalize() { return *this /= norm(); }

        template <typename T2>
        __host__ __device__ Vector<T2, Dim> cast() const
        {
            Vector<T2, Dim> result;
            for (uint8_t i = 0; i < Dim; i++)
            {
                result.data[i] = static_cast<T2>(data[i]);
            }
            return result;
        }
    };

    template <typename T, uint8_t Dim>
    Vector<T, Dim> operator*(T scalar, const Vector<T, Dim> &vec)
    {
        return vec * scalar;
    }

    using uIdx1D = Vector<uint32_t, 1>;
    using uIdx2D = Vector<uint32_t, 2>;
    using uIdx3D = Vector<uint32_t, 3>;
    using Idx1D = Vector<int, 1>;
    using Idx2D = Vector<int, 2>;
    using Idx3D = Vector<int, 3>;
    using Float1D = Vector<float, 1>;
    using Float2D = Vector<float, 2>;
    using Float3D = Vector<float, 3>;
    using Double1D = Vector<double, 1>;
    using Double2D = Vector<double, 2>;
    using Double3D = Vector<double, 3>;
}