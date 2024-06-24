#pragma once

namespace CUSTOM_TEST_KERNEL
{
    template <typename T>
    __device__ bool KERNEL_TEST(T a, T b)
    {
        if (a == b)
            return true;

        printf("ERROR left - %d, right - %d \n", static_cast<int>(a), static_cast<int>(b));
        return false;
    }

    template <>
    __device__ bool KERNEL_TEST(float a, float b)
    {
        float diff = a - b;
        if (-1e-5 < diff && diff < 1e-5)
            return true;

        printf("ERROR left - %f, right - %f \n", a, b);
        return false;
    }
}
