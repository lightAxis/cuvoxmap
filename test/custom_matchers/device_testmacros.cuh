#pragma once

#define CUVOXMAP_LOG_LOCATION printf("At location: \n%s(%d): at func: %s\n", __FILE__, __LINE__, __func__);
#define CUVOXMAP_KERNEL_TEST(type, a, b)                          \
    if (CUSTOM_TEST_KERNEL::KERNEL_TEST<type>((a), (b)) == false) \
    {                                                             \
        CUVOXMAP_LOG_LOCATION                                     \
    }
// #define TOMINATOR_KERNEL_TEST_WITHABS(type, a, b, abs)                           \
//     if (CUSTOM_TEST_KERNEL::KERNEL_TEST_WithAbs<type>((a), (b), (abs)) == false) \
//     {                                                                            \
//         CUVOXMAP_LOG_LOCATION                                                    \
//     }

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
