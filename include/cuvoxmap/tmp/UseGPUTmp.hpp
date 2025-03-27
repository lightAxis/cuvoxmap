#pragma once

#include <type_traits>

namespace cuvoxmap
{
    template <bool UseGPU>
    struct UseGPUTmp
    {
        static constexpr bool use_gpu = UseGPU;
    };

    template <typename T>
    struct is_use_gpu_tmp : std::false_type
    {
    };

    template <bool UseGPU>
    struct is_use_gpu_tmp<UseGPUTmp<UseGPU>> : std::true_type
    {
    };

    template <typename T>
    static constexpr bool is_use_gpu_tmp_v = is_use_gpu_tmp<T>::value;

    /**
     * wrapper to access the use_gpu value of a UseGPUTmp struct.
     * Only use for autocomplete and readability
     */
    template <typename T>
    struct UseGPUTmpAccessor
    {
        static_assert(is_use_gpu_tmp_v<T>, "T must be of type UseGPUTmp");

        static constexpr bool use_gpu = T::use_gpu;
    };

}