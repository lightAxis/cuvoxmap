#pragma once

#include "DimTmp.hpp"
#include "UseGPUTmp.hpp"

namespace cuvoxmap
{
    template <typename UseGPUTmp_, typename DimTmp_, typename BlockTmp_ = DimTmpNull>
    struct CuvoxmapParamTmp
    {
        static_assert(is_use_gpu_tmp_v<UseGPUTmp_>, "UseGPUTmp_ must be of type UseGPUTmp");
        static_assert(is_dim_tmp_v<DimTmp_>, "DimTmp_ must be of type DimTmp");
        static_assert(is_dim_tmp_v<BlockTmp_>, "BlockTmp_ must be of type DimTmp");

        using use_gpu_tmp = UseGPUTmp_;
        using dim_tmp = DimTmp_;
        using block_tmp = BlockTmp_;
    };

    template <typename T>
    struct is_cuvoxmap_param_tmp : std::false_type
    {
    };

    template <typename UseGPUTmp_, typename DimTmp_, typename BlockTmp_>
    struct is_cuvoxmap_param_tmp<CuvoxmapParamTmp<UseGPUTmp_, DimTmp_, BlockTmp_>> : std::true_type
    {
    };

    template <typename T>
    static constexpr bool is_cuvoxmap_param_tmp_v = is_cuvoxmap_param_tmp<T>::value;

    /**
     * wrapper to access the values of a CuvoxmapParamTmp struct.
     * Only use for autocomplete and readability
     */
    template <typename T>
    struct CuvoxmapParamTmpAccessor
    {
        static_assert(is_cuvoxmap_param_tmp_v<T>, "T must be of type CuvoxmapParamTmp");

        static constexpr bool use_gpu = UseGPUTmpAccessor<typename T::use_gpu_tmp>::use_gpu;

        static constexpr uint8_t dim = DimTmpAccessor<typename T::dim_tmp>::dim;
        static constexpr uint32_t dim_x = DimTmpAccessor<typename T::dim>::x;
        static constexpr uint32_t dim_y = DimTmpAccessor<typename T::dim>::y;
        static constexpr uint32_t dim_z = DimTmpAccessor<typename T::dim>::z;

        static constexpr uint8_t block_dim = DimTmpAccessor<typename T::block>::dim;
        static constexpr uint32_t block_x = DimTmpAccessor<typename T::block>::x;
        static constexpr uint32_t block_y = DimTmpAccessor<typename T::block>::y;
        static constexpr uint32_t block_z = DimTmpAccessor<typename T::block>::z;
    };
}