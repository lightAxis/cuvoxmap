#pragma once
#include <cinttypes>
#include <type_traits>

namespace cuvoxmap
{
    /**
     * @brief Template struct to store the dimension and the size of the dimension
     */
    template <uint8_t Dim, uint32_t X = 0, uint32_t Y = 0, uint32_t Z = 0>
    struct DimTmp
    {
        static_assert(Dim >= 0 && Dim <= 3, "Dim must be between 0 and 3");
        static constexpr uint8_t dim = Dim;
        static constexpr uint32_t x = X;
        static constexpr uint32_t y = Y;
        static constexpr uint32_t z = Z;
    };

    using DimTmpNull = DimTmp<0, 0, 0, 0>;

    template <typename T>
    struct is_dim_tmp : std::false_type
    {
    };

    template <uint8_t Dim, uint32_t X, uint32_t Y, uint32_t Z>
    struct is_dim_tmp<DimTmp<Dim, X, Y, Z>> : std::true_type
    {
    };

    template <typename T>
    static constexpr bool is_dim_tmp_v = is_dim_tmp<T>::value;

    /**
     * wrapper to access the dimension and size of a DimTmp struct.
     * Only use for autocomplete and readability
     */
    template <typename T>
    struct DimTmpAccessor
    {
        static_assert(is_dim_tmp_v<T>, "T must be of type DimTmp");

        static constexpr uint8_t dim = T::dim;
        static constexpr uint32_t x = T::x;
        static constexpr uint32_t y = T::y;
        static constexpr uint32_t z = T::z;
    };
}