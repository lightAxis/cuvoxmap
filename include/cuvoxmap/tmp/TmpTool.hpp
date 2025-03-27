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

namespace cuvoxmap
{
    namespace tmp_tool
    {
        template <uint32_t mul>
        __host__ __device__ static constexpr inline uint32_t MUl_OR_BITSHIFT(uint32_t val)
        {
            constexpr bool isPowerOfTwo = (mul & (mul - 1)) == 0;

            if constexpr (mul == 0)
            {
                return 0;
            }
            else if constexpr (isPowerOfTwo)
            {
                constexpr uint32_t shift_amount = __builtin_ctz(mul);
                return val << shift_amount;
            }
            else
            {
                return val * mul;
            }
        }

        template <uint32_t div>
        __host__ __device__ static constexpr inline uint32_t DIV_OR_BITSHIFT(uint32_t val)
        {
            constexpr bool isPowerOfTwo = (div & (div - 1)) == 0;

            static_assert(div != 0, "div must not be 0");
            if constexpr (isPowerOfTwo)
            {
                constexpr uint32_t shift_amount = __builtin_ctz(div);
                return val >> shift_amount;
            }
            else
            {
                return val / div;
            }
        }

        template <uint32_t mod>
        __host__ __device__ static constexpr inline uint32_t MOD_OR_BITSHIFT(uint32_t val)
        {
            constexpr bool isPowerOfTwo = (mod & (mod - 1)) == 0;

            static_assert(mod != 0, "mod must not be 0");
            if constexpr (isPowerOfTwo)
            {
                return val & (mod - 1);
            }
            else
            {
                return val % mod;
            }
        }
    }
}