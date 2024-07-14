#pragma once

#include <type_traits>
#include "Vector.hpp"

namespace cuvoxmap
{
    template <typename PosT, typename IdxT>
    class SuperCoverLine2D
    {
        static_assert(std::is_same<PosT, float>::value ||
                          std::is_same<PosT, double>::value,
                      "PosT must be float or double");
        static_assert(std::is_same<IdxT, int8_t>::value ||
                          std::is_same<IdxT, int16_t>::value ||
                          std::is_same<IdxT, int32_t>::value ||
                          std::is_same<IdxT, int64_t>::value,
                      "IdxT must be int8_t, int16_t, int32_t or int64_t");

    public:
        using Idx2 = Vector<IdxT, 2>;
        using Pos2 = Vector<PosT, 2>;

    private:
        Idx2 currentIdx;
        Idx2 endIdx;

        int8_t step[2];
        Idx2 tMaxMulCount;
        Pos2 tMaxOffset;
        Pos2 tDeleta;
        bool hasMore{true};

    public:
        __host__ __device__ SuperCoverLine2D() = default;
        __host__ __device__ SuperCoverLine2D(const Pos2 &startPos, const Pos2 &endPos, PosT resolution)
        {
            Pos2 delta = endPos - startPos;

            step[0] = delta[0] > 0 ? 1 : -1;
            step[1] = delta[1] > 0 ? 1 : -1;
            currentIdx = Idx2({static_cast<IdxT>(floor(startPos[0] / resolution)),
                               static_cast<IdxT>(floor(startPos[1] / resolution))});
            endIdx = Idx2({static_cast<IdxT>(floor(endPos[0] / resolution)),
                           static_cast<IdxT>(floor(endPos[1] / resolution))});
            Pos2 offset({startPos[0] - static_cast<PosT>(currentIdx[0]) * resolution,
                         startPos[1] - static_cast<PosT>(currentIdx[1]) * resolution});
            if (offset[0] < 0)
                offset[0] += resolution;
            if (offset[1] < 0)
                offset[1] += resolution;

            tMaxOffset = Pos2({delta[0] != 0 ? (step[0] > 0 ? (resolution - offset[0]) / delta[0] : offset[0] / -delta[0]) : static_cast<PosT>(1e+20f),
                               delta[1] != 0 ? (step[1] > 0 ? (resolution - offset[1]) / delta[1] : offset[1] / -delta[1]) : static_cast<PosT>(1e+20f)});
            tDeleta = Pos2({delta[0] != 0 ? resolution / fabs(delta[0]) : static_cast<PosT>(1e+20f),
                            delta[1] != 0 ? resolution / fabs(delta[1]) : static_cast<PosT>(1e+20f)});

            tMaxMulCount = Idx2::Zeros();
        }

        __host__ __device__ bool get_next_idx(Idx2 &idx)
        {
            if (!hasMore)
                return false;

            idx = currentIdx;
            bool endConditionX = (step[0] > 0) ? (currentIdx[0] > endIdx[0]) : (currentIdx[0] < endIdx[0]);
            bool endConditionY = (step[1] > 0) ? (currentIdx[1] > endIdx[1]) : (currentIdx[1] < endIdx[1]);
            if (endConditionX || endConditionY)
            {
                hasMore = false;
                return true;
            }

            // Determine which tMax is minimum
            Pos2 tMax;
            tMax[0] = tMaxOffset[0] + tDeleta[0] * static_cast<PosT>(tMaxMulCount[0]);
            tMax[1] = tMaxOffset[1] + tDeleta[1] * static_cast<PosT>(tMaxMulCount[1]);
            if (tMax[0] < tMax[1])
            {
                currentIdx[0] += step[0];
                tMaxMulCount[0]++;
            }
            else
            {
                currentIdx[1] += step[1];
                tMaxMulCount[1]++;
            }

            return true;
        }
    };
}