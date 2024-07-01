#pragma once

#include "Vector.hpp"
#include <type_traits>

namespace cuvoxmap
{
    template <typename PosT, typename IdxT>
    class SuperCoverLine3D
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
        using Idx3 = Vector<IdxT, 3>;
        using Pos3 = Vector<PosT, 3>;

    private:
        Idx3 currentIdx;
        Idx3 endIdx;

        int8_t step[3];
        Idx3 tMaxMulCount;
        Pos3 tMaxOffset;
        Pos3 tDeleta;
        bool hasMore{true};

    public:
        SuperCoverLine3D() = default;
        ~SuperCoverLine3D() = default;
        __host__ __device__ SuperCoverLine3D(const Pos3 &startPos, const Pos3 &endPos, PosT resolution)
        {
            Pos3 delta = endPos - startPos;

            step[0] = delta[0] > 0 ? 1 : -1;
            step[1] = delta[1] > 0 ? 1 : -1;
            step[2] = delta[2] > 0 ? 1 : -1;
            currentIdx = Idx3({static_cast<IdxT>(floor(startPos[0] / resolution)),
                               static_cast<IdxT>(floor(startPos[1] / resolution)),
                               static_cast<IdxT>(floor(startPos[2] / resolution))});
            endIdx = Idx3({static_cast<IdxT>(floor(endPos[0] / resolution)),
                           static_cast<IdxT>(floor(endPos[1] / resolution)),
                           static_cast<IdxT>(floor(endPos[2] / resolution))});
            Pos3 offset({startPos[0] - static_cast<PosT>(currentIdx[0]) * resolution,
                         startPos[1] - static_cast<PosT>(currentIdx[1]) * resolution,
                         startPos[2] - static_cast<PosT>(currentIdx[2]) * resolution});
            if (offset[0] < 0)
                offset[0] += resolution;
            if (offset[1] < 0)
                offset[1] += resolution;
            if (offset[2] < 0)
                offset[2] += resolution;

            tMaxOffset = Pos3({delta[0] != 0 ? (step[0] > 0 ? (resolution - offset[0]) / delta[0] : offset[0] / -delta[0]) : static_cast<PosT>(1e+20f),
                               delta[1] != 0 ? (step[1] > 0 ? (resolution - offset[1]) / delta[1] : offset[1] / -delta[1]) : static_cast<PosT>(1e+20f),
                               delta[2] != 0 ? (step[2] > 0 ? (resolution - offset[2]) / delta[2] : offset[2] / -delta[2]) : static_cast<PosT>(1e+20f)});
            tDeleta = Pos3({delta[0] != 0 ? resolution / std::abs(delta[0]) : static_cast<PosT>(1e+20f),
                            delta[1] != 0 ? resolution / std::abs(delta[1]) : static_cast<PosT>(1e+20f),
                            delta[2] != 0 ? resolution / std::abs(delta[2]) : static_cast<PosT>(1e+20f)});

            tMaxMulCount = Idx3::Zeros();
        }

        __host__ __device__ bool get_next_idx(Idx3 &idx)
        {
            if (!hasMore)
                return false;

            idx = currentIdx;
            bool endConditionX = (step[0] > 0) ? (currentIdx[0] > endIdx[0]) : (currentIdx[0] < endIdx[0]);
            bool endConditionY = (step[1] > 0) ? (currentIdx[1] > endIdx[1]) : (currentIdx[1] < endIdx[1]);
            bool endConditionZ = (step[2] > 0) ? (currentIdx[2] > endIdx[2]) : (currentIdx[2] < endIdx[2]);
            if (endConditionX || endConditionY || endConditionZ)
            {
                hasMore = false;
                return true;
            }

            // Determine which tMax is minimum
            Pos3 tMax;
            tMax[0] = tMaxOffset[0] + tDeleta[0] * static_cast<PosT>(tMaxMulCount[0]);
            tMax[1] = tMaxOffset[1] + tDeleta[1] * static_cast<PosT>(tMaxMulCount[1]);
            tMax[2] = tMaxOffset[2] + tDeleta[2] * static_cast<PosT>(tMaxMulCount[2]);
            if (tMax[0] < tMax[1] && tMax[0] < tMax[2])
            {
                currentIdx[0] += step[0];
                tMaxMulCount[0]++;
            }
            else if (tMax[1] < tMax[2])
            {
                currentIdx[1] += step[1];
                tMaxMulCount[1]++;
            }
            else
            {
                currentIdx[2] += step[2];
                tMaxMulCount[2]++;
            }

            return true;
        }
    };
}