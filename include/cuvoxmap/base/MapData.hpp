#pragma once

#include "../utils/Vector.hpp"

namespace cuvoxmap
{
    template <typename T, uint8_t Dim>
    struct MapData
    {
        T *host_data;
        T *device_data;
        Vector<uint32_t, Dim> axis_sizes;
        bool is_gpu_used;
        bool is_host_data_allocated;
        bool is_device_data_allocated;
    };
}