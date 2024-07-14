#pragma once

namespace cuvoxmap
{
    enum class eMemAllocType
    {
        HOST = 1 << 0,
        DEVICE = 1 << 1,
        HOST_AND_DEVICE = HOST | DEVICE,
        NONE = 0,
    };
}