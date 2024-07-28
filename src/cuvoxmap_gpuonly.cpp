#include <cuvoxmap/cuvoxmap.hpp>

namespace cuvoxmap
{
    void cuvoxmap2D::distance_map_update_withGPU()
    {
        throw std::runtime_error("Not implemented for CPU only build");
    }
}