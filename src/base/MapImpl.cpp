#include <cuvoxmap/base/MapImpl.hpp>
#include <stdexcept>

namespace cuvoxmap
{

    template <typename T, uint8_t Dim>
    MapImpl<T, Dim>::MapImpl(const Vector<uint32_t, Dim> &axis_sizes) : axis_sizes_(axis_sizes)
    {
        array_.resize(axis_sizes_.mul_sum());
    }

    template <typename T, uint8_t Dim>
    void MapImpl<T, Dim>::host_to_device()
    {
        throw std::runtime_error("Need to build with CUDA support");
    }

    template <typename T, uint8_t Dim>
    void MapImpl<T, Dim>::device_to_host()
    {
        throw std::runtime_error("Need to build with CUDA support");
    }

    template class MapImpl<uint8_t, 2>;
    template class MapImpl<uint8_t, 3>;
    template class MapImpl<uint16_t, 2>;
    template class MapImpl<uint16_t, 3>;
    template class MapImpl<float, 2>;
    template class MapImpl<float, 3>;
    template class MapImpl<double, 2>;
    template class MapImpl<double, 3>;
};