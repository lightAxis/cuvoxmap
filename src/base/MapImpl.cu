#include <cuvoxmap/base/MapImpl.hpp>

namespace cuvoxmap
{

    template <typename T, uint8_t Dim>
    MapImpl<T, Dim>::MapImpl(const Vector<uint32_t, Dim> &axis_sizes, eMemAllocType alloc_type)
        : axis_sizes_(axis_sizes), alloc_type_(alloc_type)
    {
        array_.resize(axis_sizes_.mul_sum(), alloc_type_);
    }

    template <typename T, uint8_t Dim>
    void MapImpl<T, Dim>::host_to_device()
    {
        array_.host_to_device();
    }

    template <typename T, uint8_t Dim>
    void MapImpl<T, Dim>::device_to_host()
    {
        array_.device_to_host();
    }

    template class MapImpl<uint8_t, 2>;
    template class MapImpl<uint8_t, 3>;
    template class MapImpl<uint16_t, 2>;
    template class MapImpl<uint16_t, 3>;
    template class MapImpl<float, 2>;
    template class MapImpl<float, 3>;
};