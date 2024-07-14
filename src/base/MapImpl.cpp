#include <cuvoxmap/base/MapImpl.hpp>
#include <stdexcept>

namespace cuvoxmap
{

    template <typename T, uint8_t Dim>
    MapImpl<T, Dim>::MapImpl(const Vector<uint32_t, Dim> &axis_sizes,
                             eMemAllocType alloc_type) : axis_sizes_(axis_sizes), alloc_type_(alloc_type)
    {
        if (static_cast<uint8_t>(alloc_type_) & static_cast<uint8_t>(eMemAllocType::DEVICE))
            throw std::runtime_error("Cannot allocated device memory. Need to build with CUDA support");

        array_.resize(axis_sizes_.mul_sum(), alloc_type_);
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

    template class MapImpl<uint8_t, 1>;
    template class MapImpl<uint8_t, 2>;
    template class MapImpl<uint8_t, 3>;
    template class MapImpl<uint16_t, 1>;
    template class MapImpl<uint16_t, 2>;
    template class MapImpl<uint16_t, 3>;
    template class MapImpl<int, 1>;
    template class MapImpl<int, 2>;
    template class MapImpl<int, 3>;
    template class MapImpl<float, 1>;
    template class MapImpl<float, 2>;
    template class MapImpl<float, 3>;
    template class MapImpl<double, 1>;
    template class MapImpl<double, 2>;
    template class MapImpl<double, 3>;
};