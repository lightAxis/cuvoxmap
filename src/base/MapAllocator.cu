#include <cuvoxmap/base/MapAllocator.hpp>
#include <cuvoxmap/base/MapImpl.hpp>
#include <cuvoxmap/base/ArrayAllocator.hpp>

namespace cuvoxmap
{
    template <typename T, uint8_t Dim>
    MapAllocator<T, Dim>::MapAllocator(const Vector<uint32_t, Dim> &axis_sizes)
    {
        impl_ = new MapImpl<T, Dim>(axis_sizes);
    }

    template <typename T, uint8_t Dim>
    MapAllocator<T, Dim>::~MapAllocator()
    {
        if (impl_ != nullptr)
            delete impl_;
    }

    template <typename T, uint8_t Dim>
    MapData<T, Dim> MapAllocator<T, Dim>::get_mapData()
    {
        MapData<T, Dim> mapData;
        mapData.axis_sizes = impl_->get_axis_sizes();
        mapData.host_data = impl_->get_host_data();
        mapData.device_data = impl_->get_device_data();
        mapData.is_gpu_used = true;
        return mapData;
    }

    template <typename T, uint8_t Dim>
    void MapAllocator<T, Dim>::host_to_device()
    {
        impl_->host_to_device();
    }
    template <typename T, uint8_t Dim>
    void MapAllocator<T, Dim>::device_to_host()
    {
        impl_->device_to_host();
    }
    template <typename T, uint8_t Dim>
    void MapAllocator<T, Dim>::fill(T value)
    {
        impl_->fill(value);
    }

    template class MapAllocator<uint8_t, 2>;
    template class MapAllocator<uint8_t, 3>;
    template class MapAllocator<uint16_t, 2>;
    template class MapAllocator<uint16_t, 3>;
    template class MapAllocator<float, 2>;
    template class MapAllocator<float, 3>;
}