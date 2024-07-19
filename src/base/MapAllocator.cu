#include <cuvoxmap/base/MapAllocator.hpp>
#include <cuvoxmap/base/MapImpl.hpp>
#include <cuvoxmap/base/ArrayAllocator.hpp>

namespace cuvoxmap
{
    template <typename T, uint8_t Dim>
    MapAllocator<T, Dim>::MapAllocator(const Vector<uint32_t, Dim> &axis_sizes, eMemAllocType alloc_type)
    {
        impl_ = new MapImpl<T, Dim>(axis_sizes, alloc_type);
    }

    template <typename T, uint8_t Dim>
    MapAllocator<T, Dim>::MapAllocator(const MapAllocator<T, Dim> &other)
    {
        impl_ = new MapImpl<T, Dim>(*other.impl_);
    }

    template <typename T, uint8_t Dim>
    MapAllocator<T, Dim> &MapAllocator<T, Dim>::operator=(const MapAllocator<T, Dim> &other)
    {
        if (this != &other)
        {
            if (impl_ != nullptr)
                delete impl_;
            impl_ = new MapImpl<T, Dim>(*other.impl_);
        }
        return *this;
    }

    template <typename T, uint8_t Dim>
    MapAllocator<T, Dim>::MapAllocator(MapAllocator<T, Dim> &&other) noexcept
    {
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }

    template <typename T, uint8_t Dim>
    MapAllocator<T, Dim> &MapAllocator<T, Dim>::operator=(MapAllocator<T, Dim> &&other) noexcept
    {
        if (this != &other)
        {
            if (impl_ != nullptr)
                delete impl_;
            impl_ = other.impl_;
            other.impl_ = nullptr;
        }
        return *this;
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
        if (!is_impl_exist())
            throw std::runtime_error("MapAllocator not initialized");

        MapData<T, Dim> mapData;
        mapData.axis_sizes = impl_->get_axis_sizes();
        mapData.host_data = impl_->get_host_data();
        mapData.device_data = impl_->get_device_data();
        mapData.is_gpu_used = true;
        mapData.is_host_data_allocated = static_cast<uint8_t>(impl_->get_alloc_type()) & static_cast<uint8_t>(eMemAllocType::HOST);
        mapData.is_device_data_allocated = static_cast<uint8_t>(impl_->get_alloc_type()) & static_cast<uint8_t>(eMemAllocType::DEVICE);
        return mapData;
    }

    template <typename T, uint8_t Dim>
    void MapAllocator<T, Dim>::host_to_device()
    {
        if (!is_impl_exist())
            throw std::runtime_error("MapAllocator not initialized");
        impl_->host_to_device();
    }
    template <typename T, uint8_t Dim>
    void MapAllocator<T, Dim>::device_to_host()
    {
        if (!is_impl_exist())
            throw std::runtime_error("MapAllocator not initialized");
        impl_->device_to_host();
    }
    template <typename T, uint8_t Dim>
    void MapAllocator<T, Dim>::fill_host(T value)
    {
        if (!is_impl_exist())
            throw std::runtime_error("MapAllocator not initialized");
        impl_->fill_host(value);
    }

    template <typename T, uint8_t Dim>
    void MapAllocator<T, Dim>::fill_device(T value)
    {
        if (!is_impl_exist())
            throw std::runtime_error("MapAllocator not initialized");
        impl_->fill_device(value);
    }

    template class MapAllocator<uint8_t, 1>;
    template class MapAllocator<uint8_t, 2>;
    template class MapAllocator<uint8_t, 3>;
    template class MapAllocator<uint16_t, 1>;
    template class MapAllocator<uint16_t, 2>;
    template class MapAllocator<uint16_t, 3>;
    template class MapAllocator<int, 1>;
    template class MapAllocator<int, 2>;
    template class MapAllocator<int, 3>;
    template class MapAllocator<float, 1>;
    template class MapAllocator<float, 2>;
    template class MapAllocator<float, 3>;
    template class MapAllocator<double, 1>;
    template class MapAllocator<double, 2>;
    template class MapAllocator<double, 3>;
}