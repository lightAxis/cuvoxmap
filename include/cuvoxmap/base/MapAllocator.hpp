#pragma once

#include "../utils/ArrayIndexing.hpp"
#include "MapData.hpp"
#include "MemAllocType.hpp"
#include "../utils/MovablePtr.hpp"

namespace cuvoxmap
{
    template <typename T, uint8_t Dim>
    class MapImpl;

    template <typename T, uint8_t Dim>
    class MapAllocator
    {
    public:
        MapAllocator() = default;
        MapAllocator(const Vector<uint32_t, Dim> &axis_sizes, eMemAllocType alloc_type);
        MapAllocator(const MapAllocator &other) = default;
        MapAllocator(MapAllocator &&other) noexcept = default;
        MapAllocator &operator=(const MapAllocator &other) = default;
        MapAllocator &operator=(MapAllocator &&other) noexcept = default;
        ~MapAllocator() = default;

        MapData<T, Dim> get_mapData();

        void host_to_device();
        void device_to_host();
        void fill_host(T value);
        void fill_device(T value);

    private:
        MovablePtr<MapImpl<T, Dim>> impl_;
        inline bool is_impl_exist() const { return impl_ != nullptr; }
    };
}