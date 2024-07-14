#pragma once

#include "../utils/ArrayIndexing.hpp"
#include "MapData.hpp"

namespace cuvoxmap
{
    template <typename T, uint8_t Dim>
    class MapImpl;

    template <typename T, uint8_t Dim>
    class MapAllocator
    {
    public:
        MapAllocator() = default;
        MapAllocator(const Vector<uint32_t, Dim> &axis_sizes);
        ~MapAllocator();

        MapData<T, Dim> get_mapData();

        void host_to_device();
        void device_to_host();
        void fill_host(T value);
        void fill_device(T value);

    private:
        MapImpl<T, Dim> *impl_{nullptr};
    };
}