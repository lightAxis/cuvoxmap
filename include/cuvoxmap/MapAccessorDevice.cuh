#pragma once

#include "base/MapData.hpp"
#include "utils/ArrayIndexing.hpp"

#include <stdexcept>

namespace cuvoxmap
{
    template <typename T, uint8_t Dim>
    class MapAccesssorDevice
    {
    public:
        MapAccesssorDevice() = default;
        explicit MapAccesssorDevice(const MapData<T, Dim> &mapData)
            : map_data_(mapData), indexing_(mapData.axis_sizes)
        {
            if (map_data_.is_gpu_used == false)
                throw std::runtime_error("Cannot use MapAccessorDevice with MapData that is not on the device. Use MapAccessorHost instead.");
        }
        ~MapAccesssorDevice() = default;

        __device__ inline T get_value(const Vector<uint32_t, Dim> &idx) const { return map_data_.device_data[indexing_.merge_device(idx)]; }
        __device__ inline void set_value(T value, const Vector<uint32_t, Dim> &idx) { map_data_.device_data[indexing_.merge_device(idx)] = value; }
        __device__ inline uint32_t merge_idx(const Vector<uint32_t, Dim> &idx) const { return indexing_.merge_device(idx); }
        __device__ inline MapData<T, Dim> get_map_data() const { return map_data_; }
        __device__ inline Vector<uint32_t, Dim> get_axis_size() const { return map_data_.axis_sizes; }

    private:
        MapData<T, Dim> map_data_;
        Indexing<Dim> indexing_;
    };
}