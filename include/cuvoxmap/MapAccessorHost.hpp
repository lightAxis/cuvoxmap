#pragma once

#include "base/MapData.hpp"
#include "utils/ArrayIndexing.hpp"

namespace cuvoxmap
{
    template <typename T, uint8_t Dim>
    class MapAccesssorHost
    {
    public:
        MapAccesssorHost() = default;
        explicit MapAccesssorHost(const MapData<T, Dim> &mapData)
            : map_data_(mapData), indexing_(mapData.axis_sizes) {}
        ~MapAccesssorHost() = default;

        inline T get_value(const std::array<uint32_t, Dim> &idx) const { return map_data_.host_data[indexing_.merge(idx)]; }
        inline T get_value(const Vector<uint32_t, Dim> &idx) const { return map_data_.host_data[indexing_.merge_device(idx)]; }
        inline void set_value(const std::array<uint32_t, Dim> &idx) { return map_data_.host_data[indexing_.merge(idx)]; }
        inline void set_value(T value, Vector<uint32_t, Dim> idx) { map_data_.host_data[indexing_.merge_device(idx)] = value; }
        inline uint32_t merge_idx(const std::array<uint32_t, Dim> &idx) const { return indexing_.merge(idx); }
        inline uint32_t merge_idx(const Vector<uint32_t, Dim> &idx) const { return indexing_.merge_device(idx); }
        inline MapData<T, Dim> get_map_data() const { return map_data_; }
        inline Vector<uint32_t, Dim> get_axis_size() const { return map_data_.axis_sizes; }

    private:
        MapData<T, Dim> map_data_;
        Indexing<Dim> indexing_;
    };
}