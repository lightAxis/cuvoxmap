#pragma once

#include "MapData.hpp"
#include "../utils/ArrayIndexing.hpp"

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

        inline T get_value(const Vector<uint32_t, Dim> &idx) const { return map_data_.host_data[indexing_.merge(idx)]; }
        inline void set_value(Vector<uint32_t, Dim> idx, T value) { map_data_.host_data[indexing_.merge(idx)] = value; }
        inline uint32_t merge_idx(const Vector<uint32_t, Dim> &idx) const { return indexing_.merge(idx); }
        inline MapData<T, Dim> get_map_data() const { return map_data_; }

    private:
        MapData<T, Dim> map_data_;
        Indexing<Dim> indexing_;
    };
}