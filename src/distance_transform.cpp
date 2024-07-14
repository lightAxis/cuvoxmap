#include <cuvoxmap/cuvoxmap.hpp>

namespace cuvoxmap
{
    using StMapT = MapType<eMap::STATE>::Type;
    using MapT = MapType<eMap::DISTANCE>::Type;

    struct ArrayPtr_s
    {
        MapT *z_buffer{nullptr};
        int *v_buffer{nullptr};
    };

    template <typename getterT, typename setterT>
    void fill_distance_host(getterT getter, setterT setter,
                            const int start, const int end, MapT *z, int *v);
    void fill_distance1_x(MapAccessorHost<StMapT, 2> state_map, MapAccessorHost<MapT, 2> &dist_temp1, const uint32_t total_voxels, ArrayPtr_s array_ptr);
    void fill_distance2_y(MapAccessorHost<MapT, 2> &dist_temp1, MapAccessorHost<MapT, 2> dist_map, const uint32_t total_voxels, ArrayPtr_s array_ptr);

    void cuvoxmap2D::distance_map_update_withCPU()
    {
        const int totalSize_y = param_.axis_sizes[1];
        const int totalSize_x = param_.axis_sizes[0];
        const int blockSize = 256;

        const int gridsize_y = (totalSize_y - 1) / blockSize + 1;
        const int gridsize_x = (totalSize_x - 1) / blockSize + 1;

        int *v_buffer = v_buffer_alloc_.get_mapData().host_data;
        MapT *z_buffer = z_buffer_alloc_.get_mapData().host_data;

        MapAccessorHost<StMapT, 2> statemap{st_map_alloc_.get_mapData()};
        MapAccessorHost<MapT, 2> temp1_distmap{temp1_dstmap_alloc_.get_mapData()};
        MapAccessorHost<MapT, 2> distmap{dst_map_alloc_.get_mapData()};

        // x axis, state map -> temp1 distmap
        for (uint32_t y = 0; y < param_.axis_sizes[1]; y++)
        {
            auto getter = [&](uint32_t x_idx)
            { return statemap.get_value(uIdx2D{x_idx, y}) & static_cast<uint8_t>(eVoxel::OCCUPIED) ? 0 : std::numeric_limits<MapT>::max(); };
            auto setter = [&](uint32_t x_idx, MapT value)
            { temp1_distmap.set_value(uIdx2D{x_idx, y}, value); };

            fill_distance_host(getter, setter, 0, param_.axis_sizes[0] - 1, z_buffer, v_buffer);
        }

        // y axis, temp1 distmap -> distmap
        for (uint32_t x = 0; x < param_.axis_sizes[0]; x++)
        {
            auto getter = [&](uint32_t y_idx)
            { return temp1_distmap.get_value(uIdx2D{x, y_idx}); };
            auto setter = [&](uint32_t y_idx, MapT value)
            { distmap.set_value(uIdx2D{x, y_idx}, sqrt(value) * param_.resolution); };

            fill_distance_host(getter, setter, 0, param_.axis_sizes[1] - 1, z_buffer, v_buffer);
        }
    }

    template <typename getterT, typename setterT>
    __device__ void fill_distance_host(getterT getter, setterT setter,
                                       const int start, const int end, MapT *z, int *v)
    {
        int k = start;
        v[start] = start;
        z[start] = -__FLT_MAX__;
        z[start + 1] = -__FLT_MAX__;

        for (int q = start + 1; q <= end; q++)
        {
            k++;
            float s;

            do
            {
                k--;
                s = ((getter(q) + q * q) - (getter(v[k]) + v[k] * v[k])) / (2 * q - 2 * v[k]);
            } while (s <= z[k]);

            k++;

            v[k] = q;
            z[k] = s;
            z[k + 1] = __FLT_MAX__;
        }

        k = start;

        float val;
        for (int q = start; q <= end; q++)
        {
            while (z[k + 1] < q)
                k++;
            val = (q - v[k]) * (q - v[k]) + getter(v[k]);
            setter(q, val);
        }
    }
}
