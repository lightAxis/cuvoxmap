#include <cuvoxmap/cuvoxmap.hpp>
#include <cuvoxmap/base/MapAccessorDevice.cuh>

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
    __device__ void fill_distance_device(getterT getter, setterT setter,
                                         const int start, const int end, MapT *z, int *v);
    __global__ void fill_distance1_x(MapAccessorDevice<StMapT, 2> state_map, MapAccessorDevice<MapT, 2> dist_temp1, const uint32_t total_voxels, ArrayPtr_s array_ptr);
    __global__ void fill_distance2_y(MapAccessorDevice<MapT, 2> dist_temp1, MapAccessorDevice<MapT, 2> dist_map, const uint32_t total_voxels, ArrayPtr_s array_ptr, MapT res);

    void cuvoxmap2D::distance_map_update_withGPU()
    {
        if (param_.use_gpu == false)
        {
            throw std::runtime_error("GPU is not enabled");
        }

        const int totalSize_x = param_.axis_sizes[0];
        const int totalSize_y = param_.axis_sizes[1];
        const int blockSize = 256;

        const int gridsize_x = (totalSize_x - 1) / blockSize + 1;
        const int gridsize_y = (totalSize_y - 1) / blockSize + 1;

        ArrayPtr_s array_ptr;
        array_ptr.v_buffer = v_buffer_alloc_.get_mapData().device_data;
        array_ptr.z_buffer = z_buffer_alloc_.get_mapData().device_data;

        MapAccessorDevice<StMapT, 2> statemap{st_map_alloc_.get_mapData()};
        MapAccessorDevice<MapT, 2> temp1_distmap{temp1_dstmap_alloc_.get_mapData()};
        MapAccessorDevice<MapT, 2> distmap{dst_map_alloc_.get_mapData()};

        fill_distance1_x<<<gridsize_x, blockSize>>>(statemap, temp1_distmap, totalSize_x, array_ptr);
        fill_distance2_y<<<gridsize_y, blockSize>>>(temp1_distmap, distmap, totalSize_y, array_ptr, param_.resolution);

        cudaDeviceSynchronize();
    }

    template <typename getterT, typename setterT>
    __device__ void fill_distance_device(getterT getter, setterT setter,
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

    __global__ void fill_distance1_x(MapAccessorDevice<StMapT, 2> state_map, MapAccessorDevice<MapT, 2> dist_temp1, const uint32_t total_voxels, ArrayPtr_s array_ptr)
    {
        const uint32_t totalIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (totalIdx >= total_voxels)
            return;

        uIdx2D map_size = state_map.get_map_data().axis_sizes;
        // const uint32_t tid_y = total_voxels / map_size[0];
        const uint32_t tid_y = totalIdx;

        MapT *z = &(array_ptr.z_buffer[totalIdx * (map_size[0] + 1)]);
        int *v = &(array_ptr.v_buffer[totalIdx * map_size[0]]);

        auto getter = [&](uint32_t tid_x)
        { return state_map.get_value(uIdx2D{tid_x, tid_y}) & static_cast<uint8_t>(eVoxel::OCCUPIED) ? 0 : __FLT_MAX__; };

        auto setter = [&](uint32_t tid_x, MapT value)
        { dist_temp1.set_value(uIdx2D{tid_x, tid_y}, value); };

        fill_distance_device(getter, setter, 0, map_size[0] - 1, z, v);
    }

    __global__ void fill_distance2_y(MapAccessorDevice<MapT, 2> dist_temp1, MapAccessorDevice<MapT, 2> dist_map, const uint32_t total_voxels, ArrayPtr_s array_ptr, MapT res)
    {
        const uint32_t totalIdx = blockIdx.x * blockDim.x + threadIdx.x;

        if (totalIdx >= total_voxels)
            return;

        uIdx2D map_size = dist_map.get_map_data().axis_sizes;
        const uint32_t tid_x = totalIdx;

        MapT *z = &(array_ptr.z_buffer[totalIdx * (map_size[1] + 1)]);
        int *v = &(array_ptr.v_buffer[totalIdx * map_size[1]]);

        auto getter = [&](uint32_t tid_y)
        { return dist_temp1.get_value(uIdx2D{tid_x, tid_y}); };

        auto setter = [&](uint32_t tid_y, MapT value)
        { dist_map.set_value(uIdx2D{tid_x, tid_y}, sqrt(value) * res); };

        fill_distance_device(getter, setter, 0, map_size[1] - 1, z, v);
    }

}