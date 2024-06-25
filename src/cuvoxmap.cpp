#include <cuvoxmap/cuvoxmap.hpp>

namespace cuvoxmap
{
    cuvoxmap2D::cuvoxmap2D(const init_s &init)
    {
        param_.x_axis_len = init.x_axis_len;
        param_.y_axis_len = init.y_axis_len;
        param_.resolution = init.resolution;

        uIdx2D dims{init.x_axis_len, init.y_axis_len};

        p_map_alloc_ = MapAllocator<float, 2>(dims);
        o_map_alloc_ = MapAllocator<uint8_t, 2>(dims);
        d_map_alloc_ = MapAllocator<float, 2>(dims);

        p_map_accessor_ = MapAccesssorHost<float, 2>(p_map_alloc_.get_mapData());
        o_map_accessor_ = MapAccesssorHost<uint8_t, 2>(o_map_alloc_.get_mapData());
        d_map_accessor_ = MapAccesssorHost<float, 2>(d_map_alloc_.get_mapData());
    }
}