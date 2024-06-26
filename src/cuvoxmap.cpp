#include <cuvoxmap/cuvoxmap.hpp>

namespace cuvoxmap
{
    cuvoxmap2D::cuvoxmap2D(const init_s &init) : pb_map_alloc_(uIdx2D{init.x_axis_len, init.y_axis_len}),
                                                 st_map_alloc_(uIdx2D{init.x_axis_len, init.y_axis_len}),
                                                 dst_map_alloc_(uIdx2D{init.x_axis_len, init.y_axis_len}),
                                                 pb_map_accessor_(pb_map_alloc_.get_mapData()),
                                                 st_map_accessor_(st_map_alloc_.get_mapData()),
                                                 dst_map_accessor_(dst_map_alloc_.get_mapData())
    {
        param_.x_axis_len = init.x_axis_len;
        param_.y_axis_len = init.y_axis_len;
        param_.resolution = init.resolution;
    }

    // void cuvoxmap2D::set_pb_map(const Idx2D &idx, float value)
    // {
    //     pb_map_accessor_.set_value(idx.cast<uint32_t>(), value);
    // }

    // void cuvoxmap2D::set_st_map(const Idx2D &idx, VoxelType value)
    // {
    // }
    // void cuvoxmap2D::set_dst_map(const Idx2D &idx, float value)
    // {
    // }

    // float cuvoxmap2D::get_pb_map(const Idx2D &Idx) const
    // {
    // }
    // cuvoxmap2D::VoxelType cuvoxmap2D::get_st_map(const Idx2D &Idx) const
    // {
    // }
    // float cuvoxmap2D::get_dst_map(const Idx2D &Idx) const
    // {
    // }
}