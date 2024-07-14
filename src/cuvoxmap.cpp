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
        param_.axis_sizes = uIdx2D{init.x_axis_len, init.y_axis_len};
        param_.resolution = init.resolution;

        glc_ = GlobLocalCvt<float, 2>(Float2D::Zeros(),
                                      param_.resolution,
                                      param_.axis_sizes);

        idx2d_ = Indexing<2>{param_.axis_sizes};
        box_ = Box<float, 2>{Float2D::Zeros(), Float2D{param_.axis_sizes[0] * param_.resolution, param_.axis_sizes[1] * param_.resolution}};
    }

}