#include <cuvoxmap/cuvoxmap.hpp>

namespace cuvoxmap
{
    uint32_t get_z_buffer_count(const uIdx2D &axis_sizes)
    {
        uint32_t max_axis = std::max(axis_sizes[0], axis_sizes[1]);
        return axis_sizes.mul_sum() + max_axis;
    }

    uint32_t get_v_buffer_count(const uIdx2D &axis_sizes)
    {
        return axis_sizes.mul_sum();
    }

    cuvoxmap2D::cuvoxmap2D(const init_s &init) : pb_map_alloc_(uIdx2D{init.x_axis_len, init.y_axis_len}, init.use_gpu ? eMemAllocType::HOST_AND_DEVICE : eMemAllocType::HOST),
                                                 st_map_alloc_(uIdx2D{init.x_axis_len, init.y_axis_len}, init.use_gpu ? eMemAllocType::HOST_AND_DEVICE : eMemAllocType::HOST),
                                                 dst_map_alloc_(uIdx2D{init.x_axis_len, init.y_axis_len}, init.use_gpu ? eMemAllocType::HOST_AND_DEVICE : eMemAllocType::HOST),
                                                 temp1_dstmap_alloc_(uIdx2D{init.x_axis_len, init.y_axis_len}, init.use_gpu ? eMemAllocType::HOST_AND_DEVICE : eMemAllocType::HOST),
                                                 v_buffer_alloc_(uIdx1D{get_v_buffer_count(uIdx2D{init.x_axis_len, init.y_axis_len})},
                                                                 init.use_gpu ? eMemAllocType::HOST_AND_DEVICE : eMemAllocType::HOST),
                                                 z_buffer_alloc_(uIdx1D{get_z_buffer_count(uIdx2D{init.x_axis_len, init.y_axis_len})},
                                                                 init.use_gpu ? eMemAllocType::HOST_AND_DEVICE : eMemAllocType::HOST),
                                                 pb_map_accessor_(pb_map_alloc_.get_mapData()),
                                                 st_map_accessor_(st_map_alloc_.get_mapData()),
                                                 dst_map_accessor_(dst_map_alloc_.get_mapData())
    {
        param_.axis_sizes = uIdx2D{init.x_axis_len, init.y_axis_len};
        param_.resolution = init.resolution;
        param_.use_gpu = init.use_gpu;

        glc_ = GlobLocalCvt<float, 2>(Float2D::Zeros(),
                                      param_.resolution,
                                      param_.axis_sizes);

        idx2d_ = Indexing<2>{param_.axis_sizes};
        box_ = Box<float, 2>{Float2D::Zeros(), Float2D{param_.axis_sizes[0] * param_.resolution, param_.axis_sizes[1] * param_.resolution}};
    }

}