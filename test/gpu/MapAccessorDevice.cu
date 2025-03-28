#include <cuvoxmap/base/MapAllocator.hpp>
#include <cuvoxmap/base/MapAccessorDevice.cuh>
#include <catch2/catch_test_macros.hpp>
#include "../custom_matchers/device_testmacros.cuh"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

__global__ void cpu_basic_kernel(cuvoxmap::MapAccessorDevice<float, 2> map, bool *res)
{
    map.set_value(cuvoxmap::uIdx2D{1, 2}, 1);
    map.set_value(cuvoxmap::uIdx2D{2, 3}, 2);

    *res = false;

    if (!CUSTOM_TEST_KERNEL::KERNEL_TEST(map.get_value(cuvoxmap::uIdx2D{1, 2}), 1.0f))
        return;
    if (!CUSTOM_TEST_KERNEL::KERNEL_TEST(map.get_value(cuvoxmap::uIdx2D{2, 3}), 2.0f))
        return;

    if (!CUSTOM_TEST_KERNEL::KERNEL_TEST(map.get_axis_size()[0], 10u))
        return;
    if (!CUSTOM_TEST_KERNEL::KERNEL_TEST(map.get_axis_size()[1], 20u))
        return;
    if (!CUSTOM_TEST_KERNEL::KERNEL_TEST(map.merge_idx(cuvoxmap::uIdx2D{1, 2}), 1 * 1u + 2 * 10u))
        return;

    cuvoxmap::MapData<float, 2> data = map.get_map_data();
    if (!CUSTOM_TEST_KERNEL::KERNEL_TEST(data.is_gpu_used, true))
        return;
    if (!CUSTOM_TEST_KERNEL::KERNEL_TEST(data.host_data != nullptr, true))
        return;
    if (!CUSTOM_TEST_KERNEL::KERNEL_TEST(data.device_data != nullptr, true))
        return;
    if (!CUSTOM_TEST_KERNEL::KERNEL_TEST(data.axis_sizes[0], 10u))
        return;
    if (!CUSTOM_TEST_KERNEL::KERNEL_TEST(data.axis_sizes[1], 20u))
        return;

    *res = true;
}

TEST_CASE("MapAccessorDevice gpu")
{
    cuvoxmap::MapAllocator<float, 2> alloc{cuvoxmap::uIdx2D{10, 20}, cuvoxmap::eMemAllocType::HOST_AND_DEVICE};
    cuvoxmap::MapAccessorDevice<float, 2> accessor{alloc.get_mapData()};

    thrust::host_vector<bool> host_res(1, false);
    thrust::device_vector<bool> dev_res(1, false);

    SECTION("cpu basic")
    {
        cpu_basic_kernel<<<1, 1>>>(accessor, thrust::raw_pointer_cast(dev_res.data()));
        cudaDeviceSynchronize();
        host_res = dev_res;
        if (host_res[0] == false)
        {
            FAIL("test failed inside CUDA kernel");
        }
    }
}