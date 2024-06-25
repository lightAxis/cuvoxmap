#include <cuvoxmap/base/MapAllocator.hpp>
#include <cuvoxmap/base/MapAccessorDevice.cuh>
#include <catch2/catch_test_macros.hpp>
#include "device_testmacros.cuh"

__global__ void cpu_basic_kernel(cuvoxmap::MapAccesssorDevice<float, 2> map)
{
    map.set_value(1, cuvoxmap::uIdx2D{1, 2});
    map.set_value(2, cuvoxmap::uIdx2D{2, 3});

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
}

TEST_CASE("MapAccessorDevice gpu")
{
    cuvoxmap::MapAllocator<float, 2> alloc{cuvoxmap::uIdx2D{10, 20}};
    cuvoxmap::MapAccesssorDevice<float, 2> accessor{alloc.get_mapData()};

    SECTION("cpu basic")
    {
        cpu_basic_kernel<<<1, 1>>>(accessor);
        cudaDeviceSynchronize();
    }
}