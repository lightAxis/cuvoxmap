#include <catch2/catch_test_macros.hpp>
#include <cuvoxmap/base/ArrayAllocator.hpp>

__global__ void test_kernel(int *ptr)
{
    ptr[0] = 100;
}

__global__ void test_kernel2(float *ptr)
{
    ptr[0] = 100.100f;
}

TEST_CASE("ArrayAllocator gpu")
{
    cuvoxmap::ArrayAllocator<int> alloc{};
    alloc.resize(100, cuvoxmap::eMemAllocType::HOST_AND_DEVICE);

    cuvoxmap::ArrayAllocator<float> alloc2{};
    alloc2.resize(100, cuvoxmap::eMemAllocType::HOST_AND_DEVICE);

    SECTION("test int basic")
    {
        auto host_ptr = alloc.get_host_ptr();
        auto device_ptr = alloc.get_device_ptr();

        REQUIRE(host_ptr != nullptr);
        REQUIRE(device_ptr != nullptr);

        alloc.host_to_device();
        alloc.device_to_host();
    }

    SECTION("test int device")
    {
        auto host_ptr = alloc.get_host_ptr();
        auto device_ptr = alloc.get_device_ptr();

        REQUIRE(host_ptr != nullptr);
        REQUIRE(device_ptr != nullptr);

        host_ptr[0] = 10;
        alloc.host_to_device();
        REQUIRE(host_ptr[0] == 10);

        test_kernel<<<1, 1>>>(device_ptr);
        cudaDeviceSynchronize();
        alloc.device_to_host();

        REQUIRE(host_ptr[0] == 100);
    }

    SECTION("test float host")
    {
        auto host_ptr2 = alloc2.get_host_ptr();
        auto device_ptr2 = alloc2.get_device_ptr();

        REQUIRE(host_ptr2 != nullptr);
        REQUIRE(device_ptr2 != nullptr);

        alloc2.host_to_device();
        alloc2.device_to_host();
    }

    SECTION("test float device")
    {
        auto host_ptr2 = alloc2.get_host_ptr();
        auto device_ptr2 = alloc2.get_device_ptr();

        REQUIRE(host_ptr2 != nullptr);
        REQUIRE(device_ptr2 != nullptr);

        host_ptr2[0] = 10;
        alloc2.host_to_device();
        REQUIRE(host_ptr2[0] == 10);

        test_kernel2<<<1, 1>>>(device_ptr2);
        cudaDeviceSynchronize();
        alloc2.device_to_host();

        REQUIRE(host_ptr2[0] == 100.100f);
    }
}