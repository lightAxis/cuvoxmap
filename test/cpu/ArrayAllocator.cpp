#include <catch2/catch_test_macros.hpp>
#include <cuvoxmap/base/ArrayAllocator.hpp>

TEST_CASE("ArrayAllocator cpu")
{
    cuvoxmap::ArrayAllocator<int> alloc{};
    alloc.resize(100);

    cuvoxmap::ArrayAllocator<float> alloc2{};
    alloc2.resize(100);

    SECTION("int test")
    {
        auto host_ptr = alloc.get_host_ptr();
        auto device_ptr = alloc.get_device_ptr();

        REQUIRE(host_ptr != nullptr);
        REQUIRE(device_ptr == nullptr);

        alloc.host_to_device();
        alloc.device_to_host();

        host_ptr[0] = 100;

        REQUIRE(host_ptr[0] == 100);
    }

    SECTION("float test")
    {
        auto host_ptr2 = alloc2.get_host_ptr();
        auto device_ptr2 = alloc2.get_device_ptr();

        REQUIRE(host_ptr2 != nullptr);
        REQUIRE(device_ptr2 == nullptr);

        alloc2.host_to_device();
        alloc2.device_to_host();

        host_ptr2[0] = 1.1f;

        REQUIRE(host_ptr2[0] == 1.1f);
    }
}