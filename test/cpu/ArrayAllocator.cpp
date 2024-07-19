#include <catch2/catch_test_macros.hpp>
#include <cuvoxmap/base/ArrayAllocator.hpp>

TEST_CASE("ArrayAllocator cpu")
{
    cuvoxmap::ArrayAllocator<int> alloc{};
    alloc.resize(100, cuvoxmap::eMemAllocType::HOST);

    cuvoxmap::ArrayAllocator<float> alloc2{};
    alloc2.resize(100, cuvoxmap::eMemAllocType::HOST);

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

    SECTION("copy test")
    {
        // copy constructor
        cuvoxmap::ArrayAllocator<float> alloc3 = alloc2;
        auto host_ptr3 = alloc3.get_host_ptr();
        auto host_ptr2 = alloc2.get_host_ptr();

        REQUIRE(host_ptr2[0] == host_ptr2[0]);
        REQUIRE(host_ptr2 != host_ptr3);

        // copy assignment operator
        cuvoxmap::ArrayAllocator<float> alloc5;
        alloc5 = alloc2;
        auto host_ptr5 = alloc5.get_host_ptr();

        REQUIRE(host_ptr2[0] == host_ptr5[0]);
        REQUIRE(host_ptr2 != host_ptr5);
    }

    SECTION("move test")
    {
        // init
        cuvoxmap::ArrayAllocator<float> alloc3;
        alloc3.resize(100, cuvoxmap::eMemAllocType::HOST);
        alloc3.get_host_ptr()[0] = 1.1f;
        auto host_ptr3 = alloc3.get_host_ptr();

        // move constructor
        cuvoxmap::ArrayAllocator<float> alloc4 = std::move(alloc3);
        auto host_ptr4 = alloc4.get_host_ptr();
        auto host_ptr3_swapped = alloc3.get_host_ptr();

        REQUIRE(host_ptr3 == host_ptr4);
        REQUIRE(host_ptr3 != host_ptr3_swapped);

        // move assignment
        cuvoxmap::ArrayAllocator<float> alloc5;
        alloc5 = std::move(alloc4);
        auto host_ptr5 = alloc5.get_host_ptr();
        auto host_ptr4_swapped = alloc4.get_host_ptr();

        REQUIRE(host_ptr4 == host_ptr5);
        REQUIRE(host_ptr4 != host_ptr4_swapped);
    }
}