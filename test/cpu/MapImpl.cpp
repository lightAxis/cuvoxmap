#include <catch2/catch_test_macros.hpp>
#include <cuvoxmap/base/MapImpl.hpp>

TEST_CASE("MapImpl_cpu")
{
    cuvoxmap::MapImpl<float, 2> map{cuvoxmap::uIdx2D{10, 20}, cuvoxmap::eMemAllocType::HOST};
    map.get_host_data()[0] = 1.1f;

    SECTION("constructor")
    {
        // copy constructor
        cuvoxmap::MapImpl<float, 2> map2{map};
        auto host_ptr2 = map2.get_host_data();
        auto host_ptr1 = map.get_host_data();

        REQUIRE(host_ptr1 != host_ptr2);
        REQUIRE(host_ptr1[0] == host_ptr2[0]);

        // copy assignment opearator
        cuvoxmap::MapImpl<float, 2> map3;
        map3 = map;

        auto host_ptr3 = map3.get_host_data();
        REQUIRE(host_ptr1 != host_ptr3);
        REQUIRE(host_ptr1[0] == host_ptr3[0]);

        // move constructor
        cuvoxmap::MapImpl<float, 2> map4{map};
        auto host_ptr4 = map4.get_host_data();
        cuvoxmap::MapImpl<float, 2> map5{std::move(map4)};
        auto host_ptr5 = map5.get_host_data();
        auto host_ptr4_swapped = map4.get_host_data();

        REQUIRE(host_ptr4 == host_ptr5);
        REQUIRE(host_ptr4_swapped != host_ptr4);

        // move assignment operator
        cuvoxmap::MapImpl<float, 2> map6;
        map6 = std::move(map5);
        auto host_ptr6 = map6.get_host_data();
        auto host_ptr5_swapped = map5.get_host_data();

        REQUIRE(host_ptr6 == host_ptr5);
        REQUIRE(host_ptr5_swapped != host_ptr5);
    }
}