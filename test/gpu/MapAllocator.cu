#include <catch2/catch_test_macros.hpp>
#include <cuvoxmap/base/MapAllocator.hpp>
#include "../custom_matchers/device_testmacros.cuh"

TEST_CASE("MapAllocator gpu")
{
    cuvoxmap::uIdx3D dims{5, 6, 7};
    cuvoxmap::MapAllocator<float, 3> mapAlloc{dims, cuvoxmap::eMemAllocType::HOST_AND_DEVICE};

    SECTION("basics")
    {
        auto mapData = mapAlloc.get_mapData();
        REQUIRE(mapData.host_data != nullptr);
        REQUIRE(mapData.device_data != nullptr);
        REQUIRE(mapData.is_gpu_used == true);

        REQUIRE(mapData.axis_sizes == cuvoxmap::uIdx3D{5, 6, 7});
    }
    SECTION("constructors")
    {
        // copy constructor
        cuvoxmap::MapAllocator<float, 3> map2 = mapAlloc;
        auto host_ptr = mapAlloc.get_mapData().host_data;
        auto host_ptr2 = map2.get_mapData().host_data;

        REQUIRE(host_ptr != host_ptr2);
        REQUIRE(host_ptr[0] == host_ptr2[0]);

        // copy assignment operator
        cuvoxmap::MapAllocator<float, 3> map3;
        map3 = mapAlloc;
        auto host_ptr3 = map3.get_mapData().host_data;

        REQUIRE(host_ptr3 != host_ptr);
        REQUIRE(host_ptr3[0] == host_ptr[0]);

        // move constructor
        cuvoxmap::MapAllocator<float, 3> map4 = mapAlloc;
        auto host_ptr4 = map4.get_mapData().host_data;
        cuvoxmap::MapAllocator<float, 3> map5 = std::move(map4);
        auto host_ptr5 = map5.get_mapData().host_data;
        REQUIRE_THROWS(map4.get_mapData().host_data);

        REQUIRE(host_ptr4 == host_ptr5);

        // move assignment operator
        cuvoxmap::MapAllocator<float, 3> map6;
        map6 = std::move(map5);
        auto host_ptr6 = map6.get_mapData().host_data;
        REQUIRE_THROWS(map5.get_mapData().host_data);

        REQUIRE(host_ptr6 == host_ptr5);
    }
}