#include <catch2/catch_test_macros.hpp>
#include <cuvoxmap/MapAllocator.hpp>
#include "device_testmacros.cuh"

TEST_CASE("MapAllocator gpu")
{
    cuvoxmap::Idx3D dims{5, 6, 7};
    cuvoxmap::MapAllocator<float, 3> mapAlloc{dims};

    SECTION("basics")
    {
        auto mapData = mapAlloc.get_mapData();
        REQUIRE(mapData.host_data != nullptr);
        REQUIRE(mapData.device_data != nullptr);
        REQUIRE(mapData.is_gpu_used == true);

        REQUIRE(mapData.axis_sizes == cuvoxmap::Idx3D{5, 6, 7});
    }
}