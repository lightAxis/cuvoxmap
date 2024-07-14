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
}