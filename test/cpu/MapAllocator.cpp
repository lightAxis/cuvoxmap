#include <catch2/catch_test_macros.hpp>
#include <cuvoxmap/base/MapAllocator.hpp>

TEST_CASE("MapAllocator cpu")
{
    cuvoxmap::uIdx3D dims{5, 6, 7};
    cuvoxmap::MapAllocator<float, 3> mapAlloc{dims, cuvoxmap::eMemAllocType::HOST};

    SECTION("basics")
    {
        auto mapData = mapAlloc.get_mapData();
        REQUIRE(mapData.host_data != nullptr);
        REQUIRE(mapData.device_data == nullptr);
        REQUIRE(mapData.is_gpu_used == false);

        REQUIRE(mapData.axis_sizes == cuvoxmap::uIdx3D{5, 6, 7});
    }
}