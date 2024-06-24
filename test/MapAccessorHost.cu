#include <cuvoxmap/MapAllocator.hpp>
#include <cuvoxmap/MapAccessorHost.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("MapAccessorHost gpu")
{
    cuvoxmap::MapAllocator<float, 2> alloc{cuvoxmap::Idx2D{10, 20}};
    cuvoxmap::MapAccesssorHost<float, 2> accessor{alloc.get_mapData()};

    SECTION("cpu basic")
    {
        accessor.set_value(1, cuvoxmap::Idx2D{1, 2});
        accessor.set_value(2, cuvoxmap::Idx2D{2, 3});

        REQUIRE(accessor.get_value(cuvoxmap::Idx2D{1, 2}) == 1);
        REQUIRE(accessor.get_value(cuvoxmap::Idx2D{2, 3}) == 2);
        REQUIRE(accessor.get_axis_size() == cuvoxmap::Idx2D{10, 20});
        REQUIRE(accessor.merge_idx({1, 2}) == 1 * 1 + 2 * 10);
        REQUIRE(accessor.merge_idx(cuvoxmap::Idx2D{1, 2}) == 1 * 1 + 2 * 10);

        cuvoxmap::MapData<float, 2> data = accessor.get_map_data();
        REQUIRE(data.is_gpu_used == true);
        REQUIRE(data.host_data != nullptr);
        REQUIRE(data.device_data != nullptr);
        REQUIRE(data.axis_sizes == cuvoxmap::Idx2D{10, 20});
    }
}