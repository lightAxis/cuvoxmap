#include <catch2/catch_test_macros.hpp>
#include "device_testmacros.cuh"
#include <cuvoxmap/utils/ArrayIndexing.hpp>

TEST_CASE("ArrayIndexing cpu", "utils")
{
    cuvoxmap::Indexing<3> indexing(2, 3, 4);

    SECTION("indexing")
    {
        REQUIRE(indexing.DIM() == 3);
        REQUIRE(indexing.getIdxSize(0) == 2);
        REQUIRE(indexing.getIdxSize(1) == 3);
        REQUIRE(indexing.getIdxSize(2) == 4);
        REQUIRE(indexing.merge({1, 2, 3}) == 23);
        REQUIRE(indexing.split(23) == std::array<uint32_t, 3UL>{1, 2, 3});

        REQUIRE(indexing.merge_device(cuvoxmap::Idx3D{1, 2, 3}) == 23);
        REQUIRE(indexing.split_device(23) == cuvoxmap::Idx3D{1, 2, 3});
    }
}

__global__ void indexing_(cuvoxmap::Indexing<3> idxing)
{
}
TEST_CASE("indexing device")
{
    cuvoxmap::Indexing<3> indexing(2, 3, 4);

    SECTION("indexing")
    {
        indexing_<<<1, 1>>>(indexing);
    }
}