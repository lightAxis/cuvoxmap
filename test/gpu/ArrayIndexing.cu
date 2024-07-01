#include <catch2/catch_test_macros.hpp>
#include "../custom_matchers/device_testmacros.cuh"
#include <cuvoxmap/utils/ArrayIndexing.hpp>

TEST_CASE("ArrayIndexing gpu host", "utils")
{
    cuvoxmap::Indexing<3> indexing(cuvoxmap::uIdx3D{2, 3, 4});

    SECTION("indexing")
    {
        REQUIRE(indexing.DIM() == 3);
        REQUIRE(indexing.getIdxSize(0) == 2);
        REQUIRE(indexing.getIdxSize(1) == 3);
        REQUIRE(indexing.getIdxSize(2) == 4);
        REQUIRE(indexing.merge(cuvoxmap::uIdx3D{1, 2, 3}) == 23);
        REQUIRE(indexing.split(23) == cuvoxmap::uIdx3D{1, 2, 3});
    }
}

__global__ void indexing_(cuvoxmap::Indexing<3> idxing)
{
}
TEST_CASE("ArrayIndexing gpu device")
{
    cuvoxmap::Indexing<3> indexing(cuvoxmap::uIdx3D{2, 3, 4});

    SECTION("indexing")
    {
        indexing_<<<1, 1>>>(indexing);
    }
}