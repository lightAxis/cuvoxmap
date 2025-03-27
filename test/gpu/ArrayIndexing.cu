#include <catch2/catch_test_macros.hpp>
#include "../custom_matchers/device_testmacros.cuh"
#include <cuvoxmap/utils/ArrayIndexing.hpp>
#include <cuvoxmap/utils/ArrayIndexingTmp.hpp>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

TEST_CASE("ArrayIndexing gpu host", "utils")
{
    cuvoxmap::Indexing<3> indexing(cuvoxmap::uIdx3D{2, 3, 4});
    using IdxTmp = cuvoxmap::IndexingTmp<cuvoxmap::DimTmp<3, 2, 3, 4>>;

    SECTION("indexing")
    {
        REQUIRE(indexing.DIM() == 3);
        REQUIRE(indexing.getIdxSize(0) == 2);
        REQUIRE(indexing.getIdxSize(1) == 3);
        REQUIRE(indexing.getIdxSize(2) == 4);
        REQUIRE(indexing.merge(cuvoxmap::uIdx3D{1, 2, 3}) == 23);
        REQUIRE(indexing.split(23) == cuvoxmap::uIdx3D{1, 2, 3});
    }

    SECTION("indexing tmp")
    {
        REQUIRE(IdxTmp::DIM() == 3);
        REQUIRE(IdxTmp::getIdxSize(0) == 2);
        REQUIRE(IdxTmp::getIdxSize(1) == 3);
        REQUIRE(IdxTmp::getIdxSize(2) == 4);
        REQUIRE(IdxTmp::getIdxSize<0>() == 2);
        REQUIRE(IdxTmp::getIdxSize<1>() == 3);
        REQUIRE(IdxTmp::getIdxSize<2>() == 4);
        REQUIRE(IdxTmp::merge(cuvoxmap::uIdx3D{1, 2, 3}) == 23);
        REQUIRE(IdxTmp::merge<cuvoxmap::DimTmp<3, 1, 2, 3>>() == 23);
        REQUIRE(IdxTmp::split(23) == cuvoxmap::uIdx3D{1, 2, 3});
        REQUIRE(IdxTmp::split<23>() == cuvoxmap::uIdx3D{1, 2, 3});
        REQUIRE(std::is_same_v<decltype(IdxTmp::split_dimtmp<23>()), cuvoxmap::DimTmp<3, 1, 2, 3>>);
        REQUIRE(std::is_same_v<IdxTmp::SplitDimTmp<23>, cuvoxmap::DimTmp<3, 1, 2, 3>>);

        [[maybe_unused]] constexpr auto dim = IdxTmp::DIM();
        [[maybe_unused]] constexpr auto idx_size_0 = IdxTmp::getIdxSize(0);
        [[maybe_unused]] constexpr auto idx_size_1 = IdxTmp::getIdxSize(1);
        [[maybe_unused]] constexpr auto idx_size_2 = IdxTmp::getIdxSize(2);
        [[maybe_unused]] constexpr auto idx_size_0_ = IdxTmp::getIdxSize<0>();
        [[maybe_unused]] constexpr auto idx_size_1_ = IdxTmp::getIdxSize<1>();
        [[maybe_unused]] constexpr auto idx_size_2_ = IdxTmp::getIdxSize<2>();
        [[maybe_unused]] constexpr auto merge_ = IdxTmp::merge<cuvoxmap::DimTmp<3, 1, 2, 3>>();
        [[maybe_unused]] constexpr auto split = IdxTmp::split_dimtmp<23>();
    }
}

__global__ void indexing_(cuvoxmap::Indexing<3> idxing, bool *res)
{
    *res = false;
    *res = true;
}

__global__ void indexing_tmp_(uint32_t idx, bool *res)
{
    using IdxTmp = cuvoxmap::IndexingTmp<cuvoxmap::DimTmp<3, 2, 3, 4>>;
    auto dim = IdxTmp::split(idx);

    *res = false;
    if (!CUSTOM_TEST_KERNEL::KERNEL_TEST(dim[0], 1u))
        return;
    if (!CUSTOM_TEST_KERNEL::KERNEL_TEST(dim[1], 2u))
        return;
    if (!CUSTOM_TEST_KERNEL::KERNEL_TEST(dim[2], 3u))
        return;
    *res = true;
}

TEST_CASE("ArrayIndexing gpu device")
{
    cuvoxmap::Indexing<3> indexing(cuvoxmap::uIdx3D{2, 3, 4});
    thrust::host_vector<bool> host_res(1, false);
    thrust::device_vector<bool> dev_res(1, false);

    SECTION("indexing")
    {
        indexing_<<<1, 1>>>(indexing, thrust::raw_pointer_cast(dev_res.data()));
        cudaDeviceSynchronize();
        host_res = dev_res;
        if (!host_res[0])
            FAIL("indexing failed");
    }

    SECTION("indexing tmp")
    {
        indexing_tmp_<<<1, 1>>>(23, thrust::raw_pointer_cast(dev_res.data()));
        cudaDeviceSynchronize();
        host_res = dev_res;
        if (!host_res[0])
            FAIL("indexing tmp failed");
    }
}