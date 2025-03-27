#include <catch2/catch_test_macros.hpp>
#include <cuvoxmap/utils/ArrayIndexing.hpp>
#include <cuvoxmap/utils/ArrayIndexingTmp.hpp>

TEST_CASE("ArrayIndexing cpu", "utils")
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

    SECTION("indexing using TMP")
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
        using splitType = IdxTmp::SplitDimTmp<23>;
    }
}

TEST_CASE("ArrayInexingBlock cpu", "utils")
{
    const cuvoxmap::uIdx3D block_indices{2, 3, 4};
    const cuvoxmap::uIdx3D grid_indices{5, 6, 7};
    cuvoxmap::IndexingBlock indexing_block{block_indices, grid_indices};

    cuvoxmap::Indexing<3> block_dim_indexing{block_indices};
    cuvoxmap::IndexingBlock indexing_block_only{block_dim_indexing};

    using BlockIndices = cuvoxmap::DimTmp<3, 2, 3, 4>;
    using GridIndices = cuvoxmap::DimTmp<3, 5, 6, 7>;
    using IdxBlockTmp = cuvoxmap::IndexingBlockTmp<BlockIndices, GridIndices>;
    using IdxBlockTmpBlockOnly = cuvoxmap::IndexingBlockTmp<BlockIndices>;

    SECTION("indexingblock")
    {
        REQUIRE(indexing_block.DIM() == 3);
        REQUIRE(indexing_block.getIdxSize(0) == 10);
        REQUIRE(indexing_block.getIdxSize(1) == 18);
        REQUIRE(indexing_block.getIdxSize(2) == 28);
        REQUIRE(indexing_block.merge(cuvoxmap::uIdx3D{5, 0, 27}) == 4387);
        REQUIRE(indexing_block.split(4387).block_idx == cuvoxmap::uIdx3D{1, 0, 3});
        REQUIRE(indexing_block.split(4387).grid_idx == cuvoxmap::uIdx3D{2, 0, 6});
    }

    SECTION("indexingblock block only")
    {
        REQUIRE(indexing_block_only.DIM() == 3);
        REQUIRE(indexing_block_only.getIdxSize(0) == 2);
        REQUIRE(indexing_block_only.getIdxSize(1) == 3);
        REQUIRE(indexing_block_only.getIdxSize(2) == 4);
        REQUIRE(indexing_block_only.merge(cuvoxmap::uIdx3D{1, 2, 3}) == 23);
        REQUIRE(indexing_block_only.split(23).block_idx == cuvoxmap::uIdx3D{1, 2, 3});
        REQUIRE(indexing_block_only.split(23).grid_idx == cuvoxmap::uIdx3D{0, 0, 0});
    }

    SECTION("indexingblock tmp")
    {
        REQUIRE(IdxBlockTmp::DIM() == 3);
        REQUIRE(IdxBlockTmp::getIdxSize(0) == 10);
        REQUIRE(IdxBlockTmp::getIdxSize(1) == 18);
        REQUIRE(IdxBlockTmp::getIdxSize(2) == 28);
        REQUIRE(IdxBlockTmp::merge(cuvoxmap::uIdx3D{5, 0, 27}) == 4387);
        REQUIRE(IdxBlockTmp::merge<cuvoxmap::DimTmp<3, 5, 0, 27>>() == 4387);
        REQUIRE(IdxBlockTmp::split(4387).block_idx == cuvoxmap::uIdx3D{1, 0, 3});
        REQUIRE(IdxBlockTmp::split(4387).grid_idx == cuvoxmap::uIdx3D{2, 0, 6});
        REQUIRE(std::is_same_v<IdxBlockTmp::SplitDimTmp<4387>::BlockDim, cuvoxmap::DimTmp<3, 1, 0, 3>>);
        REQUIRE(std::is_same_v<IdxBlockTmp::SplitDimTmp<4387>::GridDim, cuvoxmap::DimTmp<3, 2, 0, 6>>);

        [[maybe_unused]] static constexpr uint8_t dim = IdxBlockTmp::DIM();
        [[maybe_unused]] static constexpr uint32_t idx_size_0 = IdxBlockTmp::getIdxSize(0);
        [[maybe_unused]] static constexpr uint32_t idx_size_1 = IdxBlockTmp::getIdxSize(1);
        [[maybe_unused]] static constexpr uint32_t idx_size_2 = IdxBlockTmp::getIdxSize(2);
        [[maybe_unused]] const uint32_t merge0 = IdxBlockTmp::merge(cuvoxmap::uIdx3D{5, 0, 27});
        [[maybe_unused]] static constexpr uint32_t merge0_tmp = IdxBlockTmp::merge<cuvoxmap::DimTmp<3, 5, 0, 27>>();
        [[maybe_unused]] const auto splited = IdxBlockTmp::split(4387);
        using splitedType = IdxBlockTmp::SplitDimTmp<4387>;
    }

    SECTION("indexingblock block only tmp")
    {
        REQUIRE(IdxBlockTmpBlockOnly::DIM() == 3);
        REQUIRE(IdxBlockTmpBlockOnly::getIdxSize(0) == 2);
        REQUIRE(IdxBlockTmpBlockOnly::getIdxSize(1) == 3);
        REQUIRE(IdxBlockTmpBlockOnly::getIdxSize(2) == 4);
        REQUIRE(IdxBlockTmpBlockOnly::merge(cuvoxmap::uIdx3D{1, 2, 3}) == 23);
        REQUIRE(IdxBlockTmpBlockOnly::merge<cuvoxmap::DimTmp<3, 1, 2, 3>>() == 23);
        const auto asdf = IdxBlockTmpBlockOnly::split(23);
        REQUIRE(IdxBlockTmpBlockOnly::split(23).block_idx == cuvoxmap::uIdx3D{1, 2, 3});
        REQUIRE(IdxBlockTmpBlockOnly::split(23).grid_idx == cuvoxmap::uIdx3D{0, 0, 0});
        REQUIRE(std::is_same_v<IdxBlockTmpBlockOnly::SplitDimTmp<23>::BlockDim, cuvoxmap::DimTmp<3, 1, 2, 3>>);
        REQUIRE(std::is_same_v<IdxBlockTmpBlockOnly::SplitDimTmp<23>::GridDim, cuvoxmap::DimTmp<3, 0, 0, 0>>);

        [[maybe_unused]] static constexpr uint8_t dim = IdxBlockTmpBlockOnly::DIM();
        [[maybe_unused]] static constexpr uint32_t idx_size_0 = IdxBlockTmpBlockOnly::getIdxSize(0);
        [[maybe_unused]] static constexpr uint32_t idx_size_1 = IdxBlockTmpBlockOnly::getIdxSize(1);
        [[maybe_unused]] static constexpr uint32_t idx_size_2 = IdxBlockTmpBlockOnly::getIdxSize(2);
        [[maybe_unused]] const uint32_t merge0 = IdxBlockTmpBlockOnly::merge(cuvoxmap::uIdx3D{1, 2, 3});
        [[maybe_unused]] static constexpr uint32_t merge0_tmp = IdxBlockTmpBlockOnly::merge<cuvoxmap::DimTmp<3, 1, 2, 3>>();
        [[maybe_unused]] const auto splited = IdxBlockTmp::split(4387);
        using splitedType = IdxBlockTmp::SplitDimTmp<4387>;
    }
}