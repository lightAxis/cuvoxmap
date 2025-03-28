#pragma once

#include "ArrayIndexing.hpp"
#include "../tmp/DimTmp.hpp"
#include "../tmp/TmpTool.hpp"

namespace cuvoxmap
{
    template <typename DimTmp_ = DimTmpNull>
    class IndexingTmp
    {
    public:
        static_assert(is_dim_tmp_v<DimTmp_>, "DimTmp_ must be of type DimTmp");
        static_assert(DimTmp_::dim >= 1 && DimTmp_::dim <= 3, "DimTmp::dim must be between 1 and 3");
        static constexpr uint8_t dim = DimTmpAccessor<DimTmp_>::dim;
        static constexpr uint32_t x = DimTmpAccessor<DimTmp_>::x;
        static constexpr uint32_t y = DimTmpAccessor<DimTmp_>::y;
        static constexpr uint32_t z = DimTmpAccessor<DimTmp_>::z;
        using Indices = DimTmp_;

        __host__ __device__ IndexingTmp() {}
        __host__ __device__ static constexpr uint8_t DIM() { return dim; }
        __host__ __device__ static constexpr uint32_t getIdxSize(uint8_t dim) { return dim == 0 ? x : (dim == 1 ? y : z); }
        template <uint8_t axis>
        __host__ __device__ static constexpr uint32_t getIdxSize()
        {
            static_assert(axis < dim, "axis must be smaller than dim");
            return axis == 0 ? x : (axis == 1 ? y : z);
        }
        __host__ __device__ static Indexing<dim> getIndexingInstance() { return Indexing<dim>{getAxisSizesVec()}; }
        __host__ __device__ static Vector<uint32_t, dim> getAxisSizesVec() { return Vector<uint32_t, dim>{x, y, z}; }
        __host__ __device__ static uint32_t merge(const Vector<uint32_t, dim> &idx)
        {
            if constexpr (dim == 1)
            {
                return idx[0];
            }
            else if constexpr (dim == 2)
            {
                return idx[0] + x * idx[1];
            }
            else if constexpr (dim == 3)
            {
                return idx[0] + x * idx[1] + x * y * idx[2];
            }
        }
        template <typename MergeDimTmp_>
        __host__ __device__ static constexpr uint32_t merge()
        {
            static_assert(is_dim_tmp_v<MergeDimTmp_>, "MergeDimTmp_ must be of type DimTmp");
            static_assert(DimTmpAccessor<MergeDimTmp_>::dim == dim, "MergeDimTmp_ must have the same dimension as IndexingTmp");

            constexpr uint32_t merge_x = DimTmpAccessor<MergeDimTmp_>::x;
            constexpr uint32_t merge_y = DimTmpAccessor<MergeDimTmp_>::y;
            constexpr uint32_t merge_z = DimTmpAccessor<MergeDimTmp_>::z;
            if constexpr (dim == 1)
            {
                return merge_x;
            }
            else if constexpr (dim == 2)
            {
                return merge_x + tmp_tool::MUl_OR_BITSHIFT<x>(merge_y);
            }
            else if constexpr (dim == 3)
            {
                return merge_x + tmp_tool::MUl_OR_BITSHIFT<x>(merge_y) + tmp_tool::MUl_OR_BITSHIFT<x * y>(merge_z);
            }
        }

        __host__ __device__ static Vector<uint32_t, dim> split(uint32_t idx)
        {
            if constexpr (dim == 1)
            {
                return Vector<uint32_t, dim>{idx};
            }
            else if constexpr (dim == 2)
            {
                return Vector<uint32_t, dim>{tmp_tool::MOD_OR_BITSHIFT<x>(idx), tmp_tool::DIV_OR_BITSHIFT<x>(idx)};
            }
            else if constexpr (dim == 3)
            {
                return Vector<uint32_t, dim>{tmp_tool::MOD_OR_BITSHIFT<x>(idx), tmp_tool::DIV_OR_BITSHIFT<x>(tmp_tool::MOD_OR_BITSHIFT<x * y>(idx)), tmp_tool::DIV_OR_BITSHIFT<x * y>(idx)};
            }
        }
        template <uint32_t idx>
        __host__ __device__ static Vector<uint32_t, dim> split()
        {
            if constexpr (dim == 1)
            {
                return Vector<uint32_t, dim>{idx};
            }
            else if constexpr (dim == 2)
            {
                return Vector<uint32_t, dim>{tmp_tool::MOD_OR_BITSHIFT<x>(idx), tmp_tool::DIV_OR_BITSHIFT<x>(idx)};
            }
            else if constexpr (dim == 3)
            {
                return Vector<uint32_t, dim>{tmp_tool::MOD_OR_BITSHIFT<x>(idx), tmp_tool::DIV_OR_BITSHIFT<x>(tmp_tool::MOD_OR_BITSHIFT<x * y>(idx)), tmp_tool::DIV_OR_BITSHIFT<x * y>(idx)};
            }
        }

        template <uint32_t idx>
        __host__ __device__ static constexpr auto split_dimtmp()
        {
            if constexpr (dim == 1)
            {
                return DimTmp<dim, idx, 0, 0>{};
            }
            else if constexpr (dim == 2)
            {
                return DimTmp<dim, tmp_tool::MOD_OR_BITSHIFT<x>(idx), tmp_tool::DIV_OR_BITSHIFT<x>(idx), 0>{};
            }
            else if constexpr (dim == 3)
            {
                return DimTmp<dim, tmp_tool::MOD_OR_BITSHIFT<x>(idx), tmp_tool::DIV_OR_BITSHIFT<x>(tmp_tool::MOD_OR_BITSHIFT<x * y>(idx)), tmp_tool::DIV_OR_BITSHIFT<x * y>(idx)>{};
            }
        }

        template <uint32_t idx>
        using SplitDimTmp = decltype(split_dimtmp<idx>());
    };

    template <typename BlockDimTmp_, typename GridDimTmp_ = DimTmpNull>
    class IndexingBlockTmp
    {
        static_assert(is_dim_tmp_v<BlockDimTmp_>, "BlockDimTmp_ must be of type DimTmp");
        static_assert(is_dim_tmp_v<GridDimTmp_>, "GridDimTmp_ must be of type DimTmp");

        static_assert(DimTmpAccessor<BlockDimTmp_>::dim == DimTmpAccessor<GridDimTmp_>::dim ||
                          std::is_same_v<GridDimTmp_, DimTmpNull>,
                      "BlockDimTmp_ and GridDimTmp_ must have the same dimension");

    public:
        static constexpr uint8_t dim = DimTmpAccessor<BlockDimTmp_>::dim;
        static constexpr uint32_t block_x = DimTmpAccessor<BlockDimTmp_>::x;
        static constexpr uint32_t block_y = DimTmpAccessor<BlockDimTmp_>::y;
        static constexpr uint32_t block_z = DimTmpAccessor<BlockDimTmp_>::z;
        static constexpr uint32_t block_size = block_x * block_y * block_z;
        static constexpr uint32_t grid_x = std::is_same_v<GridDimTmp_, DimTmpNull> ? 1 : DimTmpAccessor<GridDimTmp_>::x;
        static constexpr uint32_t grid_y = std::is_same_v<GridDimTmp_, DimTmpNull> ? 1 : DimTmpAccessor<GridDimTmp_>::y;
        static constexpr uint32_t grid_z = std::is_same_v<GridDimTmp_, DimTmpNull> ? 1 : DimTmpAccessor<GridDimTmp_>::z;
        static constexpr uint32_t total_x = std::is_same_v<GridDimTmp_, DimTmpNull> ? DimTmpAccessor<BlockDimTmp_>::x : DimTmpAccessor<BlockDimTmp_>::x * DimTmpAccessor<GridDimTmp_>::x;
        static constexpr uint32_t total_y = std::is_same_v<GridDimTmp_, DimTmpNull> ? DimTmpAccessor<BlockDimTmp_>::y : DimTmpAccessor<BlockDimTmp_>::y * DimTmpAccessor<GridDimTmp_>::y;
        static constexpr uint32_t total_z = std::is_same_v<GridDimTmp_, DimTmpNull> ? DimTmpAccessor<BlockDimTmp_>::z : DimTmpAccessor<BlockDimTmp_>::z * DimTmpAccessor<GridDimTmp_>::z;
        static constexpr uint32_t total_size = total_x * total_y * total_z;
        using BlockDim = BlockDimTmp_;
        using BlockIndexing = IndexingTmp<BlockDim>;
        using GridDim = DimTmp<BlockDim::dim, grid_x, grid_y, grid_z>;
        using GridIndexing = IndexingTmp<GridDim>;
        using TotalDim = DimTmp<BlockDim::dim, total_x, total_y, total_z>;

        template <typename BlockDimTmp, typename GridDimTmp>
        struct BlockGridIdxTmpPair
        {
            using BlockDim = BlockDimTmp;
            using GridDim = GridDimTmp;
        };

        __host__ __device__ IndexingBlockTmp() {}
        __host__ __device__ static constexpr uint8_t DIM() { return dim; }
        __host__ __device__ static constexpr uint32_t getIdxSize(uint8_t dim) { return dim == 0 ? total_x : (dim == 1 ? total_y : total_z); }
        template <uint8_t axis>
        __host__ __device__ static constexpr uint32_t getIdxSize()
        {
            static_assert(axis < dim, "axis must be smaller than dim");
            return axis == 0 ? total_x : (axis == 1 ? total_y : total_z);
        }
        __host__ __device__ static IndexingBlock<dim> getIndexingBlockInstance() { return IndexingBlock{BlockIndexing::getIndexingInstance(), GridIndexing::getIndexingInstance()}; }
        __host__ __device__ static uint32_t merge(const Vector<uint32_t, dim> &idx)
        {
            // idx is mostly larger than block_dim
            // parse the idx of block_dim and grid_dim
            Vector<uint32_t, dim> block_idx;
            Vector<uint32_t, dim> grid_idx;
            if constexpr (dim == 1)
            {
                block_idx[0] = tmp_tool::MOD_OR_BITSHIFT<block_x>(idx[0]);
                grid_idx[0] = tmp_tool::DIV_OR_BITSHIFT<block_x>(idx[0]);
            }
            else if constexpr (dim == 2)
            {
                block_idx[0] = tmp_tool::MOD_OR_BITSHIFT<block_x>(idx[0]);
                block_idx[1] = tmp_tool::MOD_OR_BITSHIFT<block_y>(idx[1]);
                grid_idx[0] = tmp_tool::DIV_OR_BITSHIFT<block_x>(idx[0]);
                grid_idx[1] = tmp_tool::DIV_OR_BITSHIFT<block_y>(idx[1]);
            }
            else if constexpr (dim == 3)
            {
                block_idx[0] = tmp_tool::MOD_OR_BITSHIFT<block_x>(idx[0]);
                block_idx[1] = tmp_tool::MOD_OR_BITSHIFT<block_y>(idx[1]);
                block_idx[2] = tmp_tool::MOD_OR_BITSHIFT<block_z>(idx[2]);
                grid_idx[0] = tmp_tool::DIV_OR_BITSHIFT<block_x>(idx[0]);
                grid_idx[1] = tmp_tool::DIV_OR_BITSHIFT<block_y>(idx[1]);
                grid_idx[2] = tmp_tool::DIV_OR_BITSHIFT<block_z>(idx[2]);
            }

            // merge the block idx and grid idx
            return merge(block_idx, grid_idx);
        }
        __host__ __device__ static uint32_t merge(const Vector<uint32_t, dim> &block_idx, const Vector<uint32_t, dim> &grid_idx)
        {
            const uint32_t block_index = BlockIndexing::merge(block_idx);
            const uint32_t grid_index = GridIndexing::merge(grid_idx);

            // merge the block index and grid index
            return block_index + tmp_tool::MUl_OR_BITSHIFT<block_size>(grid_index);
        }

        template <typename DimTmp_>
        __host__ __device__ static constexpr uint32_t merge()
        {
            static_assert(is_dim_tmp_v<DimTmp_>, "DimTmp_ must be of type DimTmp");
            static_assert(DimTmpAccessor<DimTmp_>::dim == dim, "DimTmp_ must have the same dimension as IndexingBlockTmp");
            static_assert(DimTmpAccessor<DimTmp_>::x < total_x, "DimTmp_ must be smaller than total_x");
            static_assert(DimTmpAccessor<DimTmp_>::y < total_y, "DimTmp_ must be smaller than total_y");
            static_assert(DimTmpAccessor<DimTmp_>::z < total_z, "DimTmp_ must be smaller than total_z");

            constexpr uint32_t merge_x = DimTmpAccessor<DimTmp_>::x;
            constexpr uint32_t merge_y = DimTmpAccessor<DimTmp_>::y;
            constexpr uint32_t merge_z = DimTmpAccessor<DimTmp_>::z;
            using MergeIdxTmp = DimTmp_;

            constexpr uint32_t block_idx_x = tmp_tool::MOD_OR_BITSHIFT<block_x>(merge_x);
            constexpr uint32_t block_idx_y = tmp_tool::MOD_OR_BITSHIFT<block_y>(merge_y);
            constexpr uint32_t block_idx_z = tmp_tool::MOD_OR_BITSHIFT<block_z>(merge_z);
            using BlockIdxTmp = DimTmp<dim, block_idx_x, block_idx_y, block_idx_z>;

            constexpr uint32_t grid_idx_x = tmp_tool::DIV_OR_BITSHIFT<block_x>(merge_x);
            constexpr uint32_t grid_idx_y = tmp_tool::DIV_OR_BITSHIFT<block_y>(merge_y);
            constexpr uint32_t grid_idx_z = tmp_tool::DIV_OR_BITSHIFT<block_z>(merge_z);
            using GridIdxTmp = DimTmp<dim, grid_idx_x, grid_idx_y, grid_idx_z>;

            return merge<BlockIdxTmp, GridIdxTmp>();
        }

        template <typename BlockDimTmp, typename GridDimTmp>
        __host__ __device__ static constexpr uint32_t merge()
        {
            static_assert(is_dim_tmp_v<BlockDimTmp>, "BlockDimTmp must be of type DimTmp");
            static_assert(is_dim_tmp_v<GridDimTmp>, "GridDimTmp must be of type DimTmp");
            static_assert(DimTmpAccessor<BlockDimTmp>::dim == dim, "BlockDimTmp must have the same dimension as IndexingBlockTmp");
            static_assert(DimTmpAccessor<GridDimTmp>::dim == dim, "GridDimTmp must have the same dimension as IndexingBlockTmp");
            static_assert(DimTmpAccessor<BlockDimTmp>::x < block_x, "BlockDimTmp must be smaller than block_x");
            static_assert(DimTmpAccessor<BlockDimTmp>::y < block_y, "BlockDimTmp must be smaller than block_y");
            static_assert(DimTmpAccessor<BlockDimTmp>::z < block_z, "BlockDimTmp must be smaller than block_z");
            static_assert(DimTmpAccessor<GridDimTmp>::x < grid_x, "GridDimTmp must be smaller than grid_x");
            static_assert(DimTmpAccessor<GridDimTmp>::y < grid_y, "GridDimTmp must be smaller than grid_y");
            static_assert(DimTmpAccessor<GridDimTmp>::z < grid_z, "GridDimTmp must be smaller than grid_z");

            constexpr uint32_t block_idx = BlockIndexing::template merge<BlockDimTmp>();
            constexpr uint32_t grid_idx = GridIndexing::template merge<GridDimTmp>();

            return block_idx + tmp_tool::MUl_OR_BITSHIFT<block_size>(grid_idx);
        }
        __host__ __device__ auto static constexpr split(uint32_t idx)
        {
            const uint32_t block_index = tmp_tool::MOD_OR_BITSHIFT<block_size>(idx);
            const uint32_t grid_index = tmp_tool::DIV_OR_BITSHIFT<block_size>(idx);

            return typename IndexingBlock<dim>::BlockGridIdxPair{BlockIndexing::template split(block_index), GridIndexing::template split(grid_index)};
        }

        template <uint32_t idx>
        __host__ __device__ static constexpr auto split()
        {
            static_assert(idx < total_size, "idx must be smaller than total_size");

            constexpr uint32_t block_index = tmp_tool::MOD_OR_BITSHIFT<block_size>(idx);
            constexpr uint32_t grid_index = tmp_tool::DIV_OR_BITSHIFT<block_size>(idx);

            return BlockGridIdxTmpPair<typename BlockIndexing::template SplitDimTmp<block_index>, typename GridIndexing::template SplitDimTmp<grid_index>>{};
        }

        template <uint32_t idx>
        using SplitDimTmp = decltype(split<idx>());
    };

}