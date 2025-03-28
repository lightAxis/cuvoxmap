#pragma once

#include "Vector.hpp"

namespace cuvoxmap
{

    template <uint8_t Dim>
    class Indexing
    {
    public:
        __host__ __device__ Indexing() {}

        __host__ __device__ explicit Indexing(const Vector<uint32_t, Dim> &indices)
            : indices_(indices)
        {
        }

        __host__ __device__ constexpr uint8_t DIM() const { return Dim; }
        __host__ __device__ uint32_t getIdxSize(uint8_t dim) const { return indices_[dim]; }
        __host__ __device__ Vector<uint32_t, Dim> getIndices() const { return indices_; }
        __host__ __device__ uint32_t merge(const Vector<uint32_t, Dim> &idx) const
        {
            uint32_t index = 0;
            uint32_t multiplier = 1;
            for (int i = 0; i < Dim; i++)
            {
                index += idx[i] * multiplier;
                multiplier *= indices_[i];
            }
            return index;
        }

        __host__ __device__ Vector<uint32_t, Dim> split(uint32_t idx) const
        {
            Vector<uint32_t, Dim> indices;
            for (int i = 0; i < Dim; ++i)
            {
                indices[i] = idx % indices_[i];
                idx /= indices_[i];
            }
            return indices;
        }

        __host__ __device__ uint32_t totalSize() const
        {
            uint32_t size = 1;
            for (int i = 0; i < Dim; i++)
            {
                size *= indices_[i];
            }
            return size;
        }

    private:
        Vector<uint32_t, Dim> indices_;
    };

    /**
     * @brief IndexingBlock class. Block tiling indexing for advanced memory layout
     * block_dim_indexing is dimesions of the block itself
     * grid_dim_indexing is demensions of the total grid based on the block
     * ex) block_dim[2,3,4] and grid_dim[4,5,6] means total dim [2*4, 3*5, 4*6] size
     * memory layout is linearized first by block, and the by grid
     * ex) [block_000] [block_100] ... [block_456] = total memory layout
     * ex) [000] [100] ... [234] = each block memory layout
     */
    template <uint8_t Dim>
    class IndexingBlock
    {
    public:
        struct BlockGridIdxPair
        {
            Vector<uint32_t, Dim> block_idx;
            Vector<uint32_t, Dim> grid_idx;

            __host__ __device__ BlockGridIdxPair() {}
            __host__ __device__ BlockGridIdxPair(const Vector<uint32_t, Dim> &block_idx, const Vector<uint32_t, Dim> &grid_idx)
                : block_idx(block_idx), grid_idx(grid_idx) {}
        };

        __host__ __device__ IndexingBlock() {}
        __host__ __device__ IndexingBlock(const Indexing<Dim> &block_dim_indexing)
            : block_dim_indexing_(block_dim_indexing), grid_dim_indexing_(Vector<uint32_t, Dim>::Ones()),
              block_size_(block_dim_indexing_.totalSize()), total_indices_(block_dim_indexing.getIndices()),
              is_using_grid_(false) {}
        __host__ __device__ IndexingBlock(const Indexing<Dim> &block_dim_indexing, const Indexing<Dim> &grid_dim_indexing)
            : block_dim_indexing_(block_dim_indexing), grid_dim_indexing_(grid_dim_indexing),
              block_size_(block_dim_indexing.totalSize()), total_indices_(block_dim_indexing.getIndices().elementWiseMul(grid_dim_indexing.getIndices())),
              is_using_grid_(true) {}

        __host__ __device__ IndexingBlock(const Vector<uint32_t, Dim> &block_indices)
            : IndexingBlock(Indexing<Dim>(block_indices)) {}
        __host__ __device__ IndexingBlock(const Vector<uint32_t, Dim> &block_indices, const Vector<uint32_t, Dim> &grid_indices)
            : IndexingBlock(Indexing<Dim>(block_indices), Indexing<Dim>(grid_indices)) {}

        __host__ __device__ constexpr uint8_t DIM() const { return Dim; }
        __host__ __device__ const Indexing<Dim> &getBlockDimIndexing() const { return block_dim_indexing_; }
        __host__ __device__ const Indexing<Dim> &getGridDimIndexing() const { return grid_dim_indexing_; }
        __host__ __device__ uint32_t getIdxSize(uint8_t dim) const { return total_indices_[dim]; }
        __host__ __device__ uint32_t merge(const Vector<uint32_t, Dim> &idx) const
        {
            if (is_using_grid_ == false)
            {
                return block_dim_indexing_.merge(idx);
            }

            // idx is mostly larger than block_dim
            // parse the idx of block_dim and grid_dim
            Vector<uint32_t, Dim> block_idx;
            Vector<uint32_t, Dim> grid_idx;
            for (int i = 0; i < Dim; i++)
            {
                block_idx[i] = idx[i] % block_dim_indexing_.getIdxSize(i);
                grid_idx[i] = idx[i] / block_dim_indexing_.getIdxSize(i);
            }
            return merge(block_idx, grid_idx);
        }
        __host__ __device__ uint32_t merge(const Vector<uint32_t, Dim> &block_idx, const Vector<uint32_t, Dim> &grid_idx) const
        {
            // merge the block_idx and grid_idx
            uint32_t block_index = block_dim_indexing_.merge(block_idx);
            uint32_t grid_index = grid_dim_indexing_.merge(grid_idx);

            // merge the block_index and grid_index
            return block_index + grid_index * block_size_;
        }

        __host__ __device__ BlockGridIdxPair split(uint32_t idx) const
        {
            if (is_using_grid_ == false)
            {
                return BlockGridIdxPair(block_dim_indexing_.split(idx), Vector<uint32_t, Dim>::Zeros());
            }

            // split the idx to block_index and grid_index
            uint32_t block_index = idx % block_size_;
            uint32_t grid_index = idx / block_size_;

            // split  each index
            return BlockGridIdxPair(block_dim_indexing_.split(block_index), grid_dim_indexing_.split(grid_index));
        }

    private:
        const Indexing<Dim> block_dim_indexing_;
        const Indexing<Dim> grid_dim_indexing_;
        Vector<uint32_t, Dim> total_indices_;
        const uint32_t block_size_;
        const bool is_using_grid_;
    };
}