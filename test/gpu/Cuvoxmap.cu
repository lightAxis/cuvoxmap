#include <cuvoxmap/cuvoxmap.hpp>

#include "../custom_matchers/custom_matchers.hpp"

using namespace cuvoxmap;

TEST_CASE("Cuvoxmap_gpu")
{
    cuvoxmap::cuvoxmap2D::init_s init;
    init.x_axis_len = 10;
    init.y_axis_len = 20;
    init.resolution = 0.5f;
    init.use_gpu = true;
    cuvoxmap::cuvoxmap2D cmap{init};
    cmap.set_origin(Float2D{1.2f, 1.5f});

    using check = cuvoxmap::eCheck;
    using map = cuvoxmap::eMap;
    using frame = cuvoxmap::eFrame;
    using memType = cuvoxmap::eMemAllocType;

    SECTION("basic get set")
    {
        // using getset_prob = cuvoxmap::map_getset_s<map::PROBABILITY, check::NONE, frame::GLOBAL>;

        cmap.set_map_withIdx<getset::PRB_FAST_GLB>(cuvoxmap::Idx2D{4, 3}, 1.f);
        cmap.set_map_withIdx<getset::ST_FAST_GLB>(cuvoxmap::Idx2D{3, 4}, static_cast<uint8_t>(eVoxel::OCCUPIED));
        cmap.set_map_withIdx<getset::DST_FAST_GLB>(cuvoxmap::Idx2D{4, 4}, 2.f);

        REQUIRE(cmap.get_map_withIdx<getset::PRB_FAST_GLB>(cuvoxmap::Idx2D{4, 3}) == 1.f);
        REQUIRE(cmap.get_map_withIdx<getset::ST_FAST_GLB>(cuvoxmap::Idx2D{3, 4}) == static_cast<uint8_t>(eVoxel::OCCUPIED));
        REQUIRE(cmap.get_map_withIdx<getset::DST_FAST_GLB>(cuvoxmap::Idx2D{4, 4}) == 2.f);

        REQUIRE(cmap.get_map_withIdx<getset::PRB_FAST_LOC>(cuvoxmap::Idx2D{2, 0}) == 1.f);
        REQUIRE(cmap.get_map_withIdx<getset::ST_FAST_LOC>(cuvoxmap::Idx2D{1, 1}) == static_cast<uint8_t>(eVoxel::OCCUPIED));
        REQUIRE(cmap.get_map_withIdx<getset::DST_FAST_LOC>(cuvoxmap::Idx2D{2, 1}) == 2.f);
    }

    SECTION("basic get set with check")
    {
        cmap.set_map_withIdx<getset::PRB_CHK_GLB>(Idx2D{1, 0}, 1.f);
        cmap.set_map_withIdx<getset::ST_CHK_GLB>(Idx2D{0, 1}, static_cast<uint8_t>(eVoxel::OCCUPIED));
        cmap.set_map_withIdx<getset::DST_CHK_GLB>(Idx2D{1, 1}, 2.f);

        REQUIRE(cmap.get_map_withIdx<getset::PRB_CHK_GLB>(Idx2D{1, 0}) == 0.f);
        REQUIRE(cmap.get_map_withIdx<getset::ST_CHK_GLB>(Idx2D{0, 1}) == static_cast<uint8_t>(0));
        REQUIRE(cmap.get_map_withIdx<getset::DST_CHK_GLB>(Idx2D{1, 1}) == 0.f);

        REQUIRE(cmap.get_map_withIdx<getset::PRB_CHK_LOC>(Idx2D{-1, -3}) == 0.f);
        REQUIRE(cmap.get_map_withIdx<getset::ST_CHK_LOC>(Idx2D{-2, -2}) == static_cast<uint8_t>(0));
        REQUIRE(cmap.get_map_withIdx<getset::DST_CHK_LOC>(Idx2D{-1, -2}) == 0.f);
    }

    SECTION("fill test")
    {
        cmap.fill_all<map::DISTANCE, memType::HOST>(30.f);
        cmap.fill_all<map::PROBABILITY, memType::HOST>(0.5f);
        cmap.fill_all<map::STATE, memType::HOST>(static_cast<uint8_t>(eVoxel::UNOBSERVED));

        REQUIRE(cmap.get_map_withIdx<getset::DST_FAST_LOC>(Idx2D{0, 0}) == 30.f);
        REQUIRE(cmap.get_map_withIdx<getset::PRB_FAST_LOC>(Idx2D{0, 0}) == 0.5f);
        REQUIRE(cmap.get_map_withIdx<getset::ST_FAST_LOC>(Idx2D{0, 0}) == static_cast<uint8_t>(eVoxel::UNOBSERVED));
    }

    SECTION("line check test")
    {
        cmap.fill_all<map::STATE, memType::HOST>(static_cast<uint8_t>(eVoxel::UNKNOWN));
        cmap.set_map_withIdx<getset::ST_FAST_LOC>(Idx2D{2, 3}, static_cast<uint8_t>(eVoxel::OCCUPIED));
        cmap.set_map_withIdx<getset::ST_FAST_LOC>(Idx2D{3, 4}, static_cast<uint8_t>(eVoxel::FREE));
        cmap.set_map_withIdx<getset::ST_FAST_LOC>(Idx2D{4, 5}, static_cast<uint8_t>(eVoxel::UNOBSERVED));

        Idx2D start_lidx{1, 2};
        Float2D start_lpos = cmap.get_glob_loc_cvt().lidx_2_lpos(start_lidx);
        Float2D start_gpos = cmap.get_glob_loc_cvt().lpos_2_gpos(start_lpos);

        Idx2D end_lidx{5, 6};
        Float2D end_lpos = cmap.get_glob_loc_cvt().lidx_2_lpos(end_lidx);
        Float2D end_gpos = cmap.get_glob_loc_cvt().lpos_2_gpos(end_lpos);

        REQUIRE(cmap.check_line_state_map<linecheck::CHK_GLB_RAY>(start_gpos, end_gpos, static_cast<uint8_t>(eVoxel::OCCUPIED)) == true);
        REQUIRE(cmap.check_line_state_map<linecheck::CHK_GLB_SUP>(start_gpos, end_gpos, static_cast<uint8_t>(eVoxel::FREE)) == true);
        REQUIRE(cmap.check_line_state_map<linecheck::CHK_GLB_SUP>(start_gpos, end_gpos, static_cast<uint8_t>(eVoxel::UNOBSERVED)) == true);

        REQUIRE(cmap.check_line_state_map<linecheck::NON_LOC_RAY>(start_lpos, end_lpos, static_cast<uint8_t>(eVoxel::OCCUPIED)) == true);
        REQUIRE(cmap.check_line_state_map<linecheck::NON_LOC_SUP>(start_lpos, end_lpos, static_cast<uint8_t>(eVoxel::FREE)) == true);
        REQUIRE(cmap.check_line_state_map<linecheck::NON_LOC_SUP>(start_lpos, end_lpos, static_cast<uint8_t>(eVoxel::UNOBSERVED)) == true);

        Idx2D start2_lidx{4, 3};
        Float2D start2_lpos = cmap.get_glob_loc_cvt().lidx_2_lpos(start2_lidx);
        Float2D start2_gpos = cmap.get_glob_loc_cvt().lpos_2_gpos(start2_lpos);

        Idx2D end2_lidx{6, 5};
        Float2D end2_lpos = cmap.get_glob_loc_cvt().lidx_2_lpos(end2_lidx);
        Float2D end2_gpos = cmap.get_glob_loc_cvt().lpos_2_gpos(end2_lpos);

        REQUIRE(cmap.check_line_state_map<linecheck::CHK_GLB_RAY>(start2_gpos, end2_gpos, static_cast<uint8_t>(eVoxel::OCCUPIED)) == false);
        REQUIRE(cmap.check_line_state_map<linecheck::CHK_GLB_SUP>(start2_gpos, end2_gpos, static_cast<uint8_t>(eVoxel::FREE)) == false);
        REQUIRE(cmap.check_line_state_map<linecheck::CHK_GLB_SUP>(start2_gpos, end2_gpos, static_cast<uint8_t>(eVoxel::UNOBSERVED)) == false);

        REQUIRE(cmap.check_line_state_map<linecheck::NON_LOC_RAY>(start2_lpos, end2_lpos, static_cast<uint8_t>(eVoxel::OCCUPIED)) == false);
        REQUIRE(cmap.check_line_state_map<linecheck::NON_LOC_SUP>(start2_lpos, end2_lpos, static_cast<uint8_t>(eVoxel::FREE)) == false);
        REQUIRE(cmap.check_line_state_map<linecheck::NON_LOC_SUP>(start2_lpos, end2_lpos, static_cast<uint8_t>(eVoxel::UNOBSERVED)) == false);
    }

    SECTION("distance map update")
    {
        // cmap.fill_all<map::DISTANCE, memType::HOST>(std::numeric_limits<float>::max());
        cmap.fill_all<map::STATE, memType::HOST>(static_cast<uint8_t>(eVoxel::UNKNOWN));
        cmap.set_map_withIdx<getset::ST_FAST_LOC>(Idx2D{2, 3}, static_cast<uint8_t>(eVoxel::OCCUPIED));
        cmap.set_map_withIdx<getset::ST_FAST_LOC>(Idx2D{3, 4}, static_cast<uint8_t>(eVoxel::OCCUPIED));

        // cmap.distance_map_update_withCPU();
        // printf("\n\nCPU\n");
        // for (int x = 0; x < 10; x++)
        // {
        //     printf("\n");
        //     for (int y = 0; y < 10; y++)
        //     {
        //         printf("%f ", cmap.get_map_withIdx<getset::DST_FAST_LOC>(Idx2D{x, y}));
        //     }
        // }

        cmap.fill_all<cuvoxmap::eMap::DISTANCE, cuvoxmap::eMemAllocType::DEVICE>(9999.0f);
        cmap.set_map_withIdx<getset::ST_FAST_LOC>(Idx2D{4, 5}, static_cast<uint8_t>(eVoxel::OCCUPIED));
        cmap.host_to_device<map::STATE>();
        cmap.distance_map_update_withGPU();
        cmap.device_to_host<map::DISTANCE>();

        printf("\n\nGPU\n");
        for (int x = 0; x < 10; x++)
        {
            printf("\n");
            for (int y = 0; y < 10; y++)
            {
                printf("%f ", cmap.get_map_withIdx<getset::DST_FAST_LOC>(Idx2D{x, y}));
            }
        }
    }
}