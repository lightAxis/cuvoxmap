#include <cuvoxmap/cuvoxmap.hpp>

#include "../custom_matchers/custom_matchers.hpp"

using namespace cuvoxmap;

TEST_CASE("Cuvoxmap_cpu")
{
    cuvoxmap::cuvoxmap2D::init_s init;
    init.x_axis_len = 10;
    init.y_axis_len = 10;
    init.resolution = 0.5f;
    cuvoxmap::cuvoxmap2D cmap{init};
    cmap.set_origin(Float2D{1.2f, 1.5f});

    using check = cuvoxmap::eCheck;
    using map = cuvoxmap::eMap;
    using frame = cuvoxmap::eFrame;

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
        cmap.fill<map::DISTANCE>(30.f);
        cmap.fill<map::PROBABILITY>(0.5f);
        cmap.fill<map::STATE>(static_cast<uint8_t>(eVoxel::UNOBSERVED));

        REQUIRE(cmap.get_map_withIdx<getset::DST_FAST_LOC>(Idx2D{0, 0}) == 30.f);
        REQUIRE(cmap.get_map_withIdx<getset::PRB_FAST_LOC>(Idx2D{0, 0}) == 0.5f);
        REQUIRE(cmap.get_map_withIdx<getset::ST_FAST_LOC>(Idx2D{0, 0}) == static_cast<uint8_t>(eVoxel::UNOBSERVED));
    }
}