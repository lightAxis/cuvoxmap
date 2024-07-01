#include <cuvoxmap/utils/GlobLocalIdxCvt.hpp>
#include "../custom_matchers/custom_matchers.hpp"

using namespace Catch::Matchers;
using namespace cuvoxmap;

TEST_CASE("GlobLocalIdxCvt cpu")
{
    uIdx3D local_size{10, 7, 20};
    GlobLocalCvt<float, 3> glc{Float3D{0.7f, 0.75f, 0.5f}, 0.5f, local_size};

    SECTION("basic")
    {
        Float3D origin{0.7f, 0.75f, 0.5f};

        Float3D gpos{1.2f, 1.25f, 1.0f};
        Float3D lpos{1.2f - 0.7f, 1.25f - 0.75f, 1.0f - 0.5f};
        Idx3D gidx{2, 2, 2};
        Idx3D lidx{1, 1, 1};

        Float3D gpos2 = Float3D{0.5f / 2.f + 0.5f * 2, 0.5f / 2.f + 0.5f * 2, 0.5f / 2.f + 0.5f * 2};
        Float3D lpos2 = gpos2 - origin;

        REQUIRE_THAT(glc.get_map_origin(), FloatVecMatcher(origin));
        REQUIRE_THAT(glc.get_resolution(), WithinAbs(0.5f, 1e-6f));

        REQUIRE_THAT(glc.gpos_2_lpos(gpos), FloatVecMatcher(lpos));
        REQUIRE_THAT(glc.gpos_2_lidx(gpos), IdxVecMatcher(lidx));
        REQUIRE_THAT(glc.gpos_2_lpos(gpos), FloatVecMatcher(lpos));

        REQUIRE_THAT(glc.gidx_2_gpos(gidx), FloatVecMatcher(gpos2));
        REQUIRE_THAT(glc.gidx_2_lidx(gidx), IdxVecMatcher(lidx));
        REQUIRE_THAT(glc.gidx_2_lpos(gidx), FloatVecMatcher(lpos2));

        REQUIRE_THAT(glc.lpos_2_gidx(lpos), IdxVecMatcher(gidx));
        REQUIRE_THAT(glc.lpos_2_gpos(lpos), FloatVecMatcher(gpos));
        REQUIRE_THAT(glc.lpos_2_lidx(lpos), IdxVecMatcher(lidx));

        REQUIRE_THAT(glc.lidx_2_gidx(lidx), IdxVecMatcher(gidx));
        REQUIRE_THAT(glc.lidx_2_gpos(lidx), FloatVecMatcher(gpos2));
        REQUIRE_THAT(glc.lidx_2_lpos(lidx), FloatVecMatcher(lpos2));
    }

    SECTION("itself")
    {
        Float3D origin{0.75f, -2.0001f, 4.000001f};
        const float res = 0.25f;
        uIdx3D local_size{10, 7, 20};
        GlobLocalCvt<float, 3> glc{origin, res, local_size};

        const Float3D gpos{0.0f, -2.0f, 4.0f};
        const Idx3D gidx = glc.gpos_2_gidx(gpos);
        const Float3D lpos = glc.gpos_2_lpos(gpos);
        const Idx3D lidx = glc.gpos_2_lidx(gpos);
        const Float3D gpos2 = glc.gidx_2_gpos(gidx);
        const Float3D lpos2 = glc.gidx_2_lpos(gidx);

        REQUIRE_THAT(glc.get_map_origin(), FloatVecMatcher(origin));
        REQUIRE_THAT(glc.get_resolution(), WithinAbs(res, 1e-6f));

        REQUIRE_THAT(glc.gpos_2_lpos(gpos), FloatVecMatcher(lpos));
        REQUIRE_THAT(glc.gpos_2_lidx(gpos), IdxVecMatcher(lidx));
        REQUIRE_THAT(glc.gpos_2_lpos(gpos), FloatVecMatcher(lpos));

        REQUIRE_THAT(glc.gidx_2_gpos(gidx), FloatVecMatcher(gpos2));
        REQUIRE_THAT(glc.gidx_2_lidx(gidx), IdxVecMatcher(lidx));
        REQUIRE_THAT(glc.gidx_2_lpos(gidx), FloatVecMatcher(lpos2));

        REQUIRE_THAT(glc.lpos_2_gidx(lpos), IdxVecMatcher(gidx));
        REQUIRE_THAT(glc.lpos_2_gpos(lpos), FloatVecMatcher(gpos));
        REQUIRE_THAT(glc.lpos_2_lidx(lpos), IdxVecMatcher(lidx));

        REQUIRE_THAT(glc.lidx_2_gidx(lidx), IdxVecMatcher(gidx));
        REQUIRE_THAT(glc.lidx_2_gpos(lidx), FloatVecMatcher(gpos2));
        REQUIRE_THAT(glc.lidx_2_lpos(lidx), FloatVecMatcher(lpos2));
    }
}