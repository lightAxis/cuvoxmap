
#include <cuvoxmap/utils/RayCaster.hpp>

#include "../custom_matchers/custom_matchers.hpp"
#include <iostream>

using namespace Catch::Matchers;
using namespace cuvoxmap;

TEST_CASE("RayCaster cpu")
{

    SECTION("basic")
    {
        RayCaster<float, 3> rayCaster = RayCaster3f{Float3D{1, 1, 1}, Float3D{1, 4, 5.01}, 0.25f};

        REQUIRE_THAT(rayCaster.get_StartPos(), FloatVecMatcher<Float3D>(Float3D{1, 1, 1}));
        REQUIRE_THAT(rayCaster.get_EndPos(), FloatVecMatcher<Float3D>(Float3D{1, 4, 5.01}));
        REQUIRE_THAT(rayCaster.get_Res(), Catch::Matchers::WithinAbs(0.25f, 1e-5f));
        REQUIRE(rayCaster.isFinished() == false);
        REQUIRE(rayCaster.get_linePts_count() == 22u);
        REQUIRE(rayCaster.get_all_pts().size() == 22u);
        REQUIRE(rayCaster.isFinished() == true);
    }

    SECTION("check extraction")
    {
        RayCaster<float, 3> rayCaster{Float3D{1, 1, 1}, Float3D{1, 4, 5.01}, 0.25f};

        auto pts = rayCaster.get_all_pts();

        rayCaster.reset_to_begin();
        std::vector<Float3D> vec;
        Float3D pt;
        while (rayCaster.get_next_pt(pt))
        {
            vec.push_back(pt);
        }

        REQUIRE_THAT(vec, Catch::Matchers::RangeEquals(pts, close_enough_s<Float3D>()));
    }

    SECTION("special, len 0, count 1")
    {
        RayCaster3f ray{Float3D{1, 1, 1}, Float3D{1, 1, 1}, 0.25f};

        REQUIRE(ray.isFinished() == false);
        REQUIRE(ray.get_linePts_count() == 1u);
        std::vector<Float3D> pts = ray.get_all_pts();
        std::vector<Float3D> ans;
        ans.push_back(Float3D{1, 1, 1});
        REQUIRE_THAT(pts, Catch::Matchers::RangeEquals(ans, close_enough_s<Float3D>()));
        REQUIRE(ray.isFinished() == true);
    }

    SECTION("special, count 2")
    {
        RayCaster3f ray{Float3D{1, 1, 1}, Float3D{1, 1.24, 1}, 0.25f};
        REQUIRE(ray.isFinished() == false);
        REQUIRE(ray.get_linePts_count() == 2u);
        std::vector<Float3D> pts = ray.get_all_pts();
        std::vector<Float3D> ans;
        ans.push_back(Float3D{1, 1, 1});
        ans.push_back(Float3D{1, 1.24, 1});
        REQUIRE_THAT(pts, Catch::Matchers::RangeEquals(ans, close_enough_s<Float3D>()));
        REQUIRE(ray.isFinished() == true);
    }
}
