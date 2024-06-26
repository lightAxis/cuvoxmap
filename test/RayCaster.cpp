
#include <cuvoxmap/utils/RayCaster.hpp>

#include "custom_matchers/custom_matchers.hpp"
#include <iostream>

// CHECK_THAT(vec, Catch::Matchers::RangeEquals(vec2, close_enough_s<cuvoxmap::Float2D>()));
// CHECK_THAT(f1, FloatVecMatcher<cuvoxmap::Float2D>(f2));

TEST_CASE("RayCaster cpu")
{
    cuvoxmap::RayCaster<float, 2> rayCaster{cuvoxmap::Float2D{1.5f, 0.5f}, cuvoxmap::Float2D{0, -1}, 0.5f};

    auto adfsa = rayCaster.get_all_pts();

    rayCaster.reset_to_begin();
    std::vector<cuvoxmap::Float2D> vec;
    cuvoxmap::Float2D pt;
    while (rayCaster.get_next_pt(pt))
    {
        vec.push_back(pt);
    }

    REQUIRE_THAT(vec, Catch::Matchers::RangeEquals(adfsa, close_enough_s<cuvoxmap::Float2D>()));
}
