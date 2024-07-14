#include <cuvoxmap/utils/Box.hpp>
#include "../custom_matchers/custom_matchers.hpp"

using namespace Catch::Matchers;
using namespace cuvoxmap;

TEST_CASE("Box_cpu")
{
    Float3D lower_b{0.f, 0.f, 0.f};
    Float3D upper_b{10.0f, 10.0f, 10.0f};

    Box3f box{lower_b, upper_b};

    SECTION("both inside")
    {
        Float3D P1{1, 1, 1};
        Float3D P2{2, 2, 2};
        Float3D P1_out = P1;
        Float3D P2_out = P2;

        Float3D P1_out_ref = P1;
        Float3D P2_out_ref = P2;

        const bool res = box.cutLine(P1, P2, P1_out, P2_out);

        REQUIRE_THAT(P1_out, FloatVecMatcher(P1_out_ref));
        REQUIRE_THAT(P2_out, FloatVecMatcher(P2_out_ref));
        REQUIRE(res == true);
    }

    SECTION("both inside boundary")
    {
        Float3D P1{0, 0, 0};
        Float3D P2{10, 10, 10};
        Float3D P1_out;
        Float3D P2_out;

        Float3D P1_out_ref = P1;
        Float3D P2_out_ref = P2;

        const bool res = box.cutLine(P1, P2, P1_out, P2_out);

        REQUIRE_THAT(P1_out, FloatVecMatcher(P1_out_ref));
        REQUIRE_THAT(P2_out, FloatVecMatcher(P2_out_ref));
        REQUIRE(res == true);
    }

    SECTION("P2 outside")
    {
        Float3D P1{5, 5, 5};
        Float3D P2{5, 15, 5};
        Float3D P1_out;
        Float3D P2_out;

        Float3D P1_out_ref = P1;
        Float3D P2_out_ref{5, 10, 5};

        const bool res = box.cutLine(P1, P2, P1_out, P2_out);

        REQUIRE_THAT(P1_out, FloatVecMatcher(P1_out_ref));
        REQUIRE_THAT(P2_out, FloatVecMatcher(P2_out_ref));
        REQUIRE(res == true);
    }

    SECTION("P2 outside multiple boundary")
    {
        Float3D P1{5, 5, 5};
        Float3D P2{-15, -15, -15};
        Float3D P1_out;
        Float3D P2_out;

        Float3D P1_out_ref = P1;
        Float3D P2_out_ref{0, 0, 0};

        const bool res = box.cutLine(P1, P2, P1_out, P2_out);

        REQUIRE_THAT(P1_out, FloatVecMatcher(P1_out_ref));
        REQUIRE_THAT(P2_out, FloatVecMatcher(P2_out_ref));
        REQUIRE(res == true);
    }

    SECTION("P1 outside")
    {
        const Float3D P1{5, 15, 5};
        const Float3D P2{5, 5, 5};
        Float3D P1_out;
        Float3D P2_out;

        const Float3D P1_out_ref{5, 10, 5};
        const Float3D P2_out_ref = P2;

        const bool res = box.cutLine(P1, P2, P1_out, P2_out);

        REQUIRE_THAT(P1_out, FloatVecMatcher(P1_out_ref));
        REQUIRE_THAT(P2_out, FloatVecMatcher(P2_out_ref));
        REQUIRE(res == true);
    }

    SECTION("P1 outside multiple boundary")
    {
        const Float3D P1{15, -5, 15};
        const Float3D P2{5, 5, 5};
        Float3D P1_out;
        Float3D P2_out;

        const Float3D P1_out_ref{10, 0, 10};
        const Float3D P2_out_ref = P2;

        const bool res = box.cutLine(P1, P2, P1_out, P2_out);

        REQUIRE_THAT(P1_out, FloatVecMatcher(P1_out_ref));
        REQUIRE_THAT(P2_out, FloatVecMatcher(P2_out_ref));
        REQUIRE(res == true);
    }

    SECTION("Both outside, two point")
    {
        const Float3D P1{15, 5, 5};
        const Float3D P2{-15, 5, 5};
        Float3D P1_out;
        Float3D P2_out;

        const Float3D P1_out_ref{0, 5, 5};
        const Float3D P2_out_ref{10, 5, 5};

        const bool res = box.cutLine(P1, P2, P1_out, P2_out);

        REQUIRE_THAT(P1_out, FloatVecMatcher(P1_out_ref));
        REQUIRE_THAT(P2_out, FloatVecMatcher(P2_out_ref));
        REQUIRE(res == true);
    }

    SECTION("Both outside, two point, multiple")
    {
        const Float3D P1{15, 15, 15};
        const Float3D P2{-15, -15, -15};
        Float3D P1_out;
        Float3D P2_out;

        const Float3D P1_out_ref{0, 0, 0};
        const Float3D P2_out_ref{10, 10, 10};

        const bool res = box.cutLine(P1, P2, P1_out, P2_out);

        REQUIRE_THAT(P1_out, FloatVecMatcher(P1_out_ref));
        REQUIRE_THAT(P2_out, FloatVecMatcher(P2_out_ref));
        REQUIRE(res == true);
    }

    SECTION("Both outside, one point, edge case")
    {
        const Float3D P1{0, 0, 15};
        const Float3D P2{0, 0, -5};
        Float3D P1_out;
        Float3D P2_out;

        const Float3D P1_out_ref{0, 0, 0};
        const Float3D P2_out_ref{0, 0, 10};

        const bool res = box.cutLine(P1, P2, P1_out, P2_out);

        REQUIRE_THAT(P1_out, FloatVecMatcher(P1_out_ref));
        REQUIRE_THAT(P2_out, FloatVecMatcher(P2_out_ref));
        REQUIRE(res == true);
    }

    SECTION("Both outside, one point, edge case, multiple")
    {
        const Float3D P1{-5, 5, 5};
        const Float3D P2{5, -5, 5};
        Float3D P1_out;
        Float3D P2_out;

        const Float3D P1_out_ref{0, 0, 5};
        const Float3D P2_out_ref{0, 0, 5};

        const bool res = box.cutLine(P1, P2, P1_out, P2_out);

        REQUIRE_THAT(P1_out, FloatVecMatcher(P1_out_ref));
        REQUIRE_THAT(P2_out, FloatVecMatcher(P2_out_ref));
        REQUIRE(res == true);
    }

    SECTION("Both outside, one point, corner multiple")
    {
        const Float3D P1{5, 5, 15};
        const Float3D P2{15, 15, 5};
        Float3D P1_out;
        Float3D P2_out;

        const Float3D P1_out_ref{10, 10, 10};
        const Float3D P2_out_ref{10, 10, 10};

        const bool res = box.cutLine(P1, P2, P1_out, P2_out);

        REQUIRE_THAT(P1_out, FloatVecMatcher(P1_out_ref));
        REQUIRE_THAT(P2_out, FloatVecMatcher(P2_out_ref));
        REQUIRE(res == true);
    }

    SECTION("Both outside, zero point")
    {
        const Float3D P1{15, 15, 15};
        const Float3D P2{15, 15, 11};
        Float3D P1_out;
        Float3D P2_out;

        const Float3D P1_out_ref{10, 10, 10};
        const Float3D P2_out_ref{10, 10, 10};

        const bool res = box.cutLine(P1, P2, P1_out, P2_out);

        REQUIRE(res == false);
    }

    SECTION("Very Close, both inside")
    {
        const Float3D P1{5, 5, 5};
        const Float3D P2{5, 5, 4.9999999f};

        Float3D P1_out;
        Float3D P2_out;

        const Float3D P1_out_ref{5, 5, 5};
        const Float3D P2_out_ref{5, 5, 4.9999999f};

        const bool res = box.cutLine(P1, P2, P1_out, P2_out);

        REQUIRE_THAT(P1_out, FloatVecMatcher(P1_out_ref));
        REQUIRE_THAT(P2_out, FloatVecMatcher(P2_out_ref));
        REQUIRE(res == true);
    }

    SECTION("Very close, p1 inside")
    {
        const Float3D P1{5, 5, 10};
        const Float3D P2{5, 5, 10.00000001f};

        Float3D P1_out;
        Float3D P2_out;

        const Float3D P1_out_ref{5, 5, 10};
        const Float3D P2_out_ref{5, 5, 10};

        const bool res = box.cutLine(P1, P2, P1_out, P2_out);

        REQUIRE_THAT(P1_out, FloatVecMatcher(P1_out_ref));
        REQUIRE_THAT(P2_out, FloatVecMatcher(P2_out_ref));
        REQUIRE(res == true);
    }

    SECTION("Very close, p2 outside")
    {
        const Float3D P1{5, 5, 10.00000001f};
        const Float3D P2{5, 5, 10};

        Float3D P1_out;
        Float3D P2_out;

        const Float3D P1_out_ref{5, 5, 10};
        const Float3D P2_out_ref{5, 5, 10};

        const bool res = box.cutLine(P1, P2, P1_out, P2_out);

        REQUIRE_THAT(P1_out, FloatVecMatcher(P1_out_ref));
        REQUIRE_THAT(P2_out, FloatVecMatcher(P2_out_ref));
        REQUIRE(res == true);
    }

    SECTION("Very close, both outside")
    {
        const Float3D P1{15, 15, 10.00000001f};
        const Float3D P2{15, 15, 10.00000002f};

        Float3D P1_out;
        Float3D P2_out;

        const Float3D P1_out_ref{5, 5, 10};
        const Float3D P2_out_ref{5, 5, 10};

        const bool res = box.cutLine(P1, P2, P1_out, P2_out);

        REQUIRE(res == false);
    }
}