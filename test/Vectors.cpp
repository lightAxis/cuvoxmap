#include <catch2/catch_test_macros.hpp>
#include <cuvoxmap/utils/Vector.hpp>

int get_idx(const cuvoxmap::Vector<int, 2> &vec, uint8_t idx)
{
    return vec[idx];
}

TEST_CASE("Vectors cpu", "utils")
{
    cuvoxmap::Vector<int, 2> v1{1, 2};

    SECTION("Vector test")
    {
        REQUIRE(v1[0] == 1);
        REQUIRE(v1[1] == 2);
        REQUIRE(v1.size() == 2);
    }

    SECTION("Vector operator= copy")
    {
        cuvoxmap::Vector<int, 2> v1_copy = v1;
        REQUIRE(v1_copy[0] == 1);
        REQUIRE(v1_copy[1] == 2);
        REQUIRE(v1_copy.size() == 2);

        cuvoxmap::Vector<int, 2> v2_copy2;
        v2_copy2 = v1;
        REQUIRE(v2_copy2[0] == 1);
        REQUIRE(v2_copy2[1] == 2);
        REQUIRE(v2_copy2.size() == 2);

        cuvoxmap::Vector<int, 2> v3_copy3{4, 5};
        REQUIRE(v1 == v1_copy);
        REQUIRE(v3_copy3 != v1_copy);
    }

    SECTION("Vector edit ")
    {
        v1[0] = 3;
        v1[1] = 4;
        REQUIRE(v1[0] == 3);
        REQUIRE(v1[1] == 4);
        REQUIRE(v1.size() == 2);
    }

    SECTION("Vector function")
    {
        REQUIRE(get_idx(v1, 0) == 1);
        REQUIRE(get_idx(v1, 1) == 2);
    }

    cuvoxmap::Vector<float, 2> v2{1.1f, 2.2f};

    SECTION("Vector float")
    {
        REQUIRE(v2[0] == 1.1f);
        REQUIRE(v2[1] == 2.2f);
        REQUIRE(v2.size() == 2);
    }

    SECTION("Vector float edit")
    {
        v2[0] = 3.3f;
        v2[1] = 4.4f;
        REQUIRE(v2[0] == 3.3f);
        REQUIRE(v2[1] == 4.4f);
        REQUIRE(v2.size() == 2);
    }
}