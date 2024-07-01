#include <catch2/catch_test_macros.hpp>
#include "../custom_matchers/device_testmacros.cuh"

#include <cuvoxmap/utils/Vector.hpp>

int get_idx(const cuvoxmap::Vector<int, 2> &vec, uint8_t idx)
{
    return vec[idx];
}

TEST_CASE("Vectors host", "utils")
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

__global__ void vector_test(cuvoxmap::Vector<int, 2> vec)
{
    if (!CUSTOM_TEST_KERNEL::KERNEL_TEST(vec[0], 1))
        return;
    if (!CUSTOM_TEST_KERNEL::KERNEL_TEST(vec[1], 2))
        return;

    cuvoxmap::Vector<int, 2> vec2 = vec;
    if (!CUSTOM_TEST_KERNEL::KERNEL_TEST(vec2[0], 1))
        return;
    if (!CUSTOM_TEST_KERNEL::KERNEL_TEST(vec2[1], 2))
        return;
}

__global__ void vector_test_float(cuvoxmap::Vector<float, 2> vec)
{
    if (!CUSTOM_TEST_KERNEL::KERNEL_TEST(vec[0], 1.1f))
        return;
    if (!CUSTOM_TEST_KERNEL::KERNEL_TEST(vec[1], 2.2f))
        return;
}

TEST_CASE("Vectors device", "utils")
{
    cuvoxmap::Vector<int, 2> v{1, 2};
    vector_test<<<1, 1>>>(v);
    cudaDeviceSynchronize();

    cuvoxmap::Vector<float, 2> v2{1.1f, 2.2f};
    vector_test_float<<<1, 1>>>(v2);
    cudaDeviceSynchronize();
}
