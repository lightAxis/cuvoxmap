
#include <cuvoxmap/utils/SuperCover3D.hpp>

#include "../custom_matchers/custom_matchers.hpp"
#include <iostream>

using namespace Catch::Matchers;
using namespace cuvoxmap;

TEST_CASE("SuperCoverLine 3D cpu")
{
    SECTION("basic")
    {
        SuperCoverLine3D<float, int32_t> line{Float3D{1, 1, 1}, Float3D{1, 4, 5.01}, 0.25f};

        Idx3D idx3d;
        std::vector<Idx3D> idxs2;
        while (line.get_next_idx(idx3d) == true)
        {
            // std::cout << "idx3d: " << idx3d[0] << " " << idx3d[1] << " " << idx3d[2] << std::endl;
            idxs2.push_back(idx3d);
        }

        std::vector<Idx3D> idxs;
        idxs.push_back(Idx3D{4, 4, 4});
        idxs.push_back(Idx3D{4, 4, 5});
        idxs.push_back(Idx3D{4, 5, 5});
        idxs.push_back(Idx3D{4, 5, 6});
        idxs.push_back(Idx3D{4, 6, 6});
        idxs.push_back(Idx3D{4, 6, 7});
        idxs.push_back(Idx3D{4, 6, 8});
        idxs.push_back(Idx3D{4, 7, 8});
        idxs.push_back(Idx3D{4, 7, 9});
        idxs.push_back(Idx3D{4, 8, 9});
        idxs.push_back(Idx3D{4, 8, 10});
        idxs.push_back(Idx3D{4, 9, 10});
        idxs.push_back(Idx3D{4, 9, 11});
        idxs.push_back(Idx3D{4, 9, 12});
        idxs.push_back(Idx3D{4, 10, 12});
        idxs.push_back(Idx3D{4, 10, 13});
        idxs.push_back(Idx3D{4, 11, 13});
        idxs.push_back(Idx3D{4, 11, 14});
        idxs.push_back(Idx3D{4, 12, 14});
        idxs.push_back(Idx3D{4, 12, 15});
        idxs.push_back(Idx3D{4, 12, 16});
        idxs.push_back(Idx3D{4, 13, 16});
        idxs.push_back(Idx3D{4, 13, 17});
        idxs.push_back(Idx3D{4, 14, 17});
        idxs.push_back(Idx3D{4, 14, 18});
        idxs.push_back(Idx3D{4, 15, 18});
        idxs.push_back(Idx3D{4, 15, 19});
        idxs.push_back(Idx3D{4, 15, 20});
        idxs.push_back(Idx3D{4, 16, 20});
        idxs.push_back(Idx3D{4, 16, 21});

        REQUIRE_THAT(idxs, RangeEquals(idxs2, close_enough_s<Idx3D>()));
    }
}