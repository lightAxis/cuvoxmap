#include <cuvoxmap/cuvoxmap.hpp>

#include "../custom_matchers/custom_matchers.hpp"

TEST_CASE("Cuvoxmap_cpu")
{
    cuvoxmap::cuvoxmap2D::init_s init;
    init.x_axis_len = 10;
    init.y_axis_len = 10;
    init.resolution = 0.5f;
    cuvoxmap::cuvoxmap2D cmap{init};

    using check = cuvoxmap::eCheck;
    using map = cuvoxmap::eMap;
    cmap.set_map_withGlobIdx<map::PROBABILITY, check::NONE>(cuvoxmap::Idx2D{0, 0}, 0.f);
    cmap.set_map_withGlobIdx<map::PROBABILITY>(cuvoxmap::Idx2D{0, 0}, 0.f);
}