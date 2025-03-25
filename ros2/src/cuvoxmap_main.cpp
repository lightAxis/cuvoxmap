#include <cuvoxmap/cuvoxmap_node.hpp>

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    cuvoxmap::cuvoxmap2D::init_s init;
    init.resolution = 0.25f;
    init.use_gpu = false;
    init.x_axis_len = 10;
    init.y_axis_len = 20;
    rclcpp::spin(std::make_shared<cuvoxmapNode>(init));
    rclcpp::shutdown();
    return 0;
}