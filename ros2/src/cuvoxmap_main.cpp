#include <rclcpp/rclcpp.hpp>
#include <cuvoxmap/cuvoxmap_node.hpp>

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<cuvoxmapNode>(false));
    rclcpp::shutdown();
    return 0;
}