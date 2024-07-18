#include <rclcpp/rclcpp.hpp>

#include <cuvoxmap/base/MapAllocator.hpp>

class cuvoxmapNode : public rclcpp::Node
{
public:
    cuvoxmapNode() : Node("cuvoxmap_node"),
                     map_allocator_(cuvoxmap::uIdx2D{
                                        10,
                                        20,
                                    },
                                    cuvoxmap::eMemAllocType::HOST_AND_DEVICE)
    {
    }

private:
    cuvoxmap::MapAllocator<float, 2> map_allocator_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    rclcpp::spin(std::make_shared<cuvoxmapNode>());
    rclcpp::shutdown();
    return 0;
}