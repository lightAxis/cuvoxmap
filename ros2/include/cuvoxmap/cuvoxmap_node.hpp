#pragma once

#include <cuvoxmap/cuvoxmap.hpp>

#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>

class cuvoxmapNode : public rclcpp::Node
{
public:
    cuvoxmapNode(cuvoxmap::cuvoxmap2D::init_s init) : Node("cuvoxmap_node"),
                                                      map_(init)
    {
        map_.set_origin(cuvoxmap::Float2D{1.2f, 1.5f});
        map_.set_map_withIdx<cuvoxmap::getset::ST_FAST_LOC>(cuvoxmap::Idx2D{2, 3}, static_cast<uint8_t>(cuvoxmap::eVoxel::OCCUPIED));
        map_.set_map_withIdx<cuvoxmap::getset::ST_FAST_LOC>(cuvoxmap::Idx2D{3, 4}, static_cast<uint8_t>(cuvoxmap::eVoxel::OCCUPIED));
        // map_.set_map_withIdx<cuvoxmap::getset::ST_FAST_LOC>(cuvoxmap::Idx2D{5, 10}, static_cast<uint8_t>(cuvoxmap::eVoxel::OCCUPIED));
        // map_.set_map_withIdx<cuvoxmap::getset::ST_FAST_LOC>(cuvoxmap::Idx2D{1, 2}, static_cast<uint8_t>(cuvoxmap::eVoxel::OCCUPIED));
        auto aa = map_.get_glob_loc_cvt();
        const bool buildWithDevice = init.use_gpu;
        if (buildWithDevice)
        {
            map_.fill_all<cuvoxmap::eMap::DISTANCE, cuvoxmap::eMemAllocType::DEVICE>(9999.0f);
            map_.host_to_device<cuvoxmap::eMap::STATE>();
            map_.distance_map_update_withGPU();
            map_.device_to_host<cuvoxmap::eMap::DISTANCE>();
        }
        else
        {
            map_.fill_all<cuvoxmap::eMap::DISTANCE, cuvoxmap::eMemAllocType::HOST>(9999.0f);
            map_.distance_map_update_withCPU();
        }

        // TODO: implement the host_to_device, distance_map_update with GPU for CPU only build. but, throw std::runtime error when function called.
        // TODO: now implementation is skipped for CPU only build. causing build failure when build with device is false.

        // cuvoxmap::MapAccessorHost<float, 2> accessor{map_allocator_.get_mapData()};
        // accessor.set_value(cuvoxmap::uIdx2D{5, 10}, 1.0f);
        // accessor.set_value(cuvoxmap::uIdx2D{1, 2}, 2.0f);
        // accessor.set_value(cuvoxmap::uIdx2D{2, 4}, 0.5f);

        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("cuvoxmap_disp", 10);
        timer_ = this->create_wall_timer(std::chrono::milliseconds(100), std::bind(&cuvoxmapNode::map_display_cb, this));
    }

private:
    cuvoxmap::MapAllocator<float, 2> map_allocator_;
    cuvoxmap::cuvoxmap2D map_;

    void map_display_cb()
    {
        auto cvt = map_.get_glob_loc_cvt();
        auto idxing = cuvoxmap::Indexing<2>(cvt.get_local_size());

        sensor_msgs::msg::PointCloud2 msg;
        msg.header.frame_id = "map";
        msg.header.stamp = this->now();
        msg.height = 1;
        msg.width = cvt.get_local_size().mul_sum();

        msg.fields.reserve(4);
        sensor_msgs::msg::PointField field;
        field.count = 1;
        field.datatype = sensor_msgs::msg::PointField::FLOAT32;

        field.offset = 0;
        field.name = "x";
        msg.fields.push_back(field);

        field.offset = 4;
        field.name = "y";
        msg.fields.push_back(field);

        field.offset = 8;
        field.name = "z";
        msg.fields.push_back(field);

        field.offset = 12;
        field.name = "intensity";
        msg.fields.push_back(field);

        msg.is_bigendian = false;
        msg.point_step = 16;
        msg.row_step = msg.point_step * msg.width;
        msg.is_dense = true;
        msg.data.resize(msg.row_step * msg.height);

        for (size_t i = 0; i < msg.width; i++)
        {
            float *ptr = reinterpret_cast<float *>(msg.data.data() + i * msg.point_step);
            const cuvoxmap::Idx2D idx = (idxing.split(i)).cast<int>();
            const cuvoxmap::Float2D pos = cvt.lidx_2_gpos(idx);
            ptr[0] = pos[0];
            ptr[1] = pos[1];
            ptr[2] = static_cast<float>(0);
            ptr[3] = map_.get_map_withIdx<cuvoxmap::getset::DST_FAST_LOC>(idx);

            // std::cout << "Point " << i << ": " << ptr[0] << ", " << ptr[1] << ", " << ptr[2] << ", " << ptr[3] << std::endl;
        }

        publisher_->publish(msg);

        RCLCPP_INFO_STREAM(this->get_logger(), "Publishing point cloud" << msg.height);
    }

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
};