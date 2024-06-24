#pragma once

#include "ArrayAllocator.hpp"
#include "../utils/Vector.hpp"

namespace cuvoxmap
{
    template <typename T, uint8_t Dim>
    class MapImpl
    {
    public:
        MapImpl() = default;
        MapImpl(const Vector<uint32_t, Dim> &axis_sizes);

        ~MapImpl() = default;
        inline T *get_host_data() { return array_.get_host_ptr(); }
        inline T *get_device_data() { return array_.get_device_ptr(); }
        inline Vector<uint32_t, Dim> get_axis_sizes() const { return axis_sizes_; }

        void host_to_device();
        void device_to_host();

    private:
        // Some data members
        ArrayAllocator<T> array_;
        Vector<uint32_t, Dim> axis_sizes_;
    };
}