#include <cuvoxmap/utils/MovablePtr.hpp>
#include <cuvoxmap/base/MapImpl.hpp>

namespace cuvoxmap
{
    template <typename T>
    MovablePtr<T>::MovablePtr(const MovablePtr &other)
    {
        if (other.ptr_ != nullptr)
        {
            ptr_ = new T(*other.ptr_);
        }
    }

    template <typename T>
    MovablePtr<T> &MovablePtr<T>::operator=(const MovablePtr &other)
    {
        if (this != &other)
        {
            delete ptr_;
            if (other.ptr_ != nullptr)
            {
                ptr_ = new T(*other.ptr_);
            }
        }
        return *this;
    }

    template <typename T>
    MovablePtr<T>::MovablePtr(MovablePtr &&other) noexcept : ptr_(other.ptr_)
    {
        other.ptr_ = nullptr;
    }

    template <typename T>
    MovablePtr<T> &MovablePtr<T>::operator=(MovablePtr &&other) noexcept
    {
        if (this != &other)
        {
            delete ptr_;
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    template <typename T>
    MovablePtr<T>::~MovablePtr()
    {
        delete ptr_;
    }

    // instantiation of MovablePtr for MapImpl
    template class MovablePtr<MapImpl<uint8_t, 1>>;
    template class MovablePtr<MapImpl<uint8_t, 2>>;
    template class MovablePtr<MapImpl<uint8_t, 3>>;
    template class MovablePtr<MapImpl<uint16_t, 1>>;
    template class MovablePtr<MapImpl<uint16_t, 2>>;
    template class MovablePtr<MapImpl<uint16_t, 3>>;
    template class MovablePtr<MapImpl<int, 1>>;
    template class MovablePtr<MapImpl<int, 2>>;
    template class MovablePtr<MapImpl<int, 3>>;
    template class MovablePtr<MapImpl<float, 1>>;
    template class MovablePtr<MapImpl<float, 2>>;
    template class MovablePtr<MapImpl<float, 3>>;
    template class MovablePtr<MapImpl<double, 1>>;
    template class MovablePtr<MapImpl<double, 2>>;
    template class MovablePtr<MapImpl<double, 3>>;
}