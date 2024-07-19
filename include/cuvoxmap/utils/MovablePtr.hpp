#pragma once

namespace cuvoxmap
{

    template <typename T>
    class MovablePtr
    {
    public:
        MovablePtr() = default;
        explicit MovablePtr(T *ptr) : ptr_(ptr) {}
        MovablePtr(const MovablePtr &other);
        MovablePtr &operator=(const MovablePtr &other);
        MovablePtr(MovablePtr &&other) noexcept;
        MovablePtr &operator=(MovablePtr &&other) noexcept;
        ~MovablePtr();

        inline T *get() const { return ptr_; }
        T *operator->() const { return ptr_; }
        T &operator*() const { return *ptr_; }

        // == nullptr
        friend bool operator==(const MovablePtr &lhs, decltype(nullptr))
        {
            return lhs.ptr_ == nullptr;
        }

        friend bool operator==(decltype(nullptr), const MovablePtr &rhs)
        {
            return rhs.ptr_ == nullptr;
        }

        // != nullptr
        friend bool operator!=(const MovablePtr &lhs, decltype(nullptr))
        {
            return lhs.ptr_ != nullptr;
        }

        friend bool operator!=(decltype(nullptr), const MovablePtr &rhs)
        {
            return rhs.ptr_ != nullptr;
        }

    private:
        T *ptr_{nullptr};
    };
}