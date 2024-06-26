#pragma once

#include <cmath>

// ###### close enough ######
// Helper function to check if two floating point numbers are close enough
template <typename T>
bool close_enough(T lhs, T rhs)
{
    return std::abs(lhs - rhs) < 0.0001;
}

template <typename T>
struct close_enough_s
{
    bool operator()(const T &lhs, const T &rhs) const
    {
        return close_enough<T>(lhs, rhs);
    }
};

// Specialization for cuvoxmap::Float2D
template <>
bool close_enough<cuvoxmap::Float2D>(cuvoxmap::Float2D lhs, cuvoxmap::Float2D rhs)
{
    return close_enough(lhs.norm(), rhs.norm());
}

template <>
bool close_enough<cuvoxmap::Float3D>(cuvoxmap::Float3D lhs, cuvoxmap::Float3D rhs)
{
    return close_enough(lhs.norm(), rhs.norm());
}