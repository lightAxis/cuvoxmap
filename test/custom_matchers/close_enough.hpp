#pragma once

#include <cmath>

// ###### close enough ######
// Helper function to check if two floating point numbers are close enough
template <typename T>
bool close_enough(T lhs, T rhs)
{
    return std::abs(lhs - rhs) <= static_cast<T>(0.0001);
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
    return close_enough(lhs.normSquared(), rhs.normSquared());
}

template <>
bool close_enough<cuvoxmap::Float3D>(cuvoxmap::Float3D lhs, cuvoxmap::Float3D rhs)
{
    return close_enough(lhs.normSquared(), rhs.normSquared());
}

template <>
bool close_enough<cuvoxmap::Double2D>(cuvoxmap::Double2D lhs, cuvoxmap::Double2D rhs)
{
    return close_enough(lhs.normSquared(), rhs.normSquared());
}

template <>
bool close_enough<cuvoxmap::Double3D>(cuvoxmap::Double3D lhs, cuvoxmap::Double3D rhs)
{
    return close_enough(lhs.normSquared(), rhs.normSquared());
}

template <>
bool close_enough<cuvoxmap::Idx2D>(cuvoxmap::Idx2D lhs, cuvoxmap::Idx2D rhs)
{
    return close_enough(lhs.normSquared(), rhs.normSquared());
}

template <>
bool close_enough<cuvoxmap::Idx3D>(cuvoxmap::Idx3D lhs, cuvoxmap::Idx3D rhs)
{
    return close_enough(lhs.normSquared(), rhs.normSquared());
}

template <>
bool close_enough<cuvoxmap::uIdx2D>(cuvoxmap::uIdx2D lhs, cuvoxmap::uIdx2D rhs)
{
    return close_enough(static_cast<int64_t>(lhs.normSquared()), static_cast<int64_t>(rhs.normSquared()));
}

template <>
bool close_enough<cuvoxmap::uIdx3D>(cuvoxmap::uIdx3D lhs, cuvoxmap::uIdx3D rhs)
{
    return close_enough(static_cast<int64_t>(lhs.normSquared()), static_cast<int64_t>(rhs.normSquared()));
}