#pragma once

#include <string>
#include <sstream>

// ###### toString ######
template <typename T>
std::string toString(T val)
{
    std::ostringstream ss;
    ss << val;
    return ss.str();
}

template <>
std::string toString<cuvoxmap::Float2D>(cuvoxmap::Float2D val)
{
    std::ostringstream ss;
    ss << "Float2D(" << val[0] << ", " << val[1] << ')';
    return ss.str();
}

template <>
std::string toString<cuvoxmap::Float3D>(cuvoxmap::Float3D val)
{
    std::ostringstream ss;
    ss << "Float3D(" << val[0] << ", " << val[1] << ", " << val[2] << ')';
    return ss.str();
}

template <>
std::string toString<cuvoxmap::Double2D>(cuvoxmap::Double2D val)
{
    std::ostringstream ss;
    ss << "Double2D(" << val[0] << ", " << val[1] << ')';
    return ss.str();
}

template <>
std::string toString<cuvoxmap::Double3D>(cuvoxmap::Double3D val)
{
    std::ostringstream ss;
    ss << "Double3D(" << val[0] << ", " << val[1] << ", " << val[2] << ')';
    return ss.str();
}

template <>
std::string toString<cuvoxmap::Idx2D>(cuvoxmap::Idx2D val)
{
    std::ostringstream ss;
    ss << "Idx2D(" << val[0] << ", " << val[1] << ')';
    return ss.str();
}

template <>
std::string toString<cuvoxmap::Idx3D>(cuvoxmap::Idx3D val)
{
    std::ostringstream ss;
    ss << "Idx3D(" << val[0] << ", " << val[1] << ", " << val[2] << ')';
    return ss.str();
}

template <>
std::string toString<cuvoxmap::uIdx2D>(cuvoxmap::uIdx2D val)
{
    std::ostringstream ss;
    ss << "uIdx2D(" << val[0] << ", " << val[1] << ')';
    return ss.str();
}

template <>
std::string toString<cuvoxmap::uIdx3D>(cuvoxmap::uIdx3D val)
{
    std::ostringstream ss;
    ss << "uIdx3D(" << val[0] << ", " << val[1] << ", " << val[2] << ')';
    return ss.str();
}