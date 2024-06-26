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
