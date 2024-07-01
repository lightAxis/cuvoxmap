#pragma once

#include <catch2/catch_template_test_macros.hpp>

// ###### FloatVecMatcher ######
template <typename T>
class FloatVecMatcher : public Catch::Matchers::MatcherBase<T>
{
    T val_;
    static_assert(std::is_same_v<T, cuvoxmap::Float2D> ||
                      std::is_same_v<T, cuvoxmap::Float3D>,
                  "FloatVecMatche only supports Float2D or Float3D");

public:
    FloatVecMatcher(const T &val) : val_(val) {}

    bool match(T const &in) const override
    {
        return close_enough(in, val_);
    }

    std::string describe() const override
    {
        return toString<T>(val_);
    }
};

template <typename T>
class IdxVecMatcher : public Catch::Matchers::MatcherBase<T>
{
    T val_;
    static_assert(std::is_same_v<T, cuvoxmap::Idx2D> ||
                      std::is_same_v<T, cuvoxmap::Idx3D>,
                  "IdxVecMatcher only supports Idx2D or Idx3D");

public:
    IdxVecMatcher(const T &val) : val_(val) {}

    bool match(T const &in) const override
    {
        return close_enough(in, val_);
    }

    std::string describe() const override
    {
        return toString<T>(val_);
    }
};

// Specialize StringMaker for cuvoxmap::Float2D
namespace Catch
{
    template <>
    struct StringMaker<cuvoxmap::Float2D>
    {
        static std::string convert(cuvoxmap::Float2D const &value)
        {
            return toString(value);
        }
    };

    template <>
    struct StringMaker<cuvoxmap::Float3D>
    {
        static std::string convert(cuvoxmap::Float3D const &value)
        {
            return toString(value);
        }
    };

    template <>
    struct StringMaker<cuvoxmap::Double2D>
    {
        static std::string convert(cuvoxmap::Double2D const &value)
        {
            return toString(value);
        }
    };

    template <>
    struct StringMaker<cuvoxmap::Double3D>
    {
        static std::string convert(cuvoxmap::Double3D const &value)
        {
            return toString(value);
        }
    };

    template <>
    struct StringMaker<cuvoxmap::Idx2D>
    {
        static std::string convert(cuvoxmap::Idx2D const &value)
        {
            return toString(value);
        }
    };

    template <>
    struct StringMaker<cuvoxmap::Idx3D>
    {
        static std::string convert(cuvoxmap::Idx3D const &value)
        {
            return toString(value);
        }
    };

}