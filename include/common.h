#pragma once

#include <cstdint>

using indptr_t = int32_t;

template <typename T>
inline indptr_t FirstNotNaN(const T *data, indptr_t n)
{
    indptr_t i = 0;
    while (std::isnan(data[i]) && i < n)
    {
        ++i;
    }
    return i;
}

template <typename T>
inline indptr_t FirstNotNaN(const T *data, indptr_t n, T *out)
{
    indptr_t i = 0;
    while (std::isnan(data[i]) && i < n)
    {
        out[i++] = std::numeric_limits<T>::quiet_NaN();
    }
    return i;
}
