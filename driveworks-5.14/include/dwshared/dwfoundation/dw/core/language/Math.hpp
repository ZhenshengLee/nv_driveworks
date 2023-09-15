/////////////////////////////////////////////////////////////////////////////////////////
//
// Notice
// ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS" NVIDIA MAKES
// NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//
// NVIDIA CORPORATION & AFFILIATES assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of NVIDIA CORPORATION & AFFILIATES. No third party distribution is allowed unless
// expressly authorized by NVIDIA. Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// NVIDIA CORPORATION & AFFILIATES products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// NVIDIA CORPORATION & AFFILIATES.
//
// SPDX-FileCopyrightText: Copyright (c) 2015-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef DW_CORE_LANGUAGE_CMATH_HPP_
#define DW_CORE_LANGUAGE_CMATH_HPP_

#include "BasicTypes.hpp"
#include "Limits.hpp"
#include <type_traits>
#include <cfenv>

namespace dw
{
namespace core
{

/**
 * \defgroup compare_group Compare Group of Functions
 * @{
 */

/**
 * @brief Same as std::min but constexpr and CUDA-friendly
 *
 * @note constexpr is required for compile-time block-size computations
 * @note @p a and @p b must be comparable and return value of this function
 *       is in type @p Ta
 *
 * @tparam Ta Data type of @p a
 * @tparam Tb Data type of @p b
 *
 * @param a input value
 * @param b input value
 * @return smaller value between a and b.
 */
template <typename Ta, typename Tb>
CUDA_BOTH_INLINE constexpr auto min(Ta const a, Tb const b) -> Ta
{
    return a < b ? a : b;
}

/**
 * @brief Same as std::max but constexpr and CUDA-friendly
 *
 * @note constexpr is required for compile-time block-size computations
 * @note @p a and @p b must be comparable and return value of this function
 *       is in type @p Ta
 *
 * @tparam Ta Data type of @p a
 * @tparam Tb Data type of @p b
 *
 * @param a input value
 * @param b input value
 * @return Larger value between a and b.
 */
template <typename Ta, typename Tb>
CUDA_BOTH_INLINE constexpr auto max(Ta const a, Tb const b) -> Ta
{
    return a > b ? a : b;
}

/**
 * @brief Check if given float32_t type @p v is Not-A-Number(NAN)
 * @param v input number
 * @return true if @p is Not-A-Number, otherwise false
 */
CUDA_BOTH_INLINE bool isnan(float32_t const v)
{
#ifdef __CUDA_ARCH__
    return v != v;
#else
    return std::isnan(v);
#endif
}

/**
 * @brief Check if given float64_t type @p v is Not-A-Number(NAN)
 * @param v input number
 * @return true if @p is Not-A-Number, otherwise false
 */
CUDA_BOTH_INLINE bool isnan(float64_t const v)
{
#ifdef __CUDA_ARCH__
    return v != v;
#else
    return std::isnan(v);
#endif
}

/**
 * @brief Check if given non-float type value is Not-A-Number(NAN)
 * @tparam T data type of @p v
 * @return Always false.
 */
template <class T>
CUDA_BOTH_INLINE bool isnan(T)
{
    return false;
}

/**
 * @brief Check if given float32_t type @p v is infinite
 * @param v input number
 * @return true if @p is infinite, otherwise false
 */
CUDA_BOTH_INLINE bool isinf(float32_t const v)
{
#ifdef __CUDA_ARCH__
    /// By IEEE 754 rule, 2*Inf equals Inf
    return (2 * v == v) && (v != 0);
#else
    return std::isinf(v);
#endif
}

/**
 * @brief Check if given float64_t type @p v is infinite
 * @param v input number
 * @return true if @p is infinite, otherwise false
 */
CUDA_BOTH_INLINE bool isinf(float64_t const v)
{
#ifdef __CUDA_ARCH__
    /// By IEEE 754 rule, 2*Inf equals Inf
    return (2 * v == v) && (v != 0);
#else
    return std::isinf(v);
#endif
}

/**
 * @brief Check if given non-float type value is infinite
 * @tparam T data type of input value
 * @return Always false - non float type values cannot be infinite by nature.
 */
template <class T>
CUDA_BOTH_INLINE bool isinf(T)
{
    return false;
}

/// Compute machine epsilon around value (ignoring subnormals)
/// @param x The value used to compute epsilon around it
template <typename FloatT>
CUDA_BOTH_INLINE constexpr auto machineEpsilon(FloatT x) -> FloatT
{
    const auto absX = std::abs(x);
    auto eps        = std::nexttoward(absX, numeric_limits<FloatT>::infinity()) - absX;

    // Edge case for lowest() or max(). If lowest or max is passed in, epsilon becomes infinity,
    // as nexttowards infinity is infinity. Then, infinity - absX is still infinity.
    if (isinf(eps))
    {
        // clear errors raised by previous nexttoward call
        std::feclearexcept(FE_OVERFLOW);
        std::feclearexcept(FE_INEXACT);

        eps = absX - std::nexttoward(absX, 0);
    }

    return eps;
}

/// Same as std::abs but constexpr and supports unsigned
/// Compute absolute value of type T.
/// @param a The input value of type T.
template <typename T>
CUDA_BOTH_INLINE constexpr std::enable_if_t<std::is_signed<T>::value || std::is_floating_point<T>::value, T> const abs(T a)
{
    return a < static_cast<T>(0) ? -a : a;
}

template <typename T>
CUDA_BOTH_INLINE constexpr std::enable_if_t<std::is_unsigned<T>::value, T> abs(T a)
{
    return a;
}

/// Check if a floating point is valid divisor
/// @param var the value to check
template <typename FloatT, std::enable_if_t<std::is_floating_point<FloatT>::value, bool> = true>
CUDA_BOTH_INLINE bool isValidDivisor(FloatT const var)
{
    return !(core::abs(var) < core::numeric_limits<FloatT>::min());
}

/// Check if an integer is valid divisor
/// @param var the value to check
template <typename IntegerT, std::enable_if_t<std::is_integral<IntegerT>::value, bool> = true>
CUDA_BOTH_INLINE bool isValidDivisor(IntegerT var)
{
    return var != static_cast<IntegerT>(0);
}

/// Cuda-compatible version of std::equal
/// @param first1 First element start iterator
/// @param last1 First element last iterator
/// @param first2 Second element start iterator
template <class InputIt1, class InputIt2>
CUDA_BOTH_INLINE bool equal(InputIt1 first1, InputIt1 last1, InputIt2 first2)
{
    for (; first1 != last1; ++first1, ++first2)
    {
        if (!(*first1 == *first2))
        {
            return false;
        }
    }
    return true;
}

/// Cuda-compatible version of std::lexicographical_compare
/// @param first1 First element start iterator
/// @param last1 First element last iterator
/// @param first2 Second element start iterator
/// @param last2 Second element last iterator
template <class InputIt1, class InputIt2>
CUDA_BOTH_INLINE bool lexicographicalCompare(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2)
{
    for (; (first1 != last1) && (first2 != last2); ++first1, ++first2)
    {
        if (*first1 < *first2)
        {
            return true;
        }
        if (*first2 < *first1)
        {
            return false;
        }
    }
    return (first1 == last1) && (first2 != last2);
}

/**@}*/

} // namespace core
} // namespace dw

#endif
