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
// SPDX-FileCopyrightText: Copyright (c) 2015-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * @param[in] a input value
 * @param[in] b input value
 * @return smaller value between a and b.
 */
template <typename Ta, typename Tb>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
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
 * @param[in] a input value
 * @param[in] b input value
 * @return Larger value between a and b.
 */
template <typename Ta, typename Tb>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE constexpr auto max(Ta const a, Tb const b) -> Ta
{
    return a > b ? a : b;
}

/**
 * @brief Check if given float32_t type @p v is Not-A-Number(NAN)
 * @param[in] v input number
 * @return true if @p is Not-A-Number, otherwise false
 */
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
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
 * @param[in] v input number
 * @return true if @p is Not-A-Number, otherwise false
 */
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
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
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool isnan(T)
{
    return false;
}

/**
 * @brief Check if given float32_t type @p v is infinite
 * @param[in] v input number
 * @return true if @p is infinite, otherwise false
 */
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
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
 * @param[in] v input number
 * @return true if @p is infinite, otherwise false
 */
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
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
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool isinf(T)
{
    return false;
}

/**
 * @brief Compute machine epsilon around value (ignoring subnormals)
 * @param[in] x The value used to compute epsilon around it
*/
template <typename FloatT>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
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

/**
 * @brief Same as std::abs but constexpr and supports unsigned
 *        Compute absolute value of type T.
 * @param[in] a input value of type T.
 * @return the abs of input value
*/
template <typename T>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE constexpr std::enable_if_t<std::is_signed<T>::value || std::is_floating_point<T>::value, T> const abs(T a)
{
    return a < static_cast<T>(0) ? -a : a;
}

template <typename T>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE constexpr std::enable_if_t<std::is_unsigned<T>::value, T> abs(T a)
{
    return a;
}

/**
 * @brief Check if a floating point is valid divisor
 * @param[in] var the value to check
 * @return true if the value is valid divisor,
 *         false otherwises.
*/
template <typename FloatT, std::enable_if_t<std::is_floating_point<FloatT>::value, bool> = true>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE constexpr bool isValidDivisor(FloatT const var)
{
    return !(core::abs(var) < core::numeric_limits<FloatT>::min());
}

/**
 * @brief Check if an integer is valid divisor
 * @param[in] var the value to check
 * @return true if the value is valid divisor,
 *         false otherwises.
*/
template <typename IntegerT, std::enable_if_t<std::is_integral<IntegerT>::value, bool> = true>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE constexpr bool isValidDivisor(IntegerT var)
{
    return var != static_cast<IntegerT>(0);
}

/**
 * @brief Cuda-compatible version of std::equal
 * @param[in] first1 First element start iterator
 * @param[in] last1 First element last iterator
 * @param[in] first2 Second element start iterator
 * @return true if elements are equal,
 *         false otherwises.
*/
template <class InputIt1, class InputIt2>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
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

/**
 * @brief Cuda-compatible version of std::lexicographical_compare
 * @param[in] first1 First element start iterator
 * @param[in] last1 First element last iterator
 * @param[in] first2 Second element start iterator
 * @param[in] last2 Second element last iterator
 * @return true if first element is less than second element,
 *         false otherwises.
*/
template <class InputIt1, class InputIt2>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
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
