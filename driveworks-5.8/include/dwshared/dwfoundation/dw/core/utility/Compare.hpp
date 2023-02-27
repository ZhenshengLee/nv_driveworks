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

#ifndef DW_CORE_UTILITY_COMPARE_HPP_
#define DW_CORE_UTILITY_COMPARE_HPP_

#include <dw/core/language/BasicTypes.hpp>
#include <dw/core/platform/Limits.hpp>
#include <dw/cuda/misc/CudaMath.hpp>
#include <type_traits>
#include <cfenv>

namespace dw
{
namespace core
{
namespace util
{

/**
 * \defgroup compare_group Compare Group of Functions
 * @{
 */

/// Same as std::min but constexpr, return the minimum between Ta and Tb.
/// constexpr is required for compile-time block-size computations
/// @param a First element
/// @param b Second element
template <typename Ta, typename Tb>
CUDA_BOTH_INLINE constexpr auto min(Ta const a, Tb const b) -> Ta
{
    return a < b ? a : b;
}

/// Same as std::max but constexpr, required for compile-time block-size computations
/// @param a First element
/// @param b Second element
template <typename Ta, typename Tb>
CUDA_BOTH_INLINE constexpr auto max(Ta const a, Tb const b) -> Ta
{
    return a > b ? a : b;
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
    if (cuda::isinf(eps))
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

/// Compares a floating point type to zero allowing for numerical tolerance
/// @param a the value to compare with tolerance (default is epsilon)
/// @param tolerance the optional custom input tolerance to use for comparison
template <typename FloatT>
CUDA_BOTH_INLINE constexpr bool isFloatZero(FloatT const a, FloatT const tolerance = numeric_limits<FloatT>::epsilon())
{
    return abs(a) <= tolerance;
}

/// Check if a floating point is valid divisor
/// @param var the value to check
template <typename FloatT, std::enable_if_t<std::is_floating_point<FloatT>::value, bool> = true>
CUDA_BOTH_INLINE bool isValidDivisor(FloatT const var)
{
    return !(util::abs(var) < util::numeric_limits<FloatT>::min());
}

/// Check if an integer is valid divisor
/// @param var the value to check
template <typename IntegerT, std::enable_if_t<std::is_integral<IntegerT>::value, bool> = true>
CUDA_BOTH_INLINE bool isValidDivisor(IntegerT var)
{
    return var != static_cast<IntegerT>(0);
}

/// Check if two float values are equal against a tolerance
/// @param a the first float value
/// @param b the second float value
/// @param tolerance the tolerance to use (default is epsilon)
template <typename FloatT>
CUDA_BOTH_INLINE constexpr bool isFloatEqual(FloatT const a, FloatT b, FloatT const tolerance = numeric_limits<FloatT>::epsilon()) noexcept
{
    const bool infEq = cuda::isinf(a) && cuda::isinf(b) && a * b > 0;
    return infEq || (std::abs(a - b) <= tolerance);
}

/// Check if two float containers are equal against a tolerance.
/// @param a the first float container
/// @param b the second float container
/// @param tolerance the tolerance to use (default is epsilon)
template <typename TContainer, typename FloatT>
CUDA_BOTH_INLINE bool isFloatContainerEqual(TContainer const& a, TContainer const& b, FloatT tolerance = numeric_limits<FloatT>::epsilon()) noexcept
{
    using dw::core::util::isFloatEqual;

    auto isFloatEqualPredicate = [&](FloatT f1, FloatT f2) -> bool {
        return isFloatEqual(f1, f2, tolerance);
    };
    return a.size() == b.size() &&
           std::equal(a.begin(), a.end(),
                      b.begin(), b.end(),
                      isFloatEqualPredicate);
}

/// Check if two float containers are equal.
/// @param a the first float container
/// @param b the second float container
template <typename TContainer>
CUDA_BOTH_INLINE bool isFloatContainerEqual(TContainer const& a, TContainer const& b) noexcept
{
    using FloatT = std::remove_reference_t<decltype(*a.begin())>;
    return isFloatContainerEqual<TContainer, FloatT>(a, b);
}

/// Check if two float are equal.
/// Same as isFloatEqual but with corresponding machine epsilon.
/// @param a the first float value
/// @param b the second float value
template <typename FloatT>
CUDA_BOTH_INLINE constexpr bool isFloatEqualMachineEpsilon(FloatT a, FloatT b)
{
    return isFloatEqual(a, b, machineEpsilon(std::max(std::abs(a), std::abs(b))));
}

/// Check if two float arrays are equal against a tolerance.
/// @param a the first float array
/// @param b the second float array
/// @param tolerance the tolerance to use (default is epsilon)
template <typename FloatT, size_t DIMS>
CUDA_BOTH_INLINE bool isFloatArrayEqual(const FloatT (&a)[DIMS], const FloatT (&b)[DIMS], FloatT tolerance = numeric_limits<FloatT>::epsilon())
{
    for (size_t i = 0; i < DIMS; i++)
    {
        if (!isFloatEqual(a[i], b[i], tolerance))
        {
            return false;
        }
    }
    return true;
}

/// Check if two containers are equal against a custom predicate.
/// @param a the first container
/// @param b the second container
/// @param binaryPredicate the predicate to use
template <typename TContainer, typename BinaryPredicate>
CUDA_BOTH_INLINE bool isContainerEqual(const TContainer& a, const TContainer& b, BinaryPredicate binaryPredicate)
{
    return a.size() == b.size() &&
           std::equal(a.begin(), a.end(),
                      b.begin(), b.end(),
                      binaryPredicate);
}

/// Check if two containers are equal.
/// @param a the first container
/// @param b the second container
template <typename TContainer>
CUDA_BOTH_INLINE bool isVectorContainerEqual(const TContainer& a, const TContainer& b)
{
    using TVector               = decltype(*a.begin());
    auto isVectorEqualPredicate = [](const TVector& v1, const TVector& v2) {
        return isFloatZero((v1 - v2).squaredNorm());
    };
    return isContainerEqual(a, b, isVectorEqualPredicate);
}

/// Check if two containers have same size.
/// @param a the first container
/// @param b the second container
template <typename TContainer>
CUDA_BOTH_INLINE bool isContainerEqualSized(const TContainer& a, const TContainer& b)
{
    return (a.size() == b.size());
}

/// Check if two containers have same size but greater than a minimum size.
/// @param a the first container
/// @param b the second container
/// @param minSize the minimum size
template <typename TContainer>
CUDA_BOTH_INLINE bool isContainerEqualSizedWithMin(const TContainer& a, const TContainer& b, const typename TContainer::size_type minSize)
{
    return isContainerEqualSized(a, b) && (minSize <= a.size());
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

} // namespace util
} // namespace core
} // namespace dw

#endif
