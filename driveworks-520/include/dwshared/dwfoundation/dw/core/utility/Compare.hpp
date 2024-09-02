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

#ifndef DW_CORE_UTILITY_COMPARE_HPP_
#define DW_CORE_UTILITY_COMPARE_HPP_

#include <dwshared/dwfoundation/dw/core/language/BasicTypes.hpp>
#include <dwshared/dwfoundation/dw/core/language/Limits.hpp>
#include <dwshared/dwfoundation/dw/core/language/Math.hpp>
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

/// @brief Check if value is Zero ie. equal to 0
/// @param val the value to check
/// @return True if the value is equal to 0 else false
template <typename T>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE constexpr bool isZero(T const val) noexcept
{
    static_assert(!std::is_floating_point<T>::value, "Not suitable for floating point types. Use isFloatZero() instead.");
    static_assert(std::is_integral<T>::value, "Only integral values are expected.");
    return val == static_cast<T>(0);
}

/// Compares a floating point type to zero allowing for numerical tolerance
/// @param a the value to compare with tolerance (default is epsilon)
/// @param tolerance the optional custom input tolerance to use for comparison
template <typename FloatT>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE constexpr bool isFloatZero(FloatT const a, FloatT const tolerance = numeric_limits<FloatT>::epsilon())
{
    return abs(a) <= tolerance;
}

/// Check if two float values are equal against a tolerance
/// @param a the first float value
/// @param b the second float value
/// @param tolerance the tolerance to use (default is epsilon)
template <typename FloatT>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE constexpr bool isFloatEqual(FloatT const a, FloatT b, FloatT const tolerance = numeric_limits<FloatT>::epsilon()) noexcept
{
    const bool infEq{core::isinf(a) && core::isinf(b) && a * b > 0};
    return infEq || (core::abs(a - b) <= tolerance);
}

/// Check if two float values are equal against a relative tolerance
/// The relative tolerance is given by: relTol =  abs(a-b)/min(abs(a), abs(b))
/// This function checks if relTol < tolerance.
/// @param a the first float value
/// @param b the second float value
/// @param tolerance the tolerance to use (default is 1E-5)
/// @retval true if relTol <= tolerance false otherwise
template <class FloatT>
bool isFloatEqualRelTol(FloatT const a, FloatT const b, FloatT tolerance = static_cast<FloatT>(1E-5F))
{
    return std::abs(a - b) <= (std::min(std::abs(a), std::abs(b)) * tolerance);
}

/// @brief Check if value is Positive ie. greater than 0
/// @param val the value to check
/// @return True if the value is positive else false
template <typename T>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE constexpr bool isPositive(T const val) noexcept
{
    static_assert(std::is_arithmetic<T>::value, "Integral/floating point value is expected.");
    return val > static_cast<T>(0);
}

/// @brief Check if value is Negative ie. less than 0
/// @param val the value to check
/// @return True if the value is negative else false
template <typename T>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE constexpr bool isNegative(T const val) noexcept
{
    static_assert(std::is_arithmetic<T>::value, "Integral/floating point value is expected.");
    return val < static_cast<T>(0);
}

/// @brief Check if value is Non Positive ie. less than or equal to 0
/// @param val the value to check
/// @return True if the value is non poistive else false
template <typename T>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE constexpr bool isNonPositive(T const val) noexcept
{
    static_assert(std::is_arithmetic<T>::value, "Integral/floating point value is expected.");
    return val <= static_cast<T>(0);
}

/// @brief Check if value is Non Negative ie. greater than or equal to 0
/// @param val the value to check
/// @return True if the value is non negative else false
template <typename T>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE constexpr bool isNonNegative(T const val) noexcept
{
    static_assert(std::is_arithmetic<T>::value, "Integral/floating point value is expected.");
    return val >= static_cast<T>(0);
}

/// Check if two float containers are equal against a tolerance.
/// @param a the first float container
/// @param b the second float container
/// @param tolerance the tolerance to use (default is epsilon)
template <typename TContainer, typename FloatT>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool isFloatContainerEqual(TContainer const& a, TContainer const& b, FloatT tolerance = numeric_limits<std::remove_const_t<FloatT>>::epsilon())
{
    using dw::util::isFloatEqual;

    // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
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
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
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
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE constexpr bool isFloatEqualMachineEpsilon(FloatT a, FloatT b)
{
    return isFloatEqual(a, b, machineEpsilon(std::max(std::abs(a), std::abs(b))));
}

/// Check if two float arrays are equal against a tolerance.
/// @param a the first float array
/// @param b the second float array
/// @param tolerance the tolerance to use (default is epsilon)
template <typename FloatT, size_t DIMS>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool isFloatArrayEqual(const FloatT (&a)[DIMS], const FloatT (&b)[DIMS], FloatT tolerance = numeric_limits<FloatT>::epsilon())
{
    for (size_t i{0}; i < DIMS; i++)
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
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
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
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool isVectorContainerEqual(const TContainer& a, const TContainer& b)
{
    using TVector               = decltype(*a.begin());
    auto isVectorEqualPredicate = [](const TVector& v1, const TVector& v2) {
        return isFloatZero((v1 - v2).squaredNorm());
    };
    return isContainerEqual(a, b, isVectorEqualPredicate);
}

namespace detail
{

template <typename ContainerSizeType, class Container>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE void containerSizeCompare(bool& areEqualSized, ContainerSizeType const prevSize, Container&& c1)
{
    areEqualSized &= (c1.size() == prevSize);
}

template <typename ContainerSizeType, class Container, class... TContainers>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE void containerSizeCompare(bool& areEqualSized, ContainerSizeType const prevSize, Container&& c1, TContainers&&... cs)
{
    auto size{c1.size()};
    areEqualSized &= (c1.size() == prevSize);
    containerSizeCompare(areEqualSized, size, std::forward<TContainers>(cs)...);
}

template <typename Container, class... TContainers>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE void compareAllSizes(bool& areEqualSized, Container&& c1, TContainers&&... cs)
{
    auto size{c1.size()};
    containerSizeCompare(areEqualSized, size, std::forward<TContainers>(cs)...);
}
}

/// Check if passed containers have same size.
/// @param args container pack
template <typename... TContainers>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool areContainersEqualSized(TContainers&&... args)
{
    bool areEqualSized{true};
    detail::compareAllSizes(areEqualSized, std::forward<TContainers>(args)...);
    return areEqualSized;
}

/// Check if containers have same size but greater than a minimum size.
/// @param cs container pack
/// @param minSize the minimum size
template <typename... TContainers>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE bool areContainersEqualSizedWithMin(size_t const minSize, TContainers const&... cs)
{
    auto tuple = std::tie(cs...);
    return areContainersEqualSized(cs...) && (minSize <= std::get<0>(tuple).size());
}

/**@}*/

} // namespace util
} // namespace core
} // namespace dw

#endif
