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
// SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_CORE_LANGUAGE_CHECKED_INTEGER_CAST_HPP_
#define DW_CORE_LANGUAGE_CHECKED_INTEGER_CAST_HPP_

#include <dwshared/dwfoundation/dw/core/platform/CompilerSpecificMacros.hpp>
#include <dwshared/dwfoundation/dw/core/base/ExceptionWithStackTrace.hpp>

#include <limits>

namespace dw
{
namespace core
{

/**
 * \defgroup integercast_group SafeIntegerCast Group of Functions
 * @{
 */

/**
 * @brief Checks if both @b T and @b U are integral types.
 * 
 * @tparam T The type to check.
 * @tparam U The type to check.
 * 
 * @retval true if both @b T and @b U are integral types.
 * @retval false if both @b T and @b U are not integral types.
*/
template <class T, class U>
constexpr bool areIntegral() noexcept
{
    return std::is_integral<T>::value && std::is_integral<U>::value;
}

/**
 * @brief Checks if types @b T and @b U have the same signedness.
 * 
 * @tparam T The type to check.
 * @tparam U The type to check.
 * 
 * @retval true if @b T and @b U have the same signedness.
 * @retval false if @b T and @b U do not have the same signedness.
*/
template <class T, class U>
constexpr bool haveSameSignedness() noexcept
{
    return std::is_signed<T>::value == std::is_signed<U>::value;
}

/**
 * @brief Checks if the domain of @b TSrc is entirely contained in the domain of @b TDst .
 *
 * @tparam TSrc The source type.
 * @tparam TDst The destination type.
 * @retval true if the domain of @b TSrc is entirely contained in the domain of @b TDst .
 * @retval false if the domain of @b TSrc is not entirely contained in the domain of @b TDst .
*/
template <class TSrc, class TDst, std::enable_if_t<haveSameSignedness<TSrc, TDst>(), bool> = true>
constexpr bool isInDomain() noexcept
{
    return (std::numeric_limits<TSrc>::lowest() >= std::numeric_limits<TDst>::lowest()) &&
           (std::numeric_limits<TSrc>::max() <= std::numeric_limits<TDst>::max());
}

template <class TSrc, class TDst, std::enable_if_t<!haveSameSignedness<TSrc, TDst>(), bool> = true>
constexpr bool isInDomain() noexcept
{
    return false;
}

/**************************************************************************************************
 *
 * Narrowing integer cast with runtime checks.
 * Same as static_cast but throws if the cast leads to data loss.
 *
 * See: gsl::narrow.
 *
 * Note: Try to avoid the use of this function by avoiding the cast in the first place.
 *       For non-narrowing casts use static_cast.
 *
 **************************************************************************************************/
/**
 * @brief Narrow integer cast with runtime checks.
 *
 * @tparam TDst The destination type.
 * @tparam TSrc The source type.
 * @param[in] src Source integer to cast from.
 *
 * @throw InvalidArgumentException if cast leads to data loss.
 *
 * @return Casted destination integer.
*/
template <class TDst, class TSrc>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto narrow(TSrc const src) noexcept(false) -> typename std::enable_if<areIntegral<TDst, TSrc>() && haveSameSignedness<TDst, TSrc>() && !isInDomain<TSrc, TDst>(), TDst>::type
{
    // coverity[autosar_cpp14_a4_7_1_violation] RFD Pending: TID-2327
    // coverity[cert_int31_c_violation] RFD Pending: TID-2327
    const TDst dst{static_cast<TDst>(src)};

    const bool hasValueChanged{static_cast<TSrc>(dst) != src};
    if (hasValueChanged)
    {
        // coverity[autosar_cpp14_a5_1_1_violation] FP: nvbugs/2980306
        core::assertException<InvalidArgumentException>("core::narrow - casting lead to data loss.");
    }

    return dst;
}

template <class TDst, class TSrc>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto narrow(TSrc const src) noexcept(false) -> typename std::enable_if<areIntegral<TDst, TSrc>() && !haveSameSignedness<TDst, TSrc>() && !isInDomain<TSrc, TDst>(), TDst>::type
{
    // coverity[autosar_cpp14_a4_7_1_violation] RFD Pending: TID-2327
    // coverity[cert_int31_c_violation] RFD Pending: TID-2327
    TDst const dst{static_cast<TDst>(src)};

    bool const hasValueChanged{static_cast<TSrc>(dst) != src};
    constexpr TSrc SRC_ZERO{};
    constexpr TDst DST_ZERO{};
    bool const isSrcBelowZero{src < SRC_ZERO};
    bool const isDstBelowZero{dst < DST_ZERO};
    bool const hasSignChanged{isSrcBelowZero != isDstBelowZero}; // catch overflows
    if (hasValueChanged || hasSignChanged)
    {
        // coverity[autosar_cpp14_a5_1_1_violation] FP: nvbugs/2980306
        core::assertException<InvalidArgumentException>("core::narrow - casting lead to data loss.");
    }

    return dst;
}

/**************************************************************************************************
 *
 * Wrapper that selects narrow or static_cast depending on whether the cast has potential
 * data loss or not.
 *
 * Note: ONLY USE THIS IN TEMPLATES!
 *
 *       Otherwise use narrow or static_cast directly, so that it is easy to find narrowing
 *       casts in the code.
 *
 **************************************************************************************************/

/**
 * @brief Wrapper that selects narrow or static_cast depending on whether the cast has potential data loss or not.
 * If @b TDst and @b TSrc are both integral types and @b TSrc fits entirely into @b TDst, so it doesnot lead to data 
 * loss and will select static_cast.
 * if @b TDst and @b TSrc are both integral types and @b TSrc doesn't fit entirely into @b TDst, so potential data lost may occur
 * and it will select narrow.
 *
 * @tparam TDst The destination type.
 * @tparam TSrc The source type.
 * @param[in] src Source integer to cast from
 *
 * @return Casted destination integer
*/
template <class TDst, class TSrc>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto checkedIntegerCast(TSrc const src) noexcept(false) -> typename std::enable_if<areIntegral<TDst, TSrc>() && isInDomain<TSrc, TDst>(), TDst>::type
{
    // no need to do any checks
    return static_cast<TDst>(src);
}

template <class TDst, class TSrc>
// coverity[autosar_cpp14_a1_1_1_violation] RFD Pending: TID-2194
CUDA_BOTH_INLINE auto checkedIntegerCast(TSrc const src) noexcept(false) -> typename std::enable_if<areIntegral<TDst, TSrc>() && !isInDomain<TSrc, TDst>(), TDst>::type
{
    return narrow<TDst, TSrc>(src);
}

/**@}*/

} // core
} // dw

#endif // DW_CORE_LANGUAGE_CHECKED_INTEGER_CAST_HPP_
