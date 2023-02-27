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
// SPDX-FileCopyrightText: Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_CORE_UTILITY_CHECKED_INTEGER_CAST_HPP_
#define DW_CORE_UTILITY_CHECKED_INTEGER_CAST_HPP_

#include <dw/core/container/BaseString.hpp>
#include <dw/core/container/StringView.hpp>
#include <dw/core/base/ExceptionWithStackTrace.hpp>
#include <dw/core/platform/CompilerSpecificMacros.hpp>

namespace dw
{
namespace core
{

/**
 * \defgroup integercast_group SafeIntegerCast Group of Functions
 * @{
 */

/// Returns true if both T and U are integral types, false otherwise.
template <class T, class U>
constexpr bool areIntegral() noexcept
{
    return std::is_integral<T>::value && std::is_integral<U>::value;
}

/// Returns true if types T and U have the same signedness, false otherwise.
template <class T, class U>
constexpr bool haveSameSignedness() noexcept
{
    return std::is_signed<T>::value == std::is_signed<U>::value;
}

/// Returns true if the domain of TSrc is entirely contained in the domain of TDst, false otherwise.
template <class TSrc, class TDst, std::enable_if_t<haveSameSignedness<TSrc, TDst>(), bool> = true>
constexpr bool isInDomain() noexcept
{
    return (std::numeric_limits<TSrc>::lowest() >= std::numeric_limits<TDst>::lowest()) &&
           (std::numeric_limits<TSrc>::max() <= std::numeric_limits<TDst>::max());
}

/// Returns always false if types don't have the same signedness.
template <class TSrc, class TDst, std::enable_if_t<!haveSameSignedness<TSrc, TDst>(), bool> = true>
constexpr bool isInDomain() noexcept
{
    return false;
}

/// Create message string for exception when loss of data happens
/// @param src Source data to cast from
/// @param dst Destination data to cast to
template <class TDst, class TSrc>
FixedString<512> createDataLossMessage(TSrc const src, TDst const dst) noexcept(false)
{
    static constexpr StringView CAST_MSG{"Casting ", 8U};
    static constexpr StringView TO_MSG{" to ", 4U};
    static constexpr StringView LEADS_MSG{" leads to data loss.", 20U};
    FixedString<512> message{};
    message += CAST_MSG;
    message += src;
    message += TO_MSG;
    message += dst;
    message += LEADS_MSG;
    return message;
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
/// Narrow integer cast with runtime checks. If cast leads to data loss throws an exception.
/// TDst and TSrc shall be both integral types and shall have the same signedness.
/// @param src Source integer to cast from
template <class TDst, class TSrc>
CUDA_BOTH_INLINE auto narrow(TSrc const src) noexcept(false) -> typename std::enable_if<areIntegral<TDst, TSrc>() && haveSameSignedness<TDst, TSrc>() && !isInDomain<TSrc, TDst>(), TDst>::type
{
    const TDst dst{static_cast<TDst>(src)};

    const bool hasValueChanged{static_cast<TSrc>(dst) != src};
    if (hasValueChanged)
    {
// TODO(dwplc): RFD - tracking this violation here: https://jirasw.nvidia.com/browse/TID-405
// coverity[autosar_cpp14_a16_0_1_violation]
#ifdef __CUDACC__
        core::assertException<InvalidArgumentException>("Casting lead to data loss.");
// TODO(dwplc): RFD - tracking this violation here: https://jirasw.nvidia.com/browse/TID-405
// coverity[autosar_cpp14_a16_0_1_violation]
#else
        FixedString<512> message{createDataLossMessage(src, dst)};
        core::assertException<InvalidArgumentException>(message.c_str());
// TODO(dwplc): RFD - tracking this violation here: https://jirasw.nvidia.com/browse/TID-405
// coverity[autosar_cpp14_a16_0_1_violation]
#endif
    }

    return dst;
}

/// Narrow integer cast with runtime checks. If cast leads to data loss throws an exception.
/// TDst and TSrc shall be both integral types and shall have different signedness.
template <class TDst, class TSrc>
CUDA_BOTH_INLINE auto narrow(TSrc const src) noexcept(false) -> typename std::enable_if<areIntegral<TDst, TSrc>() && !haveSameSignedness<TDst, TSrc>() && !isInDomain<TSrc, TDst>(), TDst>::type
{
    TDst const dst{static_cast<TDst>(src)};

    bool const hasValueChanged{static_cast<TSrc>(dst) != src};
    constexpr TSrc SRC_ZERO{};
    constexpr TDst DST_ZERO{};
    bool const isSrcBelowZero{src < SRC_ZERO};
    bool const isDstBelowZero{dst < DST_ZERO};
    bool const hasSignChanged{isSrcBelowZero != isDstBelowZero}; // catch overflows
    if (hasValueChanged || hasSignChanged)
    {
// TODO(dwplc): RFD - tracking this violation here: https://jirasw.nvidia.com/browse/TID-405
// coverity[autosar_cpp14_a16_0_1_violation]
#ifdef __CUDACC__
        core::assertException<InvalidArgumentException>("Casting lead to data loss.");
// TODO(dwplc): RFD - tracking this violation here: https://jirasw.nvidia.com/browse/TID-405
// coverity[autosar_cpp14_a16_0_1_violation]
#else
        FixedString<512> message{createDataLossMessage(src, dst)};
        core::assertException<InvalidArgumentException>(message.c_str());
// TODO(dwplc): RFD - tracking this violation here: https://jirasw.nvidia.com/browse/TID-405
// coverity[autosar_cpp14_a16_0_1_violation]
#endif
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
/// Wrapper that selects narrow or static_cast depending on whether the cast has potential data loss or not.
/// TDst and TSrc shall be both integral types and TSrc fits entirely into TDst.
/// @param src Source integer to cast from
template <class TDst, class TSrc>
CUDA_BOTH_INLINE auto checkedIntegerCast(TSrc const src) noexcept(false) -> typename std::enable_if<areIntegral<TDst, TSrc>() && isInDomain<TSrc, TDst>(), TDst>::type
{
    // no need to do any checks
    return static_cast<TDst>(src);
}

/// Wrapper that selects narrow or static_cast depending on whether the cast has potential data loss or not.
/// TDst and TSrc shall be both integral types and TSrc doesn't fit entirely into TDst, so potential data lost may occur.
template <class TDst, class TSrc>
CUDA_BOTH_INLINE auto checkedIntegerCast(TSrc const src) noexcept(false) -> typename std::enable_if<areIntegral<TDst, TSrc>() && !isInDomain<TSrc, TDst>(), TDst>::type
{
    return narrow<TDst, TSrc>(src);
}

/**@}*/

} // core
} // dw

#endif // DW_CORE_UTILITY_CHECKED_INTEGER_CAST_HPP_
