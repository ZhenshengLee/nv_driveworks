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
// SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_CORE_UTILITY_STATICIF_HPP_
#define DW_CORE_UTILITY_STATICIF_HPP_

#include <dwshared/dwfoundation/dw/core/platform/CompilerSpecificMacros.hpp>

#include <type_traits>
#include <utility>

namespace dw
{
namespace core
{

/**
 * \defgroup staticif_group Static-if Group of Functions
 * @{
 */

/// @{
/// If @c Cond::value is @c true, invoke the @a action with the provided @a args.
///
template <typename Cond, typename T, typename... Args>
CUDA_BOTH_INLINE constexpr std::enable_if_t<bool(Cond::value)> staticIf(Cond, const T&& action, Args&&... args)
{
    action(std::forward<Args>(args)...);
}

/// @{
/// If @c Cond::value is @c true, invoke the @a action with the provided @a args.
///
template <typename Cond, typename T, typename... Args>
// TODO(dwplc): FP - One of the two staticIf() will always be disqualified as the condition bool(Cond::value) and !bool(Cond::value) are exclusive
// coverity[autosar_cpp14_a13_3_1_violation]
CUDA_BOTH_INLINE constexpr std::enable_if_t<!bool(Cond::value)> staticIf(Cond, const T&&, Args&&...)
{
}
/// @}

/// @{
/// Return either a @c T or @c F value based on the static @c Cond::value. Unlike using the conditional (ternary)
/// operator, the types @c T and @c F do not have to be reference-compatible -- a @c FixedString and a @c unique_ptr can
/// validly fill the arguments.
///
template <typename Cond, typename T, typename F>
// TODO(dwplc): FP - std::forward() doesn't have decltype in it. Actually the static_cast<T&&>() inside std::forward() triggers this violation. Should re-check this once we switch to QNX.
// coverity[autosar_cpp14_a5_1_7_violation]
CUDA_BOTH_INLINE constexpr std::enable_if_t<bool(Cond::value), T> staticChoose(Cond, T&& x, F&&)
{
    return std::forward<T>(x);
}

template <typename Cond, typename T, typename F>
// TODO(dwplc): FP - One of the two staticChoose() will always be disqualified as the condition bool(Cond::value) and !bool(Cond::value) are exclusive
// TODO(dwplc): FP - std::forward() doesn't have decltype in it. Actually the static_cast<T&&>() inside std::forward() triggers this violation. Should re-check this once we switch to QNX.
// coverity[autosar_cpp14_a13_3_1_violation]
// coverity[autosar_cpp14_a5_1_7_violation]
CUDA_BOTH_INLINE constexpr std::enable_if_t<!bool(Cond::value), F> staticChoose(Cond, T&&, F&& x)
{
    return std::forward<F>(x);
}
/// @}

/// If @c Cond::value is @c true, invoke the @a trueAction with the provided @a args; if @c Cond::value is @c false,
/// invoke the @a falseAction with the provided @a args.
///
template <typename Cond, typename T, typename F, typename... Args>
CUDA_BOTH_INLINE constexpr auto
staticIfElse(Cond cond, T&& trueAction, F&& falseAction, Args&&... args) -> decltype(staticChoose(cond, std::forward<T>(trueAction), std::forward<F>(falseAction))(std::forward<Args>(args)...))
{
    return staticChoose(cond, std::forward<T>(trueAction), std::forward<F>(falseAction))(std::forward<Args>(args)...);
}

/**@}*/

} // namespace dw::core
} // namespace dw

#endif /*DW_CORE_UTILITY_STATICIF_HPP_*/
