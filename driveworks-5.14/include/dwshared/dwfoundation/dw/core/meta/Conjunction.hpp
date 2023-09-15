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
// SPDX-FileCopyrightText: Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWSHARED_DW_CORE_META_CONJUNCTION_HPP_
#define DWSHARED_DW_CORE_META_CONJUNCTION_HPP_

#include <type_traits>

#include "BoolConstant.hpp"

namespace dw
{
namespace core
{
namespace meta
{

/// A conjunction is the logical @e and of all @c B traits. An empty list results in a @c true value. This metafunction
/// is short-circuiting.
///
/// @tparam B The series of traits to evaluate the @c value member of. Each @c B must have a member constant @c value
///           which is convertible to @c bool. Use of BoolConstant is helpful here.
///
/// @see CONJUNCTION_V
template <typename... B>
struct Conjunction;

template <>
struct Conjunction<> : BoolConstant<true>
{
};

template <typename B>
struct Conjunction<B> : B
{
};

template <typename BHead, typename... BRemaining>
struct Conjunction<BHead, BRemaining...> : std::conditional_t<bool(BHead::value), Conjunction<BRemaining...>, BHead>
{
};

/// @see Conjunction
// TODO(dwplc): FP - Coverity erroneously detects variable templates as an object with static storage duration.
template <typename... B>
// coverity[autosar_cpp14_a2_10_4_violation] FP: nvbugs/3477378
constexpr bool CONJUNCTION_V{Conjunction<B...>::value};

} // namespace dw::core::meta
} // namespace dw::core
} // namespace dw

#endif /*DWSHARED_DW_CORE_META_CONJUNCTION_HPP_*/
