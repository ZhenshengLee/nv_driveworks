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

#ifndef DW_FRAMEWORK_PASSDESCRIPTOR_HPP_
#define DW_FRAMEWORK_PASSDESCRIPTOR_HPP_

#include <dw/core/base/Types.h>
#include <dw/core/container/StringView.hpp>

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace dw
{
namespace framework
{

/**
 * Describe the passes of a node.
 *
 * The function is used to implement NodeConcept::describePasses().
 * Each argument of the function is created by describePass().
 */
template <typename... Args>
constexpr auto describePassCollection(const Args&&... args)
{
    return std::make_tuple(std::forward<const Args>(args)...);
}

// coverity[autosar_cpp14_a0_1_1_violation]
// coverity[autosar_cpp14_m0_1_4_violation]
static constexpr size_t PASS_NAME{0U};
// coverity[autosar_cpp14_a0_1_1_violation]
// coverity[autosar_cpp14_m0_1_4_violation]
static constexpr size_t PASS_PROCESSOR_TYPE{1U};
// coverity[autosar_cpp14_a0_1_1_violation]
// coverity[autosar_cpp14_m0_1_4_violation]
static constexpr size_t PASS_DEPENDENCIES{2U};

/**
 * Describe a specific pass of a node.
 *
 * The function is used to create the arguments for describePassCollection().
 */
constexpr std::tuple<dw::core::StringView, dwProcessorType> describePass(
    dw::core::StringView const&& name, dwProcessorType processorType)
{
    return std::make_tuple(
        std::move(name),
        std::move(processorType));
}

/**
 * Describe a specific pass of a node with custom inter-pass dependencies.
 *
 * The function is used to create the arguments for describePassCollection().
 * The dependencies argument is created by describePassDependencies().
 */
// Overloaded functions are provided for ease of use
// coverity[autosar_cpp14_a2_10_5_violation]
template <typename DependenciesT>
constexpr auto describePass(
    dw::core::StringView const&& name, dwProcessorType processorType, DependenciesT dependencies) -> std::tuple<dw::core::StringView, dwProcessorType, DependenciesT>
{
    return std::make_tuple(
        std::move(name),
        std::move(processorType),
        std::move(dependencies));
}

/**
 * Describe the custom inter-pass dependencies of a pass.
 *
 * The function is used to create the dependencies argument for describePass().
 */
template <typename... Args>
constexpr auto describePassDependencies(const Args&&... args) -> std::array<dw::core::StringView, sizeof...(Args)>
{
    return {std::forward<const Args>(args)...};
}

/// Get described passes for the passed node.
template <typename Node>
constexpr auto describePasses()
{
    return Node::describePasses();
}

/// Get the number of passes of the passed node.
template <typename Node>
constexpr std::size_t passSize()
{
    return std::tuple_size<decltype(describePasses<Node>())>::value;
}

/// Check if pass index is valid.
// Overloaded functions provide the same functionality for different argument types
// coverity[autosar_cpp14_a2_10_5_violation]
template <typename Node>
constexpr bool isValidPass(std::size_t passID)
{
    return passID < passSize<Node>();
}

namespace detail
{

/// Terminate recursion to get the pass index for a pass identified by name.
template <
    typename Node, size_t Index,
    typename std::enable_if_t<Index == passSize<Node>(), void>* = nullptr>
constexpr std::size_t passIndex(dw::core::StringView identifier)
{
    (void)identifier;
    return 0;
}

/// Recursion to get the pass index for a pass identified by name.
template <
    typename Node, size_t Index,
    typename std::enable_if_t<Index<passSize<Node>(), void>* = nullptr>
    // TODO(dwplc): FP -- The specific specialization of this templated function is selected by enable_if
    // coverity[autosar_cpp14_a2_10_5_violation]
    constexpr std::size_t passIndex(dw::core::StringView identifier)
{
    constexpr auto name = std::get<dw::framework::PASS_NAME>(std::get<Index>(describePasses<Node>()));
    if (name == identifier)
    {
        return 0;
    }
    return 1 + passIndex<Node, Index + 1>(identifier);
}

} // namespace detail

/// Get the the pass index for a pass identified by name.
// TODO(dwplc): FP -- The other passIndex() functions are defined in a namespace
// coverity[autosar_cpp14_a2_10_5_violation]
template <typename Node>
constexpr size_t passIndex(dw::core::StringView identifier)
{
    return detail::passIndex<Node, 0>(identifier);
}

/// Check if given string is a valid pass name.
template <typename Node>
constexpr bool isValidPass(dw::core::StringView identifier)
{
    constexpr size_t index = passIndex<Node>(identifier);
    return isValidPass<Node>(index);
}

/// Get the name of a pass.
template <typename Node, size_t Index>
constexpr dw::core::StringView passName()
{
    return std::get<dw::framework::PASS_NAME>(std::get<Index>(describePasses<Node>()));
}

/// Get the processor type of a pass.
template <typename Node, size_t Index>
constexpr dwProcessorType passProcessorType()
{
    return std::get<dw::framework::PASS_PROCESSOR_TYPE>(std::get<Index>(describePasses<Node>()));
}

/// Check if a pass specifies explicit dependencies.
template <typename Node, size_t Index>
constexpr bool hasPassDependencies()
{
    constexpr auto pass = std::get<Index>(describePasses<Node>());
    return std::tuple_size<decltype(pass)>() > PASS_DEPENDENCIES;
}

/// Get the dependencies of a pass.
template <
    typename Node, size_t Index,
    typename std::enable_if_t<hasPassDependencies<Node, Index>(), void>* = nullptr>
constexpr auto passDependencies()
{
    return std::get<PASS_DEPENDENCIES>(std::get<Index>(describePasses<Node>()));
}

/// Get the dependencies of a pass (which returns an empty collection for passes without explicit dependencies).
template <
    typename Node, size_t Index,
    typename std::enable_if_t<!hasPassDependencies<Node, Index>(), void>* = nullptr>
// Overloaded functions provide the same functionality for different argument types
// coverity[autosar_cpp14_a2_10_5_violation]
constexpr auto passDependencies()
{
    return std::array<dw::core::StringView, 0>();
}

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_PASSDESCRIPTOR_HPP_
