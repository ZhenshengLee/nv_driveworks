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
// SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <dwshared/dwfoundation/dw/core/container/StringView.hpp>

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
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr auto describePassCollection(const Args&&... args) -> std::tuple<Args...>
{
    return std::make_tuple(std::forward<const Args>(args)...);
}

template <size_t NumberOfDependencies>
struct PassDescriptorT
{
    dw::core::StringView name;
    dwProcessorType processorType;
    bool hasDependencies;
    std::array<dw::core::StringView, NumberOfDependencies> dependencies;

    constexpr PassDescriptorT(dw::core::StringView const&& name_, dwProcessorType processorType_)
        : name{std::move(name_)}
        , processorType{std::move(processorType_)}
        , hasDependencies{false}
        , dependencies{}
    {
        static_assert(NumberOfDependencies == 0, "PassDescriptorT constructor without dependencies only available with NumberOfDependencies == 0");
    }

    constexpr PassDescriptorT(dw::core::StringView const&& name_, dwProcessorType processorType_, std::array<dw::core::StringView, NumberOfDependencies> const&& dependencies_)
        : name{std::move(name_)}
        , processorType{std::move(processorType_)}
        , hasDependencies{true}
        , dependencies{std::move(dependencies_)}
    {
    }
};

/**
 * Describe a specific pass of a node.
 *
 * The function is used to create the arguments for describePassCollection().
 */
constexpr PassDescriptorT<0> describePass(
    dw::core::StringView const&& name, dwProcessorType processorType)
{
    return PassDescriptorT<0>(
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
template <typename DependenciesT>
// coverity[autosar_cpp14_a2_10_5_violation]
constexpr auto describePass(
    dw::core::StringView const&& name, dwProcessorType processorType, DependenciesT const&& dependencies) -> PassDescriptorT<std::tuple_size<DependenciesT>::value>
{
    return PassDescriptorT<std::tuple_size<DependenciesT>::value>(
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
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr auto describePassDependencies(const Args&&... args) -> std::array<dw::core::StringView, sizeof...(Args)>
{
    return {std::forward<const Args>(args)...};
}

/// Get described passes for the passed node.
template <typename Node>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
// coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-1984
constexpr auto describePasses()
{
    return Node::describePasses();
}

/// Get the number of passes of the passed node.
template <typename Node>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr std::size_t passSize()
{
    return std::tuple_size<decltype(describePasses<Node>())>::value;
}

/// Check if pass index is valid.
// Overloaded functions provide the same functionality for different argument types
template <typename Node>
// coverity[autosar_cpp14_a2_10_5_violation]
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
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr std::size_t passIndex(dw::core::StringView identifier)
{
    static_cast<void>(identifier);
    // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
    return 0U;
}

/// Recursion to get the pass index for a pass identified by name.
template <
    typename Node, size_t Index,
    typename std::enable_if_t<Index<passSize<Node>(), void>* = nullptr>
    // coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
    constexpr std::size_t passIndex(dw::core::StringView identifier)
{
    constexpr dw::core::StringView name{std::get<Index>(describePasses<Node>()).name};
    if (name == identifier)
    {
        // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
        return 0U;
    }
    // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
    return 1U + passIndex<Node, Index + 1>(identifier);
}

} // namespace detail

/// Get the the pass index for a pass identified by name.
template <typename Node>
// coverity[autosar_cpp14_a2_10_5_violation] RFD Pending: TID-2053
constexpr size_t passIndex(dw::core::StringView identifier)
{
    return detail::passIndex<Node, 0>(identifier);
}

/// Check if given string is a valid pass name.
template <typename Node>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr bool isValidPass(dw::core::StringView identifier)
{
    constexpr size_t index = passIndex<Node>(identifier);
    return isValidPass<Node>(index);
}

/// Get the name of a pass.
template <typename Node, size_t Index>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr dw::core::StringView passName()
{
    return std::get<Index>(describePasses<Node>()).name;
}

/// Get the processor type of a pass.
template <typename Node, size_t Index>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr dwProcessorType passProcessorType()
{
    return std::get<Index>(describePasses<Node>()).processorType;
}

/// Check if a pass specifies explicit dependencies.
template <typename Node, size_t Index>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr bool hasPassDependencies()
{
    return std::get<Index>(describePasses<Node>()).hasDependencies;
}

/// Get the dependencies of a pass (which returns an empty collection for passes without explicit dependencies).
template <typename Node, size_t Index>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
// coverity[autosar_cpp14_a7_1_5_violation]
constexpr auto passDependencies()
{
    return std::get<Index>(describePasses<Node>()).dependencies;
}

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_PASSDESCRIPTOR_HPP_
