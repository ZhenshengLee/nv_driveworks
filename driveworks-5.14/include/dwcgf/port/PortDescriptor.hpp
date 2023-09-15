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

#ifndef DW_FRAMEWORK_PORTDESCRIPTOR_HPP_
#define DW_FRAMEWORK_PORTDESCRIPTOR_HPP_

#include <dwshared/dwfoundation/dw/core/container/StringView.hpp>
#include <dwcgf/port/Port.hpp>
#include <dwshared/dwfoundation/dw/core/language/cxx20.hpp>
#include <dwshared/dwfoundation/dw/core/language/Tuple.hpp>

#include <functional>
#include <type_traits>
#include <utility>

namespace dw
{
namespace framework
{

// API needed to declare the ports of a node.

template <typename... Args>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr auto describePortCollection(Args&&... args) -> dw::core::Tuple<Args...>
{
    return dw::core::make_tuple<Args...>(std::forward<Args>(args)...);
}

enum class PortBinding : uint8_t
{
    OPTIONAL = 0,
    REQUIRED = 1
};

// coverity[autosar_cpp14_a14_1_1_violation]
template <typename PortType, size_t ArraySize>
struct PortDescriptorT
{
    // coverity[autosar_cpp14_a0_1_6_violation]
    using Type = PortType;
    dw::core::StringView typeName;
    dw::core::StringView name;
    static constexpr size_t arraySize{ArraySize};
    PortBinding binding;
    dw::core::StringView comment;

    constexpr PortDescriptorT(dw::core::StringView const&& typeName_, dw::core::StringView const&& name_, PortBinding binding_ = PortBinding::OPTIONAL, dw::core::StringView comment_ = ""_sv)
        : typeName{std::move(typeName_)}
        , name{std::move(name_)}
        , binding{std::move(binding_)}
        , comment{std::move(comment_)}
    {
    }
};

#define DW_PORT_TYPE_NAME_STRING_VIEW(TYPE_NAME_STR) TYPE_NAME_STR##_sv
#define DW_DESCRIBE_PORT(TYPE_NAME, args...) dw::framework::describePort<TYPE_NAME>(DW_PORT_TYPE_NAME_STRING_VIEW(#TYPE_NAME), ##args)

template <typename PortType>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr auto describePort(
    dw::core::StringView typeName, dw::core::StringView name, PortBinding binding = PortBinding::OPTIONAL, dw::core::StringView comment = ""_sv) -> PortDescriptorT<PortType, 0>
{
    return PortDescriptorT<PortType, 0>(
        std::move(typeName),
        std::move(name),
        std::move(binding),
        std::move(comment));
}

template <typename PortType>
// Overloaded functions are provided for ease of use
// coverity[autosar_cpp14_a2_10_5_violation]
constexpr auto describePort(
    dw::core::StringView typeName, dw::core::StringView name, dw::core::StringView comment)
{
    return describePort<PortType>(
        std::move(typeName),
        std::move(name),
        std::move(PortBinding::OPTIONAL),
        std::move(comment));
}

#define DW_DESCRIBE_PORT_ARRAY(TYPE_NAME, ARRAYSIZE, args...) dw::framework::describePortArray<TYPE_NAME, ARRAYSIZE>(DW_PORT_TYPE_NAME_STRING_VIEW(#TYPE_NAME), ##args)

template <
    typename PortType,
    size_t ArraySize,
    typename std::enable_if_t<ArraySize != 0, void>* = nullptr>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr auto describePortArray(
    dw::core::StringView typeName, dw::core::StringView name, PortBinding binding = PortBinding::OPTIONAL, dw::core::StringView comment = ""_sv) -> PortDescriptorT<PortType, ArraySize>
{
    return PortDescriptorT<PortType, ArraySize>(
        std::move(typeName),
        std::move(name),
        std::move(binding),
        std::move(comment));
}

template <
    typename PortType,
    size_t ArraySize,
    typename std::enable_if_t<ArraySize != 0, void>* = nullptr>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr auto describePortArray(
    dw::core::StringView typeName, dw::core::StringView name, dw::core::StringView comment)
{
    return describePortArray<PortType, ArraySize>(
        std::move(typeName),
        std::move(name),
        std::move(PortBinding::OPTIONAL),
        std::move(comment));
}

// API to access declared ports of a node.

template <typename Node>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
// coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-1984
constexpr auto describeInputPorts()
{
    return Node::describeInputPorts();
}

template <typename Node>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
// coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-1984
constexpr auto describeOutputPorts()
{
    return Node::describeOutputPorts();
}

template <
    typename Node,
    PortDirection Direction,
    typename std::enable_if_t<Direction == PortDirection::INPUT, void>* = nullptr>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
// coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-1984
constexpr auto describePorts()
{
    return describeInputPorts<Node>();
}

template <
    typename Node,
    PortDirection Direction,
    typename std::enable_if_t<Direction == PortDirection::OUTPUT, void>* = nullptr>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
// coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-1984
constexpr auto describePorts()
{
    return describeOutputPorts<Node>();
}

// API to query information about declared ports of a node.

// Number of input or output port descriptors
template <typename Node, PortDirection Direction>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr std::size_t portDescriptorSize()
{
    return dw::core::tuple_size<decltype(describePorts<Node, Direction>())>::value;
}

// The flag if the port described by a specific descriptor is an array
template <typename Node, PortDirection Direction, size_t DescriptorIndex>
constexpr bool descriptorPortArray()
{
    constexpr size_t array_length{dw::core::tuple_element_t<
        DescriptorIndex,
        decltype(describePorts<Node, Direction>())>::arraySize};
    return array_length > 0;
}

// The number of input or output ports described by a specific descriptor
// 1 for non-array descriptors, ARRAY_SIZE for array descriptors
template <typename Node, PortDirection Direction, size_t DescriptorIndex>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr size_t descriptorPortSize()
{
    constexpr size_t array_length{dw::core::tuple_element_t<
        DescriptorIndex,
        decltype(describePorts<Node, Direction>())>::arraySize};
    // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
    if (array_length == 0U)
    {
        // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
        return 1U;
    }
    return array_length;
}

// The binding of input or output ports described by a specific descriptor
template <typename Node, PortDirection Direction, size_t DescriptorIndex>
constexpr PortBinding descriptorPortBinding()
{
    constexpr PortBinding port_binding = dw::core::get<DescriptorIndex>(describePorts<Node, Direction>()).binding;
    return port_binding;
}

// The comment of input or output ports described by a specific descriptor
template <typename Node, PortDirection Direction, size_t DescriptorIndex>
constexpr dw::core::StringView descriptorPortComment()
{
    constexpr dw::core::StringView port_comment = dw::core::get<DescriptorIndex>(describePorts<Node, Direction>()).comment;
    return port_comment;
}

// Return type is the type of the descriptor, to be used with decltype()
template <typename Node, PortDirection Direction, size_t DescriptorIndex>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
// coverity[autosar_cpp14_a7_1_5_violation]
constexpr auto portDescriptorType()
{
    return typename dw::core::tuple_element_t<
        DescriptorIndex,
        decltype(describePorts<Node, Direction>())>::Type();
}

// Number of ports for a specific direction (sum across all descriptors)
namespace detail
{

template <
    typename Node, PortDirection Direction, size_t DescriptorIndex,
    typename std::enable_if_t<DescriptorIndex == portDescriptorSize<Node, Direction>(), void>* = nullptr>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr std::size_t portSize_()
{
    // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
    return 0U;
}

template <
    typename Node, PortDirection Direction, size_t DescriptorIndex,
    typename std::enable_if_t<DescriptorIndex<portDescriptorSize<Node, Direction>(), void>* = nullptr>
    // coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
    constexpr std::size_t portSize_()
{
    return descriptorPortSize<Node, Direction, DescriptorIndex>() + portSize_<Node, Direction, DescriptorIndex + 1>();
}

} // namespace detail

template <typename Node, PortDirection Direction>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr std::size_t portSize()
{
    return detail::portSize_<Node, Direction, 0>();
}

// Descriptor index from port index
namespace detail
{

template <
    typename Node, PortDirection Direction, size_t DescriptorIndex, size_t RemainingPortIndex,
    typename std::enable_if_t<DescriptorIndex == portDescriptorSize<Node, Direction>(), void>* = nullptr>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr std::size_t descriptorIndex_()
{
    // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
    return 0U;
}

template <
    typename Node, PortDirection Direction, size_t DescriptorIndex, size_t RemainingPortIndex,
    typename std::enable_if_t<DescriptorIndex<portDescriptorSize<Node, Direction>(), void>* = nullptr>
    // coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
    constexpr std::size_t descriptorIndex_()
{
    // coverity[autosar_cpp14_a5_1_1_violation] FP: nvbugs/3364868
    if (RemainingPortIndex < descriptorPortSize<Node, Direction, DescriptorIndex>())
    {
        // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
        return 0U;
    }
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2813925
    // coverity[autosar_cpp14_a5_1_1_violation] FP: nvbugs/3364868
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2813925
    constexpr size_t remainingPortIndex{RemainingPortIndex - descriptorPortSize<Node, Direction, DescriptorIndex>()};
    // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
    return 1U + descriptorIndex_<Node, Direction, DescriptorIndex + 1, remainingPortIndex>();
}

} // namespace detail

template <typename Node, PortDirection Direction, size_t PortIndex>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr size_t descriptorIndex()
{
    if (Direction == PortDirection::OUTPUT)
    {
        return detail::descriptorIndex_<Node, Direction, 0, PortIndex - portSize<Node, PortDirection::INPUT>()>();
    }
    return detail::descriptorIndex_<Node, Direction, 0, PortIndex>();
}

template <typename Node, PortDirection Direction, size_t PortIndex>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr dw::core::StringView portName()
{
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2813925
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2813925
    constexpr size_t index{descriptorIndex<Node, Direction, PortIndex>()};
    return dw::core::get<index>(describePorts<Node, Direction>()).name;
}

// Return type is the type of the port, to be used with decltype()
template <typename Node, PortDirection Direction, size_t PortIndex>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
// coverity[autosar_cpp14_a7_1_5_violation]
constexpr auto portType()
{
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2813925
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2813925
    constexpr size_t index{descriptorIndex<Node, Direction, PortIndex>()};
    return portDescriptorType<Node, Direction, index>();
}

// Check if port index is valid
template <typename Node, PortDirection Direction>
constexpr bool isValidPortIndex(std::size_t portID)
{
    // only temporarily for backward compatibility with enum value
    // output port indices are offset by the number of input ports
    if (Direction == PortDirection::OUTPUT)
    {
        return portID >= portSize<Node, PortDirection::INPUT>() && portID < portSize<Node, PortDirection::INPUT>() + portSize<Node, Direction>();
    }
    return portID < portSize<Node, Direction>();
}

// Array size for an array port name, 0 for non-array ports
namespace detail
{

template <
    typename Node, PortDirection Direction, size_t DescriptorIndex,
    typename std::enable_if_t<DescriptorIndex == portDescriptorSize<Node, Direction>(), void>* = nullptr>
constexpr std::size_t portArraySize_(StringView identifier)
{
    (void)identifier;
    return 0;
}

template <
    typename Node, PortDirection Direction, size_t DescriptorIndex,
    typename std::enable_if_t<DescriptorIndex<portDescriptorSize<Node, Direction>(), void>* = nullptr>
    // coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
    constexpr std::size_t portArraySize_(StringView identifier)
{
    constexpr auto descriptorName = dw::core::get<DescriptorIndex>(describePorts<Node, Direction>()).name;
    if (descriptorName == identifier)
    {
        return descriptorPortSize<Node, Direction, DescriptorIndex>();
    }
    return portArraySize_<Node, Direction, DescriptorIndex + 1>(identifier);
}

} // namespace detail

template <typename Node, PortDirection Direction>
constexpr size_t portArraySize(StringView identifier)
{
    return detail::portArraySize_<Node, Direction, 0>(identifier);
}

// Get the port index for a give port name
namespace detail
{

template <
    typename Node, PortDirection Direction, size_t DescriptorIndex,
    typename std::enable_if_t<DescriptorIndex == portDescriptorSize<Node, Direction>(), void>* = nullptr>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr std::size_t portIndex_(StringView identifier)
{
    static_cast<void>(identifier);
    // since output port indices follow input port indices
    // this must add the number of output port for invalid input port identifier
    // to avoid that for an invalid input port identifier the index of the first output port is returned
    if (Direction == PortDirection::INPUT)
    {
        return dw::framework::portSize<Node, PortDirection::OUTPUT>();
    }
    // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
    return 0U;
}

template <
    typename Node, PortDirection Direction, size_t DescriptorIndex,
    typename std::enable_if_t<DescriptorIndex<portDescriptorSize<Node, Direction>(), void>* = nullptr>
    // coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
    constexpr std::size_t portIndex_(StringView identifier)
{
    constexpr StringView descriptorName{dw::core::get<DescriptorIndex>(describePorts<Node, Direction>()).name};
    if (descriptorName == identifier)
    {
        // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
        return 0U;
    }
    return descriptorPortSize<Node, Direction, DescriptorIndex>() + portIndex_<Node, Direction, DescriptorIndex + 1>(identifier);
}

} // namespace detail

template <typename Node, PortDirection Direction>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr size_t portIndex(StringView identifier)
{
    // only temporarily for backward compatibility with enum value
    // output port indices are offset by the number of input ports
    if (Direction == PortDirection::OUTPUT)
    {
        return portSize<Node, PortDirection::INPUT>() + detail::portIndex_<Node, Direction, 0>(identifier);
    }
    return detail::portIndex_<Node, Direction, 0>(identifier);
}

// Check if given string is a valid port name
template <typename Node, PortDirection Direction>
constexpr bool isValidPortIdentifier(StringView identifier)
{
    constexpr size_t index = portIndex<Node, Direction>(identifier);
    return isValidPortIndex<Node, Direction>(index);
}

// Get the port index for a give port name
namespace detail
{

template <
    typename Node, PortDirection Direction, size_t DescriptorIndex,
    typename std::enable_if_t<DescriptorIndex == portDescriptorSize<Node, Direction>(), void>* = nullptr>
constexpr std::size_t portDescriptorIndex_(StringView identifier)
{
    (void)identifier;
    return 0;
}

template <
    typename Node, PortDirection Direction, size_t DescriptorIndex,
    typename std::enable_if_t<DescriptorIndex<portDescriptorSize<Node, Direction>(), void>* = nullptr>
    // coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
    constexpr std::size_t portDescriptorIndex_(StringView identifier)
{
    constexpr auto descriptorName = dw::core::get<DescriptorIndex>(describePorts<Node, Direction>()).name;
    if (descriptorName == identifier)
    {
        return 0;
    }
    return 1 + portDescriptorIndex_<Node, Direction, DescriptorIndex + 1>(identifier);
}

} // namespace detail

template <typename Node, PortDirection Direction>
constexpr size_t portDescriptorIndex(StringView identifier)
{
    return detail::portDescriptorIndex_<Node, Direction, 0>(identifier);
}

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_PORTDESCRIPTOR_HPP_
