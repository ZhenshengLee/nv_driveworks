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
// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <dwshared/dwfoundation/dw/core/safety/Safety.hpp>

#include <array>
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

template <typename PortType, size_t ArraySize, size_t NameSize>
struct PortDescriptorT
{
    static_assert(std::is_constructible<PortType>::value, "PortType must be constructible");

    // coverity[autosar_cpp14_a0_1_6_violation]
    using Type = PortType;
    dw::core::StringView typeName;
    dw::core::StringView name;
    // coverity[autosar_cpp14_a5_1_1_violation] FP: nvbugs/3364868
    static constexpr size_t arraySize{ArraySize};
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2813925
    // coverity[autosar_cpp14_a5_1_1_violation] FP: nvbugs/3364868
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2813925
    static constexpr size_t nameSize{NameSize};
    PortBinding binding;
    dw::core::StringView comment;

    constexpr PortDescriptorT(dw::core::StringView&& typeName_, dw::core::StringView&& name_, PortBinding binding_ = PortBinding::OPTIONAL, dw::core::StringView comment_ = ""_sv)
        : typeName{std::move(typeName_)}
        , name{std::move(name_)}
        , binding{std::move(binding_)}
        , comment{std::move(comment_)}
    {
    }
};

#define DW_PORT_TYPE_NAME_STRING_VIEW_IMPL(TYPE_NAME_STR) TYPE_NAME_STR##_sv
#define DW_PORT_TYPE_NAME_STRING_VIEW(TYPE_NAME) DW_PORT_TYPE_NAME_STRING_VIEW_IMPL(#TYPE_NAME)
#define DW_DESCRIBE_PORT(TYPE_NAME, NAME, args...) dw::framework::describePort<TYPE_NAME, NAME.size()>(DW_PORT_TYPE_NAME_STRING_VIEW(TYPE_NAME), NAME, ##args)

template <typename PortType, size_t NameSize>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr auto describePort(
    dw::core::StringView typeName, dw::core::StringView name, PortBinding binding = PortBinding::OPTIONAL, dw::core::StringView comment = ""_sv) -> PortDescriptorT<PortType, 0, NameSize>
{
    return PortDescriptorT<PortType, 0, NameSize>(
        std::move(typeName),
        std::move(name),
        std::move(binding),
        std::move(comment));
}

#define DW_DESCRIBE_PORT_ARRAY(TYPE_NAME, ARRAYSIZE, NAME, args...) dw::framework::describePortArray<TYPE_NAME, ARRAYSIZE, NAME.size()>(DW_PORT_TYPE_NAME_STRING_VIEW(TYPE_NAME), NAME, ##args)
template <
    typename PortType,
    size_t ArraySize,
    size_t NameSize,
    typename std::enable_if_t<ArraySize != 0, void>* = nullptr>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr auto describePortArray(
    dw::core::StringView typeName, dw::core::StringView name, PortBinding binding = PortBinding::OPTIONAL, dw::core::StringView comment = ""_sv) -> PortDescriptorT<PortType, ArraySize, NameSize>
{
    return PortDescriptorT<PortType, ArraySize, NameSize>(
        std::move(typeName),
        std::move(name),
        std::move(binding),
        std::move(comment));
}

template <
    typename PortType,
    size_t ArraySize,
    size_t NameSize,
    typename std::enable_if_t<ArraySize != 0, void>* = nullptr>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr auto describePortArray(
    dw::core::StringView typeName, dw::core::StringView name, dw::core::StringView comment)
{
    return describePortArray<PortType, ArraySize, NameSize>(
        std::move(typeName),
        std::move(name),
        std::move(PortBinding::OPTIONAL),
        std::move(comment));
}

// API to access declared ports of a node.

// LCOV_EXCL_START no coverage data for compile time evaluated function
template <typename Node>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
// coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
constexpr auto describeNodeInputPorts()
{
    return Node::describeInputPorts();
}
// LCOV_EXCL_STOP

template <typename Node>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
// coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
constexpr auto describeNodeOutputPorts()
{
    return Node::describeOutputPorts();
}

// LCOV_EXCL_START no coverage data for compile time evaluated function
template <
    typename Node,
    PortDirection Direction,
    typename std::enable_if_t<Direction == PortDirection::INPUT, void>* = nullptr>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
// coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
constexpr auto describePorts()
{
    return describeNodeInputPorts<Node>();
}
// LCOV_EXCL_STOP

template <
    typename Node,
    PortDirection Direction,
    typename std::enable_if_t<Direction == PortDirection::OUTPUT, void>* = nullptr>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
// coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
constexpr auto describePorts()
{
    return describeNodeOutputPorts<Node>();
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
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr bool descriptorPortArray()
{
    constexpr size_t array_length{dw::core::tuple_element_t<
        DescriptorIndex,
        decltype(describePorts<Node, Direction>())>::arraySize};
    // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
    return array_length > 0U;
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
    if (0U == array_length)
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
// coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
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
    if (PortDirection::OUTPUT == Direction)
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

namespace detail
{

// coverity[autosar_cpp14_m3_4_1_violation] RFD Pending: TID-2586
constexpr const size_t DECIMAL_BASE{10U};

constexpr size_t numberOfDigits(size_t number)
{
    static_assert(std::numeric_limits<size_t>::digits10 <= std::numeric_limits<size_t>::max(), "size_t number of digits exceeds size_t (not possible)");
    // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
    if (0U == number)
    {
        // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
        return 1U;
    }
    // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
    size_t count{0U};
    // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
    while (number > 0U)
    {
        number = number / DECIMAL_BASE;
        // without this check coverity would flag violation of cert_int30_c
        if (std::numeric_limits<size_t>::max() == count)
        {
            throw std::logic_error("size_t number of digits exceeds size_t (not possible)");
        }
        ++count;
    }
    return count;
}

constexpr size_t getArrayNameSize(size_t portNameSize, size_t arrayIndex)
{
    // port name size + '[' + number of digits of the array index + ']'
    constexpr size_t MAX_SIZE_WITHOUT_BRACKETS{std::numeric_limits<size_t>::max() - 1U - 1U};
    if (portNameSize >= MAX_SIZE_WITHOUT_BRACKETS)
    {
        throw std::runtime_error("Array name too long");
    }
    if (MAX_SIZE_WITHOUT_BRACKETS - portNameSize < numberOfDigits(arrayIndex))
    {
        throw std::runtime_error("Array name + digits for array index too long");
    }
    // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
    return portNameSize + 1U + numberOfDigits(arrayIndex) + 1U;
}

template <size_t NameSize, size_t ArraySize>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
constexpr size_t getMaximumArrayNameSize()
{
    static_assert(NameSize > 0U, "Name size must not be zero");
    static_assert(ArraySize > 0U, "Array size must not be zero");
    // number of digits for the largest array index
    // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
    return getArrayNameSize(NameSize, ArraySize - 1U);
}

template <size_t NameSize, size_t ArraySize>
class PortNamesGenerator
{
public:
    static_assert(NameSize > 0U, "Name size must not be zero");
    static_assert(ArraySize > 0U, "Array size must not be zero");
    // same size for all names even though smaller indices might be shorter
    // + 1 for a null character for each name to be defensive in case the string view is used incorrectly
    static_assert(std::numeric_limits<size_t>::max() / ArraySize > getMaximumArrayNameSize<NameSize, ArraySize>() + 1U, "The storage size exceeds size_t");
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2813925
    // coverity[autosar_cpp14_a5_1_1_violation] FP: nvbugs/3364868
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2813925
    static constexpr size_t StorageSize{ArraySize * (getMaximumArrayNameSize<NameSize, ArraySize>() + 1U)};
    constexpr PortNamesGenerator(dw::core::StringView baseName)
        : m_data()
    {
        if (baseName.size() > NameSize)
        {
            // LCOV_EXCL_START the calling code uses the size of the StringView as the template parameter
            throw ExceptionWithStatus(DW_INTERNAL_ERROR, "The passed string size ", baseName.size(), " exceeds the template parameter NameSize ", NameSize);
            // LCOV_EXCL_STOP
        }
        size_t i{0U};
        for (size_t arrayIndex{0U}; ArraySize != arrayIndex; ++arrayIndex)
        {
            // copy base port name
            for (size_t j{0U}; j < baseName.size(); ++j)
            {
                m_data.at(i) = baseName[j];
                // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
                dw::core::safeIncrement(i, 1U);
            }

            // append the array index wrapped in brackets
            const char8_t OPENING_BRACKET{'['};
            m_data.at(i) = OPENING_BRACKET;
            // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
            dw::core::safeIncrement(i, 1U);

            size_t remainingValue{arrayIndex};
            // the length of the port name isn't close to the maximum value of size_t, hence no risk of overflow
            // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
            const size_t INDEX_LAST_DIGIT{dw::core::safeAdd(i, numberOfDigits(arrayIndex) - 1U).value()};
            // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
            for (size_t j{0U}; j < numberOfDigits(arrayIndex); ++j)
            {
                // fill the array index digits in reverse order
                constexpr char8_t digits[10]{'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};
                // without this check coverity would flag violation of cert_int30_c
                if (INDEX_LAST_DIGIT < j)
                {
                    throw std::logic_error("index j must never be greater than the index of the last digit");
                }
                m_data.at(INDEX_LAST_DIGIT - j) = digits[remainingValue % DECIMAL_BASE];
                // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
                dw::core::safeIncrement(i, 1U);
                remainingValue = remainingValue / DECIMAL_BASE;
            }

            const char8_t CLOSING_BRACKET{']'};
            m_data.at(i) = CLOSING_BRACKET;
            // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
            dw::core::safeIncrement(i, 1U);
            // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
            m_data.at(i) = static_cast<char8_t>(0);
            // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
            dw::core::safeIncrement(i, 1U);
            // without this check coverity would flag violation of cert_int30_c
            // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
            if (i > std::numeric_limits<size_t>::max() - numberOfDigits(ArraySize - 1U))
            {
                throw std::logic_error("index j must never be greater than the index of the last digit");
            }
            // skip delta which this name is shorter compared to the maximum length
            // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
            i += numberOfDigits(ArraySize - 1U) - numberOfDigits(arrayIndex);
        }
    }

    dw::core::StringView getName(size_t arrayIndex) const
    {
        if (arrayIndex >= ArraySize)
        {
            throw ExceptionWithStatus(DW_OUT_OF_BOUNDS, "Array index ", arrayIndex, " out of bound for array size ", ArraySize);
        }
        // coverity[autosar_cpp14_a5_1_1_violation] FP: nvbugs/3364868
        return dw::core::StringView(&m_data.at(arrayIndex * (getMaximumArrayNameSize<NameSize, ArraySize>() + 1U)), getArrayNameSize(NameSize, arrayIndex));
    }

private:
    std::array<char8_t, StorageSize> m_data;
};

} // namespace detail

template <typename Node, PortDirection Direction, size_t PortIndex>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
dw::core::StringView portName(size_t arrayIndex)
{
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2813925
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2813925
    constexpr size_t index{descriptorIndex<Node, Direction, PortIndex>()};
    // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
    constexpr auto desc = dw::core::get<index>(describePorts<Node, Direction>());
    static_assert(desc.arraySize > 0U, "A port name with an array index argument is only applicable to array ports");
    // coverity[autosar_cpp14_a3_3_2_violation] RFD Pending: TID-2534
    static const detail::PortNamesGenerator<desc.nameSize, desc.arraySize> generatedNames{desc.name};
    return generatedNames.getName(arrayIndex);
}

// Return type is the type of the port, to be used with decltype()
template <typename Node, PortDirection Direction, size_t PortIndex>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
// coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
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
    if (PortDirection::OUTPUT == Direction)
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

// LCOV_EXCL_START no coverage data for compile time evaluated function
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
    if (PortDirection::INPUT == Direction)
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
    if (PortDirection::OUTPUT == Direction)
    {
        return portSize<Node, PortDirection::INPUT>() + detail::portIndex_<Node, Direction, 0>(identifier);
    }
    return detail::portIndex_<Node, Direction, 0>(identifier);
}
// LCOV_EXCL_STOP

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
