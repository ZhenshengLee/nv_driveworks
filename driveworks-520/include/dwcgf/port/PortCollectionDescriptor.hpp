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

#ifndef DW_FRAMEWORK_PORTCOLLECTIONDESCRIPTOR_HPP_
#define DW_FRAMEWORK_PORTCOLLECTIONDESCRIPTOR_HPP_

#include <dwcgf/port/Port.hpp>
#include <dwcgf/port/PortDescriptor.hpp>

#include <dwshared/dwfoundation/dw/core/container/BaseString.hpp>
#include <dwshared/dwfoundation/dw/core/container/StringView.hpp>
#include <dwshared/dwfoundation/dw/core/container/VectorFixed.hpp>
#include <dwshared/dwfoundation/dw/core/language/Tuple.hpp>

#include <string>
#include <typeinfo>

/// External, get the index of an input port from the string identifier.
#define NODE_GET_INPUT_PORT_INDEX_EXTERNAL(NodeT, identifier) dw::framework::portIndex<NodeT, dw::framework::PortDirection::INPUT>(identifier)
/// External, get the index of an output port from the string identifier.
#define NODE_GET_OUTPUT_PORT_INDEX_EXTERNAL(NodeT, identifier) dw::framework::portIndex<NodeT, dw::framework::PortDirection::OUTPUT>(identifier)

namespace dw
{
namespace framework
{

class PortDescriptor
{
public:
    PortDescriptor(
        PortDirection direction, dw::core::StringView name,
        dw::core::StringView typeName, size_t arraySize, bool bindingRequired,
        dw::core::StringView comment);
    PortDirection getDirection() const;

    const dw::core::StringView& getName() const;

    const dw::core::StringView& getTypeName() const;

    bool isArray() const;

    size_t getArraySize() const;

    bool isBindingRequired() const;

    const dw::core::StringView& getComment() const;

private:
    PortDirection m_direction;
    dw::core::StringView m_name;
    dw::core::StringView m_typeName;
    size_t m_arraySize;
    bool m_bindingRequired;
    dw::core::StringView m_comment;
};

// coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2813925
// coverity[autosar_cpp14_a5_1_1_violation] FP: nvbugs/4020293
// coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2813925
static constexpr const size_t MAX_PORT_DESCRIPTOR_PER_COLLECTION{256U};

class PortCollectionDescriptor
{
public:
    PortCollectionDescriptor(PortDirection direction, size_t portOffset = 0U);

    PortDirection getDirection() const;

    size_t getDescriptorSize() const;

    size_t getPortSize() const;

    const PortDescriptor& getDescriptor(size_t index) const;

    size_t getDescriptorIndex(const char* identifier) const;

    const PortDescriptor& getDescriptor(const char* identifier) const;

    size_t getPortIndex(const char* identifier) const;

    bool isValid(size_t portIndex) const;

    bool isValid(const char* identifier) const;

    void addDescriptor(const PortDescriptor& descriptor);

protected:
    PortDirection m_direction;
    size_t m_portOffset;
    dw::core::VectorFixed<PortDescriptor, MAX_PORT_DESCRIPTOR_PER_COLLECTION> m_descriptors;
};

namespace detail
{

template <
    typename NodeT, PortDirection Direction, size_t Index,
    typename std::enable_if_t<Index == dw::core::tuple_size<decltype(describePorts<NodeT, Direction>())>::value, void>* = nullptr>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
void addDescriptors(PortCollectionDescriptor& d)
{
    static_cast<void>(d);
    return;
}

template <
    typename NodeT, PortDirection Direction, size_t Index,
    typename std::enable_if_t<Index<dw::core::tuple_size<decltype(describePorts<NodeT, Direction>())>::value, void>* = nullptr>
    // coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
    void addDescriptors(PortCollectionDescriptor& d)
{
    // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
    constexpr auto t = dw::core::get<Index>(describePorts<NodeT, Direction>());
    d.addDescriptor(PortDescriptor(
        Direction,
        t.name.data(),
        t.typeName.data(),
        t.arraySize,
        PortBinding::REQUIRED == t.binding,
        t.comment.data()));
    addDescriptors<NodeT, Direction, Index + 1>(d);
}

} // namespace detail

template <
    typename NodeT, PortDirection Direction,
    typename std::enable_if_t<Direction == PortDirection::INPUT, void>* = nullptr>
// coverity[autosar_cpp14_a2_10_4_violation] FP: nvbugs/4040101
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
static PortCollectionDescriptor createPortCollectionDescriptor()
{
    PortCollectionDescriptor d{Direction};
    detail::addDescriptors<NodeT, Direction, 0>(d);
    return d;
}

template <
    typename NodeT, PortDirection Direction,
    typename std::enable_if_t<Direction == PortDirection::OUTPUT, void>* = nullptr>
// coverity[autosar_cpp14_a2_10_4_violation] FP: nvbugs/4040101
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
static PortCollectionDescriptor createPortCollectionDescriptor()
{
    PortCollectionDescriptor d{Direction, portSize<NodeT, PortDirection::INPUT>()};
    detail::addDescriptors<NodeT, Direction, 0>(d);
    return d;
}

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_PORTCOLLECTIONDESCRIPTOR_HPP_
