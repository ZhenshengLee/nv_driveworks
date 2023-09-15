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

#ifndef DW_FRAMEWORK_NODEFACTORY_HPP_
#define DW_FRAMEWORK_NODEFACTORY_HPP_

#include <dwcgf/node/Node.hpp>

#include <dwshared/dwfoundation/dw/core/logger/Logger.hpp>
#include <dwcgf/parameter/ParameterCollectionDescriptor.hpp>
#include <dwcgf/pass/PassCollectionDescriptor.hpp>
#include <dwcgf/port/ManagedPort.hpp>
#include <dwcgf/port/PortCollectionDescriptor.hpp>

#include <dwshared/dwfoundation/dw/core/container/VectorFixed.hpp>
#include <dwshared/dwfoundation/dw/core/container/StringView.hpp>

#include <iostream>
#include <map>
#include <memory>
#include <mutex>

namespace dw
{
namespace framework
{

class Node;
class ParameterProvider;

namespace detail
{

class AbstractMetaObject
{
public:
    AbstractMetaObject(const dw::core::StringView className);

    virtual ~AbstractMetaObject() = default;

    const dw::core::StringView& className() const;

    // coverity[autosar_cpp14_a2_10_5_violation] RFD Pending: TID-2053
    virtual const PortCollectionDescriptor& getInputPorts() const = 0;

    // coverity[autosar_cpp14_a2_10_5_violation] RFD Pending: TID-2053
    virtual const PortCollectionDescriptor& getOutputPorts() const = 0;

    // coverity[autosar_cpp14_a2_10_5_violation] RFD Pending: TID-2053
    virtual const ParameterCollectionDescriptor& getParameters() const = 0;

    // coverity[autosar_cpp14_a2_10_5_violation] RFD Pending: TID-2053
    virtual const PassCollectionDescriptor& getPasses() const = 0;

    // coverity[autosar_cpp14_a2_10_5_violation] RFD Pending: TID-2053
    virtual std::unique_ptr<Node> create(ParameterProvider& provider) const = 0;

    // coverity[autosar_cpp14_a2_10_5_violation] RFD Pending: TID-2053
    virtual GenericDataReference createInputPortSpecimen(const dw::core::StringView& identifier) const = 0;

    // coverity[autosar_cpp14_a2_10_5_violation] RFD Pending: TID-2053
    virtual GenericDataReference createOutputPortSpecimen(const dw::core::StringView& identifier) const = 0;

protected:
    dw::core::StringView m_className;
};

using FactoryMap = std::map<dw::core::StringView, std::unique_ptr<AbstractMetaObject>>;

FactoryMap& getFactoryMap();

std::recursive_mutex& getFactoryMapMutex();

// coverity[autosar_cpp14_a14_1_1_violation]
template <typename NodeT>
class MetaObject : public AbstractMetaObject
{
public:
    using AbstractMetaObject::AbstractMetaObject;

    // coverity[autosar_cpp14_a2_10_5_violation] RFD Pending: TID-2053
    const PortCollectionDescriptor& getInputPorts() const override
    {
        // coverity[autosar_cpp14_a3_3_2_violation]
        static const PortCollectionDescriptor descriptor{createPortCollectionDescriptor<NodeT, PortDirection::INPUT>()};
        return descriptor;
    }

    // coverity[autosar_cpp14_a2_10_5_violation] RFD Pending: TID-2053
    const PortCollectionDescriptor& getOutputPorts() const override
    {
        // coverity[autosar_cpp14_a3_3_2_violation]
        static const PortCollectionDescriptor descriptor{createPortCollectionDescriptor<NodeT, PortDirection::OUTPUT>()};
        return descriptor;
    }

    // coverity[autosar_cpp14_a2_10_5_violation] RFD Pending: TID-2053
    const ParameterCollectionDescriptor& getParameters() const override
    {
        // coverity[autosar_cpp14_a3_3_2_violation]
        static const ParameterCollectionDescriptor descriptor{createParameterCollectionDescriptor<NodeT>()};
        return descriptor;
    }

    // coverity[autosar_cpp14_a2_10_5_violation] RFD Pending: TID-2053
    const PassCollectionDescriptor& getPasses() const override
    {
        // coverity[autosar_cpp14_a3_3_2_violation]
        static const PassCollectionDescriptor descriptor{createPassCollectionDescriptor<NodeT>()};
        return descriptor;
    }

    // coverity[autosar_cpp14_a2_10_5_violation] RFD Pending: TID-2053
    std::unique_ptr<Node> create(ParameterProvider& provider) const override
    {
        return NodeT::create(provider);
    }

    // coverity[autosar_cpp14_a2_10_5_violation] RFD Pending: TID-2053
    GenericDataReference createInputPortSpecimen(const dw::core::StringView& identifier) const override
    {
        size_t const inputDescriptorIndex{getInputPorts().getDescriptorIndex(identifier.data())};
        return dw::framework::detail::createPortSpecimen<NodeT, PortDirection::INPUT>(inputDescriptorIndex);
    }

    // coverity[autosar_cpp14_a2_10_5_violation] RFD Pending: TID-2053
    GenericDataReference createOutputPortSpecimen(const dw::core::StringView& identifier) const override
    {
        size_t const outputDescriptorIndex{getOutputPorts().getDescriptorIndex(identifier.data())};
        return dw::framework::detail::createPortSpecimen<NodeT, PortDirection::OUTPUT>(outputDescriptorIndex);
    }
};

} // namespace detail

template <typename NodeT>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
void registerNode(const char* className)
{
    std::unique_ptr<detail::MetaObject<NodeT>> metaObject{std::make_unique<detail::MetaObject<NodeT>>(className)};
    if (metaObject.get() == nullptr)
    {
        throw ExceptionWithStatus(DW_BAD_ALLOC, "NodeFactory: cannot allocate meta object");
    }

    // coverity[autosar_cpp14_m0_1_3_violation] RFD Accepted: TID-1995
    std::lock_guard<std::recursive_mutex> lock{detail::getFactoryMapMutex()};
    detail::FactoryMap& factoryMap{detail::getFactoryMap()};
    if (factoryMap.find(className) != factoryMap.end())
    {
        // coverity[cert_con51_cpp_violation] FP: nvbugs/3632417
        throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "registerNode() repeatedly called for the same class name: ", className);
    }
    else
    {
        factoryMap[className] = std::move(metaObject);
    }
}

dw::core::HeapVectorFixed<dw::core::StringView> getNodeNames();

const PortCollectionDescriptor& getInputPorts(const dw::core::StringView& className);

const PortCollectionDescriptor& getOutputPorts(const dw::core::StringView& className);

const ParameterCollectionDescriptor& getParameters(const dw::core::StringView& className);

const PassCollectionDescriptor& getPasses(const dw::core::StringView& className);

std::unique_ptr<Node> createNode(const dw::core::StringView& className, ParameterProvider& provider);

GenericDataReference createInputPortSpecimen(
    const dw::core::StringView& className,
    const dw::core::StringView& identifier);

GenericDataReference createOutputPortSpecimen(
    const dw::core::StringView& className,
    const dw::core::StringView& identifier);

} // namespace framework
} // namespace dw

#define DW_REGISTER_NODE_WITH_SUFFIX_(NodeT, UniqueSuffix)     \
    namespace                                                  \
    {                                                          \
    class Proxy##UniqueSuffix                                  \
    {                                                          \
    public:                                                    \
        Proxy##UniqueSuffix()                                  \
        {                                                      \
            dw::framework::registerNode<NodeT>(#NodeT);        \
        }                                                      \
    };                                                         \
    static Proxy##UniqueSuffix g_registerNode##UniqueSuffix{}; \
    } // namespace

#define DW_REGISTER_NODE_EXPAND_(NodeT, UniqueSuffixMacro) DW_REGISTER_NODE_WITH_SUFFIX_(NodeT, UniqueSuffixMacro)

#define DW_REGISTER_NODE(NodeT) DW_REGISTER_NODE_EXPAND_(NodeT, __LINE__)

#endif //DW_FRAMEWORK_NODEFACTORY_HPP_
