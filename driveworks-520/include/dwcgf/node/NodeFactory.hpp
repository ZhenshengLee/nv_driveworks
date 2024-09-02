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
#include <vector>

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
    AbstractMetaObject(dw::core::StringView&& className);

    virtual ~AbstractMetaObject() = default;

    const dw::core::StringView& className() const;

    // coverity[autosar_cpp14_a2_10_5_violation]
    virtual const PortCollectionDescriptor& getInputPorts() const = 0;

    // coverity[autosar_cpp14_a2_10_5_violation]
    virtual const PortCollectionDescriptor& getOutputPorts() const = 0;

    // coverity[autosar_cpp14_a2_10_5_violation]
    virtual const ParameterCollectionDescriptor& getParameters() const = 0;

    // coverity[autosar_cpp14_a2_10_5_violation]
    virtual const PassCollectionDescriptor& getPasses() const = 0;

    // coverity[autosar_cpp14_a2_10_5_violation]
    virtual std::unique_ptr<Node> create(ParameterProvider& provider) const = 0;

    // coverity[autosar_cpp14_a2_10_5_violation]
    virtual GenericDataReference createInputPortSpecimen(const dw::core::StringView& identifier) const = 0;

    // coverity[autosar_cpp14_a2_10_5_violation]
    virtual GenericDataReference createOutputPortSpecimen(const dw::core::StringView& identifier) const = 0;

protected:
    dw::core::StringView m_className;
};

using FactoryMap = std::map<dw::core::StringView, std::unique_ptr<AbstractMetaObject>>;

FactoryMap& getFactoryMap();

using FactoryErrorMap = std::map<dw::core::StringView, std::vector<dw::core::StringView>>;

FactoryErrorMap& getFactoryErrorMap();

std::recursive_mutex& getFactoryMutex();

template <typename NodeT>
class MetaObject : public AbstractMetaObject
{
    static_assert(std::is_base_of<Node, NodeT>::value, "NodeT must inherit from Node");

public:
    using AbstractMetaObject::AbstractMetaObject;

    // coverity[autosar_cpp14_a2_10_5_violation]
    const PortCollectionDescriptor& getInputPorts() const override
    {
        // coverity[autosar_cpp14_a3_3_2_violation] RFD Pending: TID-2534
        static const PortCollectionDescriptor descriptor{createPortCollectionDescriptor<NodeT, PortDirection::INPUT>()}; // LCOV_EXCL_LINE branches for thrown exceptions in defensive code can't be triggered
        return descriptor;
    }

    // coverity[autosar_cpp14_a2_10_5_violation]
    const PortCollectionDescriptor& getOutputPorts() const override
    {
        // coverity[autosar_cpp14_a3_3_2_violation] RFD Pending: TID-2534
        static const PortCollectionDescriptor descriptor{createPortCollectionDescriptor<NodeT, PortDirection::OUTPUT>()}; // LCOV_EXCL_LINE branches for thrown exceptions in defensive code can't be triggered
        return descriptor;
    }

    // coverity[autosar_cpp14_a2_10_5_violation]
    const ParameterCollectionDescriptor& getParameters() const override
    {
        // coverity[autosar_cpp14_a3_3_2_violation] RFD Pending: TID-2534
        static const ParameterCollectionDescriptor descriptor{createParameterCollectionDescriptor<NodeT>()}; // LCOV_EXCL_LINE branches for thrown exceptions in defensive code can't be triggered
        return descriptor;
    }

    // coverity[autosar_cpp14_a2_10_5_violation]
    const PassCollectionDescriptor& getPasses() const override
    {
        // coverity[autosar_cpp14_a3_3_2_violation] RFD Pending: TID-2534
        static const PassCollectionDescriptor descriptor{createPassCollectionDescriptor<NodeT>()}; // LCOV_EXCL_LINE branches for thrown exceptions in defensive code can't be triggered
        return descriptor;
    }

    // coverity[autosar_cpp14_a2_10_5_violation]
    std::unique_ptr<Node> create(ParameterProvider& provider) const override
    {
        return NodeT::create(provider);
    }

    // coverity[autosar_cpp14_a2_10_5_violation]
    GenericDataReference createInputPortSpecimen(const dw::core::StringView& identifier) const override
    {
        size_t const inputDescriptorIndex{getInputPorts().getDescriptorIndex(identifier.data())};
        return dw::framework::detail::createPortSpecimen<NodeT, PortDirection::INPUT>(inputDescriptorIndex);
    }

    // coverity[autosar_cpp14_a2_10_5_violation]
    GenericDataReference createOutputPortSpecimen(const dw::core::StringView& identifier) const override
    {
        size_t const outputDescriptorIndex{getOutputPorts().getDescriptorIndex(identifier.data())};
        return dw::framework::detail::createPortSpecimen<NodeT, PortDirection::OUTPUT>(outputDescriptorIndex);
    }
};

} // namespace detail

template <typename NodeT>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
void registerNode(const char* className) noexcept
{
    std::unique_ptr<detail::MetaObject<NodeT>> metaObject{std::make_unique<detail::MetaObject<NodeT>>(className)};

    // coverity[autosar_cpp14_m0_1_3_violation] RFD Accepted: TID-1995
    std::lock_guard<std::recursive_mutex> lock{detail::getFactoryMutex()};
    detail::FactoryMap& factoryMap{detail::getFactoryMap()};
    if (factoryMap.find(metaObject->className()) != factoryMap.end())
    {
        detail::FactoryErrorMap& factoryErrorMap{detail::getFactoryErrorMap()};
        factoryErrorMap[metaObject->className()].push_back(dw::core::StringView("Repeated registration of the same class name"));
        return;
    }

    factoryMap[metaObject->className()] = std::move(metaObject);
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

bool hasRegistrationErrors(bool logErrors = true);

namespace detail
{

dw::core::HeapVectorFixed<dw::core::StringView> getNodeNamesWithErrors();

dw::core::HeapVectorFixed<dw::core::StringView> getRegistrationErrors(const dw::core::StringView& className);

} // namespace detail
} // namespace framework
} // namespace dw

#define DW_CGF_NODE_FACTORY_JOIN(a, b) a##b

#define DW_REGISTER_NODE_WITH_SUFFIX_(NodeT, UniqueSuffix)                                                         \
    namespace                                                                                                      \
    {                                                                                                              \
    class DW_CGF_NODE_FACTORY_JOIN(Proxy, UniqueSuffix)                                                            \
    {                                                                                                              \
    public:                                                                                                        \
        DW_CGF_NODE_FACTORY_JOIN(Proxy, UniqueSuffix)                                                              \
        ()                                                                                                         \
        {                                                                                                          \
            dw::framework::registerNode<NodeT>(#NodeT);                                                            \
        }                                                                                                          \
    };                                                                                                             \
    static DW_CGF_NODE_FACTORY_JOIN(Proxy, UniqueSuffix) DW_CGF_NODE_FACTORY_JOIN(g_registerNode, UniqueSuffix){}; \
    } // namespace

#define DW_REGISTER_NODE_EXPAND_(NodeT, UniqueSuffixMacro) DW_REGISTER_NODE_WITH_SUFFIX_(NodeT, UniqueSuffixMacro)

#define DW_REGISTER_NODE(NodeT) DW_REGISTER_NODE_EXPAND_(NodeT, __LINE__)

#endif //DW_FRAMEWORK_NODEFACTORY_HPP_
