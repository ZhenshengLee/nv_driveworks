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

#ifndef DW_FRAMEWORK_NODEFACTORY_HPP_
#define DW_FRAMEWORK_NODEFACTORY_HPP_

#include <dwcgf/node/Node.hpp>

#include <dw/core/logger/Logger.hpp>
#include <dwcgf/parameter/ParameterCollectionDescriptor.hpp>
#include <dwcgf/pass/PassCollectionDescriptor.hpp>
#include <dwcgf/port/PortCollectionDescriptor.hpp>

#include <dw/core/container/VectorFixed.hpp>
#include <dw/core/container/StringView.hpp>

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

    virtual const PortCollectionDescriptor& getInputPorts() const = 0;

    virtual const PortCollectionDescriptor& getOutputPorts() const = 0;

    virtual const ParameterCollectionDescriptor& getParameters() const = 0;

    virtual const PassCollectionDescriptor& getPasses() const = 0;

    virtual std::unique_ptr<Node> create(ParameterProvider& provider) const = 0;

protected:
    dw::core::StringView m_className;
};

typedef std::map<dw::core::StringView, std::unique_ptr<AbstractMetaObject>> FactoryMap;

FactoryMap& getFactoryMap();

std::recursive_mutex& getFactoryMapMutex();

template <typename NodeT>
class MetaObject : public AbstractMetaObject
{
public:
    MetaObject(const dw::core::StringView className)
        : AbstractMetaObject(std::move(className))
    {
    }

    const PortCollectionDescriptor& getInputPorts() const override
    {
        static const PortCollectionDescriptor descriptor = createPortCollectionDescriptor<NodeT, PortDirection::INPUT>();
        return descriptor;
    }

    const PortCollectionDescriptor& getOutputPorts() const override
    {
        static const PortCollectionDescriptor descriptor = createPortCollectionDescriptor<NodeT, PortDirection::OUTPUT>();
        return descriptor;
    }

    const ParameterCollectionDescriptor& getParameters() const override
    {
        static const ParameterCollectionDescriptor descriptor = createParameterCollectionDescriptor<NodeT>();
        return descriptor;
    }

    const PassCollectionDescriptor& getPasses() const override
    {
        static const PassCollectionDescriptor descriptor = createPassCollectionDescriptor<NodeT>();
        return descriptor;
    }

    std::unique_ptr<Node> create(ParameterProvider& provider) const override
    {
        return NodeT::create(provider);
    }
};

} // namespace detail

template <typename NodeT>
void registerNode(const char* className)
{
    auto metaObject = std::make_unique<detail::MetaObject<NodeT>>(className);
    if (!metaObject)
    {
        throw ExceptionWithStatus(DW_BAD_ALLOC, "NodeFactory: cannot allocate meta object");
    }

    std::lock_guard<std::recursive_mutex> lock(detail::getFactoryMapMutex());
    auto& factoryMap = detail::getFactoryMap();
    if (factoryMap.find(className) != factoryMap.end())
    {
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

} // namespace framework
} // namespace dw

#define _DW_REGISTER_NODE_WITH_SUFFIX(NodeT, UniqueSuffix)     \
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

#define _DW_REGISTER_NODE(NodeT) \
    _DW_REGISTER_NODE_WITH_SUFFIX(NodeT, UniqueSuffix)

#define _DW_REGISTER_NODE_GET_3RD_ARG(arg1, arg2, arg3, ...) arg3
#define _DW_REGISTER_NODE_MACRO_CHOOSER(...) \
    _DW_REGISTER_NODE_GET_3RD_ARG(__VA_ARGS__, _DW_REGISTER_NODE_WITH_SUFFIX, _DW_REGISTER_NODE, )

#define DW_REGISTER_NODE(...)                    \
    _DW_REGISTER_NODE_MACRO_CHOOSER(__VA_ARGS__) \
    (__VA_ARGS__)

#endif //DW_FRAMEWORK_NODEFACTORY_HPP_
