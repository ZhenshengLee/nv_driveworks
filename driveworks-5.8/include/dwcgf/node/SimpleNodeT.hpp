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

#ifndef DW_FRAMEWORK_SIMPLENODET_HPP_
#define DW_FRAMEWORK_SIMPLENODET_HPP_

#include <dwcgf/node/SimpleNode.hpp>
#include <dwcgf/pass/PassDescriptor.hpp>
#include <dwcgf/port/PortCollectionDescriptor.hpp>
#include <dwcgf/port/PortDescriptor.hpp>

/// Get the index of an input port from the string identifier.
#define NODE_GET_INPUT_PORT_INDEX(identifier) dw::framework::portIndex<NodeT, dw::framework::PortDirection::INPUT>(identifier)
/// Get the index of an output port from the string identifier.
#define NODE_GET_OUTPUT_PORT_INDEX(identifier) dw::framework::portIndex<NodeT, dw::framework::PortDirection::OUTPUT>(identifier)

/// Register a pass function with the node base class.
/**
 * The macro should be called for each pass in the constructor of a concrete node.
 * @see dw::framework::SimpleNode::registerPass
 */
#define NODE_REGISTER_PASS(identifier, ...) this->template registerPass<NodeT, dw::framework::passIndex<NodeT>(identifier)>(__VA_ARGS__)

/// Initialize a non-array input port with the node base class.
/**
 * The macro should be called for each input port in the constructor of a concrete node.
 * @see dw::framework::SimpleNode::initInputPort
 */
#define NODE_INIT_INPUT_PORT(identifier, ...) \
    this->template initInputPort<NodeT, dw::framework::portIndex<NodeT, dw::framework::PortDirection::INPUT>(identifier)>(__VA_ARGS__)
/// Initialize an array input port with the node base class.
/**
 * The macro should be called for each input port in the constructor of a concrete node.
 * @see dw::framework::SimpleNode::initInputArrayPort
 */
#define NODE_INIT_INPUT_ARRAY_PORT(identifier, ...) \
    this->template initInputArrayPort<NodeT, dw::framework::portIndex<NodeT, dw::framework::PortDirection::INPUT>(identifier)>(__VA_ARGS__)
/// Initialize a non-array output port with the node base class.
/**
 * The macro should be called for each output port in the constructor of a concrete node.
 * @see dw::framework::SimpleNode::initOutputPort
 */
#define NODE_INIT_OUTPUT_PORT(identifier, ...) \
    this->template initOutputPort<NodeT, dw::framework::portIndex<NodeT, dw::framework::PortDirection::OUTPUT>(identifier)>(__VA_ARGS__)
/// Initialize an array output port with the node base class.
/**
 * The macro should be called for each output port in the constructor of a concrete node.
 * @see dw::framework::SimpleNode::initOutputArrayPort
 */
#define NODE_INIT_OUTPUT_ARRAY_PORT(identifier, ...) \
    this->template initOutputArrayPort<NodeT, dw::framework::portIndex<NodeT, dw::framework::PortDirection::OUTPUT>(identifier)>(__VA_ARGS__)

/// Get a previously initialized non-array input port.
/**
 * @see dw::framework::SimpleNode::getInputPort()
 */
#define NODE_GET_INPUT_PORT(identifier) this->template getInputPort<NodeT, NODE_GET_INPUT_PORT_INDEX(identifier)>()
/// Get one specific input port of a previously initialized array input port.
/**
 * @see dw::framework::SimpleNode::getInputPort(size_t)
 */
#define NODE_GET_INPUT_ARRAY_PORT(identifier, index) this->template getInputPort<NodeT, NODE_GET_INPUT_PORT_INDEX(identifier)>(index)
/// Get a previously initialized non-array output port.
/**
 * @see dw::framework::SimpleNode::getOutputPort()
 */
#define NODE_GET_OUTPUT_PORT(identifier) this->template getOutputPort<NodeT, NODE_GET_OUTPUT_PORT_INDEX(identifier)>()
/// Get one specific output port of a previously initialized array output port.
/**
 * @see dw::framework::SimpleNode::getOutputPort(size_t)
 */
#define NODE_GET_OUTPUT_ARRAY_PORT(identifier, index) this->template getOutputPort<NodeT, NODE_GET_OUTPUT_PORT_INDEX(identifier)>(index)

namespace dw
{
namespace framework
{

template <typename T>
class SimpleNodeT : public SimpleNode
{
public:
    using NodeT = T;

    /// Default constructor registering the setup and teardown passes.
    SimpleNodeT()
        : SimpleNode(createAllocationParams<NodeT>())
    {
        initialize();
    }

    SimpleNodeT(NodeAllocationParams params)
        : SimpleNode(params)
    {
        initialize();
    }

    // TODO(chale): make final virtual once other users of this class are migrated.
    /// @deprecated When using SimpleProcessNodeT as the base class for a concrete node this methods shouldn't be invoked since the base class provides a default implementation for the setup pass in SimpleProcessNodeT::setupImpl.
    std::unique_ptr<Pass> createSetupPass() override
    {
        throw Exception(DW_CALL_NOT_ALLOWED, "Not meant to be called");
    }

    // TODO(chale): make final virtual once other users of this class are migrated.
    /// @deprecated When using SimpleProcessNodeT as the base class for a concrete node this methods shouldn't be invoked since the base class provides a default implementation for the setup pass in SimpleProcessNodeT::teardownImpl.
    std::unique_ptr<Pass> createTeardownPass() override
    {
        throw Exception(DW_CALL_NOT_ALLOWED, "Not meant to be called");
    }

    /// The default implementation calls SimpleNode::setup.
    virtual dwStatus setupImpl()
    {
        return this->setup();
    }

    /// The default implementation calls SimpleNode::teardown.
    virtual dwStatus teardownImpl()
    {
        return this->teardown();
    }

    /// The default implementation calls SimpleNode::resetPorts.
    dwStatus reset() override
    {
        this->resetPorts();
        return DW_SUCCESS;
    }

    /// Validate that all registered ports which have the flag binding-required are bound to a channel.
    dwStatus validate() override
    {
        if (getRegisteredInputPorts().empty() && getRegisteredOutputPorts().empty())
        {
            throw Exception(DW_NOT_IMPLEMENTED, "Not implemented");
        }

        const PortCollectionDescriptor inputPorts = createPortCollectionDescriptor<NodeT, PortDirection::INPUT>();
        dwStatus status                           = SimpleNode::validate("input", inputPorts, getRegisteredInputPorts());
        if (status != DW_SUCCESS)
        {
            return status;
        }
        const PortCollectionDescriptor outputPorts = createPortCollectionDescriptor<NodeT, PortDirection::OUTPUT>();
        return SimpleNode::validate("output", outputPorts, getRegisteredOutputPorts(), inputPorts.getPortSize());
    }

private:
    void initialize()
    {
        NODE_REGISTER_PASS("SETUP"_sv, [this]() -> dwStatus {
            return setupImpl();
        });
        NODE_REGISTER_PASS("TEARDOWN"_sv, [this]() -> dwStatus {
            return teardownImpl();
        });
    }
};

/// @deprecated Use SimpleNodeT<T> instead.
template <typename T>
class SimpleSensorNodeT : public SimpleNodeT<T>, public SimpleSensorNode
{
};

/// @deprecated Use SimpleNodeT<T> instead.
template <typename T>
class SimpleProcessNodeT : public SimpleNodeT<T>
{
};

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_SIMPLENODET_HPP_
