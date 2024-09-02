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
/// Initialize all ports of an array input port with the node base class.
/**
 * The macro should be called for each array input port in the constructor of a concrete node.
 * @see dw::framework::SimpleNode::initInputArrayPorts
 *
 * If ports of the array need to be initialized with different args, use #NODE_INIT_INPUT_ARRAY_PORT instead.
 */
#define NODE_INIT_INPUT_ARRAY_PORTS(identifier, ...) \
    this->template initInputArrayPorts<NodeT, dw::framework::portIndex<NodeT, dw::framework::PortDirection::INPUT>(identifier)>(__VA_ARGS__)
/// Initialize one port of an array input port with the node base class.
/**
 * The macro should be called instead of #NODE_INIT_INPUT_ARRAY_PORTS, if ports of the array need to be initialized with different args.
 * @see dw::framework::SimpleNode::initInputArrayPort
 */
#define NODE_INIT_INPUT_ARRAY_PORT(identifier, arrayIndex, ...) \
    this->template initInputArrayPort<NodeT, dw::framework::portIndex<NodeT, dw::framework::PortDirection::INPUT>(identifier)>(arrayIndex, ##__VA_ARGS__)
/// Initialize a non-array output port with the node base class.
/**
 * The macro should be called for each output port in the constructor of a concrete node.
 * @see dw::framework::SimpleNode::initOutputPort
 */
#define NODE_INIT_OUTPUT_PORT(identifier, ...) \
    this->template initOutputPort<NodeT, dw::framework::portIndex<NodeT, dw::framework::PortDirection::OUTPUT>(identifier)>(__VA_ARGS__)
/// Initialize all ports of an array output port with the node base class.
/**
 * The macro should be called for each array output port in the constructor of a concrete node.
 * @see dw::framework::SimpleNode::initOutputArrayPorts
 *
 * If ports of the array need to be initialized with different args, use #NODE_INIT_OUTPUT_ARRAY_PORT instead.
 */
#define NODE_INIT_OUTPUT_ARRAY_PORTS(identifier, ...) \
    this->template initOutputArrayPorts<NodeT, dw::framework::portIndex<NodeT, dw::framework::PortDirection::OUTPUT>(identifier)>(__VA_ARGS__)
/// Initialize one port of an array output port with the node base class.
/**
 * The macro should be called instead of #NODE_INIT_OUTPUT_ARRAY_PORTS, if ports of the array need to be initialized with different args.
 * @see dw::framework::SimpleNode::initOutputArrayPort
 */
#define NODE_INIT_OUTPUT_ARRAY_PORT(identifier, arrayIndex, ...) \
    this->template initOutputArrayPort<NodeT, dw::framework::portIndex<NodeT, dw::framework::PortDirection::OUTPUT>(identifier)>(arrayIndex, ##__VA_ARGS__)

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

// coverity[autosar_cpp14_a14_1_1_violation] FP: nvbugs/4356873
template <typename T>
// coverity[autosar_cpp14_a12_1_6_violation] FP: nvbugs/4016780
class SimpleNodeT : public SimpleNode
{
public:
    // coverity[autosar_cpp14_a0_1_6_violation]
    using NodeT = T;

    /// Default constructor registering the setup and teardown passes.
    SimpleNodeT()
        : SimpleNodeT(createAllocationParams<NodeT>())
    {
    }

    SimpleNodeT(NodeAllocationParams params)
        : SimpleNode(params)
    {
        initialize();
    }

    // coverity[autosar_cpp14_a12_7_1_violation] FP: nvbugs/4356873
    ~SimpleNodeT()
    {
        // can't be checked within class scope since
        // a concrete class MyClass might want to inherit from SimpleNodeT<MyClass>
        // at which point T is an incomplete type in the class scope
        static_assert(std::is_base_of<Node, T>::value, "T must inherit from Node");
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
            throw ExceptionWithStatus(DW_NOT_IMPLEMENTED, "Not implemented");
        }

        const char* INPUT_DIRECTION{"input"};
        // coverity[autosar_cpp14_a18_5_8_violation] FP: nvbugs/3498833
        const PortCollectionDescriptor inputPortDescs{createPortCollectionDescriptor<NodeT, PortDirection::INPUT>()};
        dwStatus status{SimpleNode::validate(
            INPUT_DIRECTION, inputPortDescs,
            [&](size_t portIdx) -> bool {
                const dw::core::HeapHashMap<size_t, std::shared_ptr<ManagedPortInputBase>>& registeredPorts{getRegisteredInputPorts()};
                dw::core::HeapHashMap<size_t, std::shared_ptr<ManagedPortInputBase>>::const_iterator it{registeredPorts.find(portIdx)};
                if (it == registeredPorts.end())
                {
                    throw ExceptionWithStatus(DW_NOT_IMPLEMENTED, "Input port was not registered");
                }
                if (nullptr == it->second.get())
                {
                    throw ExceptionWithStatus(DW_NOT_INITIALIZED, "Input port was not initialized");
                }
                return it->second->isBound();
            })};
        // LCOV_EXCL_START defensive code can't be triggered since SimpleNode::validate() always returns success or throws
        if (DW_SUCCESS != status)
        {
            return status;
        }
        // LCOV_EXCL_STOP
        const char* OUTPUT_DIRECTION{"output"};
        const PortCollectionDescriptor outputPortDescs{createPortCollectionDescriptor<NodeT, PortDirection::OUTPUT>()};
        size_t outputPortOffset{inputPortDescs.getPortSize()};
        return SimpleNode::validate(
            OUTPUT_DIRECTION, outputPortDescs,
            [&](size_t portIdx) -> bool {
                const dw::core::HeapHashMap<size_t, std::shared_ptr<ManagedPortOutputBase>>& registeredPorts{getRegisteredOutputPorts()};
                dw::core::HeapHashMap<size_t, std::shared_ptr<ManagedPortOutputBase>>::const_iterator it{registeredPorts.find(portIdx + outputPortOffset)};
                if (it == registeredPorts.end())
                {
                    throw ExceptionWithStatus(DW_NOT_IMPLEMENTED, "Output port was not registered");
                }
                if (nullptr == it->second.get())
                {
                    throw ExceptionWithStatus(DW_NOT_INITIALIZED, "Output port was not initialized");
                }
                return it->second->isBound();
            });
    }

private:
    void initialize()
    {
        NODE_REGISTER_PASS("SETUP"_sv, [this]() -> dwStatus {
            return setupImpl();
        },
                           {{DW_NOT_AVAILABLE, 0U}});
        NODE_REGISTER_PASS("TEARDOWN"_sv, [this]() -> dwStatus {
            return teardownImpl();
        });
    }
};

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_SIMPLENODET_HPP_
