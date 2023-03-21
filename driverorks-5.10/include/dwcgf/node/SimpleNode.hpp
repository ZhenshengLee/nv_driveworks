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
// SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_FRAMEWORK_SIMPLENODE_HPP_
#define DW_FRAMEWORK_SIMPLENODE_HPP_

#include <dw/core/base/Types.h>
#include <dwcgf/Exception.hpp>
#include <dwcgf/Types.hpp>
#include <dwcgf/channel/Channel.hpp>
#include <dwcgf/pass/Pass.hpp>
#include <dwcgf/node/Node.hpp>
#include <dwcgf/pass/PassDescriptor.hpp>
#include <dwcgf/port/ManagedPort.hpp>
#include <dwcgf/port/PortCollectionDescriptor.hpp>
#include <dwcgf/port/PortDescriptor.hpp>
#include <dwcgf/port/RegisteredPort.hpp>

#include <dw/core/container/VectorFixed.hpp>
#include <dw/core/container/BaseString.hpp>

#include <functional>
#include <map>

namespace dw
{
namespace framework
{

struct NodeAllocationParams
{
    size_t maxInputPortCount  = Node::MAX_PORT_COUNT;
    size_t maxOutputPortCount = Node::MAX_PORT_COUNT;
    size_t maxPassCount       = Node::MAX_PASS_COUNT;
};

template <typename NodeT>
NodeAllocationParams createAllocationParams()
{
    NodeAllocationParams params;
    params.maxInputPortCount  = portSize<NodeT, PortDirection::INPUT>();
    params.maxOutputPortCount = portSize<NodeT, PortDirection::OUTPUT>();
    params.maxPassCount       = passSize<NodeT>();
    return params;
}

class SimpleNode : public Node
{
public:
    static constexpr const char* PASS_SETUP_NAME    = "SETUP";
    static constexpr const char* PASS_TEARDOWN_NAME = "TEARDOWN";

    SimpleNode();
    /// Constructor which tailors the preallocated size of the internal collections for ports and passes to the need of the concrete node.
    SimpleNode(NodeAllocationParams params);
    virtual ~SimpleNode();

    dwStatus reset() override
    {
        throw ExceptionWithStatus(DW_NOT_IMPLEMENTED, "SimpleNode::reset() not implemented");
    }

    /// Associate an input port with a channel instances.
    /**
     * A concrete node shouldn't need to override this method if all input ports are initialized in the constructor using the macros like #NODE_INIT_INPUT_PORT.
     */
    dwStatus setInputChannel(ChannelObject* channel, uint8_t portID) override;

    dwStatus setInputChannel(ChannelObject*, uint8_t, dwSerializationType) override
    {
        throw ExceptionWithStatus(DW_NOT_IMPLEMENTED, "SimpleNode::setInputChannel() not implemented");
    }

    /// Associate an output port with a channel instances.
    /**
     * A concrete node shouldn't need to override this method if all output ports are initialized in the constructor using the macros like #NODE_INIT_OUTPUT_PORT.
     */
    dwStatus setOutputChannel(ChannelObject* channel, uint8_t portID) override;

    /// Gets the input channel associated with the input port
    dwStatus getInputChannel(const uint8_t portID, ChannelObject*& channel) const override;

    /// Gets the output channel associated with the output port.
    dwStatus getOutputChannel(const uint8_t portID, ChannelObject*& channel) const override;

    dwStatus validate() override
    {
        throw ExceptionWithStatus(DW_NOT_IMPLEMENTED, "SimpleNode::validate() not implemented");
    }

    /// Helper function used by dw::framework::SimpleNodeT::validate.
    /**
     * A concrete node shouldn't need to override or call this method.
     */
    dwStatus validate(const char* direction, const PortCollectionDescriptor& collection, const dw::core::HeapHashMap<size_t, std::shared_ptr<PortBase>>& ports, size_t indexOffset = 0);

    dwStatus run() override;
    size_t getPassCount() const noexcept override;
    dwStatus runPass(size_t passIndex) override;
    dwStatus getPass(Pass** pass, uint8_t index) override;
    dwStatus getPasses(VectorFixed<Pass*>& passList) override;
    dwStatus getPasses(VectorFixed<Pass*>& passList, dwProcessorType processorType, dwProcessType processType) override;

    dwStatus setName(const char* name) final;
    dwStatus getName(const char** name) override;

    dwStatus getErrorSignal(dwGraphErrorSignal*& errorSignal) override;
    dwStatus getHealthSignal(dwGraphHealthSignal*& healthSignals, bool updateFromModule = false) override;

    template <typename Func, typename PortList>
    void iteratePorts(PortList& portList, Func func)
    {
        for (auto elem : portList)
        {
            func(elem);
        }
    }

    template <typename Func>
    void iterateManagedInputPorts(Func func)
    {
        iteratePorts(m_inputPorts, [&func, this](decltype(m_inputPorts)::TElement& elem) {
            if (auto managedPort = dynamic_cast<ManagedPortInputBase*>(elem.second.get()))
            {
                func(*managedPort);
            }
            else
            {
                const char* nodeName = nullptr;
                this->getName(&nodeName);
                throw ExceptionWithStatus(DW_BAD_CAST, "SimpleNode: ports are wrong class, node ", nodeName);
            }
        });
    }

    template <typename Func>
    void iterateManagedOutputPorts(Func func)
    {
        iteratePorts(m_outputPorts, [&func, this](decltype(m_outputPorts)::TElement& elem) {
            if (auto managedPort = dynamic_cast<ManagedPortOutputBase*>(elem.second.get()))
            {
                func(*managedPort);
            }
            else
            {
                const char* nodeName = nullptr;
                this->getName(&nodeName);
                throw ExceptionWithStatus(DW_BAD_CAST, "SimpleNode: ports are wrong class node ", nodeName);
            }
        });
    }

    template <typename ModuleHandle_t>
    dwStatus setModuleHandle(ModuleHandle_t handle, dwContextHandle_t context)
    {
        dwModuleHandle_t moduleHandle;

        if (DW_NULL_HANDLE == handle)
        {
            return DW_INVALID_ARGUMENT;
        }

        dwStatus ret = getModuleHandle(&moduleHandle, handle, context);
        if (DW_SUCCESS != ret)
        {
            return ret;
        }

        return setObjectHandle(moduleHandle);
    }

    virtual dwStatus setObjectHandle(dwModuleHandle_t handle);

    /// Gets the input port associated with the input port Id.
    dwStatus getInputPort(const uint8_t portID, dw::framework::PortBase*& port) const override;

    /// Gets the output port associated with the output port Id.
    dwStatus getOutputPort(const uint8_t portID, dw::framework::PortBase*& port) const override;

protected:
    dwStatus getModuleHandle(dwModuleHandle_t* moduleHandle, void* handle, dwContextHandle_t context);

    virtual std::unique_ptr<Pass> createSetupPass()
    {
        throw ExceptionWithStatus(DW_NOT_IMPLEMENTED, "SimpleNode::createSetupPass() not implemented");
    }

    virtual std::unique_ptr<Pass> createTeardownPass()
    {
        throw ExceptionWithStatus(DW_NOT_IMPLEMENTED, "SimpleNode::createTeardownPass() not implemented");
    }

    /**
     * Simple helper to create a pass with any function implementing operator()
     *
     * Example:
     * class MyNode : public Node
     * {
     *   MyNode()
     *   {
     *     const int32_t someArg = 1;
     *
     *     auto pass1 = make_pass([someArg, this]() { return myPassWithArgs(someArg); }, ...);
     *     auto pass2 = make_pass([this]() { return myPassNoArgs(); }, ...);
     *   }
     *
     * private:
     *   dwStatus myPassWithArgs(int32_t a);
     *   dwStatus myPassNoArgs();
     * }
     *
     * @note cannot use here PassFunctionT&& + std::forward<PassFunctionT>(func) because that makes the captured
     *       this pointer by a passed lambda become invalid, see
     */
    template <typename PassFunctionT, typename... Args>
    std::unique_ptr<PassImpl<PassFunctionT>> make_pass(PassFunctionT func, Args&&... args)
    {
        return std::make_unique<PassImpl<PassFunctionT>>(*this, func, std::forward<Args>(args)...);
    }

    /// Register a pass function with the node base class.
    /**
     * The macro #NODE_REGISTER_PASS can be used for convenience to hide the template parameters of this method.
     *
     * Note that the processorType DLA_0 and DLA_1 will be replaced by DLA when STM support cuDLA and dw deprecated them
     * Issue tracked in AVCGF-569
     */
    template <
        typename NodeT, size_t PassIndex, typename PassFunctionT>
    void registerPass(PassFunctionT func, NvMediaDla* dlaEngine = nullptr)
    {
        if (!isValidPass<NodeT>(PassIndex))
        {
            throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "registerPass called with an invalid pass id: ", PassIndex);
        }
        if (m_passList.size() == 0 || m_passList.size() - 1 < PassIndex)
        {
            m_passList.resize(PassIndex + 1);
            m_passOwnershipList.resize(PassIndex + 1);
        }
        if (m_passList[PassIndex] != nullptr)
        {
            throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "registerPass called with a pass id which has been added before: ", PassIndex);
        }
        dwProcessorType processorType = passProcessorType<NodeT, PassIndex>();
        dwProcessType processType     = determineProcessType(processorType);
        if (dlaEngine == nullptr)
        {
            m_passList[PassIndex] = std::make_unique<PassImpl<PassFunctionT>>(*this, func, processorType, processType, -1, -1, -1);
        }
        else if (processorType == DW_PROCESSOR_TYPE_DLA_0)
        {
            m_passList[PassIndex] = std::make_unique<PassImpl<PassFunctionT>>(*this, func, processorType, processType, -1, -1, -1, dlaEngine);
        }
        else
        {
            throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "registerPass called with a pass which has dlaEngine but not a DLA pass: ", PassIndex);
        }
        m_passOwnershipList[PassIndex] = true;
    }

public:
    /// Register a pass function with the node base class.
    /// @deprecated Use registerPass() instead.
    void addPass(Pass* pass);

    /**
     * @brief Clears the current Health Signals from the Health Signal Array
     * @return DW_SUCCESS
     **/
    dwStatus clearHealthSignal();

    /**
     * @brief Adds the provided Health Signal to the Health Signal Array. If the array is full, the new signal will not be added.
     * @param[in] signal The Health Signal to add
     * @return DW_SUCCESS, or DW_BUFFER_FULL if the array is currently full.
     **/
    dwStatus updateHealthSignal(const dwGraphHealthSignal& signal);

    /**
     * @brief Copy health signals from the module over to the node and stores in outSignal
     * @param[out] outSignal the output module signal
     * @return DW_SUCCESS
     *         DW_NOT_AVAILABLE if there is no DW module associated with the node
     *         DW_INVALID_HANDLE if the DW module handle assiciated with the node is invalid
     **/
    dwStatus copyModuleHealthSignals(dwHealthSignal& outSignal);

    /**
     * @brief A function that allows user override to update error signal
     *          It is automatically called by dwFramework when getErrorSignal is called
     *          and when pass returns non-success return code.
     * @param[inout] signal that the node owner modifies to store current health.
     *                    It is pre-filled with the latest module health signal
     * @return DW_SUCCESS
     **/
    dwStatus reportCurrentErrorSignal(dwGraphErrorSignal& signal) override;

    /**
     * @brief A function that allows user override to update health signal
     *          It is automatically called by dwFramework during teardown
     *          and when pass returns non-success return code.
     * @param[inout] signal that the node owner modifies to store current health.
     *                    It is pre-filled with the latest module health signal
     * @return DW_SUCCESS
     **/
    dwStatus reportCurrentHealthSignal(dwGraphHealthSignal& signal) override;

private:
    inline dwProcessType determineProcessType(dwProcessorType processorType)
    {
        if (processorType == DW_PROCESSOR_TYPE_GPU ||
            processorType == DW_PROCESSOR_TYPE_DLA_0)
        {
            return DW_PROCESS_TYPE_ASYNC;
        }
        return DW_PROCESS_TYPE_SYNC;
    }

    dwStatus updateHealthSignalFromModule();

    VectorFixed<std::unique_ptr<Pass>> m_passList;
    // tracking ownership is only necessary until addPass(..., Pass* pass) is removed
    VectorFixed<bool> m_passOwnershipList;
    FixedString<MAX_NAME_LEN> m_name{};
    bool m_setupTeardownCreated = false;

    /// The error/health signal generated by this node.
    dwGraphErrorSignal m_errorSignal{};
    dwGraphHealthSignal m_healthSignal{};

    dwModuleHandle_t m_object{};
    uint32_t m_iterationCount{};
    uint32_t m_nodePeriod;

public:
    /// @deprecated Use initInputPort() / initInputArrayPort() / initOutputPort() / initOutputArrayPort() and getInputPort() / getInputPort(size_t) / getOutputPort() / getOutputPort(size_t) instead.
    template <typename TPort, typename... Args>
    std::unique_ptr<TPort> make_port(Args&&... args)
    {
        auto port = std::make_unique<TPort>(std::forward<Args>(args)...);
        port->setSyncRetriever(std::bind(&SimpleNode::getIterationCount, this));
        return port;
    }

    /// Initialize a ManagedPortInput which will be owned by the base class and can be retrieved using getInputPort().
    /**
     * The macro #NODE_INIT_INPUT_PORT can be used for convenience to hide the template parameters of this method.
     */
    template <typename NodeT, size_t PortIndex, typename... Args>
    void initInputPort(Args&&... args)
    {
        static_assert(PortIndex < portSize<NodeT, PortDirection::INPUT>(), "Invalid port index");
        using DataType = decltype(portType<NodeT, PortDirection::INPUT, PortIndex>());
        auto port      = std::make_shared<ManagedPortInput<DataType>>(std::forward<Args>(args)...);
        if (m_inputPorts.find(PortIndex) != m_inputPorts.end())
        {
            throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "Input port with the following id registered multiple times: ", PortIndex);
        }
        m_inputPorts[PortIndex] = port;
    }

    /// Initialize an array of ManagedPortInput which will be owned by the base class and can be retrieved using getInputPort(size_t).
    /**
     * The macro #NODE_INIT_INPUT_ARRAY_PORT can be used for convenience to hide the template parameters of this method.
     */
    template <typename NodeT, size_t PortIndex, typename... Args>
    void initInputArrayPort(Args&&... args)
    {
        static_assert(PortIndex < portSize<NodeT, PortDirection::INPUT>(), "Invalid port index");
        using DataType                    = decltype(portType<NodeT, PortDirection::INPUT, PortIndex>());
        constexpr size_t descriptor_index = descriptorIndex<NodeT, PortDirection::INPUT, PortIndex>();
        constexpr size_t arraySize        = descriptorPortSize<NodeT, PortDirection::INPUT, descriptor_index>();
        for (size_t i = 0; i < arraySize; ++i)
        {
            auto port = std::make_shared<ManagedPortInput<DataType>>(std::forward<Args>(args)...);
            if (m_inputPorts.find(PortIndex + i) != m_inputPorts.end())
            {
                throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "Input port with the following id registered multiple times: ", PortIndex + i);
            }
            m_inputPorts[PortIndex + i] = port;
        }
    }

    /// Initialize a ManagedPortOutput which will be owned by the base class and can be retrieved using getOutputPort().
    /**
     * The macro #NODE_INIT_OUTPUT_PORT can be used for convenience to hide the template parameters of this method.
     */
    template <typename NodeT, size_t PortIndex, typename... Args>
    void initOutputPort(Args&&... args)
    {
        static_assert(PortIndex - portSize<NodeT, PortDirection::INPUT>() < portSize<NodeT, PortDirection::OUTPUT>(), "Invalid port index");
        using DataType = decltype(portType<NodeT, PortDirection::OUTPUT, PortIndex>());
        auto port      = std::make_shared<ManagedPortOutput<DataType>>(std::forward<Args>(args)...);
        if (m_outputPorts.find(PortIndex) != m_outputPorts.end())
        {
            throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "Output port with the following id registered multiple times: ", PortIndex);
        }
        m_outputPorts[PortIndex] = port;
    }

    /// Initialize an array of ManagedPortOutput which will be owned by the base class and can be retrieved using getOutputPort(size_t).
    /**
     * The macro #NODE_INIT_OUTPUT_ARRAY_PORT can be used for convenience to hide the template parameters of this method.
     */
    template <typename NodeT, size_t PortIndex, typename... Args>
    void initOutputArrayPort(Args&&... args)
    {
        static_assert(PortIndex - portSize<NodeT, PortDirection::INPUT>() < portSize<NodeT, PortDirection::OUTPUT>(), "Invalid port index");
        using DataType                    = decltype(portType<NodeT, PortDirection::OUTPUT, PortIndex>());
        constexpr size_t descriptor_index = descriptorIndex<NodeT, PortDirection::OUTPUT, PortIndex>();
        constexpr size_t arraySize        = descriptorPortSize<NodeT, PortDirection::OUTPUT, descriptor_index>();
        for (size_t i = 0; i < arraySize; ++i)
        {
            auto port = std::make_shared<ManagedPortOutput<DataType>>(std::forward<Args>(args)...);
            if (m_outputPorts.find(PortIndex + i) != m_outputPorts.end())
            {
                throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "Output port with the following id registered multiple times: ", PortIndex + i);
            }
            m_outputPorts[PortIndex + i] = port;
        }
    }

    /// Get a previously initialized non-array ManagedPortInput.
    template <typename NodeT, size_t PortIndex>
    ManagedPortInput<decltype(portType<NodeT, PortDirection::INPUT, PortIndex>())>& getInputPort()
    {
        static_assert(PortIndex < portSize<NodeT, PortDirection::INPUT>(), "Invalid port index");
        constexpr bool isArray = descriptorPortArray<
            NodeT, PortDirection::INPUT, descriptorIndex<NodeT, PortDirection::INPUT, PortIndex>()>();
        static_assert(!isArray, "Input port is an array, must pass an array index");
        if (m_inputPorts.find(PortIndex) == m_inputPorts.end())
        {
            throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "Input port with the following id not registered: ", PortIndex);
        }
        using DataType    = decltype(portType<NodeT, PortDirection::INPUT, PortIndex>());
        using PointerType = ManagedPortInput<DataType>;
        using ReturnType  = std::shared_ptr<PointerType>;
        ReturnType port   = std::dynamic_pointer_cast<PointerType>(m_inputPorts[PortIndex]);
        if (!port)
        {
            throw ExceptionWithStatus(DW_BAD_CAST, "Failed to cast the following input port to its declared type: ", PortIndex);
        }
        return *port;
    }

    /// Get one specific ManagedPortInput from a previously initialized input array port.
    template <typename NodeT, size_t PortIndex>
    ManagedPortInput<decltype(portType<NodeT, PortDirection::INPUT, PortIndex>())>& getInputPort(size_t arrayIndex)
    {
        static_assert(PortIndex < portSize<NodeT, PortDirection::INPUT>(), "Invalid port index");
        constexpr bool isArray = descriptorPortArray<NodeT, PortDirection::INPUT, descriptorIndex<NodeT, PortDirection::INPUT, PortIndex>()>();
        static_assert(isArray, "Input port is not an array, must not pass an array index");
        constexpr size_t arraySize = descriptorPortSize<NodeT, PortDirection::INPUT, descriptorIndex<NodeT, PortDirection::INPUT, PortIndex>()>();
        if (arrayIndex >= arraySize)
        {
            throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "The array index is out of bound: ", arrayIndex);
        }
        if (m_inputPorts.find(PortIndex + arrayIndex) == m_inputPorts.end())
        {
            throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "Input port with the following id not registered: ", PortIndex + arrayIndex);
        }
        using DataType    = decltype(portType<NodeT, PortDirection::INPUT, PortIndex>());
        using PointerType = ManagedPortInput<DataType>;
        using ReturnType  = std::shared_ptr<PointerType>;
        ReturnType port   = std::dynamic_pointer_cast<PointerType>(m_inputPorts[PortIndex + arrayIndex]);
        if (!port)
        {
            throw ExceptionWithStatus(DW_BAD_CAST, "Failed to cast the following input port to its declared type: ", PortIndex + arrayIndex);
        }
        return *port;
    }

    /// Get a previously initialized non-array ManagedPortOutput.
    template <typename NodeT, size_t PortIndex>
    ManagedPortOutput<decltype(portType<NodeT, PortDirection::OUTPUT, PortIndex>())>& getOutputPort()
    {
        static_assert(PortIndex - portSize<NodeT, PortDirection::INPUT>() < portSize<NodeT, PortDirection::OUTPUT>(), "Invalid port index");
        constexpr bool isArray = descriptorPortArray<
            NodeT, PortDirection::OUTPUT, descriptorIndex<NodeT, PortDirection::OUTPUT, PortIndex>()>();
        static_assert(!isArray, "Output port is an array, must pass an array index");
        if (m_outputPorts.find(PortIndex) == m_outputPorts.end())
        {
            throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "Output port with the following id not registered: ", PortIndex);
        }
        using DataType    = decltype(portType<NodeT, PortDirection::OUTPUT, PortIndex>());
        using PointerType = ManagedPortOutput<DataType>;
        using ReturnType  = std::shared_ptr<PointerType>;
        ReturnType port   = std::dynamic_pointer_cast<PointerType>(m_outputPorts[PortIndex]);
        if (!port)
        {
            throw ExceptionWithStatus(DW_BAD_CAST, "Failed to cast the following output port to its declared type: ", PortIndex);
        }
        return *port;
    }

    /// Get one specific ManagedPortOutput from a previously initialized output array port.
    template <typename NodeT, size_t PortIndex>
    ManagedPortOutput<decltype(portType<NodeT, PortDirection::OUTPUT, PortIndex>())>& getOutputPort(size_t arrayIndex)
    {
        static_assert(PortIndex - portSize<NodeT, PortDirection::INPUT>() < portSize<NodeT, PortDirection::OUTPUT>(), "Invalid port index");
        constexpr bool isArray = descriptorPortArray<NodeT, PortDirection::OUTPUT, descriptorIndex<NodeT, PortDirection::OUTPUT, PortIndex>()>();
        static_assert(isArray, "Output port is not an array, must not pass an array index");
        constexpr size_t arraySize = descriptorPortSize<NodeT, PortDirection::OUTPUT, descriptorIndex<NodeT, PortDirection::OUTPUT, PortIndex>()>();
        if (arrayIndex >= arraySize)
        {
            throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "The array index is out of bound: ", arrayIndex);
        }
        if (m_outputPorts.find(PortIndex + arrayIndex) == m_outputPorts.end())
        {
            throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "Output port with the following id not registered: ", PortIndex + arrayIndex);
        }
        using DataType    = decltype(portType<NodeT, PortDirection::OUTPUT, PortIndex>());
        using PointerType = ManagedPortOutput<DataType>;
        using ReturnType  = std::shared_ptr<PointerType>;
        ReturnType port   = std::dynamic_pointer_cast<PointerType>(m_outputPorts[PortIndex + arrayIndex]);
        if (!port)
        {
            throw ExceptionWithStatus(DW_BAD_CAST, "Failed to cast the following output port to its declared type: ", PortIndex + arrayIndex);
        }
        return *port;
    }

    /// Default implementation of the setup pass.
    /**
     * Check that all ports which are bound to a channel have a buffer available.
     * This method is used by SimpleNodeT::setupImpl.
     */
    dwStatus setup();

    /// Default implementation of the teardown pass.
    /**
     * Call ManagedPortInput::release on all input ports.
     * This method is used by SimpleNodeT::teardownImpl.
     */
    dwStatus teardown();

    /// Default implementation to reset ports managed by the base class.
    /**
     * Call ManagedPortInput::reset on all input ports.
     * This method is used by SimpleNodeT::reset.
     */
    void resetPorts() override;

    /**
    * Sets the node's iteration count.
    * The value set can be retrieved with getIterationCount.
    */
    dwStatus setIterationCount(uint32_t iterationCount) override;

    /**
    * Sets the node's period
    * The value set can be retrieved with getNodePeriod.
    */
    dwStatus setNodePeriod(uint32_t period) override;

    const dw::core::HeapHashMap<size_t, std::shared_ptr<PortBase>>& getRegisteredInputPorts() const
    {
        return m_inputPorts;
    }

    const dw::core::HeapHashMap<size_t, std::shared_ptr<PortBase>>& getRegisteredOutputPorts() const
    {
        return m_outputPorts;
    }

    uint32_t getIterationCount() const;
    uint32_t getNodePeriod() const;
    /**
    *  Set the current state.
    *  Override setState to handle state changes
    */
    dwStatus setState(const char* state) override
    {
        static_cast<void>(state);
        return DW_SUCCESS;
    }

protected:
    dw::core::HeapHashMap<size_t, std::shared_ptr<PortBase>> m_inputPorts;
    dw::core::HeapHashMap<size_t, std::shared_ptr<PortBase>> m_outputPorts;

    std::atomic<bool> m_asyncResetFlag{false};
};

class SimpleSensorNode : public ISensorNode
{
public:
    static constexpr char LOG_TAG[] = "SimpleSensorNode";

    dwStatus start() override
    {
        throw ExceptionWithStatus(DW_NOT_IMPLEMENTED, "SimpleSensorNode::start() not implemented");
    }

    dwStatus stop() override
    {
        throw ExceptionWithStatus(DW_NOT_IMPLEMENTED, "SimpleSensorNode::stop() not implemented");
    }

    dwStatus isVirtual(bool*) override
    {
        throw ExceptionWithStatus(DW_NOT_IMPLEMENTED, "SimpleSensorNode::isVirtual() not implemented");
    }

    dwStatus setDataEventReadCallback(DataEventReadCallback) override
    {
        throw ExceptionWithStatus(DW_NOT_IMPLEMENTED, "SimpleSensorNode::setDataEventReadCallback() not implemented");
    }

    dwStatus setDataEventWriteCallback(DataEventWriteCallback) override
    {
        throw ExceptionWithStatus(DW_NOT_IMPLEMENTED, "SimpleSensorNode::setDataEventWriteCallback() not implemented");
    }
};

/// @deprecated Use SimpleNode instead.
class SimpleProcessNode : public SimpleNode
{
public:
    SimpleProcessNode();
    SimpleProcessNode(NodeAllocationParams params);
};

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_SIMPLENODE_HPP_
