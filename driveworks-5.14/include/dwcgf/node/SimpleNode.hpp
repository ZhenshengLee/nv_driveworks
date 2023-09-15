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
// SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <dwshared/dwfoundation/dw/core/container/BaseString.hpp>
#include <dwshared/dwfoundation/dw/core/container/HashContainer.hpp>
#include <dwshared/dwfoundation/dw/core/container/VectorFixed.hpp>
#include <dwshared/dwfoundation/dw/core/language/Function.hpp>

#include <memory>

namespace dw
{
namespace framework
{

struct NodeAllocationParams
{
public:
    NodeAllocationParams() = delete;
    NodeAllocationParams(
        size_t maxInputPortCount_,
        size_t maxOutputPortCount_,
        size_t maxPassCount_)
        : maxInputPortCount(maxInputPortCount_)
        , maxOutputPortCount(maxOutputPortCount_)
        , maxPassCount(maxPassCount_)
    {
    }
    size_t maxInputPortCount{};
    size_t maxOutputPortCount{};
    size_t maxPassCount{};
};

template <typename NodeT>
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
NodeAllocationParams createAllocationParams()
{
    NodeAllocationParams params{
        portSize<NodeT, PortDirection::INPUT>(),
        portSize<NodeT, PortDirection::OUTPUT>(),
        passSize<NodeT>()};
    return params;
}

class SimpleNode : public Node
{
public:
    /// Constructor which tailors the preallocated size of the internal collections for ports and passes to the need of the concrete node.
    SimpleNode(NodeAllocationParams params);

    dwStatus reset() override
    {
        throw ExceptionWithStatus(DW_NOT_IMPLEMENTED, "SimpleNode::reset() not implemented");
    }

    /// Associate an input port with a channel instances.
    /**
     * A concrete node shouldn't need to override this method if all input ports are initialized in the constructor using the macros like #NODE_INIT_INPUT_PORT.
     */
    dwStatus setInputChannel(ChannelObject* channel, uint8_t portID) override;

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
    dwStatus validate(const char* direction, const PortCollectionDescriptor& collection, dw::core::Function<bool(size_t)> isPortBound);

    dwStatus run() override;
    size_t getPassCount() const noexcept override;
    dwStatus runPass(size_t passIndex) override;
    dwStatus getPass(Pass** pass, uint8_t index) override;
    // coverity[autosar_cpp14_a2_10_5_violation]
    dwStatus getPasses(VectorFixed<Pass*>& passList) override;
    // coverity[autosar_cpp14_a2_10_5_violation]
    dwStatus getPasses(VectorFixed<Pass*>& passList, dwProcessorType processorType) override;

    dwStatus setName(const char* name) final;
    dwStatus getName(const char** name) override;

    dwStatus collectErrorSignals(dwGraphErrorSignal*& errorSignal, bool updateFromModule = true) override;
    dwStatus getModuleErrorSignal(dwErrorSignal& errorSignal) override;
    dwStatus getNodeErrorSignal(dwGraphErrorSignal& errorSignal) override;
    dwStatus collectHealthSignals(dwGraphHealthSignal*& healthSignal, bool updateFromModule = false) override;
    dwStatus getModuleHealthSignal(dwHealthSignal& healthSignal) override;
    dwStatus getNodeHealthSignal(dwGraphHealthSignal& healthSignal) override;
    dwStatus clearErrorSignal() override;
    dwStatus clearHealthSignal() override;
    dwStatus addToErrorSignal(uint32_t error, dwTime_t timestamp = 0UL) override;
    dwStatus addToHealthSignal(uint32_t error, dwTime_t timestamp = 0UL) override;

    template <typename Func, typename PortList>
    void iteratePorts(PortList& portList, Func func)
    {
        for (auto& elem : portList)
        {
            func(elem);
        }
    }

    template <typename Func>
    void iterateManagedInputPorts(Func func)
    {
        iteratePorts(m_inputPorts, [&func, this](decltype(m_inputPorts)::TElement& elem) {
            if (elem.second.get() == nullptr)
            {
                const char* nodeName{nullptr};
                this->getName(&nodeName);
                throw ExceptionWithStatus(DW_NOT_INITIALIZED, "SimpleNode: input port not initialized, node ", nodeName, ", port id ", elem.first);
            }
            func(*elem.second);
        });
    }

    template <typename Func>
    void iterateManagedOutputPorts(Func func)
    {
        iteratePorts(m_outputPorts, [&func, this](decltype(m_outputPorts)::TElement& elem) {
            if (elem.second.get() == nullptr)
            {
                const char* nodeName{nullptr};
                this->getName(&nodeName);
                throw ExceptionWithStatus(DW_NOT_INITIALIZED, "SimpleNode: output port not initialized, node ", nodeName, ", port id ", elem.first);
            }
            func(*elem.second);
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

        dwStatus ret{getModuleHandle(&moduleHandle, handle, context)};
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

    /// Register a pass function with the node base class.
    /**
     * The macro #NODE_REGISTER_PASS can be used for convenience to hide the template parameters of this method.
     *
     * Note that the processorType DLA_0 and DLA_1 will be replaced by DLA when STM support cuDLA and dw deprecated them
     * Issue tracked in AVCGF-569
     */
    template <
        typename NodeT, size_t PassIndex, typename PassFunctionT>
    void registerPass(PassFunctionT func, std::initializer_list<std::pair<dwStatus, uint32_t>> const& returnMapping = {})
    {
        // coverity[autosar_cpp14_a5_1_1_violation] FP: nvbugs/3364868
        if (!isValidPass<NodeT>(PassIndex))
        {
            throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "registerPass called with an invalid pass id: ", PassIndex);
        }
        // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
        if (m_passList.size() == 0U || m_passList.size() - 1U < PassIndex)
        {
            // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
            m_passList.resize(PassIndex + 1U);
        }
        // coverity[autosar_cpp14_a5_1_1_violation] FP: nvbugs/3364868
        if (m_passList[PassIndex] != nullptr)
        {
            throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "registerPass called with a pass id which has been added before: ", PassIndex);
        }
        dwProcessorType processorType{passProcessorType<NodeT, PassIndex>()};
        // coverity[autosar_cpp14_a5_1_1_violation] FP: nvbugs/3364868
        m_passList[PassIndex] = std::make_unique<PassImpl<PassFunctionT>>(*this, passName<NodeT, PassIndex>(), func, processorType, returnMapping);
    }

    /// Register a GPU pass function and a cuda stream with the node base class.
    /**
     * The macro #NODE_REGISTER_PASS can be used for convenience to hide the template parameters of this method.
     */
    template <
        typename NodeT, size_t PassIndex, typename PassFunctionT>
    void registerPass(PassFunctionT func, cudaStream_t const cudaStream, std::initializer_list<std::pair<dwStatus, uint32_t>> const& returnMapping = {})
    {
        static_assert(passProcessorType<NodeT, PassIndex>() == DW_PROCESSOR_TYPE_GPU, "The processor type of a pass with a cuda stream must be GPU");
        registerPass<NodeT, PassIndex>(func, returnMapping);
        m_passList[PassIndex]->m_cudaStream = cudaStream;
    }

public:
    /**
     * @brief Adds the provided Health Signal to the Health Signal Array. If the array is full, the new signal will not be added.
     * @param[in] signal The Health Signal to add
     * @return DW_SUCCESS, or DW_BUFFER_FULL if the array is currently full.
     **/
    dwStatus updateHealthSignal(const dwGraphHealthSignal& signal);

    /**
     * @brief A function that allows user override to update error signal
     *          It is automatically called by dwFramework when getErrorSignal is called
     *          and when pass returns non-success return code.
     * @param[inout] signal that the node owner modifies to store current health.
     *                    It is pre-filled with the latest module health signal
     * @return DW_SUCCESS
     **/
    dwStatus updateCurrentErrorSignal(dwGraphErrorSignal& signal) override;

    /**
     * @brief A function that allows user override to update health signal
     *          It is automatically called by dwFramework during teardown
     *          and when pass returns non-success return code.
     * @param[inout] signal that the node owner modifies to store current health.
     *                    It is pre-filled with the latest module health signal
     * @return DW_SUCCESS
     **/
    dwStatus updateCurrentHealthSignal(dwGraphHealthSignal& signal) override;

protected:
    dwGraphHealthSignal& getHealthSignal()
    {
        return m_healthSignal;
    }

private:
    VectorFixed<std::unique_ptr<Pass>> m_passList;
    FixedString<MAX_NAME_LEN> m_name;
    bool m_setupTeardownCreated;

    /// The error/health signal generated by this node.
    dwGraphErrorSignal m_errorSignal;
    dwGraphHealthSignal m_healthSignal;

    dwModuleHandle_t m_object;
    uint32_t m_iterationCount{};
    uint32_t m_nodePeriod{};

public:
    /// Initialize a ManagedPortInput which will be owned by the base class and can be retrieved using getInputPort().
    /**
     * The macro #NODE_INIT_INPUT_PORT can be used for convenience to hide the template parameters of this method.
     */
    template <typename NodeT, size_t PortIndex, typename... Args>
    void initInputPort(Args&&... args)
    {
        static_assert(PortIndex < portSize<NodeT, PortDirection::INPUT>(), "Invalid port index");
        using DataType = decltype(portType<NodeT, PortDirection::INPUT, PortIndex>());
        auto port      = std::make_shared<ManagedPortInput<DataType>>(portName<NodeT, PortDirection::INPUT, PortIndex>(), std::forward<Args>(args)...);
        if (m_inputPorts.find(PortIndex) != m_inputPorts.end())
        {
            throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "Input port with the following id registered multiple times: ", PortIndex);
        }
        m_inputPorts[PortIndex] = port;
    }

    /// Initialize an array of ManagedPortInput which will be owned by the base class and can be retrieved using getInputPort(size_t).
    /**
     * All ports of the array are initialized with the same args.
     * If ports of the array need to be initialized with different args use dw::framework::SimpleNode::initInputArrayPort instead.
     *
     * The macro #NODE_INIT_INPUT_ARRAY_PORTS can be used for convenience to hide the template parameters of this method.
     */
    template <typename NodeT, size_t PortIndex, typename... Args>
    void initInputArrayPorts(Args&&... args)
    {
        static_assert(PortIndex < portSize<NodeT, PortDirection::INPUT>(), "Invalid port index");
        constexpr size_t arraySize = descriptorPortSize<NodeT, PortDirection::INPUT, descriptorIndex<NodeT, PortDirection::INPUT, PortIndex>()>();
        for (size_t i = 0; i < arraySize; ++i)
        {
            initInputArrayPort<NodeT, PortIndex>(i, std::forward<Args>(args)...);
        }
    }

    /// Initialize one ManagedPortInput of an array which will be owned by the base class and can be retrieved using getInputPort(size_t).
    /**
     * The method should be called instead of dw::framework::SimpleNode::initInputArrayPorts, if ports of the array need to be initialized with different args.
     *
     * The macro #NODE_INIT_INPUT_ARRAY_PORT can be used for convenience to hide the template parameters of this method.
     */
    template <typename NodeT, size_t PortIndex, typename... Args>
    void initInputArrayPort(size_t arrayIndex, Args&&... args)
    {
        static_assert(PortIndex < portSize<NodeT, PortDirection::INPUT>(), "Invalid port index");
        using DataType = decltype(portType<NodeT, PortDirection::INPUT, PortIndex>());
        if (arrayIndex >= descriptorPortSize<NodeT, PortDirection::INPUT, descriptorIndex<NodeT, PortDirection::INPUT, PortIndex>()>())
        {
            throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "Invalid array index ", arrayIndex, " for array input port ", PortIndex);
        }
        auto port = std::make_shared<ManagedPortInput<DataType>>(portName<NodeT, PortDirection::INPUT, PortIndex>(), std::forward<Args>(args)...);
        if (m_inputPorts.find(PortIndex + arrayIndex) != m_inputPorts.end())
        {
            throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "Input port with the following id registered multiple times: ", PortIndex + arrayIndex);
        }
        m_inputPorts[PortIndex + arrayIndex] = port;
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
        std::shared_ptr<ManagedPortOutput<DataType>> port{std::make_shared<ManagedPortOutput<DataType>>(portName<NodeT, PortDirection::OUTPUT, PortIndex>(), std::forward<Args>(args)...)};
        if (m_outputPorts.find(PortIndex) != m_outputPorts.end())
        {
            throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "Output port with the following id registered multiple times: ", PortIndex);
        }
        m_outputPorts[PortIndex] = port;
    }

    /// Initialize an array of ManagedPortOutput which will be owned by the base class and can be retrieved using getOutputPort(size_t).
    /**
     * All ports of the array are initialized with the same args.
     * If ports of the array need to be initialized with different args use dw::framework::SimpleNode::initOutputArrayPort instead.
     *
     * The macro #NODE_INIT_OUTPUT_ARRAY_PORTS can be used for convenience to hide the template parameters of this method.
     */
    template <typename NodeT, size_t PortIndex, typename... Args>
    void initOutputArrayPorts(Args&&... args)
    {
        static_assert(PortIndex - portSize<NodeT, PortDirection::INPUT>() < portSize<NodeT, PortDirection::OUTPUT>(), "Invalid port index");
        constexpr size_t arraySize{descriptorPortSize<NodeT, PortDirection::OUTPUT, descriptorIndex<NodeT, PortDirection::OUTPUT, PortIndex>()>()};
        for (size_t i{0U}; i < arraySize; ++i)
        {
            initOutputArrayPort<NodeT, PortIndex>(i, std::forward<Args>(args)...);
        }
    }

    /// Initialize one ManagedPortOutput of an array which will be owned by the base class and can be retrieved using getOutputPort(size_t).
    /**
     * The method should be called instead of dw::framework::SimpleNode::initOutputArrayPorts, if ports of the array need to be initialized with different args.
     *
     * The macro #NODE_INIT_OUTPUT_ARRAY_PORT can be used for convenience to hide the template parameters of this method.
     */
    template <typename NodeT, size_t PortIndex, typename... Args>
    void initOutputArrayPort(size_t arrayIndex, Args&&... args)
    {
        static_assert(PortIndex - portSize<NodeT, PortDirection::INPUT>() < portSize<NodeT, PortDirection::OUTPUT>(), "Invalid port index");
        using DataType = decltype(portType<NodeT, PortDirection::OUTPUT, PortIndex>());
        if (arrayIndex >= descriptorPortSize<NodeT, PortDirection::OUTPUT, descriptorIndex<NodeT, PortDirection::OUTPUT, PortIndex>()>())
        {
            throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "Invalid array index ", arrayIndex, " for array output port ", PortIndex);
        }
        std::shared_ptr<ManagedPortOutput<DataType>> port{std::make_shared<ManagedPortOutput<DataType>>(portName<NodeT, PortDirection::OUTPUT, PortIndex>(), std::forward<Args>(args)...)};
        if (m_outputPorts.find(PortIndex + arrayIndex) != m_outputPorts.end())
        {
            throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "Output port with the following id registered multiple times: ", PortIndex + arrayIndex);
        }
        m_outputPorts[PortIndex + arrayIndex] = port;
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

    const dw::core::HeapHashMap<size_t, std::shared_ptr<ManagedPortInputBase>>& getRegisteredInputPorts() const
    {
        return m_inputPorts;
    }

    const dw::core::HeapHashMap<size_t, std::shared_ptr<ManagedPortOutputBase>>& getRegisteredOutputPorts() const
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
    dw::core::HeapHashMap<size_t, std::shared_ptr<ManagedPortInputBase>> m_inputPorts;
    dw::core::HeapHashMap<size_t, std::shared_ptr<ManagedPortOutputBase>> m_outputPorts;

    std::atomic<bool> m_asyncResetFlag;
};

// coverity[autosar_cpp14_a0_1_6_violation]
class SimpleSensorNode : public ISensorNode
{
public:
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

    dwStatus isEnabled(bool&) override
    {
        throw ExceptionWithStatus(DW_NOT_IMPLEMENTED, "SimpleSensorNode::isEnabled() not implemented");
    }
};

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_SIMPLENODE_HPP_
