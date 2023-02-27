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
// SPDX-FileCopyrightText: Copyright (c) 2017-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_FRAMEWORK_BASECLASS_NODE_HPP_
#define DW_FRAMEWORK_BASECLASS_NODE_HPP_

#include <map>

#include <dw/core/base/Types.h>
#include <dw/core/context/ObjectExtra.h>

#include <dwcgf/Types.hpp>
#include <dwcgf/channel/Channel.hpp>
#include <dwcgf/pass/Pass.hpp>
#include <dwcgf/logger/Logger.hpp>
#include <dwcgf/computegraph/GraphHealthSignal.hpp>
#include <dw/core/container/VectorFixed.hpp>
#include <dw/core/container/BaseString.hpp>

#include <string>
#include <memory>
#include <atomic>

namespace dw
{
namespace framework
{
using dw::core::FixedString;
using dw::core::VectorFixed;
class ParameterProvider;

#define _DW_CGF_STRINGIFY(x) #x
// std::char_traits<char8_t>::length() is not constexpr before C++17, use "sizeof() - 1" as WAR
#define STRING_VIEW_OF_FIXED_STRING_TEMPLATE_TYPE(x) dw::core::StringView("dw::core::FixedString<" _DW_CGF_STRINGIFY(x) ">", sizeof("dw::core::FixedString<" _DW_CGF_STRINGIFY(x) ">") - 1)

class Node
{
public:
    static constexpr size_t MAX_NAME_LEN = 128;
    using Name_t                         = FixedString<MAX_NAME_LEN>;

    static constexpr uint32_t MAX_PORT_COUNT = 256;

    static constexpr uint32_t MAX_PASS_COUNT = 256;
    static constexpr uint8_t PASS_SETUP      = std::numeric_limits<uint8_t>::max() - 1;
    static constexpr uint8_t PASS_TEARDOWN   = std::numeric_limits<uint8_t>::max();

    virtual ~Node() = default;

    /**
     * @brief Resets the state of the node.
     * @return DW_SUCCESS, DW_FAILURE
     */
    virtual dwStatus reset() = 0;

    /**
     * @brief Sets an input channel for this node with an accompanying port.
     * @param channel The channel to bind to the portID.
     * @param portID The port to bind the channel with.
     * @return DW_SUCCESS, DW_INVALID_ARGUMENT
     */
    virtual dwStatus setInputChannel(ChannelObject* channel, uint8_t portID) = 0;

    /**
     * @brief Sets an input channel for this node with an accompanying port.
     * @param channel The channel to bind to the portID.
     * @param portID The port to bind the channel with.
     * @param dataType The type of data received by this node from the channel
     * @return DW_SUCCESS, DW_INVALID_ARGUMENT
     */
    virtual dwStatus setInputChannel(ChannelObject* channel, uint8_t portID, dwSerializationType dataType) = 0;

    /**
     * @brief Sets an output channel for this node with an accompanying port.
     * @param channel The channel to bind to the portID.
     * @param portID The port to bind the channel with.
     * @return DW_SUCCESS, DW_INVALID_ARGUMENT
     */
    virtual dwStatus setOutputChannel(ChannelObject* channel, uint8_t portID) = 0;

    /**
     * @brief Checks that all mandatory ports are bound.
     * The implementation should validate that all the
     * ports are bound to the appropriate channels
     * (any required ports, that is). For example, a
     * camera node may have processed output and raw
     * output ports, but only one is required to be bound.
     * @return DW_SUCCESS, DW_FAILURE
     */
    virtual dwStatus validate() = 0;

    /**
     * @brief Runs all the passes in the node.
     * @return DW_SUCCESS, DW_FAILURE
     */
    virtual dwStatus run() = 0;

    /**
     * @brief Get number of passes in the node.
     * @return The number of passes
     */
    virtual size_t getPassCount() const noexcept = 0;

    /**
     * @brief Run one pass by ID as defined by the PassList enum class.
     * @param passID The ID of the pass to run.
     * @return DW_SUCCESS, DW_INVALID_ARGUMENT, DW_FAILURE
     */
    virtual dwStatus runPassByID(uint8_t passID) = 0;

    /**
     * @brief Run one pass by index as defined by the pass descriptors.
     * @param passIndex The index of the pass to run.
     * @return DW_SUCCESS, DW_INVALID_ARGUMENT, DW_FAILURE
     */
    virtual dwStatus runPass(size_t passIndex) = 0;

    /**
     * @brief Get all the passes in the node.
     * @param passList The output list to populate.
     * @return DW_SUCCESS, DW_INVALID_ARGUMENT
     */
    virtual dwStatus getPasses(VectorFixed<Pass*>& passList) = 0;

    /**
     * @brief Get node passes filtered by processor type and process type.
     * @param passList The output list to populate.
     * @param processorType Filter by this processor type.
     * @param processType Filter by this process type.
     * @return DW_SUCCESS, DW_INVALID_ARGUMENT
     */
    virtual dwStatus getPasses(VectorFixed<Pass*>& passList,
                               dwProcessorType processorType,
                               dwProcessType processType) = 0;

    /**
     * @brief Set the name of the node.
     * @param name The name of the node.
     * @return DW_SUCCESS, DW_INVALID_ARGUMENT
     */
    virtual dwStatus setName(const char* name) = 0;

    /**
     * @brief Get the name of the node.
     * @param name The output name.
     * @return DW_SUCCESS, DW_INVALID_ARUGMENT
     */
    virtual dwStatus getName(const char** name) = 0;

    /**
     * @brief Get the pointer to the error signal for this node.
     * @param[out] errorSignal The error signal.
     * @return DW_SUCCESS
     */
    virtual dwStatus getErrorSignal(dwGraphErrorSignal*& errorSignal) = 0;

    /**
     * @brief Get the pointer to the health signal for this node.
     * @param[out] healthSignals The health signal.
     * @param[in] updateFromModule fetch from module if set to true
     * @return DW_SUCCESS
     */
    virtual dwStatus getHealthSignal(dwGraphHealthSignal*& healthSignals, bool updateFromModule = false) = 0;

    /**
     * @brief A function that allows user override to update error signal
     *          It is automatically called by dwFramework when getErrorSignal is called
     *          and when pass returns non-success return code.
     * @param[inout] signal that the node owner modifies to store current health.
     *                    It is pre-filled with the latest module health signal
     * @return DW_SUCCESS
     **/
    virtual dwStatus reportCurrentErrorSignal(dwGraphErrorSignal& signal) = 0;

    /**
     * @brief A function that allows user override to update health signal
     *          It is automatically called by dwFramework during teardown
     *          and when pass returns non-success return code.
     * @param[inout] signal that the node owner modifies to store current health.
     *                    It is pre-filled with the latest module health signal
     * @return DW_SUCCESS
     **/
    virtual dwStatus reportCurrentHealthSignal(dwGraphHealthSignal& signal) = 0;

    /**
     * @brief Sets the node's iteration count
     * @param iterationCount The current iteration count
     * @return DW_SUCCESS
     */
    virtual dwStatus setIterationCount(uint32_t iterationCount) = 0;

    /**
     * @brief Set the current state in node.
     *       Node implementation of this API need to be thread-safe.
     * @param state The name of the new state which is about to start.
     *              The pointer is valid only for the runtime of the function and data should be stored with a deep copy.
     * @return DW_SUCCESS, DW_INVALID_ARGUMENT
     */
    virtual dwStatus setState(const char* state) = 0;

    /**
     * @brief Resets all the ports in the node
     */
    virtual void resetPorts() = 0;
};

class ISensorNode
{
public:
    /**
     * @brief Start the sensor.
     * @return DW_SUCCESS, DW_FAILURE
     */
    virtual dwStatus start() = 0;

    /**
     * @brief Stop the sensor.
     * @return DW_SUCCESS, DW_FAILURE
     */
    virtual dwStatus stop() = 0;

    /**
     * @brief Sets the affinity mask of the sensor.
     * @return DW_SUCCESS, DW_FAILURE
     */
    virtual dwStatus setAffinityMask(uint) = 0;

    /**
     * @brief Sets the thread priority of the sensor.
     * @return DW_SUCCESS, DW_FAILURE
     */
    virtual dwStatus setThreadPriority(int) = 0;

    /**
     * @brief Set start timestamp for dataset replay.
     * @return DW_SUCCESS, DW_FAILURE
     */
    virtual dwStatus setStartTime(dwTime_t) = 0;

    /**
     * @brief Set end timestamp for dataset replay.
     * @return DW_SUCCESS, DW_FAILURE
     */
    virtual dwStatus setEndTime(dwTime_t) = 0;

    /**
     * @brief distinguishes between a live and virtual sensor
     * @return DW_SUCCESS, DW_FAILURE
     */
    virtual dwStatus isVirtual(bool* isVirtualBool) = 0;

    enum class DataEventType
    {
        PRODUCE, // sensor node produces data for a node-run
        DROP,    // sensor node drops data in the next node-run
        NONE,    // sensor node does not produce data for a node-run
    };

    /**
     *  Record of data sensor data frame.
     */
    struct DataEvent
    {
        /**
         * The type of event
         */
        DataEventType dataEventType;
        /**
         *  The status of the node-run.
         *  invalid if dataEventType is DROP
         *  DW_SUCCESS if the node-run produced data.
         *  DW_TIME_OUT, DW_NOT_AVAILABLE, etc, sensor had not data for node-run.
         *  DW_END_OF_STREAM if sensor reached end of stream.
         */
        dwStatus status;
        /**
         *  The timestamp of involved data.
         *  invalid if dataEventType is NONE.
         */
        dwTime_t timestamp;
    };

    using DataEventReadCallback = dw::core::Function<bool(DataEvent&)>;
    /**
     * @brief Set read timestamp function for dataset replay.
     *        Timestamps not in the sequence returned by the callback
     *        will be dropped.
     * @return DW_SUCCESS, DW_FAILURE
     */
    virtual dwStatus setDataEventReadCallback(DataEventReadCallback cb) = 0;

    using DataEventWriteCallback = dw::core::Function<void(DataEvent)>;
    /**
     * @brief Set write timestamp function for live case.
     *        Each timestamp of data output from the node
     *        will be passed to this callback.
     * @return DW_SUCCESS, DW_FAILURE
     */
    virtual dwStatus setDataEventWriteCallback(DataEventWriteCallback cb) = 0;
};

class SensorNode
{
public:
    /// @deprecated Use ISensorNode::DataEventType instead.
    using DataEventType = ISensorNode::DataEventType;

    /// @deprecated Use ISensorNode::DataEvent instead.
    using DataEvent = ISensorNode::DataEvent;

    /// @deprecated Use ISensorNode::DataEventReadCallback instead.
    using DataEventReadCallback = ISensorNode::DataEventReadCallback;

    /// @deprecated Use ISensorNode::DataEventWriteCallback instead.
    using DataEventWriteCallback = ISensorNode::DataEventWriteCallback;

    SensorNode(Node* node)
        : m_node(node)
        , m_sensorNode(dynamic_cast<ISensorNode*>(node))
    {
        if (m_sensorNode != nullptr)
        {
            throw Exception(DW_INVALID_ARGUMENT, "Passed node pointer does not implement ISensorNode.");
        }
    }

    Node* getNode()
    {
        return m_node;
    }

    Node const* getNode() const
    {
        return m_node;
    }

    ISensorNode* getSensorNode()
    {
        return m_sensorNode;
    }

    ISensorNode const* getSensorNode() const
    {
        return m_sensorNode;
    }

private:
    Node* m_node{};
    ISensorNode* m_sensorNode{};
};

/**
 * For nodes require extra actions to be taken before shutdown should
 * implement this interface.
 * For example: some node may want to save some information to file
 * at exit of each run, the node should inherit this interface and
 * implement the save to file operations in this interface.
 */
class IContainsPreShutdownAction
{
public:
    virtual ~IContainsPreShutdownAction() = default;

    /**
     * @brief actions to be taken before node shutdown
     * @return DW_SUCCESS, DW_FAILURE
     */
    virtual dwStatus preShutdown() = 0;
};

// TODO(ajayawardane) WAR: When there is a single SSM pass in the graph, reset could potentially
// be called while node passes are running. Until STM schedule re-entry is supported, we reset the
// supported nodes in the setup pass.
class IAsyncResetable
{
public:
    virtual ~IAsyncResetable() = default;

    /**
     * @brief Set the async reset flag.
     * @return DW_SUCCESS, DW_FAILURE
     */
    virtual dwStatus setAsyncReset() = 0;
    /**
     * @brief Executes a reset if the async reset flag is set.
     * @return DW_SUCCESS, DW_FAILURE
     */
    virtual dwStatus executeAsyncReset() = 0;
};

/**
 * For nodes require extra actions to be taken after the channels are connected should
 * implement this listener.
 */
class IChannelsConnectedListener
{
public:
    virtual ~IChannelsConnectedListener() = default;

    /**
     * @brief Callback received after channels are connected
     */
    virtual void onChannelsConnected() = 0;
};

/// End of New Node Structure
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace framework
} // namespace dw

#include "impl/ExceptionSafeNode.hpp"

#endif // DW_FRAMEWORK_BASECLASS_NODE_HPP_
