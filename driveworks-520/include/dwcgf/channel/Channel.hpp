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
// SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_FRAMEWORK_CHANNEL_HPP_
#define DW_FRAMEWORK_CHANNEL_HPP_

#include "ChannelParameters.hpp"
#include "IChannelPacket.hpp"

#include <dw/core/context/Context.h>
#include <dwshared/dwfoundation/dw/core/language/Function.hpp>
#include <dwshared/dwfoundation/dw/core/container/BaseString.hpp>
#include <dw/core/health/HealthSignals.h>

#include <memory>
#include <mutex>
#include <string>

namespace dw
{
namespace framework
{

///////////////////////////////////////////////////////////////////////////////////////
// coverity[autosar_cpp14_a0_1_1_violation]
// coverity[autosar_cpp14_m0_1_4_violation]
static constexpr dwTime_t CHN_WAIT_TIMEOUT_US{50'000'000};

/**
 * Top-level interface for a specific channel instance.
 */
class ChannelObject
{
public:
    virtual ~ChannelObject() = default;

    /**
     * Common interface for NvSciSync-related setup operations.
     */
    class SyncClient
    {
    public:
        /**
         *  Retrieve the NvSciSyncObjs allocated by the Channel.
         *  The channel must be fully connected before this API is called.
         *
         *  @throws DW_INVALID_STATE if the channel is not yet connected, or if syncObjs output memory is null.
         *          DW_OUT_OF_BOUNDS if the passed span does not have sufficient size.
         * @note This will overwrite items currently in the passed output span.
         **/
        virtual void getSyncObjs(dw::core::span<NvSciSyncObj>& syncObjs) = 0;
    };

    /**
     * Interface to access the fences to wait on which were received from the Channel.
     */
    class SyncWaiter : public SyncClient
    {
    public:
        /**
         * Get the wait fences for the given data packet
         *
         * @param [in] data The data packet to get the fences for.
         * @param [out] waitFences The fences to wait on for the packet
         * @throws DW_INVALID_ARGUMENT if data is not recognized, or if waitFences output memory is null.
         *         DW_OUT_OF_BOUNDS if the passed span does not have sufficient size.
         * @note This will overwrite items currently in the passed output span.
         * @note The returned fences must be cleared by the caller after use is no longer required.
         */
        virtual void getWaitFences(void* data, dw::core::span<NvSciSyncFence>& waitFences) = 0;
    };

    /**
     * Interface to set the fences to be signaled which are to be sent over the Channel.
     */
    class SyncSignaler : public SyncClient
    {
    public:
        /**
         * Set the signal fences for the given data packet
         *
         * @param [in] data The data packet to get the fences for.
         * @param [in] postFences The fences which will be signaled.
         * @throws DW_INVALID_ARGUMENT if data is not recognized.
         *         DW_OUT_OF_BOUNDS if unsupported number of fences are passed.
         */
        virtual void setSignalFences(void* data, dw::core::span<const NvSciSyncFence> postFences) = 0;
    };

    /**
     * Common interface for both Producers and Consumers.
     */
    // coverity[autosar_cpp14_m3_4_1_violation] RFD Pending: TID-2586
    class PacketPool
    {
    public:
        SyncWaiter& getSyncWaiter();

        SyncSignaler& getSyncSignaler();

        /**
          * Wait for packets
          * @param[in] timeout Maximum time to wait for packet to become available.
          *
          * @return DW_TIME_OUT Timeout was reached and no packets became available.
          *         DW_END_OF_STREAM Upstream producer disconnected, so no further packets
          *                          will come.
          *         DW_SUCCESS
          **/
        virtual dwStatus wait(dwTime_t timeout) = 0;

        /**
          * Get all buffers allocated by the channel producer.
          **/
        virtual dw::core::VectorFixed<GenericData> getAllBuffers() = 0;
    };

    /**
     *  Child interface to produce packets on the Channel
     **/
    class Producer : public PacketPool
    {
    public:
        /**
         * Get writeable packet
         * @param[out] data The packet information is returned here on DW_SUCCESS.
         * @return DW_NOT_AVAILABLE No free packets available
         *         DW_SUCCESS
         **/
        // coverity[autosar_cpp14_a2_10_5_violation]
        virtual dwStatus get(GenericData* data) = 0;
        /**
         * Send packet
         * @param[in] data The packet data to be sent, must have come from
         *                 get() call to the same instance. On DW_SUCCESS,
         *                 write access is relinquished.
         * @return DW_BUFFER_FULL A consumer's buffers are full and was unable to receive
         *                        the packet.
         *         DW_SUCCESS
         **/
        virtual dwStatus send(void* data) = 0;

        /**
         * Release all previously acquired packet/s.
         * If data from previous send is still being held, this will release it back to the channel
         * @param[in] data The packet data to be released, this data pointer is either not yet released
         *                 by calling send or is nullptr in case of singleton channel.
         * @return
         *         DW_SUCCESS
         **/
        virtual dwStatus release(void* data) = 0;

        virtual dwTime_t getCurrentTime() = 0;
    };

    /**
     *  Child interface to consume packets from the Channel
     **/
    class Consumer : public PacketPool
    {
    public:
        /**
         * Receive a read-only packet. Recv calls should be sequenced after wait calls.
         * Calling recv without calling wait first may not result in packets being returned.
         * @param[out] data The packet information is returned here on DW_SUCCESS.
         * @return DW_END_OF_STREAM Upstream producer disconnected, so no further packets
         *                          will come.
         *         DW_NOT_AVAILABLE No packets available to return.
         *         DW_SUCCESS
         **/

        virtual dwStatus recv(GenericData* data) = 0;
        /**
         * Release a previously acquired packet
         * @param[in] data The packet data to be released, must have come from
         *                 recv() call to the same instance. On DW_SUCCESS,
         *                 read access is relinquished.
         * @return DW_BUFFER_FULL A consumer's buffers are full and was unable to receive
         *                        the packet.
         *         DW_SUCCESS
         **/

        virtual dwStatus release(void* data) = 0;
    };

    /**
     * Register a producer client.
     * @note The client interface is owned by the ChannelObject and should not be
     *       used after the ChannelObject has been destroyed.
     * @note This method shall only be called once per instance.
     * @note allocates new objects to back the requested interface.
     * @param[in] ref A reference for the data to be transfered over the channel
     * @return producer interface
     * @throws if parameters disallow producer role.
     **/
    virtual Producer* getProducer(const GenericDataReference& ref) = 0;

    /**
     * Register a consumer client.
     * @note The client interface is owned by the ChannelObject and should not be
     *       used after the ChannelObject has been destroyed.
     * @note allocates new objects to back the requested interface.
     * @param[in] ref A reference for the data to be transfered over the channel
     * @return consumer interface
     * @throws if parameters disallow consumer role.
     **/
    virtual Consumer* getConsumer(const GenericDataReference& ref) = 0;

    /**
     * Query if the channel has any clients
     * @note The channel will not have client interfaces unless requested via getProducer or getConsumer.
     **/
    virtual bool hasClients() const = 0;

    /**
     * Query if the channel and its client interfaces are connected
     * to downstream/upstream channels and are ready for use
     **/
    virtual bool isConnected() const = 0;

    /**
     * Establish connection over the channel's protocol.
     * @param[in] timeout The maximum time to attempt connection.
     *                       timeout of zero implies to do most atomic
     *                       connection attempt.
     * @return DW_TIME_OUT The time out was reached before connection
     *                     could be established.
     *         DW_CALL_NOT_ALLOWED There are no client interfaces to connect.
     *         DW_SUCCESS
     **/
    virtual dwStatus connect(dwTime_t timeout) = 0;

    /**
     * Get the parameters for this channel
     **/
    virtual const ChannelParams& getParams() const = 0;

    virtual ChannelParams& getParams() = 0;

    /**
     * Disconnect consumer endpoint
     *
     * @note only supported for NvSci producer channels
     */
    virtual void disconnectEndpoint(const char* nvsciipcEndpoint) = 0;

    /**
     * Connect consumer endpoint
     *
     * @note only supported for NvSci producer channels
     */
    virtual void connectEndpoint(const char* nvsciipcEndpoint) = 0;
};

enum class ChannelEventType
{
    CONNECTED,      ///< Channel became connected
    DISCONNECTED,   ///< Channel became or is disconnected
    ERROR,          ///< Channel encountered error
    READY,          ///< Channel is ready
    GROUP_CONNECTED ///< Channel group connected
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct ChannelEvent
{
    ChannelEventType type;                  ///< The type of the event
    uint32_t uid;                           ///< unique identifier of the channel
    uint32_t connectGroupID;                ///< connect group identifier of the event, relevant only for group connected event
    void* opaque;                           ///< The opaque pointer set for the channel tthat is ready. Only valid for type=READY
    dw::framework::OnDataReady onDataReady; ///< The callback set for the channel that is ready. Only valid for type=READY
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct ChannelError
{
    dwErrorSignal errorSignal;                   ///< The error signal.
    dw::core::FixedString<64U> nvsciipcEndpoint; ///< The endpoint, if relevant, or empty string
};

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_CHANNEL_HPP_
