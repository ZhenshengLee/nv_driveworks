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
// SPDX-FileCopyrightText: Copyright (c) 2018-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <dwcgf/Exception.hpp>
#include <dw/core/language/Function.hpp>
#include <dw/core/container/BaseString.hpp>

#include <memory>
#include <mutex>
#include <string>

namespace dw
{
namespace framework
{

///////////////////////////////////////////////////////////////////////////////////////
static constexpr dwTime_t CHN_WAIT_TIMEOUT_US = 50'000'000;

class ChannelObject
{
public:
    virtual ~ChannelObject() = default;

    class SyncClient
    {
    public:
        /**
         *  Retrieve the NvSciSyncObjs allocated by the channel.
         *  The channel must be fully connected before this API is called.
         *
         *  @throws DW_INVALID_STATE if the channel is not yet connected, or if syncObjs output memory is null.
         *          DW_OUT_OF_BOUNDS if the passed span does not have sufficient size.
         * @note This will overwrite items currently in the passed output span.         
         **/
        virtual void getSyncObjs(dw::core::span<NvSciSyncObj>& syncObjs) = 0;
    };

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
         */
        virtual void getWaitFences(void* data, dw::core::span<NvSciSyncFence>& waitFences) = 0;
    };

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

    class PacketPool
    {
    public:
        SyncWaiter& getSyncWaiter()
        {
            auto* waiter = dynamic_cast<SyncWaiter*>(this);
            if (waiter == nullptr)
            {
                throw Exception(DW_BAD_CAST, "ChannelObject::PacketPool: not a SyncWaiter");
            }
            return *waiter;
        }

        SyncSignaler& getSyncSignaler()
        {
            auto* signaler = dynamic_cast<SyncSignaler*>(this);
            if (signaler == nullptr)
            {
                throw Exception(DW_BAD_CAST, "ChannelObject::PacketPool: not a SyncSignaler");
            }
            return *signaler;
        }

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

        using OnDataReady = dw::core::Function<void()>;

        /**
         *  Set callback for when data is ready.
         *
         *  @throws DW_NOT_SUPPORTED if not supported
         **/
        virtual void setOnDataReady(void* opaque, OnDataReady onDataReady) = 0;
    };

    /**
     *  Child interface to produce packets on the channel
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
    };

    /**
     *  Child interface to consume packets on the channel
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
    virtual Producer* getProducer(GenericDataReference ref) = 0;

    /**
     * Register a consumer client.
     * @note The client interface is owned by the ChannelObject and should not be
     *       used after the ChannelObject has been destroyed.
     * @note allocates new objects to back the requested interface.
     * @param[in] ref A reference for the data to be transfered over the channel
     * @return consumer interface
     * @throws if parameters disallow consumer role.
     **/
    virtual Consumer* getConsumer(GenericDataReference ref) = 0;

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
};

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_CHANNEL_HPP_
