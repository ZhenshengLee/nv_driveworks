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
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_FRAMEWORK_ILOCKSTEP_SYNC_CLIENT_HPP_
#define DW_FRAMEWORK_ILOCKSTEP_SYNC_CLIENT_HPP_

#include <dw/core/base/Types.h>

namespace dw
{
namespace framework
{
namespace lockstep
{

enum class LockstepSyncDataStatus
{
    /// Data is not ready to be consumed by the client.
    DATA_NOT_READY,
    /// Data is ready to be consumed by the client.
    DATA_CONSUME,
    /// Data is to be dropped by the client.
    DATA_DROP
};

enum class LookstepSyncMessageType
{
    /// An advance packet with the next running timestamp of the client.
    LOCKSTEP_SYNC_ADVANCE_PKT,
    /// A data header packet corresponding to data sent by the producer.
    LOCKSTEP_SYNC_DATA_HEADER,
    /// A control packet unblocking the client.
    LOCKSTEP_SYNC_UNBLOCK
};

/**
 * A common structure for a data header as well as an advance packet.
 */
struct LockstepSyncHeader
{
    /**
     * Type of message.
     */
    LookstepSyncMessageType type;
    /**
     * Timestamp of the producer at the time of sending for a data packet
     * and the next running timstamp for an advance packet.
     */
    dwTime_t timestamp;
    /**
     * The sequence number for the data packet. This field is empty for an
     * advance packet.
     */
    uint64_t seqNum;
};

/**
 * Producer specific interface functions for lockstep sync clients
 */
class ILockstepSyncClientProducer
{
public:
    /**
     * Send the header of the data to the consumer.
     *
     * @param[in] header Header packet containing the metadata attached to the
     *                   data packet.
     */
    virtual void sendDataHeader(const LockstepSyncHeader& header) = 0;
};

/**
 * Consumer specific interface functions for lockstep sync clients
 */
class ILockstepSyncClientConsumer
{
public:
    /**
     * Validate the data received at the consumer.
     *
     * @param[in] header Header packet containing the metadata attached to the
     *                   data packet.
     * @param[out] result Informs the client whether to consume, drop or stash the data.
     *
     * @return true - The input header is valid and the resulting struct has been populated.
     *         false - The input header is invalid and the resulting struct has not been populated.
     */
    virtual bool postProcessData(LockstepSyncDataStatus& result, const LockstepSyncHeader& header) = 0;
};

/**
 * Base interface for a lockstep sync client
 *
 * A lockstep sync client is used to synchronize between two independently running entities
 * using clock synchronization. Once the receiver and producer has created the corresponding
 * lockstep sync clients, this interface is used by both entities to deterministically
 * align to each other's clock.
 */
// coverity[autosar_cpp14_a0_1_6_violation]
class ILockstepSyncClient
{
public:
    /**
     * @brief  Wait for the consumer or producer to be ready to receive data. This call blocks
     *         until the caller is ready to proceed.
     *
     * @return true - Producer is ready to send data or consumer is ready to receive and
     *                valid data is present.
     *         false - Consumer is ready to proceed but no data should be consumed. The producer
     *                 will not return false.
     */
    virtual bool wait() = 0;

    /**
     * Set the iteration count of the client so it can advance in time.
     *
     * @param[in] iterationCount iteration count of the consumer/producer
     */
    virtual void setIterationCount(uint32_t iterationCount) = 0;

    /**
     * @brief Send the next running timestamp of the consumer/producer to the
     *        receiving client.
     */
    virtual void sendNextTimestamp() = 0;

    /**
     * @brief Send a teardown message to unblock the receiving client.
     */
    virtual void sendTeardownMessage() = 0;

    /**
     * @brief Return the producer specific interface for the lockstep sync client.
     *
     * @return Producer interface
     * @throws If the lockstep sync client is not a producer.
     */
    virtual ILockstepSyncClientProducer& getProducer() = 0;

    /**
     * @brief Return the consumer specific interface for the lockstep sync client.
     *
     * @return Consumer interface
     * @throws If the lockstep sync client is not a consumer.
     */
    virtual ILockstepSyncClientConsumer& getConsumer() = 0;

    virtual ~ILockstepSyncClient() = default;
};

} // namespace lockstep
} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_ILOCKSTEP_SYNC_CLIENT_HPP_
