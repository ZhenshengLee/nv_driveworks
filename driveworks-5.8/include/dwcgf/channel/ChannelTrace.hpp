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
// SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <dw/core/base/Status.h>
#include <dw/core/language/Function.hpp>
#include <dw/core/container/Span.hpp>

#ifndef CHANNEL_TRACE_HPP
#define CHANNEL_TRACE_HPP

namespace dw
{
namespace framework
{

enum class ChannelTraceEventType
{
    WAIT,    // Consumer::wait called by client
    RECV,    // Consumer::recv called by client
    RELEASE, // Consumer::release called by client
    DROP,    // Consumer drops a packet due to mailbox mode
    GET,     // Producer::get called by client
    SEND,    // Producer::send called by client
    RETURN   // Producer receives a packet returned by consumers
};

struct ChannelPacketFingerprint
{
    /*
     * The hash of the id of the channel involved with this packet
     */
    uint32_t channelID;
    /*
     * Incremented by the producer every time get is called.
     */
    uint32_t getID;
    /*
     * The ID of the data buffers used to send this info.
     * This id indexes the packets in the order they were created and
     * added to the producer's packet pool
     */
    uint32_t bufferID;
};

/**
 *  This struct records the info relating to a single event on a channel.
 *  For example: when producer clients calls producer->get() for the first time,
 *  we expect the following trace point:
 *  eventType = GET
 *  status = DW_SUCCESS
 *  packetFingerprint = {
 *      0xDEADBEEF, // some number that is hash id of the producer
 *      0,          // the first packet filled by the producer client.
 *      0           // the first data buffer in the producer pool
 *  }
 *
 *  As application progresses, these trace points record exactly how the channel
 *  was used and what resulted at each step, including:
 *  - which packets were requested and sent by producer client
 *  - which packets were received, dropped, and/or returned by the consumer channel/client.
 *  - which calls resulted in timeouts or other failures.
 */
struct ChannelTracePoint
{
    /**
     *  The type of the event recorded in this trace point.
     */
    ChannelTraceEventType eventType;
    /**
     *  The status returned to the client (for events involving client interaction)
     */
    dwStatus status;
    /**
     *  The fingerprint of the packet involved in the event
     */
    ChannelPacketFingerprint packetFingerprint;
};

/**
 *  A channel is created with a particular mode for these traces.
 */
enum class ChannelTraceMode
{
    /**
     *  No traces enabled.
     */
    DISABLED,
    /**
     *  Traces will be recorded.
     */
    RECORD,
    /**
     *  Traces will be read and enforced by the channel.
     *  Here enforcement of a trace means:
     *  - Channel will verify that call order of its APIs is identical
     *    to what is read from previously recorded trace.
     *  - Channel will ensure that the result of each call is identical
     *    to what is read from previously recorded trace.
     *  - If the Channel is unable to replay the trace due to detected incompatibility
     *    with the trace, exceptions will be thrown.
     *  - For example, assume a client of consumer mailbox channel received packets
     *    A, B, dropped C, and D and this information was captured in a trace.
     *    Replaying the trace, it will verify that the consumer client
     *    makes the calls to receive packets A, B, and D. If the consumer
     *    tries to receive packet D before it is arrived, it will block.
     *    If the producer sends B before A is received by client, A will not be
     *    dropped. Also if the producer tries to produce a packet before consumers
     *    have released it, it will block.
     */
    REPLAY
};

/**
 *  Callback used by channel to write a tracepoint.
 *  Upstream application must implement this callback uniquely for each channel to record traces.
 */
using ChannelTraceWriteCallback = dw::core::Function<void(ChannelTracePoint)>;

/**
 *  Callback used by channel to read a tracepoint.
 *  Upstream application must implement this callback uniquely for each channel to replay traces.
 *  @return true if tracepoint was successfully read and false otherwise.
 */
using ChannelTraceReadCallback = dw::core::Function<bool(ChannelTracePoint&)>;

/**
 *  Callback used by channel to register its need to write trace points.
 *  @param [in] channelID - the id of the channel requesting a callback to write tracepoints.
 *  @return the callback to write tracepoints for the channel with id channelID.
 */
using ChannelOnRegisterTraceWriter = dw::core::Function<ChannelTraceWriteCallback(const char* channelID)>;

/**
 *  Callback used by channel to register its need to read trace points.
 *  @param [in] channelID - the id of the channel requesting a callback to read tracepoints.
 *  @return the callback to read tracepoints for the channel with id channelID.
 */
using ChannelOnRegisterTraceReader = dw::core::Function<ChannelTraceReadCallback(const char* channelID)>;

struct ChannelTraceProcessParam
{
    /**
     * The trace ID of the channel to be processed.
     */
    uint32_t channelID;
    /**
     * Iterator to read the recorded traces of the channel to be processed.
     */
    ChannelTraceReadCallback input;
    /**
     * Iterator to write the replayable traces of the channel to be processed.
     */
    ChannelTraceWriteCallback output;
};

/**
 *  @return the hashed value of an id.
 */
uint32_t ChannelTraceHashId(const char* id);

/**
 *  Process recorded traces from a set of interconnected producers/consumers into a replayable format.
 *  For every
 *  SEND event: find the corresponding RECV events.
 *  When replaying, place the corresponding RECV events for a SEND event immediately before the SEND event.
 *  Also, replace the channelID field of the RECV packet fingerprints with the channelID of the corresponding
 *  trace log it appears.
 *  For example:
 *  If producer A sends packet 0 and 1 to consumers B and C. B receives 0 and 1, C receives only 1.
 *  The recording of the trace events will look like so:
 *  In A's trace record:
 *  SEND, A, 0
 *  SEND, A, 1
 *
 *  In B's trace replay:
 *  RECV, A, 0
 *  RECV, A, 1
 *
 *  In C's trace replay:
 *  RECV, A, 1
 *
 *  After processing with this function, replaying the trace will look like so
 *  In A's trace replay:
 *  RECV, B, 0
 *  SEND, A, 0
 *  RECV, B, 1
 *  RECV, C, 1
 *  SEND, A, 1
 *
 *  In B's trace replay:
 *  RECV, A, 0
 *  RECV, A, 1
 *
 *  In C's trace replay:
 *  RECV, A, 1
 *
 *  This allows replay channel to set mask of which consumers a packet should be sent to.
 *
 *  RECV,GET,RELEASE event: copy it to the replay.
 *
 *  DROPS are removed as they are ignored in replay.
 *  @param [in,out] traces the traces to be processed into replay state
 */
void ChannelTraceProcess(dw::core::span<ChannelTraceProcessParam> traces);

} // framework
} // dw

#endif // CHANNEL_TRACE_HPP
