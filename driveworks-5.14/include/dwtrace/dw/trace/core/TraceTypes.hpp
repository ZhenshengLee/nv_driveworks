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
// SPDX-FileCopyrightText: Copyright (c) 2016-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWTRACE_CORE_TRACE_TYPES_HPP_
#define DWTRACE_CORE_TRACE_TYPES_HPP_

#include <cstdint>
#include <thread>
#include <mutex>
#include <limits>
#include <cuda_runtime_api.h>
#include <dwshared/dwfoundation/dw/core/container/VectorFixed.hpp>
#include <dwshared/dwfoundation/dw/core/container/HashContainer.hpp>
#include <dwshared/dwfoundation/dw/core/container/BaseString.hpp>
#include <dwshared/dwfoundation/dw/core/container/RingBuffer.hpp>

namespace dw
{
namespace trace
{
static constexpr uint32_t DW_TRACE_MAX_NUM_EVENTS_PER_CHAN = (20 * 1024);
static constexpr uint32_t DW_TRACE_MAX_TAG_SIZE            = 256U;
static constexpr uint32_t DW_TRACE_MAX_PAYLOAD_SIZE        = 64U;
static constexpr uint32_t DW_TRACE_SUCCESS                 = 0U;

using dwtTime_t           = uint64_t;
using dwtFixedString_t    = dw::core::FixedString<DW_TRACE_MAX_TAG_SIZE>;
using dwtStringFilepath_t = dw::core::FixedString<384>;
using dwtStringIPAddr_t   = dw::core::FixedString<16>;
template <typename typeKey, typename typeValue>
using dwtHashMap_t = dw::core::StaticHashMap<typeKey, typeValue, 384>;
template <typename typeT, size_t C = 0>
using dwtVectorFixed_t = dw::core::VectorFixed<typeT, C>;
template <typename typeT, size_t C>
using dwtBufferQueue_t                              = dw::core::RingBuffer<typeT, C>;
static constexpr dwtTime_t DWTRACE_TIMEOUT_INFINITE = std::numeric_limits<dwtTime_t>::max();

enum class TraceHeaderType
{
    NONE = 0,
    MARKER,
    RANGE_BEGIN,
    RANGE_END,
    ASYNC_RANGE_BEGIN,
    ASYNC_RANGE_END,
    CUDA_BEGIN,
    CUDA_END,
    SCOPE
};

/**
 * DWTrace channels are used for capturing similar traces in one place.
 * Channels plays important role in dwtrace post processing scripts.
 * Following channels have been created based on RR application.
 * Channels can be renamed, added or removed from DWTrace, but similar
 * changes need to be made in dwtrace post processing scripts.
 **/

enum class TraceChannel
{
    DEFAULT = 0,
    LATENCY,
    CAMERA,
    PIPELINE,
    COMM,
    RENDER,
    RADAR,
    STARTUP,
    DATA_LATENCY,
    ROADCAST,
    VDC,
    DW,
    PROFILING,
    SHUTDOWN,
    MAX_CHANNEL
};

/**
 * Tracing can be controlled through tracing levels.
 * Important/higher priority traces should have lower tracing levels.
 * Verbose/lower priority traces should have higher tracing levels.
 * For example, if Trace API is called with Level::LEVEL_10
 * then it is considered to be higher priority trace.
 **/
enum class Level
{
    NONE      = 0,
    LEVEL_10  = 10,
    LEVEL_20  = 20,
    LEVEL_30  = 30,
    LEVEL_50  = 50,
    LEVEL_70  = 70,
    LEVEL_100 = 100,
};

enum class Backend
{
    FILEBASED = 0,
    NETWORK,
    IBACKEND,
    NVTX = IBACKEND,
    FTRACE,
    MAX_BACKEND
};

struct dwtEvent_t //clang-tidy NOLINT(readability-identifier-naming)
{
    Level level;              // level of events
    TraceChannel channel;     // synchronized event channel, suggest one per thread
    TraceHeaderType type;     // event type, marker or range
    dwtFixedString_t tag;     // event tag
    float32_t duration;       // duration of particular event
    dwtFixedString_t payload; // event payload

    const char* func;            // function name of where event comes from.
    const char* deviceName;      // Name of device on which task is running.
    size_t const deviceMemFree;  // Usage of device memory at runtime
    const int32_t hostMemUsed;   // Usage of host memory at runtime (32-bit signed int, converted to size_t during post-processing)
    uint64_t const nvMapMemUsed; // Usage of nvMap memory at runtime
    uint64_t const readBytes;    // disk i/o readBytes
    uint64_t const writeBytes;   // disk i/o writeBytes
    uint64_t const ioWaitTime;   // Time spent in io wait in nanoseconds

    dwtTime_t stamp;     // time stamp
    std::thread::id tid; // thread id
    dwtEvent_t(Level const evtLevel, TraceChannel const evtChannel, TraceHeaderType const evtType,
               dwtFixedString_t const& evtTag, float32_t const evtDuration,
               dwtFixedString_t const& evtPayload, const char* const evtFunc, const char* const evtDeviceName,
               size_t const evtDeviceMemFree, const int32_t evtHostMemUsed, uint64_t const evtNvMapMemUsed, uint64_t const evtReadBytes, uint64_t const evtWriteBytes, uint64_t const evtIOWaitTime, dwtTime_t const evtStamp)
        : level(evtLevel), channel(evtChannel), type(evtType), tag(evtTag.c_str()), duration(evtDuration), payload(evtPayload.c_str()), func(evtFunc), deviceName(evtDeviceName), deviceMemFree(evtDeviceMemFree), hostMemUsed(evtHostMemUsed), nvMapMemUsed(evtNvMapMemUsed), readBytes(evtReadBytes), writeBytes(evtWriteBytes), ioWaitTime(evtIOWaitTime), stamp(evtStamp)
    {
    }
};

/** Trace channel contains traces from particular module.
 *  For example, TraceChannel::CAMERA contains traces from camera.
 *  Disabling channel stops tracing from corresponding module.
 **/
class DWTraceChannel
{
private:
    uint32_t m_id = 0;
    dwtVectorFixed_t<dwtEvent_t> m_events;

public:
    DWTraceChannel() = default;
    DWTraceChannel(uint32_t const id, uint32_t const capacity);
    ~DWTraceChannel();
    DWTraceChannel(DWTraceChannel&& other);
    DWTraceChannel& operator=(const DWTraceChannel& other) = default;

    void pushEvent(dwtEvent_t&& event);

    dwtVectorFixed_t<dwtEvent_t> const& events() { return m_events; }
    void clear() { m_events.clear(); }
};

using TraceBuf     = dwtVectorFixed_t<DWTraceChannel>;
using TraceBufPtr  = std::unique_ptr<TraceBuf>;
using avtFlushCb_t = void (*)(TraceBuf*&, void*);

} // namespace trace
} // namespace dw
#endif // DWTRACE_CORE_TRACE_TYPES_HPP_
