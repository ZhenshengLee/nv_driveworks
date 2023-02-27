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
// SPDX-FileCopyrightText: Copyright (c) 2016-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWTRACE_TRACE_HPP_
#define DWTRACE_TRACE_HPP_

// Trace Public header file
#include <dw/trace/core/TraceTypes.hpp>

namespace dw
{
namespace trace
{
static constexpr char8_t const* DW_TRACE_STR_TAG   = "[TRACE]: ";
static constexpr uint64_t DW_TRACE_CHAN_MASK_NONE  = 0;
static constexpr uint64_t DW_TRACE_CHAN_ENABLE_ALL = (0xFFFFFFFF);
static const auto DW_TRACE_CHAN_MASK               = [](uint32_t const idx) { return (1U << idx); };

/**
 * Cuda events are required for measuring execution time on GPU/DLA.
 * This field allows DWTrace to create specified number of cudaEvents at
 * initialisation time.
 * Please change the value if number of tracing API's which are using
 * cudaEvents are increased.
 **/
static constexpr size_t DW_TRACE_MAX_CUDA_EVENTS = 384;

/**
 * If not sure about tracing level at the time of using DWTrace API,
 * then use default trace level.
 **/

static constexpr Level DW_TRACE_LEVEL_DEFAULT = Level::LEVEL_30;

/**
 * Default number of channel traces before flush.
 **/
static constexpr uint32_t DW_TRACE_FLUSH_INTERVAL_DEFAULT = 5500;

/**
 * Default Max file size of generate DWTrace in MBs. After this file limit reached log
 * rotation will start.
 **/
static constexpr uint32_t DW_TRACE_MAX_FILE_SIZE_MB = 8192;

/**
 * DWTrace initialisation and configuration structure. Based on values
 * provided in following structure DWTrace will be configured.
 **/
struct TracerConfig
{
    // Enable tracing.
    bool enabled = false;
    // File path where Trace*.txt will be stored if fileBackend is enabled.
    dwtStringFilepath_t filePath = {};
    // DWTrace supports total 32 channels. Following mask allows to enable/
    // disable specific channels.
    uint64_t channelMask = 0;
    // Enable filebased backend. For this backend post processing
    // script dwTrace.py is needed to infer results.
    bool fileBackendEnabled = false;
    // Max file size of generated filebackend based dwtrace.
    // After this file limit reached log rotation will start.
    uint32_t maxFileSizeMB = 0;
    // Enable network socket backend. For this backend post processing
    bool networkBackendEnabled = false;
    // customer's ipAddr
    dwtStringIPAddr_t ipAddr = {};
    // customer's serverPort
    uint16_t serverPort = 0;
    // Enable NVTx backend.
    bool nvtxBackendEnabled = false;
    // Global tracing level, any trace which has level greater than this
    // level will be ignored.
    Level tracingLevel = Level::NONE;
    // Number of frames after which flusing to backend will be called.
    uint32_t flushInterval = 0;
    // Enable ftrace backend.
    bool ftraceBackendEnabled = false;
    // Enable memory usage trace
    bool memTraceEnabled = false;
    // Enable memory usage trace outside of STARTUP channel
    bool fullMemUsage = false;
    // Enable disk io usage(read/write bytes, io delay total) trace. Disk IO operation
    // mainly happens when running application after boot. After that data is cached(Depends upon OS and System memory)
    // For now this info will be dumped for STARTUP channels BEGIN/END, as it is expected that major IO operation happens during program init phase.
    bool diskIOStatsEnabled = false;
    // Start time of roadrunner
    dwtFixedString_t startTime = {};
    // Create Cuda Events
    bool createCudaEvents = true;

    TracerConfig() = default;
    TracerConfig(bool const enabledFlag, char8_t const* const fileLoc, uint64_t const chanMask,
                 bool const fileBackendFlag, uint32_t const fileSize, bool const networkBackendFlag, char8_t const* const ipAddress,
                 uint16_t const portNum, bool const nvtxBackendFlag, uint32_t const traceLevel,
                 uint32_t const flushEpoch, bool const ftraceFlag, bool const memTraceFlag, bool const fullMemUse, bool const diskIOStats,
                 char8_t const* const start = nullptr, bool const& createCudaEventsFlag = true)
        : enabled(enabledFlag), filePath(fileLoc), channelMask(chanMask), fileBackendEnabled(fileBackendFlag), maxFileSizeMB(fileSize), networkBackendEnabled(networkBackendFlag), ipAddr(ipAddress), serverPort(portNum), nvtxBackendEnabled(nvtxBackendFlag), tracingLevel(static_cast<Level>(traceLevel)), flushInterval((flushEpoch > DW_TRACE_MAX_NUM_EVENTS_PER_CHAN) ? DW_TRACE_MAX_NUM_EVENTS_PER_CHAN : flushEpoch), ftraceBackendEnabled(ftraceFlag), memTraceEnabled(memTraceFlag), fullMemUsage(fullMemUse), diskIOStatsEnabled(diskIOStats), startTime(start), createCudaEvents(createCudaEventsFlag)
    {
    }
};
} // namespace trace
} // namespace dw

// Supports up to 5 layers of Concatenation
#ifdef DW_USE_TRACE
#include <dw/trace/core/TraceImpl.hpp>
#include <dw/trace/wrapper/TraceWrapper.hpp>

namespace dw
{
namespace trace
{
#define DW_TRACE_GET_MACRO_2(_1, _2, NAME, ...) NAME
#define DW_TRACE_GET_MACRO_5(_1, _2, _3, _4, _5, NAME, ...) NAME

#define DW_TRACE_TAG(...) \
    dw::trace::dwtFixedString_t(DW_TRACE_GET_MACRO_5(__VA_ARGS__, DW_TRACE_TAG5, DW_TRACE_TAG4, DW_TRACE_TAG3, DW_TRACE_TAG2, DW_TRACE_TAG1)(__VA_ARGS__))
#define DW_TRACE_TAG1(STR1) (STR1)
#define DW_TRACE_TAG2(STR1, STR2) (STR1 "@" STR2)
#define DW_TRACE_TAG3(STR1, STR2, STR3) (STR1 ":" STR2 "@" STR3)
#define DW_TRACE_TAG4(STR1, STR2, STR3, STR4) (STR1 ":" STR2 ":" STR3 "@" STR4)
#define DW_TRACE_TAG5(STR1, STR2, STR3, STR4, STR5) (STR1 ":" STR2 ":" STR3 ":" STR4 "@" STR5)

#define DW_TRACE_PAYLOAD(...)                                               \
    DW_TRACE_GET_MACRO_2(__VA_ARGS__, DW_TRACE_PAYLOAD2, DW_TRACE_PAYLOAD1) \
    (__VA_ARGS__)
#define DW_TRACE_PAYLOAD1(STR1) dw::trace::singlePayload(STR1)
static inline dwtFixedString_t singlePayload(char8_t const* const str1)
{
    dwtFixedString_t const str(str1);
    return str;
}
static inline dwtFixedString_t singlePayload(int32_t const i)
{
    dwtFixedString_t str;
    str += i;
    return str;
}

#define DW_TRACE_PAYLOAD2(STR1, STR2) dw::trace::joinPayloadChars(STR1, STR2)
static inline dwtFixedString_t joinPayloadChars(char8_t const* const str1, char8_t const* const str2)
{
    dwtFixedString_t str(str1);
    str += "@";
    str += str2;
    return str;
}

// Avoid shadow declaration for nested tracing with unique variables
#define DW_CONCAT_(x, y) x##y
#define DW_CONCAT(x, y) DW_CONCAT_(x, y)
#define DW_UNIQUE_VAR(name) DW_CONCAT(name, __LINE__)

/**
 * This API configures DWTrace.
 * @param[in] chan_mask Enables specific channels in DWTrace.
 * @param[in] level Tracing API's called with level greater than this value
 *            wont appear in tracing file.
 **/
#define DW_TRACE_ENABLE(chan_mask, level) dw::trace::dwTraceEnable(chan_mask, level)

/**
 * Disables tracing for specific channels.
 * @param[in] chan_mask Disables specific channels in DWTrace.
 **/
#define DW_TRACE_DISABLE(chan_mask) dw::trace::dwTraceDisable(chan_mask)

/**
 * Enables all channels in DWTrace.
 **/
#define DW_TRACE_ENABLE_ALL() dw::trace::dwTraceEnableAll()

/**
 * API to mark specific instant in program.
 * @param[in] chan Specifies channels where trace has to be placed.
 * @param[in] tag  Unique string for every trace.
 * @param[in] level [optional] If not mentioned DW_TRACE_LEVEL_DEFAULT is used.
 * @param[in] payload [optional] Default is null string.
 **/
#define DW_TRACE_MARK(...) dw::trace::dwTraceMark(__VA_ARGS__)

/**
 * This API is used to determine time taken by specific lines or function in code.
 * Measures CPU time.
 * @param[in] chan Specifies channels where trace has to be placed.
 * @param[in] tag  Unique string for every trace.
 * @param[in] level [optional] If not mentioned DW_TRACE_LEVEL_DEFAULT is used.
 * @param[in] payload [optional] Default is null string
 **/
#define DW_TRACE_BEGIN(...) dw::trace::dwTraceBegin(__VA_ARGS__)

/**
 * This API should be paired with DW_TRACE_BEGIN. Marks end of tracing for trace
 * specified by tag string.
 * @param[in] chan Specifies channels where trace has to be placed.
 * @param[in] tag  Unique string for every trace.
 * @param[in] level [optional] If not mentioned DW_TRACE_LEVEL_DEFAULT is used.
 * @param[in] payload [optional] Default is null string
 **/
#define DW_TRACE_END(...) dw::trace::dwTraceEnd(__VA_ARGS__)

/**
 * This API is used to trace asynchronous functions.
 * Asynchronous function execution can start and stop on different thread/process.
 * Measures CPU time.
 * @param[in] chan Specifies channels where trace has to be placed.
 * @param[in] tag  Unique string for every trace.
 * @param[in] level [optional] If not mentioned DW_TRACE_LEVEL_DEFAULT is used.
 * @param[in] payload [optional] Default is null string
 **/
#define DW_TRACE_ASYNC_BEGIN(...) dw::trace::dwTraceAsyncBegin(__VA_ARGS__)

/**
 * This API should be paired with DW_TRACE_ASYNC_BEGIN.
 * @param[in] chan Specifies channels where trace has to be placed.
 * @param[in] tag  Unique string for every trace.
 * @param[in] level [optional] If not mentioned DW_TRACE_LEVEL_DEFAULT is used.
 * @param[in] payload [optional] Default is null string
 **/
#define DW_TRACE_ASYNC_END(...) dw::trace::dwTraceAsyncEnd(__VA_ARGS__)

/**
 * This API is used to determine time taken by specific tasks on GPU/DLA.
 * Uses cudaDeviceSync implicitly to wait for GPU/DLA to finish its task.
 * @param[in] chan Specifies channels where trace has to be placed.
 * @param[in] tag  Unique string for every trace.
 * @param[in] cudaStream  cudaStream on which task is queued.
 * @param[in] level [optional] If not mentioned DW_TRACE_LEVEL_DEFAULT is used.
 * @param[in] payload [optional] Default is null string
 **/
#define DW_TRACE_CUDA_BEGIN(...) dw::trace::dwTraceCudaBegin(__VA_ARGS__)

/**
 * This API should be paired with DW_TRACE_CUDA_BEGIN.
 * @param[in] chan Specifies channels where trace has to be placed.
 * @param[in] tag  Unique string for every trace.
 * @param[in] cudaStream  cudaStream on which task is queued.
 * @param[in] level [optional] If not mentioned DW_TRACE_LEVEL_DEFAULT is used.
 * @param[in] payload [optional] Default is null string
 **/
#define DW_TRACE_CUDA_END(...) dw::trace::dwTraceCudaEnd(__VA_ARGS__)

/**
 * This API is used to determine time taken by specific tasks on GPU/DLA.
 * Marks start of specific task using cudaEvent in cudaStream queue.
 * @param[in] chan Specifies channels where trace has to be placed.
 * @param[in] tag  Unique string for every trace.
 * @param[in] cudaStream  cudaStream on which task is queued.
 * @param[in] level [optional] If not mentioned DW_TRACE_LEVEL_DEFAULT is used.
 * @param[in] payload [optional] Default is null string.
 **/
#define DW_TRACE_CUDA_BEGIN_ASYNC(...) DW_TRACE_CUDA_BEGIN(__VA_ARGS__)

/**
 * Marks end of specific task using cudaEvent in cudaStream queue.
 * @param[in] chan Specifies channels where trace has to be placed.
 * @param[in] tag  Unique string for every trace.
 * @param[in] cudaStream  cudaStream on which task is queued.
 * @param[in] level [optional] If not mentioned DW_TRACE_LEVEL_DEFAULT is used.
 * @param[in] payload [optional] Default is null string.
 **/
#define DW_TRACE_CUDA_RECORD_ASYNC(...) dw::trace::dwTraceCudaRecordAsync(__VA_ARGS__)

/**
 * This API should be paired with DW_TRACE_CUDA_BEGIN_ASYNC and DW_TRACE_CUDA_RECORD_ASYNC.
 * Measures execution time on GPU/DLA using cudaEvents.
 * Uses cudaEventSynchronize implicitly to wait for GPU/DLA to finish its task.
 * This API uses start and end cuda events pushed by DW_TRACE_CUDA_BEGIN/RECORD_ASYNC API's
 * to calculate execution time on GPU/DLA.
 * (cudaStream Queue)
 * cudaStopEvent_B <-- TaskB <--- cudaStartEvent_B <--cudaStopEvent_A <--TaskA <--cudaStartEvent_A
 * ExecutionTime_TaskB = cudaEventElapsedTime(cudaEventRecord(cudaStopEvent_B) - cudaElapsedTime(cudaStopEvent_B))
 * ExecutionTime_TaskA = cudaEventElapsedTime(cudaEventRecord(cudaStopEvent_A) - cudaElapsedTime(cudaStopEvent_A))
 * @param[in] chan Specifies channels where trace has to be placed.
 * @param[in] tag  Unique string for every trace.
 * @param[in] cudaStream  cudaStream on which task is queued.
 * @param[in] level [optional] If not mentioned DW_TRACE_LEVEL_DEFAULT is used.
 * @param[in] payload [optional] Default is null string.
 **/
#define DW_TRACE_CUDA_COLLECT_ASYNC(...) dw::trace::dwTraceCudaCollectAsync(__VA_ARGS__)

/**
 * This API should be paired with DW_TRACE_CUDA_BEGIN_ASYNC and DW_TRACE_CUDA_RECORD_ASYNC.
 * Measures execution time on GPU/DLA using cudaEvents.
 * This API uses start and end cuda events pushed by DW_TRACE_CUDA_BEGIN/RECORD_ASYNC
 * to calculate execution time on GPU/DLA.
 * Unlike DW_TRACE_CUDA_COLLECT_ASYNC, this API does not use cudaEventSyncronize.
 * This API should be placed in code where we are sure that tasks on GPU/DLA have been finished.
 * (cudaStream Queue)
 * cudaStopEvent_B <-- TaskB <--- cudaStartEvent_B <--cudaStopEvent_A <--TaskA <--cudaStartEvent_A
 * ExecutionTime_TaskB = cudaEventElapsedTime(cudaEventRecord(cudaStopEvent_B) - cudaElapsedTime(cudaStopEvent_B))
 * ExecutionTime_TaskA = cudaEventElapsedTime(cudaEventRecord(cudaStopEvent_A) - cudaElapsedTime(cudaStopEvent_A))
 **/
#define DW_TRACE_CUDA_COLLECT_ALL() dw::trace::dwTraceCudaCollectAll()

/**
 * Initialise DWTrace core from application. This init will not include internal buffer
 * and backend management for file and network backends. Application need to provide buffers using
 * DW_TRACE_BIND_OUTPUT and register callback DW_TRACE_REGISTER_CB for getting info if
 * application provided buffer is full.
 * Application needs to call this API to use tracing API's.
 * @param[in] tracerCfg Specific configuration values for DWTrace
 **/
#define DW_TRACE_CORE_INIT(...) dw::trace::dwTraceCoreInit(__VA_ARGS__)

/**
 * Initialise DWTrace from application. Meant for application who donot wish for
 * handling its own file and network backend.
 * Application needs to call this API to use tracing API's.
 * @param[in] tracerCfg Specific configuration values for DWTrace
 **/
#define DW_TRACE_INIT(...)                          \
    if (!dw::trace::isDWTracingEnabled())           \
    {                                               \
        dw::trace::dwTraceCoreInit(__VA_ARGS__);    \
        dw::trace::dwTraceWrapperInit(__VA_ARGS__); \
    }

/**
 * Reset DWTrace core and DeInitialise DWTrace Wrapper.
 **/
#define DW_TRACE_RESET(...)                           \
    dw::trace::dwTraceReset(__VA_ARGS__);             \
    if (!dw::trace::isDWTracingEnabled())             \
    {                                                 \
        dw::trace::dwTraceWrapperDeinit(__VA_ARGS__); \
    }

/**
 * Reset DWTrace Wrapper
 **/
#define DW_TRACE_RESET_WRAPPER(...) \
    dw::trace::dwTraceWrapperReset(__VA_ARGS__);

#define DW_TRACE_BIND_OUTPUT(...) dw::trace::dwTraceBindOutput(__VA_ARGS__)

#define DW_TRACE_REGISTER_CB(...) dw::trace::dwTraceRegisterCallback(__VA_ARGS__)
/**
 * DWTrace uses buffer for storing traces, these traces are written to
 * backends periodically based on configuration value provided when
 * DW_TRACE_INIT is called.
 * This API can be used to force DWTrace to flush traces to backends.
 **/
#define DW_TRACE_FLUSH(isForce) dw::trace::dwTraceFlush(isForce)

/**
 * This API is used to measure CPU execution time across scope.
 * For this API, there is no need to pair with DW_TRACE_END.
 * DW_TRACE_END will be inserted automatically when program execution
 * goes out of scope.
 * @param[in] chan Specifies channels where trace has to be placed.
 * @param[in] tag  Unique string for every trace.
 * @param[in] level [optional] If not mentioned DW_TRACE_LEVEL_DEFAULT is used.
 * @param[in] payload [optional] Default is null string.
 **/
#define DW_TRACE_SCOPE(...)                                                   \
    dw::trace::CpuScopeTraceEvent DW_UNIQUE_VAR(cpuTraceEvent_)(__VA_ARGS__); \
    if (dw::trace::isDWTracingEnabled())                                      \
    {                                                                         \
        dw::trace::dwTraceBegin(__VA_ARGS__);                                 \
    }

/**
 * This API is used to measure GPU execution time across scope.
 * For this API, there is no need to pair with DW_TRACE_CUDA_END.
 * DW_TRACE_CUDA_END will be inserted automatically when program execution
 * goes out of scope.
 * @param[in] chan Specifies channels where trace has to be placed.
 * @param[in] tag  Unique string for every trace.
 * @param[in] cudaStream  cudaStream on which task is queued.
 * @param[in] level [optional] If not mentioned DW_TRACE_LEVEL_DEFAULT is used.
 * @param[in] payload [optional] Default is null string.
 **/
#define DW_TRACE_CUDA_SCOPE(...)                                                       \
    dw::trace::CudaScopeTraceEvent DW_UNIQUE_VAR(cudaTraceEvent_)(false, __VA_ARGS__); \
    if (dw::trace::isDWTracingEnabled())                                               \
    {                                                                                  \
        dw::trace::dwTraceCudaBegin(__VA_ARGS__);                                      \
    }

/**
 * This API is used to measure GPU execution time across scope.
 * For this API, there is no need to pair with DW_TRACE_CUDA_ASYNC_END.
 * DW_TRACE_CUDA_ASYNC_END will be inserted automatically when program
 * execution goes out of scope.
 * @param[in] chan Specifies channels where trace has to be placed.
 * @param[in] tag  Unique string for every trace.
 * @param[in] cudaStream  cudaStream on which task is queued.
 * @param[in] level [optional] If not mentioned DW_TRACE_LEVEL_DEFAULT is used.
 * @param[in] payload [optional] Default is null string.
 **/
#define DW_TRACE_CUDA_SCOPE_ASYNC(...)                                                \
    dw::trace::CudaScopeTraceEvent DW_UNIQUE_VAR(cudaTraceEvent_)(true, __VA_ARGS__); \
    if (dw::trace::isDWTracingEnabled())                                              \
    {                                                                                 \
        dw::trace::dwTraceCudaBegin(__VA_ARGS__);                                     \
    }
} // namespace trace
} // namespace dw

#else

namespace dw
{
namespace trace
{
#define DW_TRACE_TAG(...)
#define DW_TRACE_PAYLOAD(...)
#define DW_TRACE_INIT(...)
#define DW_TRACE_ENABLED(chan_mask, level)
#define DW_TRACE_DISABLED(chan_mask)
#define DW_TRACE_BIND_OUTPUT(...)
#define DW_TRACE_REGISTER_CB(...)
#define DW_UNIQUE_VAR(name)
#define DW_TRACE_MARK(...)
#define DW_TRACE_BEGIN(...)
#define DW_TRACE_END(...)
#define DW_TRACE_SCOPE(...)
#define DW_TRACE_ASYNC_BEGIN(...)
#define DW_TRACE_ASYNC_END(...)
#define DW_TRACE_FLUSH(isForce)
#define DW_TRACE_CUDA_BEGIN(...)
#define DW_TRACE_CUDA_END(...)
#define DW_TRACE_CUDA_BEGIN_ASYNC(...)
#define DW_TRACE_CUDA_RECORD_ASYNC(...)
#define DW_TRACE_CUDA_COLLECT_ASYNC(...)
#define DW_TRACE_CUDA_SCOPE(...)
#define DW_TRACE_CUDA_SCOPE_ASYNC(...)

} // namespace trace
} // namespace dw
#endif

namespace dw
{
namespace trace
{

// advertise performance impact of cudaEventBlockingSync (true = use it)
inline bool isBlockingSyncPreferred()
{
#ifdef VIBRANTE
    return true; // Tegra
#else
    return false; // x86
#endif
}

inline uint32_t getPreferredBlockingFlags()
{
    return dw::trace::isBlockingSyncPreferred() ? static_cast<uint32_t>(cudaEventBlockingSync) : static_cast<uint32_t>(0U);
}

} // namespace trace
} // namespace dw

#endif // DWTRACE_TRACE_HPP_
