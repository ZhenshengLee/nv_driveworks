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
// SPDX-FileCopyrightText: Copyright (c) 2018-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_FRAMEWORK_CHANNEL_PARAMETERS_HPP_
#define DW_FRAMEWORK_CHANNEL_PARAMETERS_HPP_

#include <dwshared/dwfoundation/dw/core/base/ExceptionWithStatus.hpp>
#include <dwshared/dwfoundation/dw/core/container/BaseString.hpp>
#include <dwshared/dwfoundation/dw/core/container/VectorFixed.hpp>
#include <dwshared/dwfoundation/dw/core/language/cxx23.hpp>
#include <dwshared/dwfoundation/dw/core/safety/Safety.hpp>
#include <dwshared/dwsockets/SocketClientServer.hpp>
#include <dw/core/system/NvMediaExt.h>
#include <dwcgf/channel/ChannelPacketTypes.hpp>
#include <sstream>
#include <limits>

namespace dw
{
namespace framework
{

///////////////////////////////////////////////////////////////////////////////////////
/**
 *  The backend transport type of a Channel
 */
enum class ChannelType : uint8_t
{
    // coverity[autosar_cpp14_a5_1_1_violation]
    SHMEM_LOCAL  = 0, ///< local shared memory
    SHMEM_REMOTE = 1, ///< remote shared memory
    EGLSTREAM    = 2, ///< EGL stream
    // coverity[autosar_cpp14_a5_1_1_violation]
    SOCKET = 3, ///< socket
    DDS    = 4, ///< Data Distribution Service (DDS)
    // coverity[autosar_cpp14_a5_1_1_violation]
    NVSCI = 5, ///< NvSciStream
    // coverity[autosar_cpp14_a5_1_1_violation]
    FSI = 6, ///< FSI
};

// coverity[autosar_cpp14_a0_1_3_violation]
static inline const char* ToParam(ChannelType channelType)
{
    switch (channelType)
    {
    case ChannelType::SHMEM_LOCAL:
        // coverity[autosar_cpp14_a5_1_1_violation]
        return "type=SHMEM_LOCAL";
    case ChannelType::SHMEM_REMOTE:
        // coverity[autosar_cpp14_a5_1_1_violation]
        return "type=SHMEM_REMOTE";
    case ChannelType::EGLSTREAM:
        // coverity[autosar_cpp14_a5_1_1_violation]
        return "type=EGLSTREAM";
    case ChannelType::SOCKET:
        // coverity[autosar_cpp14_a5_1_1_violation]
        return "type=SOCKET";
    case ChannelType::DDS:
        // coverity[autosar_cpp14_a5_1_1_violation]
        return "type=DDS";
    case ChannelType::NVSCI:
        // coverity[autosar_cpp14_a5_1_1_violation]
        return "type=NVSCI";
    case ChannelType::FSI:
        // coverity[autosar_cpp14_a5_1_1_violation]
        return "type=FSI";
    default:
        dw::core::unreachable();
    }
}

/**
 * The roles a channel allows.
 */
enum class ChannelRole : uint8_t
{
    DW_CHANNEL_ROLE_PRODUCER  = 0b01, ///< allows producer only
    DW_CHANNEL_ROLE_CONSUMER  = 0b10, ///< allows consumer only
    DW_CHANNEL_ROLE_COMPOSITE = 0b11, ///< allows both producer and consumer
};

inline constexpr bool IsProducer(ChannelRole role)
{
    return role == ChannelRole::DW_CHANNEL_ROLE_PRODUCER || role == ChannelRole::DW_CHANNEL_ROLE_COMPOSITE;
}

inline constexpr bool IsConsumer(ChannelRole role)
{
    return role == ChannelRole::DW_CHANNEL_ROLE_CONSUMER || role == ChannelRole::DW_CHANNEL_ROLE_COMPOSITE;
}

// coverity[autosar_cpp14_a0_1_1_violation]
// coverity[autosar_cpp14_m0_1_4_violation]
static constexpr uint16_t MAX_CHANNEL_PARAM_SIZE{1024U};
// coverity[autosar_cpp14_a0_1_1_violation]
// coverity[autosar_cpp14_m0_1_4_violation]
static constexpr uint16_t MAX_CHANNEL_ALL_PARAMS_SIZE{1024U};
// coverity[autosar_cpp14_a0_1_1_violation]
// coverity[autosar_cpp14_m0_1_4_violation]
static constexpr uint16_t MAX_CHANNEL_PRODUCERS_COUNT{2048U};
static constexpr uint16_t MAX_CHANNEL_CONSUMERS_COUNT{256U};
// coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2813925
// coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2813925
static constexpr uint16_t MAX_CHANNEL_STREAM_NAME_SIZE{64U};

using ChannelParamStr      = dw::core::FixedString<MAX_CHANNEL_PARAM_SIZE>;
using ChannelStreamNameStr = dw::core::FixedString<MAX_CHANNEL_STREAM_NAME_SIZE>;

enum class ChannelMode
{
    FIFO,
    MAILBOX,
    SINGLETON
};

struct ChannelPeerLocation
{
    uint32_t socID;
    uint32_t vmID;
};

// NOTE(eklein): This is slightly awkward. I would much prefer to put this in
// the ChannelNvSciStreamParams class directly, but I need access to the
// operator overloads inside ChannelNvSciStreamParams.
// I cannot forward declare the operators inside the class, and I cannot put the
// operators inside the enum class (as you would ideally do). The best compromise
// I could think of was to put this enum class here outside.
enum class ChannelNvSciStreamEnabledComponents : uint32_t
{
    COMPONENT_NONE = 0,

    COMPONENT_CPU     = 1 << 0,
    COMPONENT_EGL     = 1 << 1,
    COMPONENT_CUDA    = 1 << 2,
    COMPONENT_PVA     = 1 << 3,
    COMPONENT_DLA     = 1 << 4,
    COMPONENT_NVMEDIA = 1 << 5,
};

// whether the channel is created statically, or dynamically at runtime
// coverity[autosar_cpp14_a0_1_6_violation]
enum class ChannelConnectionType
{
    CONNECTION_TYPE_NONE = 0,
    CONNECTION_TYPE_STATIC,
    CONNECTION_TYPE_DYNAMIC
};

// the farthest reach this channel can achieve
// shorter reach is faster, so if topology is known upfront,
// use the shortest matching reach for maximum performance
// coverity[autosar_cpp14_a0_1_6_violation]
enum class ChannelReach
{
    REACH_NONE = 0,
    REACH_THREAD,
    REACH_PROCESS,
    REACH_VM,
    REACH_CHIP
};

inline constexpr ChannelNvSciStreamEnabledComponents
operator&(ChannelNvSciStreamEnabledComponents a, ChannelNvSciStreamEnabledComponents b)
{
    return static_cast<ChannelNvSciStreamEnabledComponents>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}

inline constexpr ChannelNvSciStreamEnabledComponents
operator|(ChannelNvSciStreamEnabledComponents a, ChannelNvSciStreamEnabledComponents b)
{
    return static_cast<ChannelNvSciStreamEnabledComponents>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

template <typename T>
// coverity[autosar_cpp14_a2_10_4_violation]
// coverity[autosar_cpp14_a2_10_5_violation]
static inline T ParseChannelParameter(const ChannelParamStr& value);

template <>
// coverity[autosar_cpp14_a2_10_4_violation]
// coverity[autosar_cpp14_a2_10_5_violation]
int64_t ParseChannelParameter(const ChannelParamStr& value)
{
    // coverity[autosar_cpp14_a5_1_1_violation]
    return dw::core::safeStrtol(value.c_str(), nullptr, 10);
}

template <>
// coverity[autosar_cpp14_a2_10_4_violation]
// coverity[autosar_cpp14_a2_10_5_violation]
size_t ParseChannelParameter(const ChannelParamStr& value)
{
    int64_t translatedSize{ParseChannelParameter<int64_t>(value)};
    // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
    if (translatedSize < 0)
    {
        throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "ParseChannelParameter: size_t is negative");
    }
    size_t result{static_cast<size_t>(translatedSize)};
    return result;
}

template <>
// coverity[autosar_cpp14_a2_10_4_violation]
// coverity[autosar_cpp14_a2_10_5_violation]
uint32_t ParseChannelParameter(const ChannelParamStr& value)
{
    size_t translatedSize{ParseChannelParameter<size_t>(value)};
    if (translatedSize > std::numeric_limits<uint32_t>::max())
    {
        throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "ParseChannelParameter: value is larger than uint32_t allows");
    }
    uint32_t result{static_cast<uint32_t>(translatedSize)};
    return result;
}

template <>
// coverity[autosar_cpp14_a2_10_4_violation]
// coverity[autosar_cpp14_a2_10_5_violation]
uint16_t ParseChannelParameter(const ChannelParamStr& value)
{
    size_t translatedSize{ParseChannelParameter<size_t>(value)};
    if (translatedSize > std::numeric_limits<uint16_t>::max())
    {
        throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "ParseChannelParameter: port is larger than uint16_t allows!");
    }
    uint16_t result{static_cast<uint16_t>(translatedSize)};
    return result;
}

template <>
// coverity[autosar_cpp14_a2_10_4_violation]
// coverity[autosar_cpp14_a2_10_5_violation]
uint8_t ParseChannelParameter(const ChannelParamStr& value)
{
    size_t translatedSize{ParseChannelParameter<size_t>(value)};
    if (translatedSize > std::numeric_limits<uint8_t>::max())
    {
        throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "ParseChannelParameter: port is larger than uint8_t allows!");
    }
    uint8_t result{static_cast<uint8_t>(translatedSize)};
    return result;
}

template <>
// coverity[autosar_cpp14_a2_10_4_violation]
// coverity[autosar_cpp14_a2_10_5_violation]
bool ParseChannelParameter(const ChannelParamStr& value)
{
    bool result;
    // coverity[autosar_cpp14_a5_1_1_violation]
    if ((value == "true") || (value == "1"))
    {
        result = true;
    }
    // coverity[autosar_cpp14_a5_1_1_violation]
    else if ((value == "false") || (value == "0"))
    {
        result = false;
    }
    else
    {
        throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "ParseChannelParameter: needs to be 'true' or 'false' or 1/0");
    }
    return result;
}

template <>
// coverity[autosar_cpp14_a2_10_4_violation]
// coverity[autosar_cpp14_a2_10_5_violation]
ChannelRole ParseChannelParameter(const ChannelParamStr& value)
{
    // coverity[autosar_cpp14_a5_1_1_violation]
    if (value == "producer")
    {
        return ChannelRole::DW_CHANNEL_ROLE_PRODUCER;
    }
    // coverity[autosar_cpp14_a5_1_1_violation]
    if (value == "consumer")
    {
        return ChannelRole::DW_CHANNEL_ROLE_CONSUMER;
    }
    // coverity[autosar_cpp14_a5_1_1_violation]
    if (value == "composite")
    {
        return ChannelRole::DW_CHANNEL_ROLE_COMPOSITE;
    }
    throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "ParseChannelParameter: role unknown!");
}

template <>
// coverity[autosar_cpp14_a2_10_4_violation]
// coverity[autosar_cpp14_a2_10_5_violation]
ChannelType ParseChannelParameter(const ChannelParamStr& value)
{
    // coverity[autosar_cpp14_a5_1_1_violation]
    if (value == "SHMEM_LOCAL")
    {
        return ChannelType::SHMEM_LOCAL;
    }
    // coverity[autosar_cpp14_a5_1_1_violation]
    if (value == "SHMEM_REMOTE")
    {
        return ChannelType::SHMEM_REMOTE;
    }
    // coverity[autosar_cpp14_a5_1_1_violation]
    if (value == "EGLSTREAM")
    {
        return ChannelType::EGLSTREAM;
    }
    // coverity[autosar_cpp14_a5_1_1_violation]
    if (value == "SOCKET")
    {
        return ChannelType::SOCKET;
    }
    // coverity[autosar_cpp14_a5_1_1_violation]
    if (value == "DDS")
    {
        return ChannelType::DDS;
    }
    // coverity[autosar_cpp14_a5_1_1_violation]
    if (value == "NVSCI")
    {
        return ChannelType::NVSCI;
    }
    // coverity[autosar_cpp14_a5_1_1_violation]
    if (value == "FSI")
    {
        return ChannelType::FSI;
    }
    throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "ParseChannelParameter: type unknown!");
}

template <>
// coverity[autosar_cpp14_a2_10_4_violation]
// coverity[autosar_cpp14_a2_10_5_violation]
ChannelMode ParseChannelParameter(const ChannelParamStr& value)
{
    ChannelMode result;
    // coverity[autosar_cpp14_a5_1_1_violation]
    if (value == "mailbox")
    {
        result = ChannelMode::MAILBOX;
    }
    // coverity[autosar_cpp14_a5_1_1_violation]
    else if (value == "singleton")
    {
        result = ChannelMode::SINGLETON;
    }
    else
    {
        throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "ParseChannelParameter: ChannelMode unknown!");
    }
    return result;
}

template <>
// coverity[autosar_cpp14_a2_10_4_violation]
// coverity[autosar_cpp14_a2_10_5_violation]
ChannelPeerLocation ParseChannelParameter(const ChannelParamStr& value)
{
    // coverity[autosar_cpp14_a5_1_1_violation]
    size_t pos{value.find(".")};
    ChannelPeerLocation result{};
    // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
    ChannelParamStr first{value.substr(0U, pos)};
    // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
    ChannelParamStr second{value.substr(dw::core::safeAdd(pos, 1U).value())};
    result.socID = ParseChannelParameter<uint32_t>(first);
    result.vmID  = ParseChannelParameter<uint32_t>(second);
    return result;
}

template <typename T>
// coverity[autosar_cpp14_a2_10_4_violation]
// coverity[autosar_cpp14_a2_10_5_violation]
void ParseChannelParameter(const ChannelParamStr& value, T& result)
{
    result = ParseChannelParameter<T>(value);
}

template <size_t Size>
// coverity[autosar_cpp14_a2_10_4_violation]
// coverity[autosar_cpp14_a2_10_5_violation]
void ParseChannelParameter(const ChannelParamStr& value, dw::core::FixedString<Size>& result)
{
    result = value;
}

template <typename T, size_t N>
// coverity[autosar_cpp14_a2_10_4_violation]
// coverity[autosar_cpp14_a2_10_5_violation]
void ParseChannelParameter(const ChannelParamStr& value, dw::core::VectorFixed<T, N>& result)
{
    size_t pos{0U};
    size_t endpos{0U};
    while (true)
    {
        // coverity[autosar_cpp14_a5_1_1_violation]
        endpos = value.find(":", pos);
        bool done{endpos == dw::core::FixedString<1>::NPOS};
        size_t count{done ? endpos : endpos - pos};
        T entry{};
        ParseChannelParameter(value.substr(pos, count), entry);
        result.push_back(entry);
        if (done)
        {
            break;
        }
        // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
        pos = endpos + 1U;
    }
}

// coverity[autosar_cpp14_a2_10_4_violation]
// coverity[autosar_cpp14_a2_10_5_violation]
// coverity[autosar_cpp14_m0_1_8_violation]
static inline void ParseChannelParameters(const ChannelParamStr&, const ChannelParamStr&)
{
    return;
}

template <typename T, typename... Others>
// coverity[autosar_cpp14_a2_10_4_violation]
// coverity[autosar_cpp14_a2_10_5_violation]
static inline void ParseChannelParameters(const ChannelParamStr& key, const ChannelParamStr& value, const char* staticKey, T& result, Others&&... others)
{
    if (key == staticKey)
    {
        ParseChannelParameter(value, result);
        return;
    }
    ParseChannelParameters(key, value, std::forward<Others>(others)...);
}

template <typename... Others>
// coverity[autosar_cpp14_a2_10_4_violation]
// coverity[autosar_cpp14_a2_10_5_violation]
static inline void ParseAllChannelParameters(const ChannelParamStr& channelParams, Others&&... others)
{
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2738197
    std::size_t key{0U};
    // coverity[autosar_cpp14_a5_1_1_violation]
    std::size_t pos{channelParams.find("=")};
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2738197
    std::size_t value{0U};
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2738197
    std::size_t valueEnd{0U};

    ChannelParamStr keyString{};
    ChannelParamStr valueString{};
    while (pos != dw::core::FixedString<1>::NPOS && value != dw::core::FixedString<1>::NPOS)
    {
        keyString = channelParams.substr(key, dw::core::safeSub(pos, key).value());
        // coverity[autosar_cpp14_a5_1_1_violation]
        value = channelParams.find(",", pos);
        if (value == dw::core::FixedString<1>::NPOS)
        {
            // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
            valueEnd = dw::core::safeSub(channelParams.length(), dw::core::safeAdd(pos, 1U).value()).value();
        }
        else
        {
            // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
            valueEnd = dw::core::safeSub(dw::core::safeSub(value, pos).value(), 1U).value();
        }
        // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
        valueString = channelParams.substr(pos + 1U, valueEnd);
        ParseChannelParameters(keyString, valueString, std::forward<Others>(others)...);

        // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
        key = value + 1U;
        // coverity[autosar_cpp14_a5_1_1_violation]
        pos = channelParams.find("=", key);
    }
}

class ChannelSocketParams
{
public:
    static inline ChannelParamStr getParamStr(const char* serverIP,
                                              uint16_t port,
                                              bool producerFifo                         = false,
                                              uint16_t numBlockingConnections           = 1U,
                                              dw::core::FixedString<8> const sockPrefix = dw::core::FixedString<8>())
    {
        std::stringstream ss{};
        ss.flags(std::ios::dec);
        ss << "type=SOCKET";
        if (serverIP != nullptr)
        {
            ss << ",ip=";
            ss << serverIP;
        }
        ss << ",id=";
        ss << port;
        ss << ",producer-fifo=";
        // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
        ss << (producerFifo ? 1U : 0U);
        ss << ",num-clients=";
        ss << numBlockingConnections;
        ss << ",sock-prefix=";
        ss << sockPrefix;
        ChannelParamStr result{ss.str().c_str()};
        return result;
    }

    ChannelSocketParams() = default;
    explicit ChannelSocketParams(const char* params)
    {
        dw::core::FixedString<MAX_CHANNEL_ALL_PARAMS_SIZE> channelParams{params};
        ParseAllChannelParameters(channelParams,
                                  // coverity[autosar_cpp14_a5_1_1_violation]
                                  "ip", m_serverIP,
                                  // coverity[autosar_cpp14_a5_1_1_violation]
                                  "producer-fifo", m_producerFifo,
                                  // coverity[autosar_cpp14_a5_1_1_violation]
                                  "id", m_port,
                                  // coverity[autosar_cpp14_a5_1_1_violation]
                                  "connect-timeout", m_connectTimeout,
                                  // coverity[autosar_cpp14_a5_1_1_violation]
                                  "sock-prefix", m_sockPrefix);
    }

    ChannelSocketParams(const ChannelSocketParams& other) = default;

    ChannelSocketParams& operator=(const ChannelSocketParams& other) = default;

    ChannelParamStr getServerIP() const { return m_serverIP; }
    uint16_t getPort() const { return m_port; }
    bool hasProducerFifo() const { return m_producerFifo; }
    dwTime_t getConnectTimeout() const { return m_connectTimeout; }
    dwshared::socketipc::SockPrefixStr getSockPrefix() const { return m_sockPrefix; }

private:
    ChannelParamStr m_serverIP; // needed for socket client connection
    uint16_t m_port{0U};
    bool m_producerFifo{false}; // Allow the socket producer to have its own fifo to queue up work
    dwTime_t m_connectTimeout{DW_TIME_INVALID};
    dwshared::socketipc::SockPrefixStr m_sockPrefix{};
};

///////////////////////////////////////////////////////////////////////////////////////

class ChannelNvSciStreamParams
{
public:
    ChannelNvSciStreamParams() = default;

    explicit ChannelNvSciStreamParams(const char* params)
        : ChannelNvSciStreamParams()
    {
        dw::core::FixedString<MAX_CHANNEL_ALL_PARAMS_SIZE> channelParams{params};
        ParseAllChannelParameters(channelParams,
                                  // coverity[autosar_cpp14_a5_1_1_violation]
                                  "streamName", m_streamNames,
                                  // coverity[autosar_cpp14_a5_1_1_violation]
                                  "limits", m_limits,
                                  // coverity[autosar_cpp14_a5_1_1_violation]
                                  "num-clients", m_localClientCount,
                                  // coverity[autosar_cpp14_a5_1_1_violation]
                                  "late-locs", m_lateLocs);
    }

    ChannelNvSciStreamParams(const ChannelNvSciStreamParams& other) = default;

    ChannelNvSciStreamParams& operator=(const ChannelNvSciStreamParams& other) = default;

    size_t getNumOutputs() const
    {
        static_assert(decltype(m_streamNames)::CAPACITY_AT_COMPILE_TIME < std::numeric_limits<size_t>::max(), "ChannelNvSciStreamParams: number of outputs over limit");
        return m_streamNames.size();
    }

    size_t getLocalClientCount() const { return m_localClientCount; }

    dw::core::span<ChannelPeerLocation const> getLateLocs() const
    {
        return dw::core::make_span<ChannelPeerLocation const>(m_lateLocs.data(), m_lateLocs.size());
    }

    // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
    ChannelStreamNameStr getStreamName(size_t index = 0U) const
    {
        if (index >= m_streamNames.size())
        {
            throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "ChannelNvSciStreamParams: stream name index out of range");
        }
        return m_streamNames[index];
    }

    // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
    int64_t getLimiterMaxPackets(size_t index = 0U) const
    {
        // Note: if no limits are denoted, return -1 and make inquirers not create limiter blocks.
        //       If not, it can lead to out of range when querying
        if (m_limits.empty())
        {
            // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
            return -1;
        }

        if (index >= m_limits.size())
        {
            throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "ChannelNvSciStreamParams: limiter maxPackets index out of range");
        }
        return m_limits[index];
    }

protected:
    dw::core::VectorFixed<ChannelStreamNameStr, 8> m_streamNames{};
    dw::core::VectorFixed<int64_t, 8> m_limits{};
    dw::core::VectorFixed<ChannelPeerLocation, 32> m_lateLocs{};
    size_t m_localClientCount{};
};

class ChannelFSIParams
{
public:
    ChannelFSIParams() = default;
    explicit ChannelFSIParams(const char* params)
    {
        ChannelParamStr channelParams{params};
        ParseAllChannelParameters(channelParams,
                                  // coverity[autosar_cpp14_a5_1_1_violation]
                                  "compat-vendor", m_compatVendor,
                                  // coverity[autosar_cpp14_a5_1_1_violation]
                                  "compat-app", m_compatApp,
                                  // coverity[autosar_cpp14_a5_1_1_violation]
                                  "num-channel", m_numChannel);
        m_compat = m_compatVendor;
        // coverity[autosar_cpp14_a5_1_1_violation]
        m_compat += ",";
        m_compat += m_compatApp;
    }

    ChannelFSIParams(const ChannelFSIParams& other) = default;

    ChannelFSIParams& operator=(const ChannelFSIParams& other) = default;

    const char* getCompat() const { return m_compat.c_str(); }
    uint8_t getNumChannel() const { return m_numChannel; }

private:
    ChannelParamStr m_compat;
    ChannelParamStr m_compatVendor;
    ChannelParamStr m_compatApp;
    uint8_t m_numChannel{0U};
};

/**
 * Class to hold the parsed Channel parameters.
 */
class ChannelParams
{
public:
    explicit ChannelParams(const char* params)
    {
        m_str = params;
        ParseAllChannelParameters(m_str,
                                  // coverity[autosar_cpp14_a5_1_1_violation]
                                  "fifo-size", m_fifoSize,
                                  // coverity[autosar_cpp14_a5_1_1_violation]
                                  "id", m_id,
                                  // coverity[autosar_cpp14_a5_1_1_violation]
                                  "uid", m_uid,
                                  // coverity[autosar_cpp14_a5_1_1_violation]
                                  "connect-group-id", m_connectGroupID,
                                  // coverity[autosar_cpp14_a5_1_1_violation]
                                  "singleton-id", m_singletonId,
                                  // coverity[autosar_cpp14_a5_1_1_violation]
                                  "mode", m_mode,
                                  // coverity[autosar_cpp14_a5_1_1_violation]
                                  "reuse", m_reuseEnabled,
                                  // coverity[autosar_cpp14_a5_1_1_violation]
                                  "debug-port", m_debugPort,
                                  // coverity[autosar_cpp14_a5_1_1_violation]
                                  "num-clients", m_clientsCount,
                                  // coverity[autosar_cpp14_a5_1_1_violation]
                                  "debug-num-clients", m_debugClientsCount,
                                  // coverity[autosar_cpp14_a5_1_1_violation]
                                  "role", m_role,
                                  // coverity[autosar_cpp14_a5_1_1_violation]
                                  "type", m_type,
                                  // coverity[autosar_cpp14_a5_1_1_violation]
                                  "data-offset", m_dataOffset,
                                  // coverity[autosar_cpp14_a5_1_1_violation]
                                  "strict", m_strictFifo,
                                  // coverity[autosar_cpp14_a5_1_1_violation]
                                  "sync-enabled", m_syncEnabled,
                                  // coverity[autosar_cpp14_a5_1_1_violation]
                                  "name", m_name);

        adjustPoolCapacity();

        // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
        if (m_clientsCount == 0U)
        {
            // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
            m_clientsCount = 1U;
        }

        if (!m_singletonId.empty())
        {
            m_mode = ChannelMode::SINGLETON;
        }

        if (m_mode == ChannelMode::MAILBOX)
        {
            m_mailboxMode = true;
        }
        else if (m_mode == ChannelMode::SINGLETON)
        {
            m_singletonMode = true;

            // Assign singletonId with id
            if (!m_id.empty() && m_singletonId.empty())
            {
                m_singletonId = m_id;
            }
        }

        // Why break this up like this? Code complexity.
        // Without this, we fail the code complexity analysis.
        ValidateMailbox();
        ValidateSingleton();

        switch (m_type)
        {
        case ChannelType::SHMEM_LOCAL:
            break;
        case ChannelType::SOCKET:
            m_socketParams = ChannelSocketParams(params);
            break;
        case ChannelType::NVSCI:
            m_nvSciStreamParams = ChannelNvSciStreamParams(params);
            break;
        case ChannelType::FSI:
            m_fsiParams = ChannelFSIParams(params);
            break;
        case ChannelType::SHMEM_REMOTE:
        case ChannelType::EGLSTREAM:
        case ChannelType::DDS:
        default:
            throw ExceptionWithStatus(DW_NOT_IMPLEMENTED, "ChannelParams: no parameters for channel type");
        }
    }

    ChannelParams(const ChannelParams& other)
    {
        *this = other;
    }

    ChannelParams& operator=(const ChannelParams& other) = default;
    ~ChannelParams()                                     = default;

    const char* getStr() const { return m_str.c_str(); }
    ChannelParamStr getId() const { return m_id; }
    ChannelParamStr getSingletonId() const { return m_singletonId; }
    uint16_t getDebugPort() const { return m_debugPort; }
    size_t getFifoSize() const { return m_fifoSize; }
    // Note(chale): Normally the fifo length governs whether a consumer
    // will receive packet. The actual packet pool's size may be larger
    // than the fifo length. non-strict mode allows consumers to receive
    // up to the entire pool size instead of just their fifo length.
    bool isStrictFifo() const { return m_strictFifo; }
    void setStrictFifo(bool strictFifo)
    {
        m_strictFifo = strictFifo;
    }
    size_t getPoolCapacity() const { return m_poolCapacity; }
    bool getMailboxMode() const { return m_mailboxMode; }
    // Note (ajayawardane): The data-offset parameter is used to describe
    // the consumption offset between the producer and the consumer
    // in the sync packet use-case. For example, if a packet is produced with the
    // sync count x, and is inteneded to be consumed when the sync count is x + 1,
    // the data-offset would be 1. This parameter is also used to identify whether
    // to use sync packets to transfer data, so needs to be included in both the
    // producer and the consumer params.
    uint32_t getDataOffset() const { return m_dataOffset; }
    bool getSyncEnabled() const { return m_syncEnabled; }
    void setMailboxMode(bool mailboxEnabled) { m_mailboxMode = mailboxEnabled; }
    bool getSingletonMode() const { return m_singletonMode; }
    bool getReuseEnabled() const { return m_reuseEnabled; }
    // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
    bool getDebugMode() const { return m_debugPort > 0U; }
    uint16_t getExpectedConnectionsCount() const { return m_clientsCount; }
    uint16_t getExpectedDebugConnectionsCount() const { return m_debugClientsCount; }
    ChannelRole getRole() const { return m_role; }
    ChannelType getType() const { return m_type; }
    uint32_t getUID() const { return m_uid; }
    uint32_t getConnectGroupID() const { return m_connectGroupID; }

    // coverity[autosar_cpp14_a2_10_5_violation]
    ChannelParamStr getName() const { return m_name; }

    const ChannelSocketParams& getSocketParams() const
    {
        if (m_type != ChannelType::SOCKET)
        {
            throw ExceptionWithStatus(DW_CALL_NOT_ALLOWED, "ChannelParams: getSocketParams: channel is not of type SOCKET");
        }
        return m_socketParams;
    }

    const ChannelNvSciStreamParams& getNvSciStreamParams() const
    {
        if (m_type != ChannelType::NVSCI)
        {
            throw ExceptionWithStatus(DW_CALL_NOT_ALLOWED, "ChannelParams: getNvSciStreamParams: channel is not of type NVSCI");
        }
        return m_nvSciStreamParams;
    }

    const ChannelFSIParams& getFSIParams() const
    {
        if (m_type != ChannelType::FSI)
        {
            throw ExceptionWithStatus(DW_CALL_NOT_ALLOWED, "ChannelParams: getFSIParams: channel is not of type FSI");
        }
        return m_fsiParams;
    }

private:
    void ValidateMailbox()
    {
        if (m_singletonMode)
        {
            // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
            if (m_fifoSize != 1U)
            {
                throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "ChannelParams: Singleton and mailbox modes are incompatible with a fifo setting other than 1");
            }
        }
        if (!m_mailboxMode && m_reuseEnabled)
        {
            throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "ChannelParams: reuse=true specified when mode!=mailbox. Not valid");
        }
        if (m_mailboxMode && m_singletonMode)
        {
            throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "ChannelParams: Singleton mode is incompatible mailbox mode");
        }
    }

    void ValidateSingleton()
    {
        // Assign singletonId with id in singleton mode
        if (m_singletonMode && m_singletonId.empty())
        {
            m_singletonId = m_id;
        }
        if (!m_singletonMode && !m_singletonId.empty())
        {
            throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "ChannelParams: Singleton mode requires both the mode set AND singletonId set");
        }
        if (m_singletonMode && (m_type != ChannelType::SHMEM_LOCAL))
        {
            throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "ChannelParams: Singleton mode is only valid for SHMEM_LOCAL channels");
        }
    }

    void adjustPoolCapacity()
    {
        // This deserves a comment. The LocalShmem pools (in the fifo-case)
        // need the following number of slots, in the worst case:
        //  1 for getImpl to return at any time
        //  1 per consumer for in-flight data
        //  fifo-size to store a full fifo worth of packets
        // So assuming max consumers, we'd need fifo-size + MAX_CHANNEL_CONSUMERS_COUNT + 1
        if (dw::core::safeAdd(m_fifoSize, MAX_CHANNEL_CONSUMERS_COUNT).value() >= m_poolCapacity)
        {
            // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
            m_poolCapacity = m_fifoSize + MAX_CHANNEL_CONSUMERS_COUNT + 1U;
        }
    }

private:
    ChannelParamStr m_str{};
    ChannelParamStr m_id{};
    ChannelParamStr m_singletonId{};
    ChannelParamStr m_name{};

    // fifo size of the socket client that implies the maximum number of packets
    // a client can hold simultaneously
    size_t m_fifoSize{1U};
    // This deserves a comment. The LocalShmem pools (in the fifo-case)
    // need the following number of slots, in the worst case:
    //  1 for getImpl to return at any time
    //  1 per consumer for in-flight data
    //  fifo-size to store a full fifo worth of packets
    // So assuming a fifo-size of 1 and max consumers, we'd need MAX_CHANNEL_CONSUMERS_COUNT + 2
    size_t m_poolCapacity{MAX_CHANNEL_CONSUMERS_COUNT + 2U};
    ChannelMode m_mode{ChannelMode::FIFO};
    uint32_t m_dataOffset{0U};
    uint32_t m_uid{0U};
    uint32_t m_connectGroupID{0U};
    bool m_syncEnabled{false};
    bool m_mailboxMode{false};
    bool m_singletonMode{false};
    bool m_reuseEnabled{false};
    ChannelRole m_role{ChannelRole::DW_CHANNEL_ROLE_COMPOSITE};
    ChannelType m_type{ChannelType::SHMEM_LOCAL};
    bool m_strictFifo{true};

    uint16_t m_clientsCount{1U};      // number of clients for blocking mode for socket
    uint16_t m_debugClientsCount{1U}; // number of debug clients for blocking mode for socket

    uint16_t m_debugPort{0U};

    ChannelSocketParams m_socketParams{};
    ChannelNvSciStreamParams m_nvSciStreamParams{};
    ChannelFSIParams m_fsiParams{};
};

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_CHANNEL_HPP_
