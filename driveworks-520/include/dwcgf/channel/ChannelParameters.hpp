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

#ifndef DW_FRAMEWORK_CHANNEL_PARAMETERS_HPP_
#define DW_FRAMEWORK_CHANNEL_PARAMETERS_HPP_

#include <dwshared/dwfoundation/dw/core/base/ExceptionWithStatus.hpp>
#include <dwshared/dwfoundation/dw/core/container/BaseString.hpp>
#include <dwshared/dwfoundation/dw/core/container/HashContainer.hpp>
#include <dwshared/dwfoundation/dw/core/container/StringView.hpp>
#include <dwshared/dwfoundation/dw/core/container/VectorFixed.hpp>
#include <dwshared/dwfoundation/dw/core/language/cxx23.hpp>
#include <dwshared/dwfoundation/dw/core/safety/SafeStrOps.hpp>
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
    SHMEM_LOCAL,  ///< local shared memory
    SHMEM_REMOTE, ///< remote shared memory
    EGLSTREAM,    ///< EGL stream
    SOCKET,       ///< socket
    DDS,          ///< Data Distribution Service (DDS)
    NVSCI,        ///< NvSciStream
    FSI,          ///< FSI
};

// coverity[autosar_cpp14_a0_1_3_violation]
inline const char* ToParam(ChannelType channelType)
{
    switch (channelType)
    {
    case ChannelType::SHMEM_LOCAL:
    {
        const char* value{"type=SHMEM_LOCAL"};
        return value;
    }
    case ChannelType::SHMEM_REMOTE:
    {
        const char* value{"type=SHMEM_REMOTE"};
        return value;
    }
    case ChannelType::EGLSTREAM:
    {
        const char* value{"type=EGLSTREAM"};
        return value;
    }
    case ChannelType::SOCKET:
    {
        const char* value{"type=SOCKET"};
        return value;
    }
    case ChannelType::DDS:
    {
        const char* value{"type=DDS"};
        return value;
    }
    case ChannelType::NVSCI:
    {
        const char* value{"type=NVSCI"};
        return value;
    }
    case ChannelType::FSI:
    {
        const char* value{"type=FSI"};
        return value;
    }
    // LCOV_EXCL_START all valid enumerators are covered by cases and enforced by the compiler
    default:
        dw::core::unreachable();
        // LCOV_EXCL_STOP
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
    return ChannelRole::DW_CHANNEL_ROLE_PRODUCER == role || ChannelRole::DW_CHANNEL_ROLE_COMPOSITE == role;
}

inline constexpr bool IsConsumer(ChannelRole role)
{
    return ChannelRole::DW_CHANNEL_ROLE_CONSUMER == role || ChannelRole::DW_CHANNEL_ROLE_COMPOSITE == role;
}

// coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2813925
// coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2813925
static constexpr uint16_t MAX_CHANNEL_PARAM_SIZE{1024U};
// coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2813925
// coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2813925
static constexpr uint16_t MAX_CHANNEL_ALL_PARAMS_SIZE{1024U};
// coverity[autosar_cpp14_a0_1_1_violation]
// coverity[autosar_cpp14_m0_1_4_violation]
static constexpr uint16_t MAX_CHANNEL_PRODUCERS_COUNT{2048U};
static constexpr uint16_t MAX_CHANNEL_CONSUMERS_COUNT{256U};
// coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2813925
// coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2813925
static constexpr uint16_t MAX_CHANNEL_STREAM_NAME_SIZE{64U};
// coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2813925
// coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2813925
static constexpr uint16_t MAX_CHANNEL_STREAM_NAMES{8U};

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
// coverity[autosar_cpp14_a2_10_4_violation] FP: nvbugs/4040101
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
static inline T ParseChannelParameter(const ChannelParamStr& value);

template <>
// coverity[autosar_cpp14_a2_10_4_violation] FP: nvbugs/4040101
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
int64_t ParseChannelParameter(const ChannelParamStr& value)
{
    const int32_t BASE_10{10};
    return dw::core::safeStrtol(value.c_str(), nullptr, BASE_10);
}

template <>
// coverity[autosar_cpp14_a2_10_4_violation] FP: nvbugs/4040101
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
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
// coverity[autosar_cpp14_a2_10_4_violation] FP: nvbugs/4040101
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
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
// coverity[autosar_cpp14_a2_10_4_violation] FP: nvbugs/4040101
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
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
// coverity[autosar_cpp14_a2_10_4_violation] FP: nvbugs/4040101
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
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
// coverity[autosar_cpp14_a2_10_4_violation] FP: nvbugs/4040101
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
bool ParseChannelParameter(const ChannelParamStr& value)
{
    // coverity[autosar_cpp14_a3_3_2_violation] RFD Pending: TID-2534
    static const dw::core::StaticHashMap<ChannelParamStr, bool, 4> MAPPING{
        {"true", true},
        {"1", true},
        {"false", false},
        {"0", false},
    };
    if (!MAPPING.contains(value))
    {
        throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "ParseChannelParameter: needs to be 'true' or 'false' or 1/0");
    }
    return MAPPING.at(value);
}

template <>
// coverity[autosar_cpp14_a2_10_4_violation] FP: nvbugs/4040101
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
ChannelRole ParseChannelParameter(const ChannelParamStr& value)
{
    // coverity[autosar_cpp14_a3_3_2_violation] RFD Pending: TID-2534
    static const dw::core::StaticHashMap<ChannelParamStr, ChannelRole, 3> MAPPING{
        // LCOV_EXCL_START no coverage data available for these lines
        {"producer", ChannelRole::DW_CHANNEL_ROLE_PRODUCER},
        {"consumer", ChannelRole::DW_CHANNEL_ROLE_CONSUMER},
        {"composite", ChannelRole::DW_CHANNEL_ROLE_COMPOSITE},
        // LCOV_EXCL_STOP
    };
    if (!MAPPING.contains(value))
    {
        throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "ParseChannelParameter: role unknown!");
    }
    return MAPPING.at(value);
}

template <>
// coverity[autosar_cpp14_a2_10_4_violation] FP: nvbugs/4040101
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
ChannelType ParseChannelParameter(const ChannelParamStr& value)
{
    // coverity[autosar_cpp14_a3_3_2_violation] RFD Pending: TID-2534
    static const dw::core::StaticHashMap<ChannelParamStr, ChannelType, 7> MAPPING{
        // LCOV_EXCL_START no coverage data available for these lines
        {"SHMEM_LOCAL", ChannelType::SHMEM_LOCAL},
        {"SHMEM_REMOTE", ChannelType::SHMEM_REMOTE},
        {"EGLSTREAM", ChannelType::EGLSTREAM},
        {"SOCKET", ChannelType::SOCKET},
        {"DDS", ChannelType::DDS},
        {"NVSCI", ChannelType::NVSCI},
        {"FSI", ChannelType::FSI},
        // LCOV_EXCL_STOP
    };
    if (!MAPPING.contains(value))
    {
        throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "ParseChannelParameter: type unknown!");
    }
    return MAPPING.at(value);
}

template <>
// coverity[autosar_cpp14_a2_10_4_violation] FP: nvbugs/4040101
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
ChannelMode ParseChannelParameter(const ChannelParamStr& value)
{
    // coverity[autosar_cpp14_a3_3_2_violation] RFD Pending: TID-2534
    static const dw::core::StaticHashMap<ChannelParamStr, ChannelMode, 2> MAPPING{
        // LCOV_EXCL_START no coverage data available for these lines
        {"mailbox", ChannelMode::MAILBOX},
        {"singleton", ChannelMode::SINGLETON},
        // LCOV_EXCL_STOP
    };
    if (!MAPPING.contains(value))
    {
        throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "ParseChannelParameter: ChannelMode unknown!");
    }
    return MAPPING.at(value);
}

template <>
// coverity[autosar_cpp14_a2_10_4_violation] FP: nvbugs/4040101
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
ChannelPeerLocation ParseChannelParameter(const ChannelParamStr& value)
{
    const char* PEER_LOCATION_SEPARATOR{"."};
    size_t pos{value.find(PEER_LOCATION_SEPARATOR)};
    ChannelPeerLocation result{};
    // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
    ChannelParamStr first{value.substr(0U, pos)};
    // the parameter string length isn't close to the maximum value of size_t, hence no risk of overflow
    // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
    ChannelParamStr second{value.substr(dw::core::safeAdd(pos, 1U).value())};
    result.socID = ParseChannelParameter<uint32_t>(first);
    result.vmID  = ParseChannelParameter<uint32_t>(second);
    return result;
}

template <typename T>
// coverity[autosar_cpp14_a2_10_4_violation] FP: nvbugs/4040101
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
void ParseChannelParameter(const ChannelParamStr& value, T& result)
{
    result = ParseChannelParameter<T>(value);
}

template <size_t Size>
// coverity[autosar_cpp14_a2_10_4_violation] FP: nvbugs/4040101
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
void ParseChannelParameter(const ChannelParamStr& value, dw::core::FixedString<Size>& result)
{
    result = value;
}

template <typename T, size_t N>
// coverity[autosar_cpp14_a2_10_4_violation] FP: nvbugs/4040101
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
void ParseChannelParameter(const ChannelParamStr& value, dw::core::VectorFixed<T, N>& result)
{
    const char* ELEMENT_SEPARATOR{":"};
    size_t pos{0U};
    size_t endpos{0U};
    while (true)
    {
        endpos = value.find(ELEMENT_SEPARATOR, pos);
        bool done{dw::core::FixedString<1>::NPOS == endpos};
        size_t count{done ? endpos : endpos - pos};
        T entry{};
        ParseChannelParameter(value.substr(pos, count), entry);
        static_cast<void>(result.push_back(entry));
        if (done)
        {
            break;
        }
        // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
        pos = endpos + 1U;
    }
}

template <typename T>
// coverity[autosar_cpp14_a2_10_4_violation] FP: nvbugs/4040101
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
static inline void ParseChannelParameters(const ChannelParamStr& key, const ChannelParamStr& value, dw::core::StringView staticKey, T& result)
{
    if (dw::core::StringView(key.data(), key.size()) == staticKey)
    {
        ParseChannelParameter(value, result);
    }
}

template <
    typename T, typename... Others,
    std::enable_if_t<sizeof...(Others) != 0>* = nullptr>
// coverity[autosar_cpp14_a2_10_4_violation] FP: nvbugs/4040101
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
static inline void ParseChannelParameters(const ChannelParamStr& key, const ChannelParamStr& value, dw::core::StringView staticKey, T& result, Others&&... others)
{
    if (dw::core::StringView(key.data(), key.size()) == staticKey)
    {
        ParseChannelParameter(value, result);
        return;
    }
    ParseChannelParameters(key, value, std::forward<Others>(others)...);
}

template <typename... Others>
// coverity[autosar_cpp14_a2_10_4_violation] FP: nvbugs/4040101
// coverity[autosar_cpp14_a2_10_5_violation] FP: nvbugs/3907242
static inline void ParseAllChannelParameters(const ChannelParamStr& channelParams, Others&&... others)
{
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2738197
    std::size_t key{0U};
    const char* KEY_VALUE_SEPARATOR{"="};
    std::size_t pos{channelParams.find(KEY_VALUE_SEPARATOR)};
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2738197
    std::size_t value{0U};
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2738197
    std::size_t valueEnd{0U};

    const char* PARAMETER_SEPARATOR{","};
    ChannelParamStr keyString{};
    ChannelParamStr valueString{};
    while (dw::core::FixedString<1>::NPOS != pos && dw::core::FixedString<1>::NPOS != value)
    {
        // loop logic ensures that index 'pos' is always greater than the index 'key', hence no underflow possible
        keyString = channelParams.substr(key, dw::core::safeSub(pos, key).value());
        value     = channelParams.find(PARAMETER_SEPARATOR, pos);
        if (dw::core::FixedString<1>::NPOS == value)
        {
            // condition above ensures no overflow possible
            // loop logic ensures that pos + 1 is never greater than the parameter length, hence no underflow possible
            // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
            valueEnd = dw::core::safeSub(channelParams.length(), dw::core::safeAdd(pos, 1U).value()).value();
        }
        else
        {
            // loop logic ensures that index 'value' is always greater than the index 'pos', hence no underflow possible
            // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
            valueEnd = dw::core::safeSub(dw::core::safeSub(value, pos).value(), 1U).value();
        }
        // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
        valueString = channelParams.substr(pos + 1U, valueEnd);
        ParseChannelParameters(keyString, valueString, others...);

        // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
        key = value + 1U;
        pos = channelParams.find(KEY_VALUE_SEPARATOR, key);
    }
}

class ChannelSocketParams
{
public:
    static inline ChannelParamStr getParamStr(
        const char* serverIP,
        uint16_t port,
        bool producerFifo                         = false,
        uint16_t numBlockingConnections           = 1U,
        dw::core::FixedString<8> const sockPrefix = dw::core::FixedString<8>())
    {
        std::stringstream ss{};
        static_cast<void>(ss.flags(std::ios::dec));
        ss << "type=SOCKET";
        if (nullptr != serverIP)
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
                                  dw::core::StringView{"ip"}, m_serverIP,
                                  dw::core::StringView{"producer-fifo"}, m_producerFifo,
                                  dw::core::StringView{"id"}, m_port,
                                  dw::core::StringView{"connect-timeout"}, m_connectTimeout,
                                  dw::core::StringView{"sock-prefix"}, m_sockPrefix);
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
                                  dw::core::StringView{"streamName"}, m_streamNames,
                                  dw::core::StringView{"limits"}, m_limits,
                                  dw::core::StringView{"connectPrios"}, m_connectPrios,
                                  dw::core::StringView{"num-clients"}, m_localClientCount,
                                  dw::core::StringView{"late-locs"}, m_lateLocs);
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

    uint32_t getConnectPrio(size_t index = 0U) const
    {
        if (m_connectPrios.empty())
        {
            // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
            return 0U;
        }

        if (index >= m_connectPrios.size())
        {
            throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "ChannelNvSciStreamParams: connect prio index out of range");
        }
        return m_connectPrios[index];
    }

protected:
    dw::core::VectorFixed<ChannelStreamNameStr, MAX_CHANNEL_STREAM_NAMES> m_streamNames{};
    dw::core::VectorFixed<uint32_t, MAX_CHANNEL_STREAM_NAMES> m_connectPrios{};
    dw::core::VectorFixed<int64_t, MAX_CHANNEL_STREAM_NAMES> m_limits{};
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
                                  dw::core::StringView{"compat-vendor"}, m_compatVendor,
                                  dw::core::StringView{"compat-app"}, m_compatApp,
                                  dw::core::StringView{"num-channel"}, m_numChannel);
        m_compat = m_compatVendor;
        m_compat += dw::core::StringView{","};
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
                                  dw::core::StringView{"fifo-size"}, m_fifoSize,
                                  dw::core::StringView{"id"}, m_id,
                                  dw::core::StringView{"uid"}, m_uid,
                                  dw::core::StringView{"connect-group-id"}, m_connectGroupID,
                                  dw::core::StringView{"singleton-id"}, m_singletonId,
                                  dw::core::StringView{"mode"}, m_mode,
                                  dw::core::StringView{"reuse"}, m_reuseEnabled,
                                  dw::core::StringView{"debug-port"}, m_debugPort,
                                  dw::core::StringView{"num-clients"}, m_clientsCount,
                                  dw::core::StringView{"debug-num-clients"}, m_debugClientsCount,
                                  dw::core::StringView{"role"}, m_role,
                                  dw::core::StringView{"type"}, m_type,
                                  dw::core::StringView{"data-offset"}, m_dataOffset,
                                  dw::core::StringView{"strict"}, m_strictFifo,
                                  dw::core::StringView{"sync-enabled"}, m_syncEnabled,
                                  dw::core::StringView{"name"}, m_name,
                                  dw::core::StringView{"producer-fifo"}, m_producerFifo,
                                  dw::core::StringView{"sync-object-id"}, m_syncObjectId);
        adjustPoolCapacity();

        // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
        if (0U == m_clientsCount)
        {
            // coverity[autosar_cpp14_a5_1_1_violation] RFD Accepted: TID-2056
            m_clientsCount = 1U;
        }

        if (!m_singletonId.empty())
        {
            m_mode = ChannelMode::SINGLETON;
        }

        if (ChannelMode::MAILBOX == m_mode)
        {
            m_mailboxMode = true;
        }
        else if (ChannelMode::SINGLETON == m_mode)
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
        setParameters(params);
    }

    ChannelParams(const ChannelParams& other) = default;

    ChannelParams& operator=(const ChannelParams& other) = default;
    ~ChannelParams()                                     = default;

    const char* getStr() const { return m_str.c_str(); }
    ChannelParamStr getId() const { return m_id; }
    ChannelParamStr getSingletonId() const { return m_singletonId; }
    const ChannelParamStr& getSyncObjectId() const { return m_syncObjectId; }
    uint16_t getDebugPort() const { return m_debugPort; }
    bool hasProducerFifo() const { return m_producerFifo; }
    size_t getFifoSize() const { return m_fifoSize; }
    void setFifoSize(size_t fifoSize) { m_fifoSize = fifoSize; }
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
        if (ChannelType::SOCKET != m_type)
        {
            throw ExceptionWithStatus(DW_CALL_NOT_ALLOWED, "ChannelParams: getSocketParams: channel is not of type SOCKET");
        }
        return m_socketParams;
    }

    const ChannelNvSciStreamParams& getNvSciStreamParams() const
    {
        if (ChannelType::NVSCI != m_type && ChannelType::SHMEM_REMOTE != m_type)
        {
            throw ExceptionWithStatus(DW_CALL_NOT_ALLOWED, "ChannelParams: getNvSciStreamParams: channel is not of type NVSCI or SHMEM_REMOTE");
        }
        return m_nvSciStreamParams;
    }

    const ChannelFSIParams& getFSIParams() const
    {
        if (ChannelType::FSI != m_type)
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
            if (1U != m_fifoSize)
            {
                throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "ChannelParams: Singleton and mailbox modes are incompatible with a fifo setting other than 1");
            }
        }
        if (!m_mailboxMode && m_reuseEnabled)
        {
            throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "ChannelParams: reuse=true specified when mode!=mailbox. Not valid");
        }
    }

    void ValidateSingleton()
    {
        if (m_singletonMode && (ChannelType::SHMEM_LOCAL != m_type))
        {
            throw ExceptionWithStatus(DW_INVALID_ARGUMENT, "ChannelParams: Singleton mode is only valid for SHMEM_LOCAL channels");
        }
    }

    void setParameters(const char* params)
    {
        switch (m_type)
        {
        case ChannelType::SHMEM_LOCAL:
            break;
        case ChannelType::SOCKET:
            m_socketParams = ChannelSocketParams(params);
            break;
        case ChannelType::NVSCI:
        case ChannelType::SHMEM_REMOTE:
            m_nvSciStreamParams = ChannelNvSciStreamParams(params);
            break;
        case ChannelType::FSI:
            m_fsiParams = ChannelFSIParams(params);
            break;
        case ChannelType::EGLSTREAM:
        case ChannelType::DDS:
        default:
            throw ExceptionWithStatus(DW_NOT_IMPLEMENTED, "ChannelParams: no parameters for channel type");
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
        // the fifo size isn't close to the maximum value of size_t, hence no risk of overflow
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
    ChannelParamStr m_syncObjectId{};

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
    bool m_producerFifo{false};
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
