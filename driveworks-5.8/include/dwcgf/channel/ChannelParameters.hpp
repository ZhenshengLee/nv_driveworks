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

#ifndef DW_FRAMEWORK_CHANNEL_PARAMETERS_HPP_
#define DW_FRAMEWORK_CHANNEL_PARAMETERS_HPP_

#include <dwcgf/Types.hpp>
#include <dwcgf/Exception.hpp>
#include <dw/core/base/Types.h>
#include <dw/core/container/BaseString.hpp>
#include <dw/core/container/VectorFixed.hpp>
#include <dw/core/system/NvMediaExt.h>
#include <dwcgf/channel/ChannelPacketTypes.hpp>
#include <sstream>
#include <limits>

namespace dw
{
namespace framework
{

///////////////////////////////////////////////////////////////////////////////////////
using ChannelType = enum ChannelType : uint8_t {
    DW_CHANNEL_TYPE_SHMEM_LOCAL  = 0,
    DW_CHANNEL_TYPE_SHMEM_REMOTE = 1,
    DW_CHANNEL_TYPE_EGLSTREAM    = 2,
    DW_CHANNEL_TYPE_SOCKET       = 3,
    DW_CHANNEL_TYPE_DDS          = 4,
    DW_CHANNEL_TYPE_NVSCI        = 5,
};

static inline const char* ToParam(ChannelType channelType)
{
    const char* result;
    switch (channelType)
    {
    case DW_CHANNEL_TYPE_SHMEM_LOCAL:
        result = "type=SHMEM_LOCAL";
        break;
    case DW_CHANNEL_TYPE_SHMEM_REMOTE:
        result = "type=SHMEM_REMOTE";
        break;
    case DW_CHANNEL_TYPE_EGLSTREAM:
        result = "type=EGLSTREAM";
        break;
    case DW_CHANNEL_TYPE_SOCKET:
        result = "type=SOCKET";
        break;
    case DW_CHANNEL_TYPE_DDS:
        result = "type=DDS";
        break;
    case DW_CHANNEL_TYPE_NVSCI:
        result = "type=NVSCI";
        break;
    default:
        result = "";
        break;
    }
    return result;
}

enum class ChannelRole : uint8_t
{
    DW_CHANNEL_ROLE_UNKNOWN   = 0b00,
    DW_CHANNEL_ROLE_PRODUCER  = 0b01,
    DW_CHANNEL_ROLE_CONSUMER  = 0b10,
    DW_CHANNEL_ROLE_COMPOSITE = 0b11, // Contains both producer and consumer
};

inline constexpr bool IsProducer(ChannelRole role)
{
    return static_cast<uint8_t>(role) & static_cast<uint8_t>(ChannelRole::DW_CHANNEL_ROLE_PRODUCER);
}

inline constexpr bool IsConsumer(ChannelRole role)
{
    return static_cast<uint8_t>(role) & static_cast<uint8_t>(ChannelRole::DW_CHANNEL_ROLE_CONSUMER);
}

static constexpr uint16_t MAX_CHANNEL_PARAM_SIZE      = 256;
static constexpr uint16_t MAX_CHANNEL_ALL_PARAMS_SIZE = 256;
static constexpr uint16_t MAX_CHANNEL_PRODUCERS_COUNT = 1024;
static constexpr uint16_t MAX_CHANNEL_CONSUMERS_COUNT = 256;

using ChannelParamStr = dw::core::FixedString<MAX_CHANNEL_PARAM_SIZE>;

enum class ChannelMode
{
    FIFO,
    MAILBOX,
    SINGLETON
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
enum class ChannelConnectionType
{
    CONNECTION_TYPE_NONE = 0,
    CONNECTION_TYPE_STATIC,
    CONNECTION_TYPE_DYNAMIC
};

// the farthest reach this channel can achieve
// shorter reach is faster, so if topology is known upfront,
// use the shortest matching reach for maximum performance
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
static inline T ParseChannelParameter(const ChannelParamStr& value);

template <>
ChannelNvSciStreamEnabledComponents ParseChannelParameter(const ChannelParamStr& value)
{
    std::size_t start = 0;
    std::size_t end;
    ChannelNvSciStreamEnabledComponents result{};
    do
    {
        end = value.find("|", start);

        ChannelParamStr valueSubString = (end != dw::core::FixedString<1>::NPOS) ? value.substr(start, end - start) : value.substr(start, value.length() - start);

        if (valueSubString == "CPU")
        {
            result = result | ChannelNvSciStreamEnabledComponents::COMPONENT_CPU;
        }
        else if (valueSubString == "EGL")
        {
            result = result | ChannelNvSciStreamEnabledComponents::COMPONENT_EGL;
        }
        else if (valueSubString == "CUDA")
        {
            result = result | ChannelNvSciStreamEnabledComponents::COMPONENT_CUDA;
        }
        else if (valueSubString == "PVA")
        {
            result = result | ChannelNvSciStreamEnabledComponents::COMPONENT_PVA;
        }
        else if (valueSubString == "DLA")
        {
            result = result | ChannelNvSciStreamEnabledComponents::COMPONENT_DLA;
        }
        else if (valueSubString == "NVMEDIA")
        {
            result = result | ChannelNvSciStreamEnabledComponents::COMPONENT_NVMEDIA;
        }
        else
        {
            throw Exception(DW_INVALID_ARGUMENT, "ChannelNvSciStreamParams: enabledComponents sub-value unrecongnized: ", valueSubString);
        }

        start = end + 1;
    } while (end != dw::core::FixedString<1>::NPOS);

    return result;
}

template <>
ChannelConnectionType ParseChannelParameter(const ChannelParamStr& value)
{
    ChannelConnectionType result;
    if (value == "dynamic")
    {
        result = ChannelConnectionType::CONNECTION_TYPE_DYNAMIC;
    }
    else if (value == "static")
    {
        result = ChannelConnectionType::CONNECTION_TYPE_STATIC;
    }
    else
    {
        throw Exception(DW_INVALID_ARGUMENT, "ChannelNvSciStreamParams: connection phase value unrecongnized: ", value);
    }
    return result;
}

template <>
ChannelReach ParseChannelParameter(const ChannelParamStr& value)
{
    ChannelReach result;
    if (value == "thread")
    {
        result = ChannelReach::REACH_THREAD;
    }
    else if (value == "process")
    {
        result = ChannelReach::REACH_PROCESS;
    }
    else if (value == "vm")
    {
        result = ChannelReach::REACH_VM;
    }
    else if (value == "chip")
    {
        result = ChannelReach::REACH_CHIP;
    }
    else
    {
        throw Exception(DW_INVALID_ARGUMENT, "ChannelNvSciStreamParams: reach value unrecongnized: ", value);
    }

    return result;
}

template <>
int64_t ParseChannelParameter(const ChannelParamStr& value)
{
    auto translatedSize = strtol(value.c_str(), nullptr, 10);
    return translatedSize;
}

template <>
size_t ParseChannelParameter(const ChannelParamStr& value)
{
    auto translatedSize = ParseChannelParameter<int64_t>(value);
    if (translatedSize < 0)
    {
        throw Exception(DW_INVALID_ARGUMENT, "ParseChannelParameter: size_t is negative");
    }
    size_t result = static_cast<size_t>(translatedSize);
    return result;
}

template <>
uint32_t ParseChannelParameter(const ChannelParamStr& value)
{
    auto translatedSize = ParseChannelParameter<size_t>(value);
    uint32_t result     = static_cast<uint32_t>(translatedSize);
    return result;
}

template <>
uint16_t ParseChannelParameter(const ChannelParamStr& value)
{
    auto translatedSize = ParseChannelParameter<size_t>(value);
    if (translatedSize > 0xFFFF)
    {
        throw Exception(DW_INVALID_ARGUMENT, "ChannelSocketParams: port is larger than uint16_t allows!");
    }
    uint16_t result = static_cast<uint16_t>(translatedSize);
    return result;
}

template <>
bool ParseChannelParameter(const ChannelParamStr& value)
{
    bool result;
    if ((value == "true") || (value == "1"))
    {
        result = true;
    }
    else if ((value == "false") || (value == "0"))
    {
        result = false;
    }
    else
    {
        throw Exception(DW_INVALID_ARGUMENT, "ParseChannelParameter: needs to be 'true' or 'false' or 1/0");
    }
    return result;
}

template <>
ChannelRole ParseChannelParameter(const ChannelParamStr& value)
{
    ChannelRole result{};
    if (value == "producer")
    {
        result = ChannelRole::DW_CHANNEL_ROLE_PRODUCER;
    }
    else if (value == "consumer")
    {
        result = ChannelRole::DW_CHANNEL_ROLE_CONSUMER;
    }
    else if (value == "composite")
    {
        result = ChannelRole::DW_CHANNEL_ROLE_COMPOSITE;
    }
    else
    {
        throw Exception(DW_INVALID_ARGUMENT, "ParseChannelParameter: role unknown!");
    }
    return result;
}

template <>
ChannelType ParseChannelParameter(const ChannelParamStr& value)
{
    ChannelType result{};
    if (value == "SHMEM_LOCAL")
    {
        result = ChannelType::DW_CHANNEL_TYPE_SHMEM_LOCAL;
    }
    else if (value == "SHMEM_REMOTE")
    {
        result = ChannelType::DW_CHANNEL_TYPE_SHMEM_REMOTE;
    }
    else if (value == "EGLSTREAM")
    {
        result = ChannelType::DW_CHANNEL_TYPE_EGLSTREAM;
    }
    else if (value == "SOCKET")
    {
        result = ChannelType::DW_CHANNEL_TYPE_SOCKET;
    }
    else if (value == "DDS")
    {
        result = ChannelType::DW_CHANNEL_TYPE_DDS;
    }
    else if (value == "NVSCI")
    {
        result = ChannelType::DW_CHANNEL_TYPE_NVSCI;
    }
    else
    {
        throw Exception(DW_INVALID_ARGUMENT, "ParseChannelParameter: type unknown!");
    }

    return result;
}

template <>
ChannelMode ParseChannelParameter(const ChannelParamStr& value)
{
    ChannelMode result;
    if (value == "mailbox")
    {
        result = ChannelMode::MAILBOX;
    }
    else if (value == "singleton")
    {
        result = ChannelMode::SINGLETON;
    }
    else
    {
        throw Exception(DW_INVALID_ARGUMENT, "ParseChannelParameter: ChannelMode unknown!");
    }
    return result;
}

template <typename T>
void ParseChannelParameter(const ChannelParamStr& value, T& result)
{
    result = ParseChannelParameter<T>(value);
}

template <size_t Size>
void ParseChannelParameter(const ChannelParamStr& value, dw::core::FixedString<Size>& result)
{
    result = value;
}

template <typename T, size_t N>
void ParseChannelParameter(const ChannelParamStr& value, dw::core::VectorFixed<T, N>& result)
{
    size_t pos    = 0U;
    size_t endpos = 0U;
    bool done     = false;
    while (!done)
    {
        endpos       = value.find(":", pos);
        done         = endpos == dw::core::FixedString<1>::NPOS;
        size_t count = done ? endpos : endpos - pos;
        T entry{};
        ParseChannelParameter(value.substr(pos, count), entry);
        result.push_back(entry);
        pos = endpos + 1;
    }
}

static inline void ParseChannelParameters(const ChannelParamStr&, const ChannelParamStr&)
{
    return;
}

template <typename T, typename... Others>
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
static inline void ParseAllChannelParameters(const ChannelParamStr& channelParams, Others&&... others)
{
    std::size_t key      = 0;
    std::size_t pos      = channelParams.find("=");
    std::size_t value    = 0;
    std::size_t valueEnd = 0;

    ChannelParamStr keyString;
    ChannelParamStr valueString;
    while (pos != dw::core::FixedString<1>::NPOS && value != dw::core::FixedString<1>::NPOS)
    {
        keyString   = channelParams.substr(key, pos - key);
        value       = channelParams.find(",", pos);
        valueEnd    = (value == dw::core::FixedString<1>::NPOS) ? (channelParams.length() - (pos + 1)) : (value - pos - 1);
        valueString = channelParams.substr(pos + 1, valueEnd);
        ParseChannelParameters(keyString, valueString, std::forward<Others>(others)...);

        key = value + 1;
        pos = channelParams.find("=", key);
    }
}

class ChannelSocketParams
{
public:
    static inline ChannelParamStr getParamStr(const char* serverIP,
                                              uint16_t port,
                                              bool producerFifo                         = false,
                                              uint16_t numBlockingConnections           = 1,
                                              dw::core::FixedString<8> const sockPrefix = dw::core::FixedString<8>())
    {
        std::stringstream ss;
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
        ss << static_cast<uint32_t>(producerFifo);
        ss << ",num-clients=";
        ss << numBlockingConnections;
        ss << ",sock-prefix=";
        ss << sockPrefix;
        ChannelParamStr result(ss.str().c_str());
        return result;
    }

    ChannelSocketParams() = default;
    explicit ChannelSocketParams(const char* params)
    {
        dw::core::FixedString<MAX_CHANNEL_ALL_PARAMS_SIZE> channelParams(params);
        ParseAllChannelParameters(channelParams,
                                  "ip", m_serverIP,
                                  "producer-fifo", m_producerFifo,
                                  "id", m_port,
                                  "sock-prefix", m_sockPrefix);
    }

    ChannelSocketParams(const ChannelSocketParams& other) = default;

    ChannelSocketParams& operator=(const ChannelSocketParams& other) = default;

    ChannelParamStr getServerIP() const { return m_serverIP; }
    uint16_t getPort() const { return m_port; }
    bool hasProducerFifo() const { return m_producerFifo; }
    dw::core::FixedString<8> getSockPrefix() const { return m_sockPrefix; }

private:
    ChannelParamStr m_serverIP; // needed for socket client connection
    uint16_t m_port                       = 0;
    bool m_producerFifo                   = false; // Allow the socket producer to have its own fifo to queue up work
    dw::core::FixedString<8> m_sockPrefix = dw::core::FixedString<8>();
};

///////////////////////////////////////////////////////////////////////////////////////

class ChannelNvSciStreamParams
{
public:
    ChannelNvSciStreamParams() = default;

    explicit ChannelNvSciStreamParams(const char* params)
        : ChannelNvSciStreamParams()
    {
        dw::core::FixedString<MAX_CHANNEL_ALL_PARAMS_SIZE> channelParams(params);
        ParseAllChannelParameters(channelParams,
                                  "streamName", m_streamNames,
                                  "enabledComponents", m_enabledComponents,
                                  "connectionType", m_connectionType,
                                  "timeoutUsec", m_timeoutUsec,
                                  "reach", m_reaches);
    }

    ChannelNvSciStreamParams(const ChannelNvSciStreamParams& other) = default;

    ChannelNvSciStreamParams& operator=(const ChannelNvSciStreamParams& other) = default;

    dwTime_t getTimeoutUsec() const { return m_timeoutUsec; }

    ChannelNvSciStreamEnabledComponents getEnabledComponents() const { return m_enabledComponents; }
    bool isEnabledComponentCpu() const { return ((m_enabledComponents & ChannelNvSciStreamEnabledComponents::COMPONENT_CPU) == ChannelNvSciStreamEnabledComponents::COMPONENT_CPU); }
    bool isEnabledComponentEgl() const { return ((m_enabledComponents & ChannelNvSciStreamEnabledComponents::COMPONENT_EGL) == ChannelNvSciStreamEnabledComponents::COMPONENT_EGL); }
    bool isEnabledComponentCuda() const { return ((m_enabledComponents & ChannelNvSciStreamEnabledComponents::COMPONENT_CUDA) == ChannelNvSciStreamEnabledComponents::COMPONENT_CUDA); }
    bool isEnabledComponentPva() const { return ((m_enabledComponents & ChannelNvSciStreamEnabledComponents::COMPONENT_PVA) == ChannelNvSciStreamEnabledComponents::COMPONENT_PVA); }
    bool isEnabledComponentDla() const { return ((m_enabledComponents & ChannelNvSciStreamEnabledComponents::COMPONENT_DLA) == ChannelNvSciStreamEnabledComponents::COMPONENT_DLA); }
    bool isEnabledComponentNvmedia() const { return ((m_enabledComponents & ChannelNvSciStreamEnabledComponents::COMPONENT_NVMEDIA) == ChannelNvSciStreamEnabledComponents::COMPONENT_NVMEDIA); }

    ChannelConnectionType getChannelConnectionType() const { return m_connectionType; }
    bool isConnectionTypeStatic() const { return (m_connectionType == ChannelConnectionType::CONNECTION_TYPE_STATIC); }
    bool isConnectionTypeDynamic() const { return (m_connectionType == ChannelConnectionType::CONNECTION_TYPE_DYNAMIC); }
    uint16_t getNumOutputs() const
    {
        static_assert(decltype(m_streamNames)::CAPACITY_AT_COMPILE_TIME < std::numeric_limits<uint16_t>::max(), "ChannelNvSciStreamParams: number of outputs over limit");
        return static_cast<uint16_t>(m_streamNames.size());
    }

    bool isMulticast() const
    {
        return m_streamNames.size() > 1U;
    }

    const char* getStreamName(uint16_t index = 0) const
    {
        if (index >= m_streamNames.size())
        {
            throw Exception(DW_INVALID_ARGUMENT, "ChannelNvSciStreamParams: stream name index out of range");
        }
        return m_streamNames[index].c_str();
    }

    ChannelReach getChannelReach(uint16_t index = 0) const
    {
        if (m_reaches.size() == 0)
        {
            return ChannelReach::REACH_PROCESS;
        }
        if (index >= m_reaches.size())
        {
            throw Exception(DW_INVALID_ARGUMENT, "ChannelNvSciStreamParams: reach index out of range");
        }
        return m_reaches[index];
    }

protected:
    dwTime_t m_timeoutUsec = 5000 * 1000;
    // bitmask of ChannelNvSciStreamEnabledComponents enum
    ChannelNvSciStreamEnabledComponents m_enabledComponents = ChannelNvSciStreamEnabledComponents::COMPONENT_CPU;
    ChannelConnectionType m_connectionType                  = ChannelConnectionType::CONNECTION_TYPE_DYNAMIC;
    dw::core::VectorFixed<dw::core::FixedString<64>, 8> m_streamNames{};
    dw::core::VectorFixed<ChannelReach, 8> m_reaches{};
};

class ChannelParams
{
public:
    explicit ChannelParams(const char* params)
        : m_str(params)
    {
        ParseAllChannelParameters(m_str,
                                  "fifo-size", m_fifoSize,
                                  "id", m_id,
                                  "singleton-id", m_singletonId,
                                  "trace-name", m_traceId, // TODO(chale): should be trace-id but assumption that id field is
                                                           // is only parameter with 'id' as a substring is hardcoded in many places
                                  "mode", m_mode,
                                  "reuse", m_reuseEnabled,
                                  "debug-port", m_debugPort,
                                  "num-clients", m_clientsCount,
                                  "debug-num-clients", m_debugClientsCount,
                                  "role", m_role,
                                  "type", m_type,
                                  "data-offset", m_dataOffset,
                                  "strict", m_strictFifo,
                                  "sync-enabled", m_syncEnabled);

        adjustPoolCapacity();

        if (m_clientsCount == 0)
        {
            m_clientsCount = 1;
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
        case DW_CHANNEL_TYPE_SHMEM_LOCAL:
            break;
        case DW_CHANNEL_TYPE_SOCKET:
            m_socketParams = ChannelSocketParams(params);
            break;
        case DW_CHANNEL_TYPE_NVSCI:
            m_nvSciStreamParams = ChannelNvSciStreamParams(params);
            break;
        case DW_CHANNEL_TYPE_SHMEM_REMOTE:
        case DW_CHANNEL_TYPE_EGLSTREAM:
        case DW_CHANNEL_TYPE_DDS:
        default:
            throw Exception(DW_NOT_IMPLEMENTED, "ChannelParams: no parameters for channel type");
            break;
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
    bool getDebugMode() const { return m_debugPort > 0; }
    uint16_t getExpectedConnectionsCount() const { return m_clientsCount; }
    uint16_t getExpectedDebugConnectionsCount() const { return m_debugClientsCount; }
    ChannelRole getRole() const { return m_role; }
    ChannelType getType() const { return m_type; }
    ChannelParamStr getTraceId() const { return m_traceId; }

    const ChannelSocketParams& getSocketParams() const
    {
        if (m_type != ChannelType::DW_CHANNEL_TYPE_SOCKET)
        {
            throw Exception(DW_CALL_NOT_ALLOWED, "ChannelParams: getSocketParams: channel is not of type SOCKET");
        }
        return m_socketParams;
    }

    const ChannelNvSciStreamParams& getNvSciStreamParams() const
    {
        if (m_type != ChannelType::DW_CHANNEL_TYPE_NVSCI)
        {
            throw Exception(DW_CALL_NOT_ALLOWED, "ChannelParams: getNvSciStreamParams: channel is not of type NVSCI");
        }
        return m_nvSciStreamParams;
    }

private:
    void ValidateMailbox()
    {
        if (m_singletonMode)
        {
            if (m_fifoSize != 1)
            {
                throw Exception(DW_INVALID_ARGUMENT, "ChannelParams: Singleton and mailbox modes are incompatible with a fifo setting other than 1");
            }
        }
        if (!m_mailboxMode && m_reuseEnabled)
        {
            throw Exception(DW_INVALID_ARGUMENT, "ChannelParams: reuse=true specified when mode!=mailbox. Not valid");
        }
        if (m_mailboxMode && m_singletonMode)
        {
            throw Exception(DW_INVALID_ARGUMENT, "ChannelParams: Singleton mode is incompatible mailbox mode");
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
            throw Exception(DW_INVALID_ARGUMENT, "ChannelParams: Singleton mode requires both the mode set AND singletonId set");
        }
        if (m_singletonMode && (m_type != ChannelType::DW_CHANNEL_TYPE_SHMEM_LOCAL))
        {
            throw Exception(DW_INVALID_ARGUMENT, "ChannelParams: Singleton mode is only valid for SHMEM_LOCAL channels");
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
        if (m_fifoSize + MAX_CHANNEL_CONSUMERS_COUNT + 1 > m_poolCapacity)
        {
            m_poolCapacity = m_fifoSize + MAX_CHANNEL_CONSUMERS_COUNT + 1;
        }
    }

private:
    ChannelParamStr m_str{};
    ChannelParamStr m_id{};
    ChannelParamStr m_singletonId{};
    ChannelParamStr m_traceId{};

    // fifo size of the socket client that implies the maximum number of packets
    // a client can hold simultaneously
    size_t m_fifoSize = 1;
    // This deserves a comment. The LocalShmem pools (in the fifo-case)
    // need the following number of slots, in the worst case:
    //  1 for getImpl to return at any time
    //  1 per consumer for in-flight data
    //  fifo-size to store a full fifo worth of packets
    // So assuming a fifo-size of 1 and max consumers, we'd need MAX_CHANNEL_CONSUMERS_COUNT + 2
    size_t m_poolCapacity = MAX_CHANNEL_CONSUMERS_COUNT + 2;
    ChannelMode m_mode    = ChannelMode::FIFO;
    uint32_t m_dataOffset = 0;
    bool m_syncEnabled    = false;
    bool m_mailboxMode    = false;
    bool m_singletonMode  = false;
    bool m_reuseEnabled   = false;
    ChannelRole m_role    = ChannelRole::DW_CHANNEL_ROLE_COMPOSITE;
    ChannelType m_type    = ChannelType::DW_CHANNEL_TYPE_SHMEM_LOCAL;
    bool m_strictFifo     = true;

    uint16_t m_clientsCount      = 1; // number of clients for blocking mode for socket
    uint16_t m_debugClientsCount = 1; // number of debug clients for blocking mode for socket

    uint16_t m_debugPort = 0;

    ChannelSocketParams m_socketParams{};
    ChannelNvSciStreamParams m_nvSciStreamParams{};
};

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_CHANNEL_HPP_
