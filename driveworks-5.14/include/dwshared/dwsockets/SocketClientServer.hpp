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
// SPDX-FileCopyrightText: Copyright (c) 2014-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef SOCK_IPC_SOCKETCLIENTSERVER_HPP_
#define SOCK_IPC_SOCKETCLIENTSERVER_HPP_

#include <unistd.h>
#include <poll.h>
#include <sys/socket.h>
#include <memory>
#include <netdb.h>
#include <netinet/in.h>

// C++ interface
#include <dwshared/dwfoundation/dw/core/container/VectorFixed.hpp>
#include <dwshared/dwfoundation/dw/core/container/HashContainer.hpp>
#include <dwshared/dwfoundation/dw/core/logger/Logger.hpp>
#include <dwshared/dwfoundation/dw/core/container/BaseString.hpp>
#include <dwshared/dwfoundation/dw/core/container/StringView.hpp>
#include <dwshared/dwfoundation/dw/core/utility/Constants.hpp>

#include "impl/SocketWrapper.hpp"

#ifdef VIBRANTE_V5Q

//Bug 200684998
static_assert(FD_SETSIZE >= 1000U, "FD_SETSIZE is less than 1000 on QNX");
#endif

namespace dwshared
{
namespace socketipc
{

using sv            = dw::core::StringView;
using SockPrefixStr = dw::core::FixedString<32U>;

/**
 * @brief Get the the system time from epoch in micro seconds
 *
 * @return int64_t
 */
int64_t getCurrentTimeUS() noexcept;

class SocketConstants
{
public:
    static constexpr uint32_t ONE_SECOND_US{1000000U};
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2813925
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2813925
    static constexpr int32_t ONE_MS_US{1000U};
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2813925
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2813925
    static constexpr uint32_t MAX_BIND_ATTEMPTS{10U};
    static constexpr int32_t INVALID_SOCKET_HANDLE{-1};
    static constexpr uint32_t MAX_ERRMSG_LEN{256U};
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2813925
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2813925
    static constexpr uint32_t MAX_HOSTNAME_LEN{256U};
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2813925
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2813925
    static constexpr uint32_t MAX_PORTSTR_LEN{5U};
};

inline void logStreamError(const void* const ptrObject, dw::core::FixedString<SocketConstants::MAX_ERRMSG_LEN> const& logMessage, bool const throwEx = true)
{
    if (ptrObject != nullptr)
    {
        char8_t errMsg[SocketConstants::MAX_ERRMSG_LEN];
#ifndef __QNX__
        LOGSTREAM_ERROR(ptrObject) << logMessage.c_str() << " - " << ::strerror_r(errno, errMsg, SocketConstants::MAX_ERRMSG_LEN) << dw::core::Logger::State::endl;
#else
        int32_t errNum{::strerror_r(errno, errMsg, SocketConstants::MAX_ERRMSG_LEN)};
        LOGSTREAM_ERROR(ptrObject) << logMessage.c_str() << sv{" - "} << errNum << sv{" - "} << errMsg << dw::core::Logger::State::endl;
#endif
    }

    if (throwEx)
    {
        throw dw::core::ExceptionBase(logMessage.c_str());
    }
}

// coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
inline void logStreamErrorWithPort(const void* const ptrObject, dw::core::FixedString<SocketConstants::MAX_ERRMSG_LEN> const& logMessage, uint16_t port, bool const throwEx = true)
{
    if (ptrObject != nullptr)
    {
        char8_t errMsg[SocketConstants::MAX_ERRMSG_LEN];
#ifndef __QNX__
        LOGSTREAM_ERROR(ptrObject) << logMessage.c_str() << " - " << port << " - "
                                   << ::strerror_r(errno, errMsg, SocketConstants::MAX_ERRMSG_LEN) << dw::core::Logger::State::endl;
#else
        int32_t errNum{::strerror_r(errno, errMsg, SocketConstants::MAX_ERRMSG_LEN)};
        LOGSTREAM_ERROR(ptrObject) << logMessage.c_str() << sv{" - "} << errNum << sv{" - "} << port
                                   << sv{" - "} << errMsg << dw::core::Logger::State::endl;
#endif
    }

    if (throwEx)
    {
        throw dw::core::ExceptionBase(logMessage.c_str(), port);
    }
}

/** 
 * Socket Status definitions
 */
enum class Sock_Status : int32_t
{
    /// Socket buffer is full
    BUFFER_FULL,
    /// Cannot create the socket object
    CANNOT_CREATE_OBJECT,
    /// Socket stream has finished
    END_OF_STREAM,
    /// Generic Failure
    FAILURE,
    /// Internal Error
    INTERNAL_ERROR,
    /// Invalid Argument
    INVALID_ARGUMENT,
    /// Socket operation successfully completed
    SUCCESS,
    /// Socket timeout
    TIME_OUT,
};

static constexpr int64_t SOCK_TIMEOUT_INFINITE{0x0123456789ABCDEF};

// Simple class to wrap socket
class UniqueSocketDescriptorHandle
{
public:
    explicit UniqueSocketDescriptorHandle(int32_t const hSocket) noexcept
        : m_hSocket(hSocket)
    {
    }

    // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
    UniqueSocketDescriptorHandle() noexcept
        : UniqueSocketDescriptorHandle(SocketConstants::INVALID_SOCKET_HANDLE)
    {
    }

    ~UniqueSocketDescriptorHandle()
    {
        reset();
    }

    UniqueSocketDescriptorHandle& operator=(UniqueSocketDescriptorHandle const&) = delete; // no copies!
    UniqueSocketDescriptorHandle(UniqueSocketDescriptorHandle const&)            = delete; // no copies!

    // move
    // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
    UniqueSocketDescriptorHandle(UniqueSocketDescriptorHandle&& other) noexcept
        : m_hSocket(other.m_hSocket)
    {
        other.m_hSocket = SocketConstants::INVALID_SOCKET_HANDLE;
    }

    // move assignment
    // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
    UniqueSocketDescriptorHandle& operator=(UniqueSocketDescriptorHandle&& right) noexcept
    {
        if (this != &right)
        { // different, do the move
            int32_t const sHandle{right.m_hSocket};
            right.m_hSocket = SocketConstants::INVALID_SOCKET_HANDLE;
            reset(sHandle);
        }
        return (*this);
    }

    // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
    int32_t release() noexcept
    {
        int32_t const tmp{m_hSocket};
        m_hSocket = SocketConstants::INVALID_SOCKET_HANDLE;
        return tmp;
    }

    int32_t get() const noexcept
    {
        return m_hSocket;
    }

    void reset(int32_t const& hSocket)
    {
// TODO(owner): Closing the old socket here causes a lot of autosar A15-0-2 violations, basically on all exception throw statements.
// When an exception is thrown after a move operation on UniqueSocketDescriptorHandle, basic exception safety could not be guaranteed
// in cases where old socket has been deallocated and the new one has failed initialization.
// As an example, see the constructor function: SocketConnection::SocketConnection(UniqueSocketDescriptorHandle connectionSocket)
#ifndef __COVERITY__
        if (m_hSocket > 0)
        {
            // Further transmissions will be disallowed after calling
            // shutdown(2) system call. This also avoids the possible protocol error
            // (error: connection reset by peer) that might occur if there
            // was any data left out in the last transaction.
            static_cast<void>(::shutdown(m_hSocket, SHUT_RDWR));
            static_cast<void>(::close(m_hSocket));
        }
#endif
        m_hSocket = hSocket;
    }

    void reset() noexcept
    {
        constexpr int32_t NULL_SOCK_ID{SocketConstants::INVALID_SOCKET_HANDLE};
        reset(NULL_SOCK_ID); // invalid value
    }

    // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
    bool isValid() const noexcept
    {
        return m_hSocket > dw::core::util::ZERO;
    }

private:
    int32_t m_hSocket;
};

/// Simple class to manage memory allocated by getaddrinfo
class AddrInfo
{
public:
    AddrInfo() = default;

    /// Constructor to store the passed address
    /// @param addr the address to store
    explicit AddrInfo(addrinfo* const addr)
        : m_addr(addr) {}
    ~AddrInfo();

    AddrInfo(AddrInfo&&) = default;
    AddrInfo& operator=(AddrInfo&&) = default;
    AddrInfo(AddrInfo const&)       = default;
    AddrInfo& operator=(AddrInfo const&) = default;

    /// Return the stored address
    const struct addrinfo* getAddr() const { return m_addr; }

private:
    struct addrinfo* m_addr;
};

/**
 * @brief A structure contains data for POSIX sockaddr_in
 */
struct IpcSockAddrIn
{
    /// Port
    uint16_t port;
    /// IP address
    uint32_t ip;
};

/// Store data of the socket connection of a server or client.
class SocketConnection
{
public:
    /// Create a new socket connection object passing the socket handle.
    /// @param connectionSocket the socket handle.
    explicit SocketConnection(UniqueSocketDescriptorHandle connectionSocket);
    ~SocketConnection();

    SocketConnection(SocketConnection&&) = default;
    SocketConnection& operator=(SocketConnection&&) = default;
    SocketConnection(SocketConnection const&)       = delete;
    SocketConnection& operator=(SocketConnection const&) = delete;

    /// Hold the Message Status of TX/RX performed on this connection.
    class MessageStatus
    {

    public:
        /// Create a new message status.
        /// @param status the socket status.
        /// @param size the size of the TX message.
        MessageStatus(Sock_Status const status, uint64_t const size) noexcept
            : m_status(status)
            , m_transmissionSize(size)
            , m_rxtime{}
        {
        }

        /// Returns the current socket status.
        /// @return the socket status.
        Sock_Status getStatus() const noexcept { return m_status; }
        /// Set a new socket status
        /// @param status the new socket status.
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        void setStatus(Sock_Status const status) noexcept { m_status = status; }
        /// Get the size of the TX on this connection.
        /// @return the TX size
        uint64_t getTransmissionSize() const noexcept { return m_transmissionSize; }
        /// Set the size of the TX on this connection.
        /// @param size the TX size
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        void setTransmissionSize(uint64_t const size) noexcept { m_transmissionSize = size; }
        /// Get packet arrival kernel timestamp in microseconds from system time epoch.
        /// @return the packet arrival timestamp.
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        int64_t getRxTime() const { return m_rxtime; }
        /// Set the packet arrival kernel timestamp in microseconds from system time epoch.
        /// @param time the packet arrival timestamp.
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        void setRxTime(int64_t const& time) { m_rxtime = time; }

    private:
        Sock_Status m_status;
        uint64_t m_transmissionSize;
        int64_t m_rxtime; // Packet arrival kernel timestamp in microsecond
    };

    /// Sends content of the buffer through this socket connection to the default host address/port.
    /// @param buffer the buffer to send.
    /// @param bufferSize the size of the buffer to send.
    /// @param timeoutUs the max transmission time allowed (us) before arising a connection timeout error. timeoutUs = 0 for non-blocking communication.
    /// @return the message status for this TX.
    MessageStatus send(const void* const buffer, size_t const bufferSize, int64_t const timeoutUs = SOCK_TIMEOUT_INFINITE);
    /// Sends content of the buffer through this socket connection to a specific host/port.
    /// @param buffer the buffer to send.
    /// @param bufferSize the size of the buffer to send.
    /// @param host the address of the host.
    /// @param sendPort the port of the host.
    /// @param timeoutUs the max transmission time allowed (us) before arising a connection timeout error. timeoutUs = 0 for non-blocking communication.
    /// @return the message status for this TX.
    MessageStatus sendTo(const void* const buffer, size_t const bufferSize, char8_t const* const host, uint16_t const sendPort, int64_t const timeoutUs = SOCK_TIMEOUT_INFINITE);

    /// Reads data into the buffer and returns number of read bytes if successful.
    /// Using recv(), the data in the socket reception buffer is removed.
    /// @param buffer the buffer to store received data.
    /// @param bufferSize the size of the buffer.
    /// @param timeoutUs the max reception time allowed (us) before arising a connection timeout error. timeoutUs = 0 for non-blocking communication.
    /// @return the message status for this RX.
    MessageStatus recv(void* const buffer, size_t const bufferSize, int64_t const timeoutUs = SOCK_TIMEOUT_INFINITE);
    /// Peeks data into the buffer and returns number of read bytes if successful.
    /// Using peek(), the data in the socket reception buffer is not removed.
    /// @param buffer the buffer to store received data.
    /// @param bufferSize the size of the buffer.
    /// @param timeoutUs the max reception time allowed (us) before arising a connection timeout error.
    /// @return the message status for this RX.
    MessageStatus peek(uint8_t* const buffer, size_t const bufferSize, int64_t const timeoutUs);

    /// Set the socket bound to this connection as non-blocking.
    /// @param sockfd the socket descriptor.
    static void setNonBlocking(int32_t const sockfd);
    /// Set the socket bound to this connection as blocking.
    /// @param sockfd the socket descriptor.
    static void unsetNonBlocking(int32_t const sockfd);

    /// Returns the socket descriptor bound to this connection.
    /// @return the socket descriptor.
    int32_t get_sockfd() const noexcept
    {
        return m_connectionSocket.get();
    }

    /// Get the socket status by address.
    /// @param addr the socket address.
    /// @return the socket status.
    Sock_Status getSockaddr(IpcSockAddrIn* const addr) const;

    /// Set a new TX timeout to the socket bound to this connection.
    /// @param timeoutUs the timeout (uS)
    // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
    inline void setSendTimeout(int64_t const timeoutUs)
    {
        if (timeoutUs == m_sendTimeoutUs)
        {
            return;
        }
        timeval const tv{timeoutToTimeval(timeoutUs)};
        errno = dw::core::util::ZERO;
        if (::setsockopt(m_connectionSocket.get(), SOL_SOCKET, SO_SNDTIMEO, &tv, static_cast<socklen_t>(sizeof(tv))) < dw::core::util::ZERO_U)
        {
            dwshared::socketipc::logStreamError(nullptr, "SocketConnection: can't set send timeout");
        }
        m_sendTimeoutUs = timeoutUs;
    }

    /// Set a new RX timeout to the socket bound to this connection.
    /// @param timeoutUs the timeout (uS)
    // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
    inline void setRecvTimeout(int64_t const timeoutUs)
    {
        if (timeoutUs == m_recvTimeoutUs)
        {
            return;
        }
        timeval const tv{timeoutToTimeval(timeoutUs)};
        errno = dw::core::util::ZERO;
        if (::setsockopt(m_connectionSocket.get(), SOL_SOCKET, SO_RCVTIMEO, &tv, static_cast<socklen_t>(sizeof(tv))) < dw::core::util::ZERO_U)
        {
            dwshared::socketipc::logStreamError(nullptr, "SocketConnection: can't set recv timeout");
        }
        m_recvTimeoutUs = timeoutUs;
    }

    /// Set the UDP Broadcast flag for this connection.
    /// @param udpBroadcast the UDP broadcast flag (true or false)
    // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
    void setUDPBroadcast(bool const udpBroadcast) noexcept
    {
        m_udpBroadcastEnabled = udpBroadcast;
    }

    /// Check if the UDP Broadcast flag for this connection has been set.
    /// @return true or false
    // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
    bool isUDPBroadcast() const noexcept
    {
        return m_udpBroadcastEnabled;
    }

    /// Get the number of TX packet skipped so far due to a connection issue.
    /// @return the number of TX packet skipped.
    // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
    uint64_t getSkipCount() const noexcept { return m_skipCount; }

    /// Increment the number of TX packet skipped so far due to a connection issue.
    // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
    void incSkipCount() noexcept
    {
        if (m_skipCount < std::numeric_limits<uint64_t>::max())
        {
            m_skipCount += dw::core::util::ONE_U;
        }
    }

private:
    dwshared::socketipc::SendRecvReturnStruct peekBlock(uint8_t* const buffer, size_t const bufferSize);
    dwshared::socketipc::SendRecvReturnStruct peekBlockWithTimeout(uint8_t* const buffer, size_t const bufferSize, int64_t const timeoutUs) const;
    dwshared::socketipc::SendRecvReturnStruct peekNonBlock(uint8_t* const buffer, size_t const bufferSize) const;
    dwshared::socketipc::SendRecvReturnStruct processRecvWithTimeout(void* const buffer, size_t const bufferSize, int64_t const timeoutUs);
    static timeval timeoutToTimeval(int64_t const timeoutUs) noexcept
    {
        timeval tv{
            static_cast<decltype(timeval::tv_sec)>(timeoutUs / SocketConstants::ONE_SECOND_US),
            static_cast<decltype(timeval::tv_usec)>(timeoutUs % SocketConstants::ONE_SECOND_US)};
#ifdef VIBRANTE_V5Q
        constexpr int64_t MAX_QNX_SOCKET_TIMEOUT{31};
        // Some of the QNX socket timeouts are restricted to 32 seconds or under (eg., SO_SNDTIMEO)
        if (tv.tv_sec > MAX_QNX_SOCKET_TIMEOUT)
        {
            tv.tv_sec = MAX_QNX_SOCKET_TIMEOUT;
        }
#endif
        // QNX setsockopt has undefined behaviour when passing 0 as send or receive timeout value
        // Linux implementation usually makes it an infinite timeout, which we don't want either.
        if ((tv.tv_sec == dw::core::util::ZERO) && (tv.tv_usec == dw::core::util::ZERO))
        {
            tv.tv_usec = dw::core::util::ONE;
        }

        return tv;
    }

    // Convert from errno to MessageStatus for ::send()
    MessageStatus getSendErrorStatus(int32_t const error);

    // Counter for packet skip
    uint64_t m_skipCount{0UL};

    UniqueSocketDescriptorHandle m_connectionSocket;
    int64_t m_sendTimeoutUs{dw::core::util::ZERO};
    int64_t m_recvTimeoutUs{dw::core::util::ZERO};
    int64_t m_receivedBytes{dw::core::util::ZERO};
    bool m_udpBroadcastEnabled{false};
};

/// Connection pool management to keep multiple connections open at the same time.
class SocketConnectionPool
{
public:
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2813925
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2813925
    static constexpr size_t INVALID_CONNECTION_SLOT{0xBADDBADD};

    constexpr static size_t MAX_PARTIAL_BUFFER_SIZE{0x600000};

    /// Stores remaning data for partial transfers for one socket connection
    struct PartialPacket
    {
        uint8_t m_data[MAX_PARTIAL_BUFFER_SIZE]{}; // buffer to store remaining partial packet from sent()
        size_t m_offset{};                         // starting index of remaining data
        size_t m_size{};                           // size in bytes, of remaining data
        IpcSockAddrIn peerAddr{};                  // IP address of peer connection, for debug purpose
        /// Check if there is valid remaining data
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        bool hasBytes() { return m_size > dw::core::util::ZERO_U; }
        /// Begining pointer of remaining data
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        uint8_t* begin()
        {
            return m_offset <= MAX_PARTIAL_BUFFER_SIZE ? &m_data[m_offset] : m_data;
        }
        /// Size of remaining data in bytes
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        size_t size() { return m_size; }
        /// Function to process remaining data and update offset and size
        template <typename F>
        SocketConnection::MessageStatus update(F&& func)
        {
            // coverity[autosar_cpp14_a8_5_2_violation] FP: nvbugs/3904083
            auto status = func(begin(), size());
            m_offset    = dw::core::safeAdd(m_offset, status.getTransmissionSize()).value();
            m_size      = dw::core::safeSub(m_size, status.getTransmissionSize()).value();
            return status;
        };
    };

    /// Stores remaning data for a group of socket collections, indexed by socket fd
    class PartialPacketMap
    {
    public:
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        explicit PartialPacketMap(size_t mapSize)
            : m_map(mapSize){};
        ~PartialPacketMap() = default;

        /// Check if a socket has valid PartialPacket
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        bool contains(int32_t key)
        {
            std::lock_guard<std::mutex> lock{m_mutex};
            return m_map.contains(key);
        }

        /// Access PartialPacket in the map
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        PartialPacket& operator[](int32_t key)
        {
            std::lock_guard<std::mutex> lock{m_mutex};
            return m_map[key];
        }

        /// Remove one PartialPacket from the map
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        bool erase(int32_t key)
        {
            std::lock_guard<std::mutex> lock{m_mutex};
            return m_map.erase(key);
        }

        /// Thread safe wrapper call to PartialPacket update API
        template <typename F>
        SocketConnection::MessageStatus update(int32_t key, F&& func)
        {
            // coverity[autosar_cpp14_m0_1_3_violation] FP: nvbugs/3785302
            // coverity[autosar_cpp14_m0_1_9_violation] FP: nvbugs/3785302
            std::lock_guard<std::mutex> lock{m_mutex};
            return m_map[key].update(func);
        };

    private:
        std::mutex m_mutex;
        dw::core::HeapHashMap<int32_t, PartialPacket> m_map;
    };

    using PartialPacketMapPtr = std::shared_ptr<PartialPacketMap>;

    /// Create a connection pool with the given size.
    explicit SocketConnectionPool(size_t const poolSize, PartialPacketMapPtr partialPacketMapPtr = nullptr);
    ~SocketConnectionPool();

    SocketConnectionPool(SocketConnectionPool&&) = default;
    SocketConnectionPool& operator=(SocketConnectionPool&&) = default;
    SocketConnectionPool(SocketConnectionPool const&)       = default;
    SocketConnectionPool& operator=(SocketConnectionPool const&) = default;

    /// Acquires a free pool slot, or INVALID_CONNECTION_SLOT if no slot is free.
    /// @return the index of the free slot, or INVALID_CONNECTION_SLOT.
    size_t getFreeSlot() const;
    /// Add a connection to the pool at the given slot.
    /// @param connection the connection to add.
    /// @param index the slot of the pool where place the connection.
    void addToPool(std::weak_ptr<SocketConnection> const connection, size_t const index);

    /// Broadcasts content of the buffer to all valid connections.
    /// @param buffer the buffer to broadcast.
    /// @param bufferSize the size of the buffer.
    /// @param timeoutUs the max TX time allowed (us) before arising a connection timeout error.
    /// @return the socket status.
    Sock_Status broadcast(uint8_t const* const buffer, size_t const bufferSize, int64_t const timeoutUs = SOCK_TIMEOUT_INFINITE);

    /// Counters to record error cases
    struct Metrics
    {
        uint64_t dropCount;                   // Counter for packet drop
        uint64_t partialTransferCount;        // Counter for partial transfer cases
        uint64_t partialTransferFailureCount; // Counter for partial transfer retry failures
        uint64_t pendingPartialTransferCount; // Counter for valid partial packet yet to be sent
    };

    /// Return metrics struct
    /// @return metrics collected by connection pool
    const Metrics& getMetrics() const
    {
        return m_metrics;
    }

private:
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2813925
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2813925
    constexpr static int32_t SOCK_SELECT_AT_BROADCAST_TIMEOUT{200000};
    // Log interval in (2^n)-1 format
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2813925
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2813925
    constexpr static size_t REPEATED_ERR_LOG_INTERVAL_MINUS1{511U};

    using PollfdVector = dw::core::VectorFixed<struct ::pollfd, 8192U>;

    dw::core::VectorFixed<std::weak_ptr<SocketConnection>> m_connectionPool;
    /// pollfd vector to temporarily save all the FDs of connections which are ready to broadcast
    PollfdVector m_pollfdVector;

    /// Metrics counters for error cases
    Metrics m_metrics{};

    bool isBroadcastPossible();
    bool isBroadcastConnectionValid(std::shared_ptr<SocketConnection> con);
    /// Try to send current packet for one socket connection
    SocketConnection::MessageStatus sendPacket(dwshared::socketipc::SocketConnection& connection, uint8_t const* const buffer, size_t const bufferSize, int64_t const timeoutUs, size_t index);
    /// Save remaining data in case of partial transfer
    void savePartialPacket(dwshared::socketipc::SocketConnection& connection, uint8_t const* const buffer, size_t const bufferSize, size_t transmittedSize);

    PartialPacketMapPtr m_partialPacketMap{};
};

/// Class to initialize a socket on the server side.
class SocketServer
{
public:
    /// Constructor : create the server socket and bind it to the specified port.
    /// If socket creation fails it arises a dw::core::ExceptionBase exception.
    /// @param port The port to listen for client connections.
    /// @param connectionPoolSize The number of connections that this server can accept.
    /// @param reusePort If true, allow multiple sockets to bind to the used port.
    /// @param partialPacketMapPtr If valid, handle partial transfers by resending remaining data in next broadcast.
    SocketServer(uint16_t const port, size_t const connectionPoolSize, bool const reusePort = true, SocketConnectionPool::PartialPacketMapPtr partialPacketMapPtr = nullptr);
    SocketServer(uint16_t const port, size_t const connectionPoolSize, bool const reusePort, SockPrefixStr const& sockPrefix, SocketConnectionPool::PartialPacketMapPtr partialPacketMapPtr = nullptr);
    ~SocketServer() = default;

    SocketServer(SocketServer&&) = default;
    SocketServer& operator=(SocketServer&&) = default;
    SocketServer(SocketServer const&)       = delete;
    SocketServer& operator=(SocketServer const&) = delete;

    /// Store an accepted connection with the relative socket status.
    class AcceptStatus
    {
    public:
        /// Create a new AcceptStatus for a given socket status.
        /// @param status The socket status to store.
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        explicit AcceptStatus(Sock_Status const status) noexcept
            : m_status(status)
            , m_connection(nullptr)
        {
        }

        /// Retrieve the stored socket status.
        /// @return The socket status.
        Sock_Status getStatus() const noexcept { return m_status; }

        /// Store a new socket status.
        /// @param status The new socket status.
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        void setStatus(Sock_Status const status) noexcept { m_status = status; }

        /// Bind a connection to this accept status.
        /// @param connection The shared pointer to the connection to bind.
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        void setConnection(std::shared_ptr<SocketConnection> connection) { m_connection = std::move(connection); }

        /// Retrieve the bound connection.
        /// @return The shared pointer to the bound connection.
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        std::shared_ptr<SocketConnection> getConnection() { return m_connection; }
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        void resetConnection() { m_connection.reset(); }

    private:
        Sock_Status m_status;
        std::shared_ptr<SocketConnection> m_connection;
    };

    /// Wait for a client socket to connect, until the timeout elapse.
    /// @param timeoutUs The timeout after which the server will stop accepting connections.
    /// @return The status of the accepted connection.
    AcceptStatus accept(int64_t const timeoutUs);

    /// Get a free connection slot from connection pool
    /// @return The slot number of avaiable connection.
    size_t getConnectionSlot();

    /// Same as accept, but does not add connection to connection pool
    /// @param timeoutUs The timeout after which the server will stop accepting connections.
    /// @return The status of the accepted connection.
    SocketServer::AcceptStatus acceptWithoutRegister(int64_t const timeoutUs);

    /// Add a connection returned by acceptWithoutRegister to connection pool
    /// @param connection The socket connection returned by acceptWithoutRegister.
    /// @param connectionSlot The slot number returned by getConnectionSlot.
    void addToPool(std::weak_ptr<SocketConnection> const connection, size_t const connectionSlot);

    /// Create an UDP socket on the specified port.
    /// @param port The port to use.
    /// @return The socket id of the created socket, or -1 if an error occurred.
    static int32_t createUDPSocket(uint16_t const port);

    /// Create an UDP socket on the specified port.
    /// @param port The port to use.
    /// @return The socket address structure.
    static sockaddr_in getSocketAddress(uint16_t const port);

    /// Broadcasts content of the buffer to all valid connections.
    /// @param buffer The buffer to send.
    /// @param bufferSize The size of the buffer.
    /// @param timeoutUs The time available to send out all the data.
    /// @return The socket status related to this transmission.
    Sock_Status broadcast(uint8_t const* const buffer, size_t const bufferSize, int64_t const timeoutUs = SOCK_TIMEOUT_INFINITE);

    /// Get the metrics from connection pool
    /// @return metrics from connection pool
    // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
    const SocketConnectionPool::Metrics& getConnectionPoolMetrics() const
    {
        return m_connectionPool.getMetrics();
    }

protected:
    // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
    int32_t getServerSocket() const noexcept { return m_serverSocket.get(); }
    SocketServer::AcceptStatus acceptImpl(int64_t const timeoutUs);

private:
    UniqueSocketDescriptorHandle m_serverSocket;
    SocketConnectionPool m_connectionPool;
    uint16_t m_port; // debug only; no specific use case
    SockPrefixStr m_sockPrefix;
};

/// Class to initialize a socket on the client side.
class SocketClient
{
public:
    /// Socket Client Constructor, shall store the connectionPoolSize.
    /// @param connectionPoolSize The size of the connection pool
    explicit SocketClient(size_t const connectionPoolSize);
    ~SocketClient() = default;

    SocketClient(SocketClient&&) = default;
    SocketClient& operator=(SocketClient&&) = default;
    SocketClient(SocketClient const&)       = delete;
    SocketClient& operator=(SocketClient const&) = delete;

    /// Connect a connection to a server until the timeout is reached.
    class ConnectStatus
    {

    public:
        /// Create a connection status object, passing the current socket status.
        /// @param status the socket status.
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        explicit ConnectStatus(Sock_Status const status) noexcept
            : m_status(status)
            , m_connection(nullptr)
        {
        }

        /// Returns the current socket status.
        /// @return the socket status.
        Sock_Status getStatus() const { return m_status; }
        /// Set a new socket status
        /// @param status the new socket status
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        void setStatus(Sock_Status const status) { m_status = status; }

        /// Bind a new socket connection to this object.
        /// @param connection the shared pointer to the new socket connection.
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        void setConnection(std::shared_ptr<SocketConnection> connection) { m_connection = std::move(connection); }

        /// Returns the current socket connection.
        /// @return the shared pointer to the socket connection.
        std::shared_ptr<SocketConnection> getConnection() { return m_connection; }

        /// Reset the socket connection.
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        void resetConnection() { m_connection.reset(); }

        /// Check if the connection with the server has been established.
        /// @return true if connection established, false otherwise.
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        bool isSuccessConnection() const noexcept { return m_status == Sock_Status::SUCCESS && m_connection != nullptr; }

    private:
        Sock_Status m_status;
        std::shared_ptr<SocketConnection> m_connection;
    };

    /// Connect to a specific server socket and port, using TCP protocol.
    /// @param host the server address.
    /// @param port the server port.
    /// @param timeoutUs the timeout after which this client will arise a connection error.
    /// @param sockPrefix an optional socket prefix.
    /// @return the connection status for this client socket.
    ConnectStatus connect(char8_t const* const host, uint16_t const port, int64_t const timeoutUs,
                          SockPrefixStr const& sockPrefix = SockPrefixStr());

    /// Connect to a specific server socket and port, using UDP protocol.
    /// @param host the server address.
    /// @param remotePort the server port.
    /// @param enableBroadcast if true, messages from this client will be broadcast to all connected client sockets.
    /// @param timeoutUs the timeout after which this client will arise a connection error.
    /// @param sockPrefix an optional socket prefix.
    /// @return the connection status for this client socket.
    ConnectStatus connectUDP(char8_t const* const host, uint16_t const remotePort, bool const enableBroadcast, int64_t const timeoutUs, SockPrefixStr const& sockPrefix = SockPrefixStr());

    /// Bind this socket to a specific server, using UDP protocol.
    /// @param host the server address.
    /// @param localPort the client port.
    /// @param enableBroadcast if true, messages from this client will be broadcast to all connected client sockets.
    /// @param timeoutUs the timeout after which this client will arise a connection error.
    /// @param sockPrefix an optional socket prefix.
    /// @return the connection status for this client socket.
    ConnectStatus bindUDP(char8_t const* const host, uint16_t const localPort, bool const enableBroadcast, int64_t const timeoutUs, SockPrefixStr const& sockPrefix = SockPrefixStr());

    /// Broadcasts content of the buffer to all valid connections.
    /// @param buffer the buffer to send.
    /// @param bufferSize the size of the buffer to send.
    /// @param timeoutUs the max transmission time allowed (us) before arising a connection timeout error.
    /// @return the socket status
    Sock_Status broadcast(uint8_t const* const buffer, size_t const bufferSize, int64_t const timeoutUs = SOCK_TIMEOUT_INFINITE);

private:
    uint32_t m_protocol;

    /**
     *  Enumerate valid states during a connection attempt
     */
    enum class ConnectState : uint32_t
    {
        /// Socket attempting to connect
        CONNECT,
        /// Socket connection is in progress
        INPROGRESS,
        /// Connection has been reset.
        RESET,
        /// Connection succeeded
        SUCCEEDED,
        /// Connection error
        ERROR
    };

    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2813925
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2813925
    constexpr static size_t MAX_CONN_ATTEMPT{3U};
    SocketConnectionPool m_connectionPool;

    ConnectStatus connectOrBindUDP(bool const connectOnly,
                                   char8_t const* const host,
                                   uint16_t const port,
                                   bool const enableBroadcast,
                                   int64_t const timeoutUs,
                                   SockPrefixStr const& sockPrefix = SockPrefixStr());

    void attemptConnect(ConnectStatus& ret,
                        const struct addrinfo* const gaiResults,
                        int64_t const timeoutUs,
                        SockPrefixStr const& sockPrefix = SockPrefixStr());

    void attemptConnectUDP(ConnectStatus& ret,
                           char8_t const* const host,
                           uint16_t const port);

    void attemptUDP(bool const connect,
                    ConnectStatus& ret,
                    const struct addrinfo* const gaiResults,
                    bool const enableBroadcast,
                    SockPrefixStr const& sockPrefix = SockPrefixStr());

    void handleConnect(ConnectStatus& ret,
                       UniqueSocketDescriptorHandle& connectionSocket,
                       const struct addrinfo* const servAddr,
                       int64_t const timeoutUs);

    static bool handleConnectInProgress(ConnectStatus& ret,
                                        const UniqueSocketDescriptorHandle& connectionSocket,
                                        int64_t const timeoutUs); // returns requirement to continue trying

    // Helper functions to initialize a socket during connection
    // attempts

    static void resetConnectSocket(ConnectStatus& ret,
                                   UniqueSocketDescriptorHandle& connectionSocket,
                                   const struct addrinfo& servAddr,
                                   SockPrefixStr const& sockPrefix = SockPrefixStr());

    /// @brief Helper function to set socket options
    ///
    /// @param[in] socket Socket ID
    inline void setSockOptions(const int32_t socket);

    void bindAndCheckConnectSocket(ConnectStatus& ret,
                                   const UniqueSocketDescriptorHandle& connectionSocket,
                                   const struct addrinfo& servAddr);

    void resetBindAndCheckConnectSocket(ConnectStatus& ret,
                                        UniqueSocketDescriptorHandle& connectionSocket,
                                        const struct addrinfo& servAddr,
                                        SockPrefixStr const& sockPrefix = SockPrefixStr());

    // Connect state machine

    static ConnectState handleConnectState(const UniqueSocketDescriptorHandle& connectionSocket,
                                           const struct addrinfo& servAddr);

    static ConnectState handleInProgressState(ConnectStatus& ret,
                                              const UniqueSocketDescriptorHandle& connectionSocket,
                                              int64_t const timeoutUs);

    ConnectState handleResetState(ConnectStatus& ret,
                                  UniqueSocketDescriptorHandle& connectionSocket,
                                  const struct addrinfo& serv_addr);
};
} // namespace socketipc
} // namespace dwshared

#endif // SOCK_IPC_SOCKETCLIENTSERVER_HPP_
