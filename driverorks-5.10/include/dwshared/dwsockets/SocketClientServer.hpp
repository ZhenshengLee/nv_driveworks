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
// SPDX-FileCopyrightText: Copyright (c) 2014-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <sys/select.h>
#include <sys/socket.h>
#include <memory>
#include <netdb.h>
#include <netinet/in.h>

// C++ interface
#include <dw/core/container/VectorFixed.hpp>
#include <dw/core/logger/Logger.hpp>
#include <dw/core/container/BaseString.hpp>

#include "impl/SocketWrapper.hpp"

#ifdef VIBRANTE_V5Q

//Bug 200684998
static_assert(FD_SETSIZE >= 1000U, "FD_SETSIZE is less than 1000 on QNX");
#endif

namespace dwshared
{
namespace socketipc
{

/**
 * @brief Get the the system time from epoch in micro seconds
 *
 * @return int64_t
 */
int64_t getCurrentTimeUS() noexcept;

class SocketConstants
{
public:
    static constexpr uint32_t ONE_SECOND_US        = 1000000U;
    static constexpr uint32_t MAX_BIND_ATTEMPTS    = 10U;
    static constexpr int32_t INVALID_SOCKET_HANDLE = -1;
    static constexpr uint32_t MAX_ERRMSG_LEN       = 256U;
    static constexpr uint32_t MAX_HOSTNAME_LEN     = 256U;
    static constexpr uint32_t MAX_PORTSTR_LEN      = 5U;
};

inline void logStreamError(const void* const ptrObject, dw::core::FixedString<SocketConstants::MAX_ERRMSG_LEN> const& logMessage, bool const throwEx = true)
{
    if (ptrObject != nullptr)
    {
        char8_t errMsg[SocketConstants::MAX_ERRMSG_LEN];
#ifndef __QNX__
        LOGSTREAM_ERROR(ptrObject) << logMessage.c_str() << " - " << ::strerror_r(errno, errMsg, SocketConstants::MAX_ERRMSG_LEN) << dw::core::Logger::State::endl;
#else
        int32_t errNum = ::strerror_r(errno, errMsg, SocketConstants::MAX_ERRMSG_LEN);
        LOGSTREAM_ERROR(ptrObject) << logMessage.c_str() << " - " << errNum << " - " << errMsg << dw::core::Logger::State::endl;
#endif
    }

    if (throwEx)
    {
        throw dw::core::ExceptionBase(logMessage.c_str());
    }
}

inline void logStreamErrorWithPort(const void* const ptrObject, dw::core::FixedString<SocketConstants::MAX_ERRMSG_LEN> const& logMessage, uint16_t port, bool const throwEx = true)
{
    if (ptrObject != nullptr)
    {
        char8_t errMsg[SocketConstants::MAX_ERRMSG_LEN];
#ifndef __QNX__
        LOGSTREAM_ERROR(ptrObject) << logMessage.c_str() << " - " << port << " - "
                                   << ::strerror_r(errno, errMsg, SocketConstants::MAX_ERRMSG_LEN) << dw::core::Logger::State::endl;
#else
        int32_t errNum = ::strerror_r(errno, errMsg, SocketConstants::MAX_ERRMSG_LEN);
        LOGSTREAM_ERROR(ptrObject) << logMessage.c_str() << " - " << errNum << " - " << port
                                   << " - " << errMsg << dw::core::Logger::State::endl;
#endif
    }

    if (throwEx)
    {
        throw dw::core::ExceptionBase(logMessage.c_str(), port);
    }
}

enum class Sock_Status : int32_t
{
    BUFFER_FULL,
    CANNOT_CREATE_OBJECT,
    END_OF_STREAM,
    FAILURE,
    INTERNAL_ERROR,
    INVALID_ARGUMENT,
    SUCCESS,
    TIME_OUT,
};

static constexpr int64_t SOCK_TIMEOUT_INFINITE = 0x0123456789ABCDEF;

inline void SOCK_FD_SET(int32_t const handle, fd_set& fdset)
{
    // TODO(dwplc): Violation originated from QNX certified headers usage (Supporting permit SWE-DRC-518-SWSADP)
    // coverity[autosar_cpp14_a3_9_1_violation]
    // coverity[autosar_cpp14_a5_2_5_violation]
    // coverity[autosar_cpp14_a5_2_2_violation]
    // coverity[autosar_cpp14_a7_1_1_violation]
    // coverity[autosar_cpp14_m5_0_3_violation]
    // coverity[autosar_cpp14_m5_0_10_violation]
    // coverity[autosar_cpp14_m5_0_20_violation]
    // coverity[autosar_cpp14_m5_0_21_violation]
    // coverity[autosar_cpp14_m5_0_8_violation]
    // coverity[autosar_cpp14_m5_0_9_violation]
    // coverity[autosar_cpp14_m5_0_14_violation]
    // coverity[autosar_cpp14_m5_8_1_violation]
    // coverity[autosar_cpp14_m6_2_1_violation]
    // coverity[autosar_cpp14_m5_0_20_violation]
    // coverity[autosar_cpp14_m5_0_3_violation]
    // coverity[autosar_cpp14_m5_8_1_violation]
    // coverity[cert_arr30_c_violation]
    // coverity[cert_int34_c_violation]
    // coverity[cert_int31_c_violation]
    // coverity[cert_ctr50_cpp_violation]
    // clang-tidy NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
    FD_SET(handle, &fdset);
}

// TODO(dwplc): FP - it claims parameter is not modified and should be const but it is actually modified within a macros
// coverity[autosar_cpp14_m7_1_2_violation]
inline void SOCK_FD_ZERO(fd_set& fdset) noexcept
{
    // TODO(dwplc): Violation originated from QNX certified headers usage (Supporting permit SWE-DRC-518-SWSADP)
    // coverity[autosar_cpp14_a3_9_1_violation]
    // coverity[autosar_cpp14_a5_0_2_violation]
    // coverity[autosar_cpp14_a7_1_7_violation]
    // coverity[autosar_cpp14_m7_4_3_violation]
    // coverity[autosar_cpp14_m8_0_1_violation]
    // coverity[autosar_cpp14_a12_0_2_violation]
    // clang-tidy NOLINTNEXTLINE(readability-isolate-declaration)
    FD_ZERO(&fdset);
}

// Simple class to wrap socket
class UniqueSocketDescriptorHandle
{
public:
    explicit UniqueSocketDescriptorHandle(int32_t const hSocket) noexcept
        : m_hSocket(hSocket)
    {
    }

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
    UniqueSocketDescriptorHandle(UniqueSocketDescriptorHandle&& other) noexcept
        : m_hSocket(other.m_hSocket)
    {
        other.m_hSocket = SocketConstants::INVALID_SOCKET_HANDLE;
    }

    // move assignment
    UniqueSocketDescriptorHandle& operator=(UniqueSocketDescriptorHandle&& right) noexcept
    {
        if (this != &right)
        { // different, do the move
            int32_t const sHandle = right.m_hSocket;
            right.m_hSocket       = SocketConstants::INVALID_SOCKET_HANDLE;
            reset(sHandle);
        }
        return (*this);
    }

    int32_t release() noexcept
    {
        int32_t const tmp = m_hSocket;
        m_hSocket         = SocketConstants::INVALID_SOCKET_HANDLE;
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
            static_cast<void>(::close(m_hSocket));
        }
#endif
        m_hSocket = hSocket;
    }

    void reset() noexcept
    {
        constexpr int32_t NULL_SOCK_ID = SocketConstants::INVALID_SOCKET_HANDLE;
        reset(NULL_SOCK_ID); // invalid value
    }

    bool isValid() const noexcept
    {
        return m_hSocket > 0;
    }

private:
    int32_t m_hSocket;
};

/// Simple class to manage memory allocated by getaddrinfo
class AddrInfo
{
public:
    AddrInfo() = default;
    explicit AddrInfo(addrinfo* const addr)
        : m_addr(addr) {}
    ~AddrInfo();

    AddrInfo(AddrInfo&&) = default;
    AddrInfo& operator=(AddrInfo&&) = default;
    AddrInfo(AddrInfo const&)       = default;
    AddrInfo& operator=(AddrInfo const&) = default;

    const struct addrinfo* getAddr() const { return m_addr; }

private:
    struct addrinfo* m_addr;
};

/**
 * @brief A structure contains data for POSIX sockaddr_in
 */
using IpcSockAddrIn = struct IpcSockAddrIn
{
    uint16_t port;
    uint32_t ip;
};

class SocketConnection
{
public:
    explicit SocketConnection(UniqueSocketDescriptorHandle connectionSocket);
    ~SocketConnection();

    SocketConnection(SocketConnection&&) = default;
    SocketConnection& operator=(SocketConnection&&) = default;
    SocketConnection(SocketConnection const&)       = delete;
    SocketConnection& operator=(SocketConnection const&) = delete;

    // Message Transfer
    class MessageStatus
    {

    public:
        MessageStatus(Sock_Status const status, uint64_t const size) noexcept
            : m_status(status)
            , m_transmissionSize(size)
            , m_rxtime{}
        {
        }

        Sock_Status getStatus() const noexcept { return m_status; }
        void setStatus(Sock_Status const status) noexcept { m_status = status; }
        uint64_t getTransmissionSize() const noexcept { return m_transmissionSize; }
        void setTransmissionSize(uint64_t const size) noexcept { m_transmissionSize = size; }

        // Get packet arrival kernel timestamp in microseconds
        int64_t getRxTime() const { return m_rxtime; }
        void setRxTime(int64_t const& time) { m_rxtime = time; }

    private:
        Sock_Status m_status;
        uint64_t m_transmissionSize;
        int64_t m_rxtime; // Packet arrival kernel timestamp in microsecond
    };
    // sends content of the buffer
    MessageStatus send(const void* const buffer, size_t const bufferSize, int64_t const timeoutUs = SOCK_TIMEOUT_INFINITE);
    MessageStatus sendTo(const void* const buffer, size_t const bufferSize, char8_t const* const host, uint16_t const sendPort, int64_t const timeoutUs = SOCK_TIMEOUT_INFINITE);

    // reads into buffer (returns number of read bytes if successful)
    MessageStatus recv(void* const buffer, size_t const bufferSize, int64_t const timeoutUs = SOCK_TIMEOUT_INFINITE);
    // SOCK_TIMEOUT_INFINITE timeout shall be totally blocking
    MessageStatus peek(uint8_t* const buffer, size_t const bufferSize, int64_t const timeoutUs);
    static void setNonBlocking(int32_t const sockfd);
    static void unsetNonBlocking(int32_t const sockfd);
    int32_t get_sockfd() const noexcept
    {
        return m_connectionSocket.get();
    }
    Sock_Status getSockaddr(IpcSockAddrIn* const addr) const;

    inline void setSendTimeout(int64_t const timeoutUs)
    {
        if (timeoutUs == m_sendTimeoutUs)
        {
            return;
        }
        timeval const tv = timeoutToTimeval(timeoutUs);
        errno            = 0;
        if (::setsockopt(m_connectionSocket.get(), SOL_SOCKET, SO_SNDTIMEO, &tv, static_cast<socklen_t>(sizeof(tv))) < 0)
        {
            dwshared::socketipc::logStreamError(nullptr, "SocketConnection: can't set send timeout");
        }
        m_sendTimeoutUs = timeoutUs;
    }

    inline void setRecvTimeout(int64_t const timeoutUs)
    {
        if (timeoutUs == m_recvTimeoutUs)
        {
            return;
        }
        timeval const tv = timeoutToTimeval(timeoutUs);
        errno            = 0;
        if (::setsockopt(m_connectionSocket.get(), SOL_SOCKET, SO_RCVTIMEO, &tv, static_cast<socklen_t>(sizeof(tv))) < 0)
        {
            dwshared::socketipc::logStreamError(nullptr, "SocketConnection: can't set recv timeout");
        }
        m_recvTimeoutUs = timeoutUs;
    }

    void setUDPBroadcast(bool const udpBroadcast) noexcept
    {
        m_udpBroadcastEnabled = udpBroadcast;
    }
    bool isUDPBroadcast() const noexcept
    {
        return m_udpBroadcastEnabled;
    }

    uint64_t getSkipCount() const noexcept { return m_skipCount; }
    void incSkipCount() noexcept
    {
        if (m_skipCount < std::numeric_limits<uint64_t>::max())
        {
            m_skipCount += 1U;
        }
    }

private:
    dwshared::socketipc::SendRecvReturnStruct peekBlock(uint8_t* const buffer, size_t const bufferSize);
    dwshared::socketipc::SendRecvReturnStruct peekBlockWithTimeout(uint8_t* const buffer, size_t const bufferSize, int64_t const timeoutUs) const;
    dwshared::socketipc::SendRecvReturnStruct peekNonBlock(uint8_t* const buffer, size_t const bufferSize) const;
    dwshared::socketipc::SendRecvReturnStruct processRecvWithTimeout(void* const buffer, size_t const bufferSize, int64_t const timeoutUs);
    static timeval timeoutToTimeval(int64_t const timeoutUs) noexcept
    {
        // TODO(dwplc): FP - "tv" suggested as const, but its value changes for VIBRANTE_V5Q
        // coverity[autosar_cpp14_a7_1_1_violation]
        timeval tv = {
            .tv_sec  = static_cast<decltype(timeval::tv_sec)>(timeoutUs / SocketConstants::ONE_SECOND_US),
            .tv_usec = static_cast<decltype(timeval::tv_usec)>(timeoutUs % SocketConstants::ONE_SECOND_US)};

#ifdef VIBRANTE_V5Q
        // Some of the QNX socket timeouts are restricted to 32 seconds or under (eg., SO_SNDTIMEO)
        if (tv.tv_sec > 31)
        {
            tv.tv_sec = 31;
        }
#endif
        // QNX setsockopt has undefined behaviour when passing 0 as send or receive timeout value
        // Linux implementation usually makes it an infinite timeout, which we don't want either.
        if ((tv.tv_sec == 0) && (tv.tv_usec == 0))
        {
            tv.tv_usec = 1;
        }

        return tv;
    }

    // Convert from errno to MessageStatus for ::send()
    MessageStatus getSendErrorStatus(int32_t const error);

    // Counter for packet skip
    uint64_t m_skipCount = 0UL;

    UniqueSocketDescriptorHandle m_connectionSocket;
    int64_t m_sendTimeoutUs    = 0;
    int64_t m_recvTimeoutUs    = 0;
    int64_t m_receivedBytes    = 0;
    bool m_udpBroadcastEnabled = false;
};

class SocketConnectionPool
{
public:
    static constexpr size_t INVALID_CONNECTION_SLOT = 0xBADDBADD;
    explicit SocketConnectionPool(size_t const poolSize);
    ~SocketConnectionPool();

    SocketConnectionPool(SocketConnectionPool&&) = default;
    SocketConnectionPool& operator=(SocketConnectionPool&&) = default;
    SocketConnectionPool(SocketConnectionPool const&)       = default;
    SocketConnectionPool& operator=(SocketConnectionPool const&) = default;

    // acquires a free pool slot, or nullptr if no slot is free
    size_t getFreeSlot() const;
    void addToPool(std::weak_ptr<SocketConnection> const connection, size_t const index);

    // broadcasts content of the buffer to all valid connections
    Sock_Status broadcast(uint8_t const* const buffer, size_t const bufferSize, int64_t const timeoutUs = SOCK_TIMEOUT_INFINITE);

private:
    constexpr static int32_t SOCK_SELECT_AT_BROADCAST_TIMEOUT = 200000;
    // Log interval in (2^n)-1 format
    constexpr static size_t REPEATED_ERR_LOG_INTERVAL_MINUS1 = 511U;

    dw::core::VectorFixed<std::weak_ptr<SocketConnection>> m_connectionPool;
    // Counter for packet drop
    uint64_t m_dropCount = 0U;

    bool isBroadcastPossible(fd_set& wfdset);
    bool isBroadcastConnectionValid(std::shared_ptr<SocketConnection> connection, fd_set const& wfdset);
};

class SocketServer
{
public:
    SocketServer(uint16_t const port, size_t const connectionPoolSize, bool const reusePort = true);
    ~SocketServer() = default;

    SocketServer(SocketServer&&) = default;
    SocketServer& operator=(SocketServer&&) = default;
    SocketServer(SocketServer const&)       = delete;
    SocketServer& operator=(SocketServer const&) = delete;

    // accept an incoming connection until the timeout is reached
    class AcceptStatus
    {
    public:
        explicit AcceptStatus(Sock_Status const status) noexcept
            : m_status(status)
            , m_connection(nullptr)
        {
        }

        Sock_Status getStatus() const noexcept { return m_status; }
        void setStatus(Sock_Status const status) noexcept { m_status = status; }

        void setConnection(std::shared_ptr<SocketConnection> connection) { m_connection = std::move(connection); }
        std::shared_ptr<SocketConnection> getConnection() { return m_connection; }
        void resetConnection() { m_connection.reset(); }

    private:
        Sock_Status m_status;
        std::shared_ptr<SocketConnection> m_connection;
    };
    AcceptStatus accept(int64_t const timeoutUs);
    static int32_t createUDPSocket(uint16_t const port);
    static sockaddr_in getSocketAddress(uint16_t const port);

    // broadcasts content of the buffer to all valid connections
    Sock_Status broadcast(uint8_t const* const buffer, size_t const bufferSize, int64_t const timeoutUs = SOCK_TIMEOUT_INFINITE);

protected:
    int32_t getServerSocket() const noexcept { return m_serverSocket.get(); }

private:
    UniqueSocketDescriptorHandle m_serverSocket;
    SocketConnectionPool m_connectionPool;
    uint16_t m_port; // debug only; no specific use case
};

class SocketClient
{
public:
    explicit SocketClient(size_t const connectionPoolSize);
    ~SocketClient() = default;

    SocketClient(SocketClient&&) = default;
    SocketClient& operator=(SocketClient&&) = default;
    SocketClient(SocketClient const&)       = delete;
    SocketClient& operator=(SocketClient const&) = delete;

    // connect a connection to a server until the timeout is reached
    class ConnectStatus
    {

    public:
        explicit ConnectStatus(Sock_Status const status) noexcept
            : m_status(status)
            , m_connection(nullptr)
        {
        }

        Sock_Status getStatus() const { return m_status; }
        void setStatus(Sock_Status const status) { m_status = status; }

        void setConnection(std::shared_ptr<SocketConnection> connection) { m_connection = std::move(connection); }
        std::shared_ptr<SocketConnection> getConnection() { return m_connection; }
        void resetConnection() { m_connection.reset(); }
        bool isSuccessConnection() const noexcept { return m_status == Sock_Status::SUCCESS && m_connection != nullptr; }

    private:
        Sock_Status m_status;
        std::shared_ptr<SocketConnection> m_connection;
    };
    ConnectStatus connect(char8_t const* const host, uint16_t const port, int64_t const timeoutUs,
                          dw::core::FixedString<8> const& sockPrefix = dw::core::FixedString<8>());
    ConnectStatus connectUDP(char8_t const* const host, uint16_t const remotePort, bool const enableBroadcast, int64_t const timeoutUs, dw::core::FixedString<8> const& sockPrefix = dw::core::FixedString<8>());
    ConnectStatus bindUDP(char8_t const* const host, uint16_t const localPort, bool const enableBroadcast, int64_t const timeoutUs, dw::core::FixedString<8> const& sockPrefix = dw::core::FixedString<8>());

    // broadcasts content of the buffer to all valid connections
    Sock_Status broadcast(uint8_t const* const buffer, size_t const bufferSize, int64_t const timeoutUs = SOCK_TIMEOUT_INFINITE);

private:
    // Enumerate valid states during a connection attempt

    uint32_t m_protocol;

    enum class ConnectState : uint32_t
    {
        CONNECT,
        INPROGRESS,
        RESET,
        SUCCEEDED,
        ERROR
    };

    constexpr static size_t MAX_CONN_ATTEMPT = 3U;
    SocketConnectionPool m_connectionPool;

    ConnectStatus connectOrBindUDP(bool const connectOnly,
                                   char8_t const* const host,
                                   uint16_t const port,
                                   bool const enableBroadcast,
                                   int64_t const timeoutUs,
                                   dw::core::FixedString<8> const& sockPrefix = dw::core::FixedString<8>());

    void attemptConnect(ConnectStatus& ret,
                        const struct addrinfo* const gaiResults,
                        int64_t const timeoutUs,
                        dw::core::FixedString<8> const& sockPrefix = dw::core::FixedString<8>());

    void attemptConnectUDP(ConnectStatus& ret,
                           char8_t const* const host,
                           uint16_t const port);

    void attemptUDP(bool const connect,
                    ConnectStatus& ret,
                    const struct addrinfo* const gaiResults,
                    bool const enableBroadcast,
                    dw::core::FixedString<8> const& sockPrefix = dw::core::FixedString<8>());

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
                                   dw::core::FixedString<8> const& sockPrefix = dw::core::FixedString<8>());

    void bindAndCheckConnectSocket(ConnectStatus& ret,
                                   const UniqueSocketDescriptorHandle& connectionSocket,
                                   const struct addrinfo& servAddr);

    void resetBindAndCheckConnectSocket(ConnectStatus& ret,
                                        UniqueSocketDescriptorHandle& connectionSocket,
                                        const struct addrinfo& servAddr,
                                        dw::core::FixedString<8> const& sockPrefix = dw::core::FixedString<8>());

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
