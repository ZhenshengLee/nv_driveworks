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
// SPDX-FileCopyrightText: Copyright (c) 2014-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/// @brief Get the the system time from epoch in micro seconds
///
/// @return int64_t
int64_t getCurrentTimeUS() noexcept;

/// @brief  SocketConstants is designed to manage the common constants in this SW unit
class SocketConstants
{
public:
    /// us value of 1 second
    static constexpr uint32_t ONE_SECOND_US{1000000U};
    /// ms value of 1 second
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2813925
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2813925

    static constexpr int32_t ONE_MS_US{1000U};
    /// Maximal Bind value of socket connection
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2813925
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2813925
    static constexpr uint32_t MAX_BIND_ATTEMPTS{10U};
    /// Invalid Socket Handle
    static constexpr int32_t INVALID_SOCKET_HANDLE{-1};
    /// Maximal Error Message length
    static constexpr uint32_t MAX_ERRMSG_LEN{256U};
    /// Maximal Host Name length
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2813925
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2813925
    static constexpr uint32_t MAX_HOSTNAME_LEN{256U};
    /// Maximal Port length
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2813925
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2813925
    static constexpr uint32_t MAX_PORTSTR_LEN{5U};
};

/// @brief The function is particularly useful for logging errors that occur in a stream-based system, such as network applications.
///
/// @param[in] ptrObject The pointer to the object that is causing the error.
/// @param[in] logMessage A fixed string that contains the error message.
/// @param[in] throwEx A flag that indicates whether to throw an exception or not.
/// @throw dw::core::ExceptionBase if the @c throwEx is set.
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

/// @brief The function is particularly useful for logging errors that occur in a stream-based system, such as network applications.
///
/// @param[in] ptrObject The pointer to the object that is causing the error.
/// @param[in] logMessage A fixed string that contains the error message.
/// @param[in] port the port number of the socket connection.
/// @param[in] throwEx A flag that indicates whether to throw an exception or not.
/// @throw dw::core::ExceptionBase if the @c throwEx is set.
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

/// @brief Socket Status definitions

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

/// Infinite value of sock timeout
static constexpr int64_t SOCK_TIMEOUT_INFINITE{0x0123456789ABCDEF};

/// @brief SocketHelper is designed as Wrapper class to process the socket prefix on QNX
class SocketHelper
{
public:
    /// Helper function to set sock prefix on QNX system when a valid sock prefix is provided.
    /// @param[in] sockPrefix A fixed string that contains the sock prefix info.
    /// @param[in] logModule A fixed string that contains log provided by caller module.
    /// @throw dw::core::ExceptionBase if sock prefix is not set successfully.
    static void setSockPrefix(SockPrefixStr const& sockPrefix, dw::core::FixedString<SocketConstants::MAX_ERRMSG_LEN> const& logModule)
    {
#if defined(__QNX__) && (VIBRANTE_PDK_DECIMAL >= 6000000)
        // API setsockprefix/clearsockprefix is introduced from QNX SDP7.1
        if (!sockPrefix.empty())
        {
            SockPrefixStr currentSockPrefix{setsockprefix(nullptr)};
            if (currentSockPrefix != sockPrefix)
            {
                setsockprefix(sockPrefix.c_str());

                // check if sock prefix is set successfully
                currentSockPrefix = SockPrefixStr{setsockprefix(nullptr)};
                if (currentSockPrefix != sockPrefix)
                {
                    dw::core::FixedString<SocketConstants::MAX_ERRMSG_LEN> errorMsg{"Failed to set sock prefix in "};
                    errorMsg += logModule;
                    throw dw::core::ExceptionBase(errorMsg.c_str());
                }
            }
        }
#else
        static_cast<void>(sockPrefix);
        static_cast<void>(logModule);
#endif
    }

    /// Helper function to clear sock prefix on QNX system if a valid sock prefix is set previously by setsockprefix().
    static void clearSockPrefix()
    {
#if defined(__QNX__) && (VIBRANTE_PDK_DECIMAL >= 6000000)
        SockPrefixStr currentSockPrefix{setsockprefix(nullptr)};
        if (!currentSockPrefix.empty())
        {
            clearsockprefix();
        }
#endif
    }

}; // SocketHelper

/// @brief UniqueSocketDescriptorHandle is designed as Wrapper handle to maintain the life cycle of socket fd.
/// It will shutdown and close the socket fd automatically once do deconstruct.
class UniqueSocketDescriptorHandle
{
public:
    /// @brief Construct a Socket Wrapper instance.
    /// This function creates a Socket Wrapper instance,
    /// which will then create a UniqueSocketDescriptorHandle with a specific socket ID.
    ///
    /// @param[in] hSocket the socket ID to wrap in the class.
    explicit UniqueSocketDescriptorHandle(int32_t const hSocket) noexcept
        : m_hSocket(hSocket)
    {
    }

    /// @brief Construct a Socket Wrapper instance.
    /// This function constructs a Socket Wrapper instance,
    /// which then creates a UniqueSocketDescriptorHandle with an invalid socket ID by default.
    ///
    // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
    UniqueSocketDescriptorHandle() noexcept
        : UniqueSocketDescriptorHandle(SocketConstants::INVALID_SOCKET_HANDLE)
    {
    }

    /// @brief Deconstruct a Socket Wrapper instance.
    /// This function will shutdown and close the socket id wrapped in the class instance.
    ///
    ~UniqueSocketDescriptorHandle()
    {
        reset();
    }

    UniqueSocketDescriptorHandle& operator=(UniqueSocketDescriptorHandle const&) = delete; // no copies!
    UniqueSocketDescriptorHandle(UniqueSocketDescriptorHandle const&)            = delete; // no copies!

    /// @brief Construct a Socket Wrapper instance.
    /// This function deconstructs a Socket Wrapper instance,
    /// which then shuts down and closes the socket that was wrapped in the class instance.
    ///
    /// @param[in] other Socket Wrapper instance which stored the specific socket ID.
    // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
    UniqueSocketDescriptorHandle(UniqueSocketDescriptorHandle&& other) noexcept
        : m_hSocket(other.m_hSocket)
    {
        other.m_hSocket = SocketConstants::INVALID_SOCKET_HANDLE;
    }

    /// @brief Move assignment a Socket Wrapper instance.
    /// This function moves the assignment of a Socket Wrapper instance,
    /// which then moves the UniqueSocketDescriptorHandle with a specific socket ID.
    ///
    /// @param[in] right Socket Wrapper instance which stored the specific socket ID.
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

    /// @brief Release the stored socket id.
    /// This function releases the stored socket ID and sets the stored value to invalid socket ID.
    ///
    // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
    int32_t release() noexcept
    {
        int32_t const tmp{m_hSocket};
        m_hSocket = SocketConstants::INVALID_SOCKET_HANDLE;
        return tmp;
    }

    /// @brief Get the stored socket id.
    /// This function returns the stored socket ID.
    ///
    int32_t get() const noexcept
    {
        return m_hSocket;
    }

    /// @brief Reset the specific socket id.
    /// This function resets the specific socket ID by
    /// shutting down and resetting the input socket ID.
    ///
    /// @param[in] hSocket The specific socket ID to reset.
    void reset(int32_t const& hSocket)
    {
// TODO(owner): Closing the old socket here causes a lot of autosar A15-0-2 violations, basically on all exception throw statements.
// When an exception is thrown after a move operation on UniqueSocketDescriptorHandle, basic exception safety could not be guaranteed
// in cases where old socket has been deallocated and the new one has failed initialization.
// As an example, see the constructor function: SocketConnection::SocketConnection(UniqueSocketDescriptorHandle connectionSocket)
#ifndef __COVERITY__
        if (m_hSocket >= 0)
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

    /// @brief Reset the invalid socket id.
    /// This function resets the invalid socket ID by shutting down and closing the invalid socket ID.
    ///
    void reset() noexcept
    {
        constexpr int32_t NULL_SOCK_ID{SocketConstants::INVALID_SOCKET_HANDLE};
        reset(NULL_SOCK_ID); // invalid value
    }

    /// @brief Get if the stored socket id is valid.
    /// This function returns a boolean value indicating whether the stored socket ID is valid.
    ///
    /// @return return true if the socket id is valid, else return false.
    // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
    bool isValid() const noexcept
    {
        return m_hSocket >= dw::core::util::ZERO;
    }

private:
    /// Stored specific socket id
    int32_t m_hSocket;
};

/// @brief AddrInfo is a Simple class designed to manage memory allocated by getaddrinfo
class AddrInfo
{
public:
    AddrInfo() = default;

    /// @brief Constructor to store the passed address
    /// @param[in] addr the address to store
    explicit AddrInfo(addrinfo* const addr)
        : m_addr(addr) {}
    ~AddrInfo();

    AddrInfo(AddrInfo&&) = default;
    AddrInfo& operator=(AddrInfo&&) = default;
    AddrInfo(AddrInfo const&)       = default;
    AddrInfo& operator=(AddrInfo const&) = default;

    /// @brief This function returns the stored address.
    ///
    /// @return The addrinfo data stored in the instance
    const struct addrinfo* getAddr() const { return m_addr; }

private:
    /// The addrinfo data stored in the instance
    struct addrinfo* m_addr;
};

/// @brief A structure contains data for POSIX sockaddr_in
struct IpcSockAddrIn
{
    /// Port
    uint16_t port;
    /// IP address
    uint32_t ip;
};

/// @brief SocketConnection is a designed handle that serves to maintain the session between a server and a client.
/// It effectively preserves the context and status following the establishment of a connection between the server and the client.
/// The SocketConnection encapsulates and offers methods such as send, sendTo, recv, peek, and others to
/// facilitate communication between the server and client via data packets.
/// It is essential for both the server and the client in order to sustain a bidirectional communication session.

/// @note As known, TCP is a connection-oriented network protocol, while UDP is a connectionless network protocol.
/// The "connection" here refers to an abstract handle used to maintain the communication context between a client and a server,
/// and does not represent a connection-oriented service at the network protocol layer.
class SocketConnection
{
public:
    /// @brief Create a new socket connection object passing the socket handle.
    ///
    /// @param[in] connectionSocket the socket handle.
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
        /// @brief Create a new message status.
        ///
        /// @param[in] status the socket status.
        /// @param[in] size the size of the TX message.
        MessageStatus(Sock_Status const status, uint64_t const size) noexcept
            : m_status(status)
            , m_transmissionSize(size)
            , m_rxtime{}
        {
        }

        /// @brief Return the current socket status.
        ///
        /// @return the socket status.
        /// include (Sock_Status::FAILURE | TIME_OUT | FAILURE | INVALID_ARGUMENT | BUFFER_FULL | END_OF_STREAM).
        Sock_Status getStatus() const noexcept { return m_status; }
        /// @brief Set a internal socket status as input value.
        ///
        /// @param status the new socket status.
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        void setStatus(Sock_Status const status) noexcept { m_status = status; }
        /// @brief Get the size of the TX on this connection.
        ///
        /// @return the TX size
        uint64_t getTransmissionSize() const noexcept { return m_transmissionSize; }
        /// @brief Set the size of the TX on this connection.
        ///
        /// @param[in] size the TX size
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        void setTransmissionSize(uint64_t const size) noexcept { m_transmissionSize = size; }
        /// @brief Get packet arrival kernel timestamp in microseconds from system time epoch.
        ///
        /// @return the packet arrival timestamp.
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        int64_t getRxTime() const { return m_rxtime; }
        /// @brief Set the packet arrival kernel timestamp in microseconds from system time epoch.
        ///
        /// @param[in] time the packet arrival timestamp.
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        void setRxTime(int64_t const& time) { m_rxtime = time; }

    private:
        /// The status of the connection
        Sock_Status m_status;
        /// Size of the tranmission
        uint64_t m_transmissionSize;
        /// Packet arrival kernel timestamp in microsecond
        int64_t m_rxtime;
    };

    /// @brief To send the content of the buffer through this SocketConnection to the specific address/port maintained by the SocketConnection.
    ///
    /// It will use the POSIX send() method to send the content of the buffer through this SocketConnection to the specific address/port maintained by the SocketConnection.
    /// This function can be used by both server and client instances that own the connection object.
    /// Before calling this method, the user must establish the connection.
    /// The connection should be established and in a connected state to ensure successful transmission of the buffer content.
    ///
    /// @param[in] buffer the buffer to send.
    /// @param[in] bufferSize the size of the buffer to send.
    /// @param[in] timeoutUs the max transmission time allowed (us) before arising a connection timeout error(range 0 ~ SOCK_TIMEOUT_INFINITE[0x0123456789ABCDEF]).
    /// @return the message status for this TX
    /// include socket connection status(Sock_Status::SUCCESS | END_OF_STREAM | TIME_OUT | INVALID_ARGUMENT | BUFFER_FULL),
    /// And the TX bytes in this operation.

    MessageStatus send(const void* const buffer, size_t const bufferSize, int64_t const timeoutUs = SOCK_TIMEOUT_INFINITE);

    /// @brief To send the content of the buffer through this SocketConnection to a specific address/port set by the input parameters.
    ///
    /// It will use the POSIX sendTo() method to send the content of the buffer through this SocketConnection to the specific address/port set by the input parameters.
    /// This function can be used by both server and client instances that own the connection object.
    /// The sendTo() method is particularly useful for connectionless socket communication. When calling this method,
    /// the user should ensure to set the target address and port accordingly to specify the destination for the data transmission.
    ///
    /// @param[in] buffer the buffer to send.
    /// @param[in] bufferSize the size of the buffer to send.
    /// @param[in] host the address of the host.
    /// @param[in] sendPort the port of the host.
    /// @param[in] timeoutUs the max transmission time allowed (us) before arising a connection timeout error(range 0 ~ SOCK_TIMEOUT_INFINITE[0x0123456789ABCDEF]).
    /// @return the message status for this TX
    /// include socket connection status(Sock_Status::SUCCESS | END_OF_STREAM | TIME_OUT | INVALID_ARGUMENT | BUFFER_FULL),
    /// And the TX bytes in this operation.
    MessageStatus sendTo(const void* const buffer, size_t const bufferSize, char8_t const* const host, uint16_t const sendPort, int64_t const timeoutUs = SOCK_TIMEOUT_INFINITE);

    /// @brief Read the data into the buffer and returns number of read bytes if successful.
    ///
    /// This function reads the data from the socket reception buffer into the buffer and returns the number of bytes read if the operation is successful.
    /// The POSIX recv() method is used to remove the data from the socket reception buffer.
    /// This function can be used by both server and client instances who own the connection object. The user must establish the connection before calling this method.
    ///
    /// @param[out] buffer the buffer to store received data.
    /// @param[in] bufferSize the size of the buffer.
    /// @param[in] timeoutUs the max reception time allowed (us) before arising a connection timeout error(range 0 ~ SOCK_TIMEOUT_INFINITE[0x0123456789ABCDEF]).
    /// @return the message status for this RX.
    /// include socket connection status(Sock_Status::SUCCESS | END_OF_STREAM | TIME_OUT | INVALID_ARGUMENT | BUFFER_FULL),
    /// RX bytes in this operation.
    MessageStatus recv(void* const buffer, size_t const bufferSize, int64_t const timeoutUs = SOCK_TIMEOUT_INFINITE);

    /// @brief Peek the data into the buffer and returns count of bytes successfully read.
    ///
    /// This function attempts to peek the data into the buffer and returns the number of bytes read if the operation is successful.
    /// The function uses NonBlock, Block, or Block with a timeout of 3 models to peek the data in the receive buffer,
    /// depending on the value of timeoutUS (which can be dw::core::util::ZERO, SOCK_TIMEOUT_INFINITE, or a custom value).
    ///
    /// @param[out] buffer the buffer to store received data.
    /// @param[in] bufferSize the size of the buffer.
    /// @param[in] timeoutUs the max reception time[us] allowed before arising a connection timeout error(range 0 ~ SOCK_TIMEOUT_INFINITE[0x0123456789ABCDEF]).
    /// @return the message status for this RX.
    /// include socket connection status(Sock_Status::SUCCESS | END_OF_STREAM | TIME_OUT | INVALID_ARGUMENT | FAILURE),
    /// And the RX bytes in this operation.
    MessageStatus peek(uint8_t* const buffer, size_t const bufferSize, int64_t const timeoutUs);

    /// @brief Set the socket bound to this connection as non-blocking.
    ///
    /// @param[in] sockfd the socket descriptor.
    static void setNonBlocking(int32_t const sockfd);
    /// @brief Set the socket bound to this connection as blocking.
    /// @param[in] sockfd the socket descriptor.
    static void unsetNonBlocking(int32_t const sockfd);

    /// @brief Return the socket descriptor id bound to this connection.
    ///
    /// @return the socket descriptor.
    int32_t get_sockfd() const noexcept
    {
        return m_connectionSocket.get();
    }

    /// @brief Get the socket address for the connection instance.
    ///
    /// @param[out] addr The output socket address.
    /// @return The status if the input pointer is valid.
    /// Sock_Status::SUCCESS -- get socket addr succeeded
    /// Sock_Status::FAILURE -- get socket addr failed
    Sock_Status getSockaddr(IpcSockAddrIn* const addr) const;

    /// @brief Set a new TX timeout value to the socket bound to this connection.
    ///
    /// @param[in] timeoutUs the timeout[us](range: 0 ~ SOCK_TIMEOUT_INFINITE[0x0123456789ABCDEF]).
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

    /// @brief Set a new RX timeout value to the socket bound to this connection.
    ///
    /// @param[in] timeoutUs the timeout[us] (range: 0 ~ SOCK_TIMEOUT_INFINITE[0x0123456789ABCDEF]).
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

    /// @brief Set the UDP Broadcast flag for this connection.
    ///
    /// @param[in] udpBroadcast the UDP broadcast flag (true or false)
    // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
    void setUDPBroadcast(bool const udpBroadcast) noexcept
    {
        m_udpBroadcastEnabled = udpBroadcast;
    }

    /// @brief Check if the UDP Broadcast flag for this connection has been set.
    ///
    /// @return return true if broadcast enabled, else return false.
    // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
    bool isUDPBroadcast() const noexcept
    {
        return m_udpBroadcastEnabled;
    }

    /// @brief Get the number of TX packet skipped so far due to a connection issue.
    ///
    /// @return the number of TX packet skipped.
    // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
    uint64_t getSkipCount() const noexcept { return m_skipCount; }

    /// @brief Increment the number of TX packet skipped so far due to a connection issue.
    ///
    // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
    void incSkipCount() noexcept
    {
        if (m_skipCount < std::numeric_limits<uint64_t>::max())
        {
            m_skipCount += dw::core::util::ONE_U;
        }
    }

private:
    /// @brief Peek the data with block behavior into the buffer and returns the count of bytes successfully read.
    ///
    /// This function attempts to peek the data into the buffer and returns the count of bytes read if the operation is successful.
    /// The method has block behavior, meaning it will not return until data has been received or an error occurs.
    ///
    /// @param[out] buffer the buffer to store received data.
    /// @param[in] bufferSize the size of the buffer.
    /// @return the status for this peek operation.
    /// include bytes in this peek operation, error code, RX time in microseconds from system time epoch.
    dwshared::socketipc::SendRecvReturnStruct peekBlock(uint8_t* const buffer, size_t const bufferSize);

    /// @brief Peek the data with block timeout into the buffer and returns the count of bytes successfully read.
    ///
    /// This function attempts to peek the data into the buffer and returns the count of bytes read if the operation is successful.
    /// However, if there is no data available to peek, the function will return after a timeout, even if an error occurs.
    ///
    /// @param[out] buffer the buffer to store received data.
    /// @param[in] bufferSize the size of the buffer.
    /// @param[in] timeoutUs the max reception time allowed (us) before arising a connection timeout error(range 0 ~ SOCK_TIMEOUT_INFINITE[0x0123456789ABCDEF]).
    /// @return the status for this peek operation.
    /// include bytes in this peek operation, error code, RX time in microseconds from system time epoch.
    dwshared::socketipc::SendRecvReturnStruct peekBlockWithTimeout(uint8_t* const buffer, size_t const bufferSize, int64_t const timeoutUs) const;

    /// @brief Peek the data with non-block behavior into the buffer and returns the count of bytes successfully read.
    ///
    /// This function attempts to peek the data into the buffer and returns the count of bytes read if the operation is successful.
    /// The method has block behavior, meaning it will immediately return if no data is received.
    ///
    /// @param[out] buffer the buffer to store received data.
    /// @param[in] bufferSize the size of the buffer.
    /// @return the status for this peek operation.
    /// include bytes in this peek operation, error code, RX time in microseconds from system time epoch.
    dwshared::socketipc::SendRecvReturnStruct peekNonBlock(uint8_t* const buffer, size_t const bufferSize) const;

    /// @brief Read the data into the buffer with timeout.
    ///
    /// This function calls the POSIX recv() method to read data from the socket reception buffer into the buffer.
    /// The specified timeout parameter determines the maximum time to wait for data.
    /// The data read from the buffer will be removed, meaning it will be consumed during the read operation.
    /// The function will continue trying to receive data from the socket or connection until either data is received or the timeout period expires.
    /// Once data is received or the timeout occurs, the function will break the loop and return.
    ///
    /// @param[out] buffer the buffer to store received data.
    /// @param[in] bufferSize the size of the buffer.
    /// @param[in] timeoutUs the max reception time allowed[us] before arising a connection timeout error(range: 0 ~ SOCK_TIMEOUT_INFINITE[0x0123456789ABCDEF]).
    /// @return the status for this peek operation.
    /// include bytes in this recv operation, error code, RX time in microseconds from system time epoch.
    dwshared::socketipc::SendRecvReturnStruct processRecvWithTimeout(void* const buffer, size_t const bufferSize, int64_t const timeoutUs);

    /// @brief Convert the int timeout value to the POSIX timeval type.
    ///
    /// @param[in] timeoutUs The int64 timeout value to be convert(range: no limitation).
    /// @return The POSIX timeval instance convert from input timeoutUs.
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

    /// @brief  Convert from the errno to MessageStatus.
    ///
    /// @param[in] error error number to be convert.
    /// @return MessageStatus convert from error number
    /// include (Sock_Status::FAILURE | TIME_OUT | INVALID_ARGUMENT | BUFFER_FULL | END_OF_STREAM).
    MessageStatus getSendErrorStatus(int32_t const error);

    /// Counter for packet skip
    uint64_t m_skipCount{0UL};

    /// Counter for packet skip
    UniqueSocketDescriptorHandle m_connectionSocket;
    /// Timeout value of send function
    int64_t m_sendTimeoutUs{dw::core::util::ZERO};
    /// Timeout value of recv function
    int64_t m_recvTimeoutUs{dw::core::util::ZERO};
    /// Bytes value of recv function
    int64_t m_receivedBytes{dw::core::util::ZERO};
    /// Flag if broadcast enabled
    bool m_udpBroadcastEnabled{false};
};

/// @brief SocketConnectionPool is designed to efficiently manage the connections of the server/client, enabling the simultaneous maintenance of multiple open connections.
/// Its primary responsibility is to allocate slots for newly established connections and add them to the connection pool.
/// By utilizing the SocketConnectionPool, both the server and client can engage in bidirectional broadcasting and partially transfer data packets as needed.
class SocketConnectionPool
{
public:
    /// Invalid Connection Slot
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2813925
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2813925
    static constexpr size_t INVALID_CONNECTION_SLOT{0xBADDBADD};

    /// The maximal partial buffer size
    constexpr static size_t MAX_PARTIAL_BUFFER_SIZE{0x100000};

    /// Store the remaining data for partial transfers in a single socket connection.
    struct PartialPacket
    {
        /// buffer to store remaining partial packet from sent()
        uint8_t m_data[MAX_PARTIAL_BUFFER_SIZE]{};
        /// starting index of remaining data
        size_t m_offset{};
        /// size in bytes, of remaining data
        size_t m_size{};
        /// IP address of peer connection, for debug purpose
        IpcSockAddrIn peerAddr{};

        /// @brief  Check if there is valid remaining data.
        ///
        /// @return return true if has bytes, else return false.
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        bool hasBytes() { return m_size > dw::core::util::ZERO_U; }

        /// @brief Beginning pointer of remaining data.
        ///
        /// @return The offset pointer at the beginning remaining data.
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        uint8_t* begin()
        {
            return m_offset <= MAX_PARTIAL_BUFFER_SIZE ? &m_data[m_offset] : m_data;
        }

        /// @brief Get the size of remaining data in bytes.
        ///
        /// @return The size of remaining data in bytes.
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        size_t size() { return m_size; }

        /// @brief Function to process the remaining data and update offset and size.
        ///
        /// @param[in] func Input function to call in the update method.
        /// @return The MessageStatus return by the input function.
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

    /// Store the remaining data for a group of socket collections, indexed by socket fd
    class PartialPacketMap
    {
    public:
        /// @brief Construct a partial packet map instance.
        ///
        /// @param[in] mapSize Size of the partial packet map instance.
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        explicit PartialPacketMap(size_t mapSize)
            : m_map(mapSize){};
        ~PartialPacketMap() = default;

        /// @brief Check if a socket has valid PartialPacket.
        ///
        /// @param[in] key The key to search specific PartialPacket in the map.
        /// @return return true if contains the key, else return false.
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        bool contains(int32_t key)
        {
            std::lock_guard<std::mutex> lock{m_mutex};
            return m_map.contains(key);
        }

        /// @brief Access the PartialPacket in the map.
        ///
        /// @param[in] key The key to search specific PartialPacket in the map.
        /// @return The PartialPacket associated with the key in the map.
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        PartialPacket& operator[](int32_t key)
        {
            std::lock_guard<std::mutex> lock{m_mutex};
            return m_map[key];
        }

        /// @brief Remove one PartialPacket from the map
        ///
        /// @param[in] key The key to search specific PartialPacket in the map.
        /// @return Returns true if an element has been erased, else turn false.
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        bool erase(int32_t key)
        {
            std::lock_guard<std::mutex> lock{m_mutex};
            return m_map.erase(key);
        }

        /// @brief Thread safe wrapper call to PartialPacket update API.
        ///
        /// @param[in] key The key to search specific PartialPacket in the map.
        /// @param[in] func Input function to call in the update method.
        /// @return The MessageStatus return by the input function.
        template <typename F>
        SocketConnection::MessageStatus update(int32_t key, F&& func)
        {
            // coverity[autosar_cpp14_m0_1_3_violation] FP: nvbugs/3785302
            // coverity[autosar_cpp14_m0_1_9_violation] FP: nvbugs/3785302
            std::lock_guard<std::mutex> lock{m_mutex};
            return m_map[key].update(func);
        };

    private:
        /// Mutex lock for the map
        std::mutex m_mutex;
        /// Hash map instance
        dw::core::HeapHashMap<int32_t, PartialPacket> m_map;
    };

    using PartialPacketMapPtr = std::shared_ptr<PartialPacketMap>;

    /// @brief Create a connection pool with the given size.
    ///
    /// @param[in] poolSize Pool size of the connection Pool(range 1 ~ INVALID_CONNECTION_SLOT[0xBADDBADD]).
    /// @param[in] partialPacketMapPtr Partial Packet Map handle pass to the connection pool.
    /// @throw dw::core::ExceptionBase if the poolSize is larger INVALID_CONNECTION_SLOT[0xBADDBADD]
    explicit SocketConnectionPool(size_t const poolSize, PartialPacketMapPtr partialPacketMapPtr = nullptr);
    ~SocketConnectionPool();

    SocketConnectionPool(SocketConnectionPool&&) = default;
    SocketConnectionPool& operator=(SocketConnectionPool&&) = default;
    SocketConnectionPool(SocketConnectionPool const&)       = default;
    SocketConnectionPool& operator=(SocketConnectionPool const&) = default;

    /// @brief Acquire a free pool slot, or INVALID_CONNECTION_SLOT if no slot is free.
    ///
    /// @return the index of the free slot, or INVALID_CONNECTION_SLOT.
    size_t getFreeSlot() const;

    /// @brief Add a connection to the pool at the given slot.
    ///
    /// @param[in] connection the connection to add.
    /// @param[in] index the slot of the pool where place the connection.
    void addToPool(std::weak_ptr<SocketConnection> const connection, size_t const index);

    /// @brief The function broadcasts the data packets in the input buffer to all valid connections in the SocketConnectionPool.
    /// It iterates through the socket connection statuses in the pool one by one.
    /// If there are valid connections in the pool, it calls the sendPacket() method to send the data packets to those connections.
    ///
    /// @param[in] buffer the buffer to broadcast.
    /// @param[in] bufferSize the size of the buffer(recommend range: 0 ~ mtu size[1500 by default]).
    /// @param[in] timeoutUs the max TX time allowed[us] before arising a connection timeout error(range: 0 ~ SOCK_TIMEOUT_INFINITE[0x0123456789ABCDEF]).
    /// @return the socket status.
    /// Sock_Status::SUCCESS -- broadcast operation succeeded,
    /// means successfully broadcast the data packets to the destination or no valid connections in pool to broadcast.
    /// Sock_Status::FAILURE -- broadcast operation failed,
    /// If the return value of the send() or sendPacket() function is not Sock_Status::SUCCESS or the transmission size does not match the bufferSize,
    /// the function will return Sock_Status::FAILURE, user can re-try to do broadcast.
    Sock_Status broadcast(uint8_t const* const buffer, size_t const bufferSize, int64_t const timeoutUs = SOCK_TIMEOUT_INFINITE);

    /// Counters to record error cases
    struct Metrics
    {
        /// Counter for packet drop
        uint64_t dropCount;
        /// Counter for partial transfer cases
        uint64_t partialTransferCount;
        /// Counter for partial transfer retry failures
        uint64_t partialTransferFailureCount;
        /// Counter for valid partial packet yet to be sent
        uint64_t pendingPartialTransferCount;
    };

    /// @brief Return the metrics struct
    ///
    /// @return metrics collected by connection pool
    const Metrics& getMetrics() const
    {
        return m_metrics;
    }

private:
    /// Timeout value for select method ta broadcast model
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2813925
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2813925
    constexpr static int32_t SOCK_SELECT_AT_BROADCAST_TIMEOUT{200000};
    /// Log interval in (2^n)-1 format
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2813925
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2813925
    constexpr static size_t REPEATED_ERR_LOG_INTERVAL_MINUS1{511U};

    /// Connection Pool to maintain the connections in a fixed vector
    dw::core::VectorFixed<std::weak_ptr<SocketConnection>> m_connectionPool;
    /// pollfd vector to temporarily save all the FDs of connections which are ready to broadcast
    dw::core::VectorFixed<struct ::pollfd> m_pollfdVector;
    /// Metrics counters for error cases
    Metrics m_metrics{};

    /// @brief Return if broadcast operation is possible for the connection pool
    ///
    /// @return return true if broadcast is possible, else return false.
    bool isBroadcastPossible();

    /// @brief Return if the given connection is possible for broadcast
    ///
    /// @param[in] con connection for broadcast validation.
    /// @return return true if broadcast connection is valid, else return false.
    bool isBroadcastConnectionValid(std::shared_ptr<SocketConnection> con);

    /// @brief To send the current packet to a SocketConnection
    ///
    /// This function will Check the partial packet map for any remaining data packets that were not fully transmitted in previous attempts.
    /// If there are any remaining packets, resend them first and update the partial packet map accordingly.
    /// Next, attempt to send the current data packet to the destination through the SocketConnection.
    /// If the size of the transmission is less than the size of the current data buffer, it means that not all data could be sent in this attempt.
    /// In this case, store the remaining data packets in the partial packet map.
    /// These packets will be sent in the next transmission attempt.
    ///
    /// @param[in] connection specific connection instance for sending.
    /// @param[in] buffer data buffer to send
    /// @param[in] bufferSize buffer size to send
    /// @param[in] timeoutUs timeout value for send packet(range: 0 ~ SOCK_TIMEOUT_INFINITE[0x0123456789ABCDEF]).
    /// @param[in] index index of the connection in the pool to reset after end of stream
    /// @return return MessageStatus has 3 members
    /// - Sock_Status -- include the 5 states (Sock_Status::SUCCESS | END_OF_STREAM | TIME_OUT | INVALID_ARGUMENT | BUFFER_FULL)
    /// - transmissionSize -- the size actually send
    /// - rxtime -- Packet arrival kernel timestamp in microsecond
    SocketConnection::MessageStatus sendPacket(dwshared::socketipc::SocketConnection& connection, uint8_t const* const buffer, size_t const bufferSize, int64_t const timeoutUs, size_t index);

    /// @brief To handle partial transfer and save remaining data.
    ///
    /// In the case of a broadcast operation, partial transfer is supported.
    /// This means that only a part of the data in the buffer is transferred to the destination.
    /// If there is remaining data in the buffer that was not transferred, it is saved in the partial packet map.
    /// This map keeps track of the data that needs to be resent in the next broadcast operation.
    ///
    /// @param[in] connection specific connection.
    /// @param[in] buffer data buffer to save
    /// @param[in] bufferSize buffer size to save
    /// @param[in] transmittedSize transmitted data size of the total data buffer, to calculate the remain size
    void savePartialPacket(dwshared::socketipc::SocketConnection& connection, uint8_t const* const buffer, size_t const bufferSize, size_t transmittedSize);

    /// Partial packet map
    PartialPacketMapPtr m_partialPacketMap{};
};

/// @brief SocketServer is designed to initialize and launch a socket server in the C/S (Client/Server) network model.
/// In the server process, the SocketServer maintains a SocketConnectionPool and provides the broadcast method for broadcast communication.
/// In the case of server/client point-to-point communication, the user can obtain a single connection instance by using the accept()/acceptWithoutRegister() method
/// and use that instance for point-to-point communication.
///
/// - For the TCP case:
/// The SocketServer opens a TCP socket, binds it to a specific IP address and port, and listens on that socket.
/// When a client connects to the server, the SocketServer provides the accept method to accept the valid connection and adds it to the connection pool.
/// It then returns an AcceptStatus instance, which encapsulates the connection handle.
/// The user can use this returned connection handle for bidirectional transmission of data packets.
///
/// - For the UDP case:
/// After constructing and initializing the SocketServer, the user should call the createUDPSocket() method to open a UDP socket and bind it to a specific port.
/// Afterward, the SocketServer can follow the aforementioned flow to facilitate point-to-point or broadcast communication.
///
class SocketServer
{
public:
    /// @brief  Constructor : create the server socket and bind it to the specified port.
    ///
    /// @param[in] port The port to listen for client connections.
    /// @param[in] connectionPoolSize The number of connections that this server can accept.
    /// @param[in] reusePort If true, allow multiple sockets to bind to the same port.
    /// @param[in] partialPacketMapPtr If valid, handle partial transfers by resending remaining data in next broadcast.
    /// The SocketConnectionPool's broadcast method supports partial packet transmission due to network bandwidth limitations.
    /// If the size of a data packet to be sent exceeds the maximum transmission size,
    /// the remaining data will be saved in the internal buffer of @b partialPacketMapPtr and resent in the next transmission.
    ///This allows for efficient utilization of network resources.
    /// @throw dw::core::ExceptionBase If socket creation fails

    SocketServer(uint16_t const port, size_t const connectionPoolSize, bool const reusePort = true, SocketConnectionPool::PartialPacketMapPtr partialPacketMapPtr = nullptr);
    /// @brief  Constructor : create the server socket and bind it to the specified port.
    ///
    /// @param[in] port The port to listen for client connections.
    /// @param[in] connectionPoolSize The number of connections that this server can accept.
    /// @param[in] reusePort If true, allow multiple sockets to bind to the same port.
    /// @param[in] sockPrefix A string stand for the socket environment variable.
    /// @param[in] partialPacketMapPtr If valid, handle partial transfers by resending remaining data in next broadcast.
    /// The SocketConnectionPool's broadcast method supports partial packet transmission due to network bandwidth limitations.
    /// If the size of a data packet to be sent exceeds the maximum transmission size,
    /// the remaining data will be saved in the internal buffer of @b partialPacketMapPtr and resent in the next transmission.
    /// This allows for efficient utilization of network resources.
    /// @throw dw::core::ExceptionBase If socket creation fails
    SocketServer(uint16_t const port, size_t const connectionPoolSize, bool const reusePort, SockPrefixStr const& sockPrefix, SocketConnectionPool::PartialPacketMapPtr partialPacketMapPtr = nullptr);
    ~SocketServer() = default;

    SocketServer(SocketServer&&) = default;
    SocketServer& operator=(SocketServer&&) = default;
    SocketServer(SocketServer const&)       = delete;
    SocketServer& operator=(SocketServer const&) = delete;

    /// @brief Store an accepted SocketConnection handle and its socket status.
    /// This class also provides the APIs to access the SocketConnection and its status.
    class AcceptStatus
    {
    public:
        /// @brief Create a new AcceptStatus for a given socket status.
        ///
        /// @param[in] status The socket status to store.
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        explicit AcceptStatus(Sock_Status const status) noexcept
            : m_status(status)
            , m_connection(nullptr)
        {
        }

        /// @brief Retrieve the stored socket status.
        ///
        /// @return The socket status.
        /// include (Sock_Status::FAILURE | TIME_OUT | FAILURE | INVALID_ARGUMENT | BUFFER_FULL | END_OF_STREAM).
        Sock_Status getStatus() const noexcept { return m_status; }

        /// @brief Set a internal socket status as input value.
        ///
        /// @param[in] status The new socket status.
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        void setStatus(Sock_Status const status) noexcept { m_status = status; }

        /// @brief Move set the internal connection handle with the input value.
        ///
        /// @param[in] connection The shared pointer to the connection to bind.
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        void setConnection(std::shared_ptr<SocketConnection> connection) { m_connection = std::move(connection); }

        /// @brief Retrieve the bound connection.
        ///
        /// @return The shared pointer to the bound connection.
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        std::shared_ptr<SocketConnection> getConnection() { return m_connection; }

        /// @brief Reset the connection in the AcceptStatus.
        /// After performing the reset, the connection handle is deconstructed.
        ///
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        void resetConnection() { m_connection.reset(); }

    private:
        /// connection status value
        Sock_Status m_status;
        /// Connection maintained in the AcceptStatus wrapper
        std::shared_ptr<SocketConnection> m_connection;
    };

    /// @brief The function waits for the client socket to connect, with a specified timeout.
    ///
    /// This function waits for a client socket to connect until timeout.
    /// The accept() method allocates a connection slot from the SocketConnectionPool if the accept operation is successful,
    /// the new connection will be added to the pool.
    ///
    /// @sa dwshared::socketipc::SocketConnectionPool
    ///
    /// @param[in] timeoutUs The timeout(unit: us) value after which the server will stop accepting connections(range: 0 ~ SOCK_TIMEOUT_INFINITE[0x0123456789ABCDEF]).
    /// if a new connection is successfully accepted, the timeout event will not be triggered.
    /// @return return AcceptStatus, has 2 members.
    /// - socket connection status(Sock_Status::SUCCESS | TIME_OUT | BUFFER_FULL),
    /// - SocketConnection handle that wraps the socket fd and provides the send/recv etc. methods
    /// @sa SocketServer::AcceptStatus
    AcceptStatus accept(int64_t const timeoutUs);

    /// @brief Get a free connection slot from connection pool
    ///
    /// @return The slot number of available connection.
    /// if failed, it will return the INVALID_CONNECTION_SLOT.
    size_t getConnectionSlot();

    /// @brief Similar to accept, but does not add connection to connection pool.
    ///
    /// This function waits for a client socket to connect until the timeout elapses.
    /// However, unlike accept(), this function only accepts a connection and wraps it inside the AcceptStatus return value to the user,
    /// rather than adding it to the connection pool.
    ///
    /// @param[in] timeoutUs The timeout[us] after which the server will stop accepting connections(range: 0 ~ SOCK_TIMEOUT_INFINITE[0x0123456789ABCDEF]).
    /// @return The status of the accepted connection.
    /// include socket connection status(Sock_Status::SUCCESS | TIME_OUT),
    /// And the connection handle wrapped in the AcceptStatus.
    SocketServer::AcceptStatus acceptWithoutRegister(int64_t const timeoutUs);

    /// @brief Add a connection which returned by accept operation to SocketConnectionPool
    /// It is a wrapper function implemented by SocketConnectionPool::addToPool()
    ///
    /// @sa dwshared::socketipc::SocketConnectionPool
    ///
    /// @param[in] connection The socket connection returned by acceptWithoutRegister.
    /// @param[in] connectionSlot The slot number returned by getConnectionSlot.
    ///
    void addToPool(std::weak_ptr<SocketConnection> const connection, size_t const connectionSlot);

    /// @brief Create an UDP socket on the specific port.
    ///
    /// This function creates an UDP socket and binds it to a specific port.
    ///
    /// @param[in] port The port to use.
    /// @param[in] sockPrefix an optional socket prefix for QNX system.
    /// @return The socket id of the created socket, or -1 if an error occurred.
    static int32_t createUDPSocket(uint16_t const port, SockPrefixStr const& sockPrefix = SockPrefixStr());

    /// @brief Get the Socket address structure of LocalHost
    ///
    /// @param[in] port The port to use.
    /// @return The socket address structure instance.
    /// @sa netinet/in.h POSIX header file to check the struct sockaddr_in
    static sockaddr_in getSocketAddress(uint16_t const port);

    /// @brief Broadcasts the content of the SocketServer to all valid connections in the SocketConnectionPool.
    /// It is a wrapper function implemented by SocketConnectionPool::broadcast()
    ///
    /// @sa dwshared::socketipc::SocketConnectionPool
    ///
    /// @param[in] buffer The buffer to send.
    /// @param[in] bufferSize The size of the buffer(recommend range: 0 ~ mtu size[1500 by default]).
    /// @param[in] timeoutUs The time[us] available to send out all the data(range: 0 ~ SOCK_TIMEOUT_INFINITE[0x0123456789ABCDEF]).
    /// @return The socket status related to this transmission.
    /// Sock_Status::SUCCESS -- broadcast operation succeeded,
    /// means successfully broadcast the data packets to the destination or no valid connections in pool to broadcast.
    /// Sock_Status::FAILURE -- broadcast operation failed,
    /// If the return value of the send() or sendPacket() function is not Sock_Status::SUCCESS or the transmission size does not match the bufferSize,
    /// the function will return Sock_Status::FAILURE, user can re-try to do broadcast.
    Sock_Status broadcast(uint8_t const* const buffer, size_t const bufferSize, int64_t const timeoutUs = SOCK_TIMEOUT_INFINITE);

    /// @brief Get the metrics from connection pool
    ///
    /// @note the metrics is to record all kinds of error Counters such as
    /// dropCount | partialTransferCount | partialTransferFailureCount | pendingPartialTransferCount
    ///
    /// @return metrics from connection pool
    /// which include the Counter for drop | partial transfer cases | partial transfer retry failures | valid partial packet yet to be sent
    // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
    const SocketConnectionPool::Metrics& getConnectionPoolMetrics() const
    {
        return m_connectionPool.getMetrics();
    }

protected:
    /// @brief Get the socket handle stored in the socket server
    ///
    /// @return return the stored raw socket fd.
    // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
    int32_t getServerSocket() const noexcept { return m_serverSocket.get(); }

    /// @brief The implementation flow of the accept function in the socket server.
    ///
    /// The acceptImpl() method performs a poll operation with a timeout, and if it successfully returns,
    /// it calls the POSIX accept() method to establish a connection and return it to the caller.
    /// If accept failed, it will return the failed code wrapped in the AcceptStatus.
    ///
    /// @param[in] timeoutUs The timeout[us] after which the server will stop accepting connections(range: 0 ~ SOCK_TIMEOUT_INFINITE[0x0123456789ABCDEF]).
    /// @return return AcceptStatus, has 2 members.
    /// - socket connection status(Sock_Status::SUCCESS | TIME_OUT),
    /// - SocketConnection handle that wraps the socket fd and provides the send/recv etc. methods
    SocketServer::AcceptStatus acceptImpl(int64_t const timeoutUs);

private:
    /// socket handle maintained by socket server
    UniqueSocketDescriptorHandle m_serverSocket;
    /// socket connections pool maintained by socket server
    /// @sa dwshared::socketipc::SocketConnectionPool
    SocketConnectionPool m_connectionPool;
    /// port of the socket server
    uint16_t m_port; // debug only; no specific use case
    /// prefix value of the socket, it contains the environment variable of specific network interface setting
    SockPrefixStr m_sockPrefix;
};

/// @brief SocketClient is designed to initialize and launch a socket client in the C/S (Client/Server) network model.
///
/// In the client process, the SocketClient maintains a SocketConnectionPool and provides the broadcast method for broadcast communication.
/// For server/client point-to-point communication, the user can obtain a single connection instance by using the connect()/connectUDP()/bindUDP()
/// method and utilize it for point-to-point communication.
///
/// - For the TCP case:
/// The SocketClient uses the connect() method to establish a connection with the TCP server.
/// If successful, it opens a new socket.
/// The SocketConnectionPool allocates a slot, adds the newly established connection associated with the new socket to the connection pool,
/// and returns a ConnectStatus instance that encapsulates the connection handle.
/// The user can utilize this returned connection handle for bidirectional transmission of data packets.
///
/// - For the UDP case:
/// After constructing and initializing the SocketClient, the user should call the connectUDP()/bindUDP() method to open a UDP socket and bind it to a specific port.
/// The SocketConnectionPool allocates a slot, adds the newly established connection associated with the new socket to the connection pool,
/// and returns a ConnectStatus instance that encapsulates the connection handle.
/// Following this, the SocketClient can follow the aforementioned flow to facilitate point-to-point or broadcast communication.
///
class SocketClient
{
public:
    /// @brief Socket Client Constructor
    ///
    /// This function creates a SocketClient instance, initializes the protocol,
    /// sets the internal m_requestNoDelayTcp flag to false by default through the input requestNoDelayTcp parameter,
    /// and creates a connection pool with connectionPoolSize.
    ///
    /// @param[in] connectionPoolSize The size of the connection pool(range 1 ~ INVALID_CONNECTION_SLOT[0xBADDBADD])
    /// @param[in] requestNoDelayTcp It is set as false by default. If true, request to set TCP_NODELAY socket option.
    explicit SocketClient(size_t const connectionPoolSize, bool const requestNoDelayTcp = false);
    ~SocketClient() = default;

    SocketClient(SocketClient&&) = default;
    SocketClient& operator=(SocketClient&&) = default;
    SocketClient(SocketClient const&)       = delete;
    SocketClient& operator=(SocketClient const&) = delete;

    /// @brief Store an connected SocketConnection handle and its socket status.
    /// This class also provides the APIs to access the SocketConnection and its status.
    class ConnectStatus
    {

    public:
        /// @brief Create a ConnectStatus object, set its status as the current input socket status.
        ///
        /// @param[in] status the socket status.
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        explicit ConnectStatus(Sock_Status const status) noexcept
            : m_status(status)
            , m_connection(nullptr)
        {
        }

        /// @brief Return the current socket status.
        ///
        /// @return the socket status.
        /// include (Sock_Status::FAILURE | TIME_OUT | FAILURE | INVALID_ARGUMENT | BUFFER_FULL | END_OF_STREAM)
        Sock_Status getStatus() const { return m_status; }

        /// @brief Set a internal socket status as input value.
        ///
        /// @param[in] status the new socket status
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        void setStatus(Sock_Status const status) { m_status = status; }

        /// @brief Bind a new socket connection to this object.
        ///
        /// @param[in] connection the shared pointer to the new socket connection.
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        void setConnection(std::shared_ptr<SocketConnection> connection) { m_connection = std::move(connection); }

        /// @brief Return the current socket connection.
        ///
        /// @return the shared pointer to the socket connection.
        std::shared_ptr<SocketConnection> getConnection() { return m_connection; }

        /// @brief Reset the socket connection.
        ///
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        void resetConnection() { m_connection.reset(); }

        /// @brief Check if the connection with the server has been established.
        ///
        /// @return return true if connection established, otherwise return false.
        // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
        bool isSuccessConnection() const noexcept { return m_status == Sock_Status::SUCCESS && m_connection != nullptr; }

    private:
        /// connection status value
        Sock_Status m_status;
        /// Connection maintained in the ConnectStatus wrapper
        std::shared_ptr<SocketConnection> m_connection;
    };

    /// @brief Connect to a specific server socket and port, using TCP protocol.
    ///
    /// This function connects to a specific server socket and port using the TCP protocol.
    /// The connect() method retrieves a connection slot from the connection pool and adds the new connection to the pool if the connect operation is successful.
    ///
    /// @param[in] host the server address.
    /// @param[in] port the server port.
    /// @param[in] timeoutUs the timeout[us] after which this client will arise a connection error(range: 0 ~ SOCK_TIMEOUT_INFINITE[0x0123456789ABCDEF]).
    /// @param[in] sockPrefix an optional socket prefix.
    /// @return the connection status for this client socket.
    /// include socket connection status(Sock_Status::SUCCESS | FAILURE | BUFFER_FULL | INVALID_ARGUMENT),
    /// And the connection handle wrapped in the ConnectStatus.
    ConnectStatus connect(char8_t const* const host, uint16_t const port, int64_t const timeoutUs,
                          SockPrefixStr const& sockPrefix = SockPrefixStr());

    /// @brief Connect to a specific server socket and port, using UDP protocol.
    ///
    /// This function connects to a specific server socket and port using the UDP protocol.
    /// The connectUDP() method retrieves a connection slot from the connection pool by calling connectOrBindUDP(),
    /// and adds the new connection to the pool if the connect operation is successful.
    ///
    /// @param[in] host the server address.
    /// @param[in] remotePort the server port.
    /// @param[in] enableBroadcast if true, messages from this client will be broadcast to all connections in the pool.
    /// @param[in] timeoutUs the timeout[us] after which this client will arise a connection error(range: 0 ~ SOCK_TIMEOUT_INFINITE[0x0123456789ABCDEF]).
    /// @param[in] sockPrefix an optional socket prefix.
    /// @return the connection status for this client socket.
    /// include socket connection status(Sock_Status::SUCCESS | FAILURE | BUFFER_FULL | INVALID_ARGUMENT),
    /// And the connection handle wrapped in the ConnectStatus.
    ConnectStatus connectUDP(char8_t const* const host, uint16_t const remotePort, bool const enableBroadcast, int64_t const timeoutUs, SockPrefixStr const& sockPrefix = SockPrefixStr());

    /// @brief Bind the socket to a specific server, using UDP protocol.
    ///
    /// To bind the socket to a specific server using the UDP protocol, you can follow these steps:
    /// 1. Call the bindUDP() method.
    /// 2. Inside the bindUDP() method, retrieve a connection slot from the connection pool by calling connectOrBindUDP().
    /// 3. If the bind operation is successful, add the new connection to the pool.
    /// 4. If the bind operation fails, return the failure code wrapped in the returned ConnectStatus instance.
    ///
    /// @param[in] host the server address.
    /// @param[in] localPort the client port.
    /// @param[in] enableBroadcast if true, messages from this client will be broadcast to all connected client sockets.
    /// @param[in] timeoutUs the timeout[us] after which this client will arise a connection error(range: 0 ~ SOCK_TIMEOUT_INFINITE[0x0123456789ABCDEF]).
    /// @param[in] sockPrefix an optional socket prefix.
    /// @return return ConnectStatus, has 2 members.
    /// - socket connection status(Sock_Status::SUCCESS | FAILURE | BUFFER_FULL | INVALID_ARGUMENT),
    /// - SocketConnection handle that wraps the socket fd and provides the send/recv etc. methods

    ConnectStatus bindUDP(char8_t const* const host, uint16_t const localPort, bool const enableBroadcast, int64_t const timeoutUs, SockPrefixStr const& sockPrefix = SockPrefixStr());

    /// @brief Broadcasts the content of the SocketClient to all valid connections in the SocketConnectionPool.
    /// It is a wrapper function implemented by SocketConnectionPool::broadcast()
    ///
    /// @sa dwshared::socketipc::SocketConnectionPool
    ///
    /// @param[in] buffer the buffer to send.
    /// @param[in] bufferSize the size of the buffer to send(recommend range: 0 ~ mtu size[1500 by default]).
    /// @param[in] timeoutUs the max transmission time allowed[us] before arising a connection timeout error(range: 0 ~ SOCK_TIMEOUT_INFINITE[0x0123456789ABCDEF]).
    /// @return the socket status
    /// Sock_Status::SUCCESS -- broadcast operation succeeded,
    /// means successfully broadcast the data packets to the destination or no valid connections in pool to broadcast.
    /// Sock_Status::FAILURE -- broadcast operation failed,
    /// If the return value of the send() or sendPacket() function is not Sock_Status::SUCCESS or the transmission size does not match the bufferSize,
    /// the function will return Sock_Status::FAILURE, user can re-try to do broadcast.
    Sock_Status broadcast(uint8_t const* const buffer, size_t const bufferSize, int64_t const timeoutUs = SOCK_TIMEOUT_INFINITE);

private:
    /// Protocol of current socket client instance
    uint32_t m_protocol;

    /// Flag to request to set TCP_NODELAY socket option. When this parameter is set to true and m_protocol equals SOCK_STREAM, TCP_NoDelay will be set
    bool m_requestNoDelayTcp;

    /// @brief Enumerate valid states during a connection attempt
    /// During a connection attempt, the SocketClient uses a internal state machine
    /// that goes through the following 5 valid states:
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

    /// Maximal connections to attempt to connect
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2813925
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2813925
    constexpr static size_t MAX_CONN_ATTEMPT{3U};
    /// Connection pool for the socket client
    /// @sa dwshared::socketipc::SocketConnectionPool
    SocketConnectionPool m_connectionPool;

    /// @brief The function is designed to connect or bind to a UDP server using a specified connection method.
    ///
    /// The connectOrBindUDP() method retrieves a connection slot from the connection pool and adds the new connection to the pool if the connect operation is successful.
    /// The function chooses whether to use the connect or bind method depending on the connectOnly flag. If the function returns a Sock_Status::SUCCESS connection status,
    /// it will add the connection to the connection pool. If the function returns a Sock_Status::INTERNAL_ERROR connection status,
    /// it will return the Sock_Status::INTERNAL_ERROR.
    ///
    /// @param[in] connectOnly Flag to decide use connect(true) or bind(false) method
    /// @param[in] host the server address.
    /// @param[in] port the server port for connect, the client port for bind.
    /// @param[in] enableBroadcast if true, messages from this client will be broadcast to all connected client sockets.
    /// @param[in] timeoutUs the timeout[us] after which this client will arise a connection error(range: 0 ~ SOCK_TIMEOUT_INFINITE[0x0123456789ABCDEF]).
    /// @param[in] sockPrefix an optional socket prefix.
    /// @return the connection status
    /// include socket connection status(Sock_Status::SUCCESS | FAILURE | BUFFER_FULL | INVALID_ARGUMENT),
    /// And the connection handle wrapped in the ConnectStatus.
    ConnectStatus connectOrBindUDP(bool const connectOnly,
                                   char8_t const* const host,
                                   uint16_t const port,
                                   bool const enableBroadcast,
                                   int64_t const timeoutUs,
                                   SockPrefixStr const& sockPrefix = SockPrefixStr());

    /// @brief The implementation to establish a connection to a specific server socket and port using the TCP protocol.
    ///
    /// The attemptConnect() method that is designed to connect to a TCP server using a specified connection method and handle the connection states.
    /// Inside the attemptConnect() method, include a states machine to handle the following 5 states(CONNECT | INPROGRESS | RESET | SUCCEEDED | ERROR)
    /// during the connection attempt.
    ///
    /// @param[out] ret the ConnectStatus stored in.
    /// @param[in] gaiResults the server address info stored in.
    /// @param[in] timeoutUs the timeout[us] after which this client will arise a connection error(range: 0 ~ SOCK_TIMEOUT_INFINITE[0x0123456789ABCDEF]).
    /// @param[in] sockPrefix an optional socket prefix.
    void attemptConnect(ConnectStatus& ret,
                        const struct addrinfo* const gaiResults,
                        int64_t const timeoutUs,
                        SockPrefixStr const& sockPrefix = SockPrefixStr());

    /// @brief The implementation to Connect to a specific server socket and port, using UDP protocol.
    ///
    /// The function calls POSIX connect() method to connect to the UDP server using the socket address.
    /// If the connection is successful, the function stores the result in the ret object and returns without performing any further operations.
    /// If the connection is not successful, the function throws an error and sets the ret object's status to Sock_Status::FAILURE.
    ///
    /// @param[out] ret the ConnectStatus stored in.
    /// @param[in] host the host server address.
    /// @param[in] port the server port for connect
    /// @throw dw::core::ExceptionBase If connection is not successful raised by dwshared::socketipc::logStreamError()
    void attemptConnectUDP(ConnectStatus& ret,
                           char8_t const* const host,
                           uint16_t const port);

    /// @brief The function is the implementation to connect or bind to a UDP server using a specified connection method.
    /// The function will choose the connect or bind method depends on the connect flag.
    ///
    /// @param[in] connect Flag to decide use connect(true) or bind(false) method
    /// @param[out] ret the ConnectStatus stored in.
    /// @param[in] gaiResults stored the server address info.
    /// @param[in] enableBroadcast if true, messages from this client will be broadcast to all connected client sockets.
    /// @param[in] sockPrefix an optional socket prefix.
    void attemptUDP(bool const connect,
                    ConnectStatus& ret,
                    const struct addrinfo* const gaiResults,
                    bool const enableBroadcast,
                    SockPrefixStr const& sockPrefix = SockPrefixStr());

    /// @brief The implementation flow for handling an attempt to connect:
    ///
    /// This method handles the flow of attempting a connection until either a timeout or success occurs.
    /// It utilizes a state machine to manage the five states: CONNECT, INPROGRESS, RESET, SUCCEEDED, or ERROR.
    /// The state machine is designed to handle the transition between states during the connection attempt.
    ///
    /// @sa Figure 6 of dwshared_dwsockets__swud for the design of the state machine
    ///
    /// @param[out] ret the ConnectStatus stored in.
    /// Returns ret, with ret.status one of:
    /// Sock_Status::SUCCESS -- connect succeeded
    /// Sock_Status::FAILURE -- connect failed
    /// @param[in] connectionSocket socket handle descriptor for the connection
    /// @param[in] servAddr stored the server address info.
    /// @param[in] timeoutUs the timeout[us] after which this client will arise a connection error(range: 0 ~ SOCK_TIMEOUT_INFINITE[0x0123456789ABCDEF]).
    /// @param[in] sockPrefix an optional socket prefix.
    void handleConnect(ConnectStatus& ret,
                       UniqueSocketDescriptorHandle& connectionSocket,
                       const struct addrinfo* const servAddr,
                       int64_t const timeoutUs,
                       SockPrefixStr const& sockPrefix = SockPrefixStr());

    /// @brief The implementation flow for handling the INPROGRESS state in the connection attempt state machine.
    ///
    /// This method is specifically designed to handle the INPROGRESS state within the connection attempt state machine.
    /// It utilizes the POSIX poll() method to continuously poll the connection socket file descriptor (fd)
    /// until the connection attempt either succeeds or returns with a specific reason, such as a timeout or interruption by the operating system (OS) signal.
    ///
    /// @sa Figure 6 of dwshared_dwsockets__swud for the design of the state machine
    ///
    /// @param[out] ret the ConnectStatus stored in.
    /// Returns ret, with ret.status one of:
    /// Sock_Status::SUCCESS -- connect succeeded
    /// Sock_Status::FAILURE -- connect failed
    /// Sock_Status::TIME_OUT -- connect timeout
    /// @param[in] connectionSocket socket handle descriptor for the connection
    /// @param[in] timeoutUs the timeout[us] after which this client will arise a connection error(range: 0 ~ SOCK_TIMEOUT_INFINITE[0x0123456789ABCDEF]).
    /// @return return true if poll failed or timeout, else return false
    static bool handleConnectInProgress(ConnectStatus& ret,
                                        const UniqueSocketDescriptorHandle& connectionSocket,
                                        int64_t const timeoutUs);

    /// @brief To reset a socket that will be used to connect to a server.
    ///
    /// If a call to the POSIX connect() method returns ECONNREFUSED, it indicates that the TCP connection was refused.
    /// In such cases, the socket needs to be closed and reopened before attempting another connection.
    /// This function closes any existing socket and replaces it with a new one that has been configured for non-blocking I/O.
    /// After calling this method, the user can obtain a new socket connection handle that has been reset.
    /// This handle can then be used to perform bind or connect operations again.
    ///
    /// @param[out] ret the ConnectStatus stored in.
    /// Returns ret, with ret.status one of:
    /// Sock_Status::SUCCESS        -- successful reset
    /// Sock_Status::INTERNAL_ERROR -- system error occurred
    /// @param[out] connectionSocket socket handle descriptor to store the reset socket fd.
    /// @param[in] servAddr the server address info stored in.
    /// @param[in] sockPrefix an optional socket prefix.
    static void resetConnectSocket(ConnectStatus& ret,
                                   UniqueSocketDescriptorHandle& connectionSocket,
                                   const struct addrinfo& servAddr,
                                   SockPrefixStr const& sockPrefix = SockPrefixStr());

    /// @brief Helper function to set socket options
    /// It will enable kernel timestamping by set SO_TIMESTAMPING(or SO_TIMESTAMP for QNX System)
    /// If m_requestNoDelayTcp flag is true, it will enable TCP socket the TCP_NODELAY attribute for no delay model.
    ///
    /// @param[in] socket Socket fd
    inline void setSockOptions(const int32_t socket);

    /// @brief To bind a connect socket and check the assigned port by the operating system (OS)
    ///
    /// Call the POSIX bind() method to bind the socket file descriptor (fd) of the connection with a specific IP Address.
    /// After the bind operation, check whether the assigned port is valid.
    /// This step ensures that the port is available and can be used for the connection.
    ///
    /// @param[out] ret the ConnectStatus stored in.
    /// Returns ret, with ret.status one of:
    /// Sock_Status::SUCCESS        -- OS bound the socket to a valid port
    /// Sock_Status::FAILURE        -- OS bound the socket to the server port in the SOCK_STREAM case
    /// Sock_Status::INTERNAL_ERROR -- system error occurred
    /// @param[in] connectionSocket socket handle descriptor for the connection
    /// @param[in] servAddr the server address info stored in.
    void bindAndCheckConnectSocket(ConnectStatus& ret,
                                   const UniqueSocketDescriptorHandle& connectionSocket,
                                   const struct addrinfo& servAddr);

    /// @brief Reset the connection and bind the new socket to a port until
    /// it has been bound to a port that is different from the one being connected to.
    ///
    /// @param[out] ret the ConnectStatus stored in.
    /// Returns ret, with ret.status one of:
    /// Sock_Status::SUCCESS        -- successful reset and bind
    /// Sock_Status::INTERNAL_ERROR -- system error occurred
    /// @param[in] connectionSocket socket handle descriptor for the connection
    /// @param[in] servAddr the server address info stored in.
    /// @param[in] sockPrefix an optional socket prefix.
    void resetBindAndCheckConnectSocket(ConnectStatus& ret,
                                        UniqueSocketDescriptorHandle& connectionSocket,
                                        const struct addrinfo& servAddr,
                                        SockPrefixStr const& sockPrefix = SockPrefixStr());

    /// @brief Handling the CONNECT state in the connection attempt state machine.
    ///
    /// When the connection attempt state machine starts, it enters the CONNECT state. This function is called to handle the CONNECT state.
    /// In the CONNECT state, the function calls the POSIX connect() method to attempt a connection to the server IP address.
    /// The function then returns the state that determines the next step in the state transition process.
    ///
    /// @sa Figure 6 of dwshared_dwsockets__swud for the design of the state machine
    ///
    /// @param[in] connectionSocket socket handle descriptor for the connection
    /// @param[in] servAddr the server address info stored in.
    /// @return return the ConnectState
    /// - ConnectState::SUCCEEDED -- connect succeeded, goto the SUCCEEDED state
    /// - ConnectState::INPROGRESS -- connect failed with EINPROGRESS error, goto the INPROGRESS state
    /// - ConnectState::RESET -- connect failed with other errors, goto the RESET state
    static ConnectState handleConnectState(const UniqueSocketDescriptorHandle& connectionSocket,
                                           const struct addrinfo& servAddr);

    /// @brief To handle the INPROGRESS state in the connection attempt state machine.
    ///
    /// When the connection attempt state machine enters the INPROGRESS state, it calls the handleConnectInProgress() method to handle the specific flow for this state.
    /// If the handleConnectInProgress() method returns true, indicating a successful connection, the state machine transitions to the SUCCEEDED state in the next step.
    /// If the handleConnectInProgress() method returns a timeout error value (Sock_Status::TIME_OUT), the state machine remains in the INPROGRESS state and continues to call the handleConnectInProgress() method.
    /// If the handleConnectInProgress() method returns any other error value, the state machine transitions to the RESET state in the next step.
    ///
    /// @sa Figure 6 of dwshared_dwsockets__swud for the design of the state machine
    ///
    /// @param[out] ret the ConnectStatus stored in.
    /// Returns ret, with ret.status one of:
    /// Sock_Status::SUCCESS -- connect succeeded
    /// Sock_Status::FAILURE -- connect failed
    /// @param[in] connectionSocket socket handle descriptor for the connection
    /// @param[in] timeoutUs the timeout[us] after which this client will arise a connection error(range: 0 ~ SOCK_TIMEOUT_INFINITE[0x0123456789ABCDEF]).
    /// @return return the ConnectState
    /// - ConnectState::SUCCEEDED -- connect succeeded, goto the SUCCEEDED state
    /// - ConnectState::INPROGRESS -- connect failed with EINPROGRESS error, goto the INPROGRESS state
    /// - ConnectState::RESET -- connect failed with other errors, goto the RESET state
    static ConnectState handleInProgressState(ConnectStatus& ret,
                                              const UniqueSocketDescriptorHandle& connectionSocket,
                                              int64_t const timeoutUs);

    /// @brief To handle the RESET state in the connection attempt state machine.
    ///
    /// In the RESET state handling flow, the first step is to reset the socket file descriptor (fd).
    /// If the reset operation is successful, the state machine sets the state to ConnectState::CONNECT in the next step of the state transition.
    ///
    /// @sa Figure 6 of dwshared_dwsockets__swud for the design of the state machine
    ///
    /// @param[out] ret the ConnectStatus stored in.
    /// @param[in] connectionSocket socket handle descriptor for the connection
    /// @param[in] serv_addr the server address info stored in.
    /// @param[in] sockPrefix an optional socket prefix.
    /// @return return the ConnectState
    /// - ConnectState::CONNECT -- reset succeeded, goto the CONNECT state
    /// - ConnectState::ERROR -- reset failed with some errors, will goto the exit
    ConnectState handleResetState(ConnectStatus& ret,
                                  UniqueSocketDescriptorHandle& connectionSocket,
                                  const struct addrinfo& serv_addr,
                                  SockPrefixStr const& sockPrefix = SockPrefixStr());
};
} // namespace socketipc
} // namespace dwshared

#endif // SOCK_IPC_SOCKETCLIENTSERVER_HPP_
