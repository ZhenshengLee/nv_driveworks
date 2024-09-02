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

#ifndef SOCK_IPC_WRAPPER_HPP_
#define SOCK_IPC_WRAPPER_HPP_

#include <sys/time.h>

namespace dwshared
{
namespace socketipc
{

/// Return structure for socket functions sendto(), send() and recv()
struct SendRecvReturnStruct
{
    /// Number of bytes received
    ssize_t bytes;
    /// Error code
    int32_t error;
    /// RX time in microseconds from system time epoch
    int64_t rxtime;
};

/// Return structure for socket function connect()
struct ConnectReturnStruct
{
    /// Status code
    int32_t status;
    /// Error code
    int32_t error;
};

// Class to wrap socket calls, introduced to properly manage errno.
// coverity[autosar_cpp14_a0_1_6_violation] FP: nvbugs/3866413
class SocketWrapper
{
public:
    /**
     * @brief Send the content of the buffer through this socket connection to a specific host/port  using the POSIX sendTo() method.
     *
     * This method can be used for connectionless socket communication. When calling this method, the user should set the target address and port.
     *
     * @param[in] sockId the socket ID to use.
     * @param[in] buffer the buffer to send.
     * @param[in] bufferSize the buffer size.
     * @param[in] flags the flags to use to sendTo in socket.h.
     * @param[in] addr the address of the target.
     * @param[in] sockLen The size of the address pointed to by addr.
     * @return the SendRecvReturnStruct containing the number of bytes received, error codes and RX time in microseconds from system time epoch.
     */
    // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
    static SendRecvReturnStruct sendTo(int32_t const sockId, void const* const buffer, size_t const bufferSize, int32_t const flags,
                                       const struct sockaddr* const addr, socklen_t const sockLen)
    {
        SendRecvReturnStruct retStruct{};

        errno = dw::core::util::ZERO;
        ssize_t const bytes{::sendto(sockId, buffer, bufferSize, flags, addr, sockLen)};

        retStruct.bytes = bytes;
        retStruct.error = errno;

        return retStruct;
    }

    /**
     * @brief Send the content of the buffer through this socket connection to the default host address/port using the POSIX send() method.
     *
     * The user must establish a connection before calling this method.
     *
     * @param[in] sockId the socket ID to use.
     * @param[in] buffer the buffer to send.
     * @param[in] bufferSize the buffer size.
     * @param[in] flags the flags to use to send in socket.h.
     * @return the SendRecvReturnStruct containing the number of bytes received, error codes and RX time in microseconds from system time epoch.
     */
    // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
    static SendRecvReturnStruct send(int32_t const sockId, void const* const buffer, size_t const bufferSize, int32_t const flags)
    {
        SendRecvReturnStruct retStruct{};

        errno = dw::core::util::ZERO;
        ssize_t const bytes{::send(sockId, buffer, bufferSize, flags)};

        retStruct.bytes = bytes;
        retStruct.error = errno;

        return retStruct;
    }

    /**
     * @brief Read the data into the buffer and return the number of bytes read if the operation is successful.
     * The user must establish a connection before calling this method.
     *
     * @param[in] sockId the socket ID to use.
     * @param[in] buffer the buffer to store received data.
     * @param[in] bufferSize the size of the buffer.
     * @param[in] flags the flags to use to recvmsg in socket.h.
     * @return the SendRecvReturnStruct containing the number of bytes received, error codes and RX time in microseconds from system time epoch.
     */
    // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
    static SendRecvReturnStruct recv(int32_t const sockId, void* const buffer, size_t const bufferSize, int32_t const flags)
    {
        SendRecvReturnStruct retStruct{};
        iovec data{buffer, bufferSize};

        // TODO(dwplc): Violation originated from QNX certified headers usage (Supporting permit SWE-DRC-518-SWSADP)
        // coverity[cert_int31_c_violation] Deviation Record: SWE-DRC-518-SWSADR
        uint8_t auxData[CMSG_SPACE(sizeof(struct timeval))] = {0};

        msghdr msg{};
        msg.msg_iov        = &data;
        msg.msg_iovlen     = dw::core::util::ONE_U;
        msg.msg_control    = auxData;
        msg.msg_controllen = static_cast<decltype(msg.msg_controllen)>(sizeof(auxData));
        msg.msg_flags      = dw::core::util::ZERO;

        errno = dw::core::util::ZERO;
        ssize_t const bytes{::recvmsg(sockId, &msg, flags)};

        // parse message header for timestamp
        // coverity[cert_pos39_c_violation] FP: nvbugs/3732079
        if ((bytes > dw::core::util::ZERO_U) && (msg.msg_controllen > dw::core::util::ZERO_U))
        {
            retStruct.rxtime = getTimeFromHeader(msg);
        }

        retStruct.bytes = bytes;
        retStruct.error = errno;

        return retStruct;
    }

    /**
     * @brief Return the time from a msghdr struct.
     *
     * @param[in] msgHdr the message header to use to retreive the time.
     * @return the time from the message header struct.
     */
    static int64_t getTimeFromHeader(msghdr& msgHdr)
    {
        int64_t result{-1};

        // coverity[autosar_cpp14_a5_2_2_violation] RFD Accepted: TID-1864
        // coverity[autosar_cpp14_m5_2_9_violation] Deviation Record: SWE-DRC-518-SWSADR
        cmsghdr* cmsg{CMSG_FIRSTHDR(&msgHdr)}; // clang-tidy NOLINTNEXTLINE(cppcoreguidelines-pro-type-cstyle-cast)
        while (cmsg != nullptr)
        {
#ifndef __QNX__
            // coverity[autosar_cpp14_a5_2_2_violation] Deviation Record: SWE-DRC-518-SWSADR
            if (cmsg->cmsg_type == SCM_TIMESTAMPING && cmsg->cmsg_len == CMSG_LEN(sizeof(timeval)))
#else
            // TODO(dwplc): Violation originated from QNX certified headers usage (Supporting permit SWE-DRC-518-SWSADP)
            // coverity[autosar_cpp14_m5_0_21_violation] Deviation Record: SWE-DRC-518-SWSADR
            // coverity[autosar_cpp14_m5_0_3_violation] Deviation Record: SWE-DRC-518-SWSADR
            // coverity[cert_int31_c_violation] Deviation Record: SWE-DRC-518-SWSADR
            if (cmsg->cmsg_type == SCM_TIMESTAMP && cmsg->cmsg_len == CMSG_LEN(sizeof(timeval)))
#endif
            {
                // coverity[autosar_cpp14_a5_2_4_violation] RFD Accepted: TID-1371
                // coverity[autosar_cpp14_m5_0_15_violation] Deviation Record: SWE-DRC-518-SWSADR
                // coverity[autosar_cpp14_a5_2_2_violation] RFD Accepted: TID-1864
                // coverity[cert_int31_c_violation] Deviation Record: SWE-DRC-518-SWSADR
                const timeval& tv{*(reinterpret_cast<timeval*>(CMSG_DATA(cmsg)))}; // clang-tidy NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                result = dw::core::safeAdd(dw::core::safeMul(tv.tv_sec, 1'000'000L).value(), tv.tv_usec).value();
            }
            // coverity[autosar_cpp14_a5_2_2_violation] Deviation Record: SWE-DRC-518-SWSADR
            // coverity[autosar_cpp14_m5_0_15_violation] Deviation Record: SWE-DRC-518-SWSADR
            // coverity[cert_int31_c_violation] Deviation Record: SWE-DRC-518-SWSADR
            // coverity[autosar_cpp14_m5_2_9_violation] Deviation Record: SWE-DRC-518-SWSADR
            cmsg = CMSG_NXTHDR(&msgHdr, cmsg);
        }
        return result;
    }

    /**
     * @brief Open a connection on a socket to a specified address.
     *
     * @param[in] sockId the socket ID to use.
     * @param[in] addr the address to connect to.
     * @param[in] sockLen the socket length.
     * @return A ConnectReturnStruct containing status and error information.
     **/
    // coverity[autosar_cpp14_m0_1_10_violation] RFD Accepted: TID-1835
    static ConnectReturnStruct connect(int32_t const sockId, const struct sockaddr* const addr, socklen_t const sockLen)
    {
        ConnectReturnStruct retStruct{};

        errno            = dw::core::util::ZERO;
        retStruct.status = ::connect(sockId, addr, sockLen);
        retStruct.error  = errno;

        return retStruct;
    }

}; // SocketErrorHandler

} // namespace socketipc
} // namespace dwshared

#endif
