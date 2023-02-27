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

#ifndef SOCK_IPC_WRAPPER_HPP_
#define SOCK_IPC_WRAPPER_HPP_

namespace dwshared
{
namespace socketipc
{

// Return structure for socket functions ::sendto, ::send and ::recv
struct SendRecvReturnStruct
{
    ssize_t bytes;
    int32_t error;
    int64_t rxtime;
};

// Return structure for socket function ::connect
struct ConnectReturnStruct
{
    int32_t status;
    int32_t error;
};

// Class to wrap socket calls, introduced to properly manage errno.
class SocketWrapper
{
public:
    static SendRecvReturnStruct sendTo(int32_t const sockId, void const* const buffer, size_t const bufferSize, int32_t const flags,
                                       const struct sockaddr* const addr, socklen_t const sockLen)
    {
        SendRecvReturnStruct retStruct{};

        errno               = 0;
        ssize_t const bytes = ::sendto(sockId, buffer, bufferSize, flags, addr, sockLen);

        retStruct.bytes = bytes;
        retStruct.error = errno;

        return retStruct;
    }

    static SendRecvReturnStruct send(int32_t const sockId, void const* const buffer, size_t const bufferSize, int32_t const flags)
    {
        SendRecvReturnStruct retStruct{};

        errno               = 0;
        ssize_t const bytes = ::send(sockId, buffer, bufferSize, flags);

        retStruct.bytes = bytes;
        retStruct.error = errno;

        return retStruct;
    }

    static SendRecvReturnStruct recv(int32_t const sockId, void* const buffer, size_t const bufferSize, int32_t const flags)
    {
        SendRecvReturnStruct retStruct{};
        iovec data{buffer, bufferSize};

        // TODO(dwplc): Violation originated from QNX certified headers usage (Supporting permit SWE-DRC-518-SWSADP)
        // coverity[autosar_cpp14_m5_0_21_violation]
        // coverity[cert_int31_c_violation]
        uint8_t auxData[CMSG_SPACE(sizeof(struct timeval))] = {0};

        msghdr msg{};
        msg.msg_iov        = &data;
        msg.msg_iovlen     = 1;
        msg.msg_control    = auxData;
        msg.msg_controllen = static_cast<decltype(msg.msg_controllen)>(sizeof(auxData));
        msg.msg_flags      = 0;

        errno               = 0;
        ssize_t const bytes = ::recvmsg(sockId, &msg, flags);

        // parse message header for timestamp
        // coverity[cert_pos39_c_violation] FP: 3732079
        if (bytes > 0 && msg.msg_controllen > 0)
        {
            retStruct.rxtime = getTimeFromHeader(msg);
        }

        retStruct.bytes = bytes;
        retStruct.error = errno;

        return retStruct;
    }

    // coverity[autosar_cpp14_m7_1_2_violation]
    static int64_t getTimeFromHeader(msghdr& msgHdr)
    {
        int64_t result = -1;

        // coverity[autosar_cpp14_a5_2_2_violation]
        // clang-tidy NOLINTNEXTLINE(cppcoreguidelines-pro-type-cstyle-cast)
        cmsghdr* cmsg = CMSG_FIRSTHDR(&msgHdr);
        while (cmsg != nullptr)
        {
#ifndef __QNX__
            if (cmsg->cmsg_type == SCM_TIMESTAMPING && cmsg->cmsg_len == CMSG_LEN(sizeof(timeval)))
#else
            // TODO(dwplc): Violation originated from QNX certified headers usage (Supporting permit SWE-DRC-518-SWSADP)
            // coverity[autosar_cpp14_m5_0_21_violation]
            // coverity[autosar_cpp14_m5_0_3_violation]
            // coverity[cert_int31_c_violation]
            if (cmsg->cmsg_type == SCM_TIMESTAMP && cmsg->cmsg_len == CMSG_LEN(sizeof(timeval)))
#endif
            {
                // coverity[autosar_cpp14_a5_2_2_violation]
                //coverity[autosar_cpp14_m5_0_21_violation]
                // coverity[autosar_cpp14_a3_9_1_violation]
                // coverity[cert_int31_c_violation]
                // clang-tidy NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                const timeval& tv = *(reinterpret_cast<timeval*>(CMSG_DATA(cmsg)));
                result            = dw::core::safeAdd(dw::core::safeMul(tv.tv_sec, 1'000'000L).value(), tv.tv_usec).value();
            }
            // coverity[autosar_cpp14_a5_2_2_violation]
            // coverity[autosar_cpp14_a3_9_1_violation]
            // coverity[autosar_cpp14_m5_0_20_violation]
            //coverity[autosar_cpp14_m5_0_21_violation]
            // coverity[autosar_cpp14_m5_0_3_violation]
            // coverity[cert_int31_c_violation]
            cmsg = CMSG_NXTHDR(&msgHdr, cmsg);
        }
        return result;
    }

    static ConnectReturnStruct connect(int32_t const sockId, const struct sockaddr* const addr, socklen_t const sockLen)
    {
        ConnectReturnStruct retStruct{};

        errno            = 0;
        retStruct.status = ::connect(sockId, addr, sockLen);
        retStruct.error  = errno;

        return retStruct;
    }

}; // SocketErrorHandler

} // socketipc
} // dwshared

#endif
