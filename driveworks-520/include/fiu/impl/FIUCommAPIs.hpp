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
// SPDX-FileCopyrightText: Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.
//
// Parser Version: 0.7.1
// SSM Version:    0.8.2
//
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef FIU_COMM_APIS_H
#define FIU_COMM_APIS_H

#include <queue>
#include <pthread.h>
#include <unistd.h>
#include <atomic>
#include <sys/socket.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <map>
#include <netdb.h>
#include <arpa/inet.h>
#include <sys/ioctl.h>
#include <fiu/impl/FIUCommon.hpp>
#include <fiu/impl/ObjectManager.hpp>

namespace fiu2
{

/** @struct Holds a FI command
 *
 *  @var messageID ID of the message that is sent [Future Use]
 *  @var command Command ID that is associated with this message
 *  @var payloadSize Size of the payload that follows this message
 */
struct FIMessage
{
    uint64_t messageID{};
    uint8_t command{};
    size_t payloadSize{};
};

/** @struct Holds a FI command and payload
 *
 *  @var msg FIMessage with the command and payload
 *  @var poolIndex Index of the memory pool in which the memory has been allocated
 *  @var buffer Pointer to the memory buffer
 */
struct FIQueueMessage
{
    FIMessage msg{};
    uint32_t poolIndex{};
    unsigned char* buffer{};
};

/// Total Connections allowed
constexpr int MAX_CONNECTIONS_ALLOWED{16};

/// Default fd for a socket
constexpr int INVALID_SOCKET_FD{-1};

/// Total Fragments to work with
constexpr uint8_t FRAGMENTION_LIMIT{16};

/// Max message size accepted
constexpr uint32_t FI_MAX_MESSAGE_SIZE{2048};

/// Max number of reconnection attempts during connection event
constexpr uint8_t MAX_RECONNECT_ATTEMPTS{3};

/// ID of message related memory allocation buffers
constexpr uint32_t FI_MSG_MEM_ID{199991};

/// Command IDs for messages
constexpr uint8_t FI_MSG_PING{1};
constexpr uint8_t FI_MSG_TEXT{2};

/**
 * Returns a duplicated file descriptior
 *
 * @param[in] fd File descriptor of the socket
 * @return New duplicated file descriptor
 */
int getNonZeroFd(int fd);

/**
 * Write data to an open socket
 *
 * The standard protocol used to send data is as follows:
 *    - Sender first sends a FIMessage structure
 *    - The message structure contains the command ID
 *    - It also contains a payloadSize
 *    - payloadSize > 0 indicates that a payload buffer of the specifed
 *      size is sent following the FIMessage structure
 *    - Payload buffer is transmitted immediate after the FIMessage
 *
 * @param[in] fd File descriptor of the socket to be written to
 * @param[in] command Command that needs to be transmitted
 * @param[in] payload incase data needs to be transmitted along with the command
 * @param[in] payloadSize Size of the payload
 * @return No of rytes Read
 */
bool sendPacket(int& fd, uint8_t command, const void* payload, size_t payloadSize = 0);

/**
 * Reads a fragmented packet
 *
 * API first reads a FIMessage from the socket, figured out if it needs to
 * read a subsequent payload buffer and performs the appropriate action
 *
 * @param[in] fd File descriptor of socket to be read from
 * @param[in] buffer Pointer to the buffer to which the data has to be copied to
 * @return No of bytes read
 */
int getPacket(int& fd, unsigned char* buffer, int packetSize);

/**
 * Debug function that prints the message that it receives
 *
 * @param[in] msg FIMessage that is received
 * @param[in] buffer Data that is transferred through the command
 */
void printMessage(FIMessage& msg, char* buffer);

/**
 * Computes the new port for the middleman comm infrastructure from the
 * external port
 *
 * @param[in] givenPortID Port of the external socket
 */
int computeMiddleManServerPort(int givenPortID);
}

#endif