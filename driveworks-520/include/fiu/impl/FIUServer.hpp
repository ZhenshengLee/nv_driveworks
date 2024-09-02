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
// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.
//
// FI Tool Version:    0.2.0
//
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef FIU_SERVER_H
#define FIU_SERVER_H
#include <fiu/impl/FIUCommAPIs.hpp>

namespace fiu2
{

///////// TODO: Replace this with Fixed containers //////////
using FIQueue = std::queue<FIQueueMessage>;
/////////////////////////////////////////////////////////////

class FIUServer
{
public:
    /**
     * Destructor
     */
    ~FIUServer();

    /**
     * Constructor
     *
     * @param[in] objectManager Reference to the object manager for memory mgmt
     * @param[in] port Port that the client needs to connect & listen to
     * @param[in] shutdownFlag Reference to the caller's shutdown flag
     * @param[in] maxMsgSize Maximum message size allowed
     */
    FIUServer(ObjectManager& objectManagerPtr,
              int port,
              std::atomic<bool>& shutdownFlag,
              int maxMsgSize = FI_MAX_MESSAGE_SIZE);

    /**
     * Sets up the server
     *
     * @return True if the server has been setup correctly
     */
    bool setupServer();

    /**
     * Broadcasts message to all the connected faults
     *
     * @param[in] command FI command that needs to be broadcast
     * @param[in] payload Pointer to the buffer that needs to be transmitted
     * @param[in] payloadSize Size of the buffer that needs to be transmitted
     * @return True if data has been transfered
     */
    bool broadcast(uint8_t command, const void* payload, size_t payloadSize);

    /**
     * Transmits data to the remote server
     *
     * @param[in] index Index of the client to which that data needs to be sent
     * @param[in] command FI command that needs to be broadcast
     * @param[in] payload Pointer to the buffer that needs to be transmitted
     * @param[in] payloadSize Size of the buffer that needs to be transmitted
     * @return true if data is transmitted successfully
     */
    bool send(int index, uint8_t command, const void* payload, size_t payloadSize);

    /**
     * Iterates over all the connections using the select api
     */
    void iterateOverClientData();

    /**
     * Fetches data from the socket and copies into the buffer
     *
     * @param[out] &fiMsg Reference to the FIQueueMessage object
     * @return True if data has been fetched
     */
    bool getNextPacket(FIQueueMessage& fiMsg);

    /**
     * Returns the size of the message queue
     *
     * @return messageQueue size
     */
    uint32_t getMessageQueueSize()
    {
        //////////// CRITICAL SECTION /////////////
        FILock lg(m_messageQueueLock);
        return m_messageQueue.size();
        ///////////////////////////////////////////
    }

    /**
     * Returns the reference to the shutdown flag
     *
     * @return Reference to the shutdown flag
     */
    std::atomic<bool>& getShutdownFlag()
    {
        return m_shutdownFlag;
    }

    /**
     * Returns maximum message size
     *
     * @return Maximum message size
     */
    int getMaxMessageSize()
    {
        return m_maxMessageSize;
    }

    /**
     * Returns if the server has been setup correctly
     *
     * @return true if the server is valid
     */
    bool isServerValid()
    {
        return m_isServerValid;
    }

private:
    /// Messages are queued in this data structure
    FIQueue m_messageQueue{};

    /// Lock used to protect m_messageQueue from race conditions
    SpinLock m_messageQueueLock{};

    /// Server socket port
    int m_listeningPort{};

    /// Maximum message size allowed
    int m_maxMessageSize{};

    /// Array of all the file descriptors for the select api
    struct pollfd m_pollFDs[MAX_CONNECTIONS_ALLOWED]{};

    /// Total live connected identified so far
    uint32_t m_maxLiveConnectionsRecorded{0};

    /// Object manager used to reserve and release memory buffers
    ObjectManager& m_objectManagerObj;

    /// File descriptor for the server
    int m_serverFD{INVALID_SOCKET_FD};

    /// Socket address object for the server object
    struct sockaddr_in m_address
    {
    };

    /// Total live connections
    uint32_t m_totalLiveConnections{1};

    /// Reference to the shutdownFlag object
    std::atomic<bool>& m_shutdownFlag;

    /// True if the server is valid; else false
    bool m_isServerValid{true};

    /// Variable holds the size of the ip address
    const int m_addrlen{sizeof(m_address)};

    /**
     * Clean up and ignore the message if there is too much fragmentation
     *
     * @param[in] fragmentationCount Total number of fragments that the message was broken down through TCP
     */
    void dealWithTooMuchFragmentation(uint8_t fragmentationCount);

    /**
     * Accepts a new TCP connection
     *
     * @param[in] newSocket Socket ID of the new socket connection
     * @param[in] sockOpt Socket option varialbe used across the socket read function
     */
    void acceptNewConnection(int& newSocket, int& sockOpt);
};

using FIUServerPtr = std::shared_ptr<FIUServer>;
};

#endif
