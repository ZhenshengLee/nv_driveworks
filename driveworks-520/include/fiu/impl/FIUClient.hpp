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
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef FIU_CLIENT_H
#define FIU_CLIENT_H

#include <fiu/impl/FIUCommAPIs.hpp>

namespace fiu2
{

class FIUClient
{
public:
    /**
     * Destructor
     *
     * Used to close sockets mostly
     */
    ~FIUClient();

    /**
     * Constructor
     *
     * @param[in] objectManager Reference to the object manager for memory mgmt
     * @param[in] name Name of the client
     * @param[in] port Port that the client needs to connect & listen to
     * @param[in] ip IP Address of the remote server the client needs to connect to
     */
    FIUClient(ObjectManager& objectManager, const char* name, uint16_t port, const char* ip);

    /**
     * Transmits data to the remote server
     *
     * @param[in] command Command ID of the data packet
     * @param[in] payload Buffer that needs to be transmitted
     * @param[in] sizeOfPayload Size of the buffer that needs to be transmitted
     * @return true if data is transmitted successfully
     */
    bool sendData(uint8_t command, const void* payload = NULL, size_t sizeOfPayload = 0);

    /**
     * Receives data from the remote server
     *
     * @param[out] fiMsg Reference to the FIQueueMessage object
     * @return true if data has been received successfully
     */
    bool recvData(FIQueueMessage& fiMsg);

    /**
     * Get the name of the client
     *
     * @return returns the name of the client
     */
    std::string getName()
    {
        return m_clientName;
    }

    /**
     * Checks if the client is connected and is ready to transmit data
     *
     * @return returns true if client is ready to transmit data
     */
    bool checkIfClientIsReady();

    /**
     * Establishes a connection with the remote server
     *
     * @return returns true if client is connected with the remote system
     */
    bool establishConnection();

private:
    /// Pointer to the object manager
    ObjectManager& m_objectManagerPtr;

    /// Pointer to the client name
    const char* m_clientName{};

    /// Port to which the client has to connect and listen to
    int m_listenPort{INVALID_SOCKET_FD};

    /// IP Address to which the client has to connect to
    const char* m_ipAddress{};

    /// File descriptor to the connected socket
    int m_sockfd{INVALID_SOCKET_FD};

    /// Holds if the client is ready
    bool m_isClientReady{false};

    /// Variable that holds the server info
    struct sockaddr_in m_servaddr
    {
    };
};

using FIUClientPtr = std::shared_ptr<FIUClient>;
};

#endif
