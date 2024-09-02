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

#ifndef FIU_CLIENT_THREAD_H
#define FIU_CLIENT_THREAD_H
#include <fiu/impl/FIUClient.hpp>

namespace fiu2
{

class FIUClientThread;

/**
 * Starts FI client object
 *
 * @param[in] *args Caller sends the FIUClientThread pointer
 * @return NULL for thread terminaltion
 */
inline static void* startTCPClient(void* args);

class FIUClientThread
{
public:
    /**
     * Destructor
     */
    ~FIUClientThread()
    {
        stopClient();
    }

    /**
     * Constructor
     *
     * @param[in] nm Name of the fi client
     * @param[in] port Port the client needs to connect to the middle man server
     * @param[in] ip IP address of server
     * @param[in] m_shutdownFlag Reference to the caller's shutdown flag
     */
    FIUClientThread(const char* nm, int port, const char* ip, std::atomic<bool>& sFlag)
        : m_name(nm), m_ipAddress(ip), m_serverConnectionPort(port), m_shutdownFlag(sFlag)
    {
    }

    /**
     * Starts the FI Client thread
     */
    void startFIClient()
    {
        if (!m_clientPtr)
        {
            m_clientPtr = fiu2::FIUClientPtr(new fiu2::FIUClient(*getFIUManagerSingleton().getObjectManager(), m_name, m_serverConnectionPort, m_ipAddress));
            if (pthread_create(&m_clientThread, NULL, &startTCPClient, reinterpret_cast<void*>(this)))
            {
                FI_ERROR("Unable to start the client: ", m_name);
            }
        }
        else
        {
            FI_ERROR("Attempting to create a duplicate client: ", m_name);
        }
    }

    /**
     * Stops the client
     */
    void stopClient()
    {
        pthread_cancel(m_clientThread);

        // As pthread cancellation is not synchronous with pthread_cancel() call
        // POSIX doesn't guarantee the thread is immediately cancelled right after
        // this pthread_cancel call, although most times it indeed ends very quickly.
        // Better to have a join to guarantee this thread will never have a chance to
        // access the destroyed SSMServer instance.
        auto ret = pthread_join(m_clientThread, nullptr);
        if (ret != 0)
        {
            FI_ERROR("Error : failed to join canceled SSMServer thread: ", ret);
        }
        FI_DEBUG("SSMServer thread exit ");
    }

    /**
     * Returns the message count present in the queue
     *
     * @return total message count
     */
    uint64_t getMessageCount()
    {
        return m_totalMessageCount;
    }

    /**
     * Check if the client is connected to the server
     *
     * @return true if the client is connected to the server
     */
    bool isClientConnected()
    {
        return m_isClientReady;
    }

    /**
     * API that listens to messages
     */
    void processMessages()
    {
        bool isValidMessage{false};
        fiu2::FIQueueMessage fiMsg{};
        nlohmann::json jsonStr;

        // Iterate over a blocking read until shutdown is called
        while (!m_shutdownFlag && m_isClientReady)
        {
            isValidMessage = false;

            if (m_clientPtr->recvData(fiMsg))
            {
                m_totalMessageCount++;
                if (fiMsg.msg.command == FI_MSG_TEXT && fiMsg.msg.payloadSize > 0)
                {
                    FI_PERF("FIUClientThread Received Message!!!");
                    printMessage(fiMsg.msg, (char*)fiMsg.buffer);
                    try
                    {
                        jsonStr        = nlohmann::json::parse(fiMsg.buffer);
                        isValidMessage = true;
                    }
                    catch (const std::exception& exc)
                    {
                        FI_ERROR(exc.what());
                    }

                    // Release memory
                    getFIUManagerSingleton().getObjectManager()->releaseMemory(fiMsg.buffer, FI_MSG_MEM_ID, fiMsg.poolIndex);
                    if (isValidMessage)
                    {
                        if (!getFIUManagerSingleton().parseJSONCommands(jsonStr))
                        {
                            FI_ERROR("Unabled to process JSONString", jsonStr);
                        }
                        else
                        {
                            FI_PERF("#####################################");
                            FI_PERF("FIU Command Processed Successfully!!!");
                            FI_PERF("#####################################");
                        }
                    }
                }
            }
        }
    }

    /**
     * API that listens to messages
     */
    void listenToMessages()
    {
        uint32_t countAttempts{0};

        // First connect with the server
        while (!m_shutdownFlag && !m_isClientReady)
        {
            m_isClientReady = m_clientPtr->checkIfClientIsReady();
            if (m_isClientReady)
            {
                break;
            }
            countAttempts++;
            usleep(100);
            if (countAttempts > 1000)
            {
                FI_ERROR("Error : Unabled to connect with FIServer process: ", m_ipAddress, m_serverConnectionPort, m_name);
                countAttempts = 0;
            }
        }
        processMessages();
    }

private:
    /// Name of the FI Client
    const char* m_name{};

    /// IP Address of the remote server that needs to connect
    const char* m_ipAddress{};

    /// Flag that indicates if the client is ready
    bool m_isClientReady{false};

    /// Total messages this client has received so far
    uint64_t m_totalMessageCount{};

    /// Port of the middle man server
    int m_serverConnectionPort{INVALID_SOCKET_FD};

    /// Reference to the shutdownFlag object
    std::atomic<bool>& m_shutdownFlag;

    /// Pointer to the client object
    FIUClientPtr m_clientPtr{};

    /// Thread that manages the client thread
    pthread_t m_clientThread{};
};

/**
 * Thread function that runs the fi client
 *
 * @param[in] *args Caller sends the middle man server pointer
 */
void* startTCPClient(void* args)
{
    pthread_setname_np(pthread_self(), "fiu::tcpClient");
    FIUClientThread* fiClientThread = reinterpret_cast<FIUClientThread*>(args);
    fiClientThread->listenToMessages();
    return NULL;
}

using FIUClientThreadPtr = std::shared_ptr<FIUClientThread>;
};

#endif
