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

#ifndef FIU_SERVER_THREAD_H
#define FIU_SERVER_THREAD_H
#include <fiu/impl/FIUServer.hpp>

namespace fiu2
{

constexpr int FIU_HOSTNAME_STRING_SIZE{256};

/**
 * Thread function that runs the external server
 *
 * @param[in] *args Caller sends the FIUServerThread pointer
 */
void* runExternalServer(void* args);

/**
 * Thread function that runs the middle man server
 *
 * @param[in] *args Caller sends the middle man server pointer
 */
inline static void* runMiddleManServer(void* args)
{
    pthread_setname_np(pthread_self(), "fiu::tcpServer");
    FIUServerPtr fiServer = *static_cast<FIUServerPtr*>(args);
    fiServer->iterateOverClientData();
    return NULL;
}

class FIUServerThread
{
public:
    /**
     * Destructor
     */
    ~FIUServerThread()
    {
        stopServer();
    }

    /**
     * Constructor
     *
     * @param[in] fiuServerObject Reference to the fiu server object
     * @param[in] serverPort Port for the middle man server
     */
    FIUServerThread(FIUServerPtr& fiuServerObject, int serverPort)
        : m_externalServerPort(serverPort), m_middleManServerPtr(fiuServerObject)
    {
    }

    /**
     * Gets the Middle man server pointer
     *
     * @return pointer to the middle man
     */
    FIUServerPtr getFIMiddleManServerPtr()
    {
        return m_middleManServerPtr;
    }

    /**
     * Gets the port for the external server
     *
     * @return external server port
     */
    int getExternalServerPort()
    {
        return m_externalServerPort;
    }

    /**
     * Starts the middle man FI server
     */
    void startFIServer()
    {
        if (pthread_create(&m_middleManServerThread, NULL, &runMiddleManServer, static_cast<void*>(&m_middleManServerPtr)))
        {
            FI_ERROR("Unable to start the Middle Man Server thread");
        }
    }

    /**
     * Starts the external server to which external clients can connect to
     */
    void startExternalServer()
    {
        if (pthread_create(&m_externalServerThread, NULL, &runExternalServer, static_cast<void*>(this)))
        {
            FI_ERROR("Unable to start the FIServer external thread");
        }
    }

    /**
     * Stops all the server threads
     */
    void stopServer()
    {
        pthread_cancel(m_middleManServerThread);
        pthread_cancel(m_externalServerThread);

        // As pthread cancellation is not synchronous with pthread_cancel() call
        // POSIX doesn't guarantee the thread is immediately cancelled right after
        // this pthread_cancel call, although most times it indeed ends very quickly.
        // Better to have a join to guarantee this thread will never have a chance to
        // access the destroyed FIServer instances.
        auto ret = pthread_join(m_middleManServerThread, nullptr);
        if (ret != 0)
        {
            FI_ERROR("Error : failed to join canceled MiddleManServer thread: ", ret);
        }

        ret = pthread_join(m_externalServerThread, nullptr);
        if (ret != 0)
        {
            FI_ERROR("Error : failed to join canceled External server thread: ", ret);
        }

        FI_DEBUG("FIServerThread exit ");
    }

private:
    /// Pointer to the middle man server
    FIUServerPtr m_middleManServerPtr{};

    /// Thread object that runs the middle man server
    pthread_t m_middleManServerThread{};

    /// Thread object that runs the external server
    pthread_t m_externalServerThread{};

    /// External server port
    int m_externalServerPort{INVALID_SOCKET_FD};
};
};

#endif
