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
// SPDX-FileCopyrightText: Copyright (c) 2020-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <ssm/comm.hpp>
#include <ssm/SSMHistogram.hpp>

namespace SystemStateManager
{

class SSMServer
{
public:
    ~SSMServer();
    SSMServer( std::string queueName,
               int port,
               int maxMessageSize = MAX_MESSAGE_SIZE);

    pthread_t *startServer();
    bool setupServer();

    bool writeMessage(StateUpdate &su);
    bool readMessages(StateUpdate &su);

    bool writeUserData(UserDataPktInternal &pkt);
    bool readUserData(UserDataPktInternal &pkt);

    bool writeHeadRouteData(UserDataPktInternal &pkt);
    bool readHeadRouteData(UserDataPktInternal &pkt);

    bool writeSehError(UserDataPktInternal &pkt);
    bool readSehError(UserDataPktInternal &pkt);

    void printHistogram(std::string str = "");

    const std::string& getName() const {
        return queueName;
    }

    inline void setSWCCountToInit(int swcCount) {
        totalInitCountSoFar = 0;
        swcInitCountRequired = swcCount;
    }
    void waitForInitialization();

private:

    void incrementInitCount();
    int getPacket(int index, char *buffer, int packetSize);

    // Error Checking functions
    void processMutexLockError(const int lock);
    void processMutexUNLockError(const int lock);
    void processClockError(const int gettimeErr);
    void processCondTimeWaitError(const int condErr);

private:
    std::string queueName;
    int clientSocketArr[MAX_CONNECTIONS_ALLOWED] {};
    int newSocket {INVALID_SOCKET_FD};
    int listenPort {};
    int maxMessageSize {};
    SSMHistogram msgOverhead {};
    SSMHistogram garbageCollectiongOverhead {};
    struct pollfd pollFDs[MAX_CONNECTIONS_ALLOWED];

    int valread {0};
    int serverFD {INVALID_SOCKET_FD};
    struct sockaddr_in address;
    int addrlen {sizeof(address)};
    int opt {1};

    std::atomic<bool> queueFlag {false};
    std::mutex queueMutex;
    std::queue<StateUpdate> msgQueue;

    std::atomic<bool> userDataQueueFlag {false};
    std::mutex userDataMutex;
    std::queue<UserDataPktInternal> userDataQueue;

    std::mutex sehErrorMutex;
    std::queue<UserDataPktInternal> sehErrorQueue;

    std::atomic<bool> headRouteDataQueueFlag {false};
    std::mutex headRouteDataMutex;
    std::queue<UserDataPktInternal> headRouteDataQueue;

    char *buffer {};

    // Counters to accumate stats
    // DO NOT use them for logic
    uint64_t connectionCounter {0};
    uint64_t closeCounter {0};
    int maxLiveConnectionsRecorded {0};
    int swcInitCountRequired {0};
    int totalInitCountSoFar {0};

    pthread_mutex_t initMutex;
    pthread_cond_t initCond;
};

inline static void *startTCPServer(void *args)
{
    pthread_setname_np(pthread_self(), "ssm::tcpServer");
    SSMServer *ssmServer = static_cast<SSMServer*>(args);
    ssmServer->setupServer();
    return NULL;
}


class SSMServerThread
{
public:
    SSMServerThread(void *ssmServerObject) {
        ssmServer = ssmServerObject;
    }

    void startServer() {
        pthread_create(&psm, NULL, &startTCPServer, ssmServer);
    }

    void stopServer() {
        pthread_cancel(psm);

        // As pthread cancellation is not synchronous with pthread_cancel() call
        // POSIX doesn't guarantee the thread is immediately cancelled right after
        // this pthread_cancel call, although most times it indeed ends very quickly.
        // Better to have a join to guarantee this thread will never have a chance to
        // access the destroyed SSMServer instance.
        auto ret = pthread_join(psm, nullptr);
        if (ret != 0) {
            SSM_ERROR("Error : failed to join canceled SSMServer thread: " + std::to_string(ret) + ", srv: " + static_cast<const SSMServer*>(ssmServer)->getName());
        }
        SSM_DEBUG("SSMServer thread exit, " + static_cast<const SSMServer*>(ssmServer)->getName());
    }

private:
    void *ssmServer {};
    pthread_t psm {};
};

}
