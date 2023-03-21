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

#include <ssm/SSMIncludes.hpp>
#include <ssm/commServer.hpp>
#include <ssm/commClient.hpp>

namespace SystemStateManager
{

typedef struct _QueueBuff {
    unsigned char buff[MAX_DATA_PKG_SIZE];
} QueueBuffer;// __attribute__ ((aligned (64)));

// TODO: Replace std::map with a fixed Map [https://jirasw.nvidia.com/browse/AVRR-1383]
typedef std::queue<QueueBuffer> MessageQueue;
typedef struct _msgQueueStruct {
    SpinLock queueSpinLock;
    MessageQueue queue;
} MessageQueueStruct;

typedef std::map<std::string, MessageQueueStruct> MessageQueueMap;


class SSMBuffer
{

protected:
    SSMBuffer() { }

    static SSMBuffer* ssmBuffer[MAX_SSM_BUFFERS_ALLOWED];

public:

    // SSMBuffer should not be cloneable.
    SSMBuffer(SSMBuffer &other) = delete;

    // SSMBuffers should not be assignable.
    void operator=(const SSMBuffer &) = delete;

    // Mark the state machine as ready
    bool markAsReady(std::string name);

    // Mark the state machine as ready
    bool isSMReady(std::string name);

    // Displays the list of state machines
    void printStateMachines();

    // Displays the list of clients that are ready
    void printClientsReadyList();

    /**
     * This is the static method that controls the access to the SSMBuffer
     * instance. On the first run, it creates a SSMBuffer object and places it
     * into the static field. On subsequent runs, it returns the client existing
     * object stored in the static field.
     */
    static SSMBuffer *getInstance(int instanceID);

    // Use this to function to reset the buffer in back to back tests
    static void resetBuffer();

    void resetMessageQueue();

    bool addMessageQueue(std::string channelName);

    bool sendData(std::string targetSMName,
                  void *ptr,
                  int size,
                  const void *extraPayload,
                  int sizeOfExtraPayload);

    bool send(std::string targetSMName,
              const void *ptr,
              int size);

    bool readMessages(std::string stateMachineName, StateUpdate &su);

    bool readUserData(std::string stateMachineName, UserDataPktInternal &pkt);

private:
    int maxAllowedQueueSize {MAX_ALLOWED_MSG_QUEUE_SIZE};
    SpinLock messageQueueSpinLock;
    MessageQueueMap messageQueueMap;
    StringSet readyStateMachineSet;
};

}
