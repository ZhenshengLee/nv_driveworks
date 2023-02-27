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

class SSMClient
{
public:
    ~SSMClient();
    SSMClient(std::string name, int port, std::string ip, int maxSize);
    bool send(void *ptr,
              int size,
              const void *extraPayload = NULL,
              int sizeOfExtraPayload = 0);
    std::string getName() { return queueName; }
    bool checkIfClientIsReady();
    void printHistogram();
    bool establishConnection();

private:
    SSMHistogram msgOverhead;
    std::string queueName;
    int maxAllowedSize;
    int listenPort;
    std::string ipAddress;
    bool isClientReady {false};
    int sockfd {INVALID_SOCKET_FD};
    struct sockaddr_in servaddr {};
    unsigned char *payloadBuffer {};
};

typedef std::shared_ptr<SSMClient> SSMClientPtr;

}
