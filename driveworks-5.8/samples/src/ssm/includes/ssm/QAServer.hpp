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

namespace SystemStateManager
{

class QAServer
{
public:
    ~QAServer();
    QAServer(int port);

    bool setupServer();
    void getMessage(std::string &message);
    bool sendMessage(std::string &message);
    bool queryHeadStateMachine(std::string message, std::string &reply);

private:
    int new_socket {0};
    int listenPort;

    int valread {0};
    int server_fd {0};
    int clientFD {0};
    struct sockaddr_in address;
    int addrlen {sizeof(address)};
    int opt {1};

    std::atomic<bool> queueFlag {false};
    std::mutex queueMutex;
    std::queue<StateUpdate> msgQueue;
    char *buffer {};

};

}
