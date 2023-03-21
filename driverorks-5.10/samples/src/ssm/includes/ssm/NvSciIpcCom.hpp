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
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#pragma once
#include <ssm/comm.hpp>

#include <string.h>
#include <iostream>
#include <memory>

#include <nvscisync.h>
#include <nvscierror.h>

namespace SystemStateManager
{

#define MAX_ENDPOINT 100
#define NVSCIIPC_MAX_ENDPOINT_NAME 64U
#define DEFAULT_ITERATIONS 128UL

typedef struct EndPoint
{
    std::string name{};
    NvSciIpcEndpoint h;
    struct NvSciIpcEndpointInfo info;
    int32_t fd;
} EndPoint;

class NvSciIpcCom
{
public:
    NvSciIpcCom(std::string name);
    ~NvSciIpcCom();
    NvSciError initResources();
    void releaseResources();
    NvSciError waitEvent(uint32_t mask, uint32_t* events);
    bool sendData(StateUpdate *up, int size, const void *extraPayload, int sizeOfExtraPayload);
    bool write(void *buffer, int32_t size, int32_t *bytesWritten);
    bool readData();
    bool read(void *buffer, int sizeToRead, int *sizeRead);
    bool getStopValue() { return m_Stop; }
    bool connect();
    void disConnect();

private:
    bool m_Stop{false};
    EndPoint endPoint{};
    std::unique_ptr<uint8_t[]> payloadBuffer {};
};

}