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
// SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_FRAMEWORK_CONNECTION_HPP_
#define DW_FRAMEWORK_CONNECTION_HPP_

#include <dw/core/base/Types.h>

namespace dw
{
namespace framework
{

const uint64_t CONNECTION_DISCONNECTED = static_cast<uint64_t>(-1);

using Connection = struct Connection
{
    uint64_t srcNodeId;
    uint64_t dstNodeId;
    uint8_t srcPortId;
    uint8_t dstPortId;
    uint64_t channelId;
    static Connection createInputOnly(uint64_t dstNodeId,
                                      uint8_t dstPortId,
                                      uint64_t channelId)
    {
        Connection c{};
        c.srcNodeId = CONNECTION_DISCONNECTED;
        c.srcPortId = 0;
        c.dstNodeId = dstNodeId;
        c.dstPortId = dstPortId;
        c.channelId = channelId;
        return c;
    }

    static Connection createOutputOnly(uint64_t srcNodeId,
                                       uint8_t srcPortId,
                                       uint64_t channelId)
    {
        Connection c{};
        c.dstNodeId = CONNECTION_DISCONNECTED;
        c.dstPortId = 0;
        c.srcNodeId = srcNodeId;
        c.srcPortId = srcPortId;
        c.channelId = channelId;
        return c;
    }
};
}
}
#endif // DW_FRAMEWORK_CONNECTION_HPP_
