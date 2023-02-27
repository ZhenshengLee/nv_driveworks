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
// SPDX-FileCopyrightText: Copyright (c) 2018-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_FRAMEWORK_SOFTISP_NODE_HPP_
#define DW_FRAMEWORK_SOFTISP_NODE_HPP_

#include <dw/isp/SoftISP.h>

#include <dwcgf/node/Node.hpp>
#include <dwframework/dwnodes/common/ChannelPacketTypes.hpp>

namespace dw
{
namespace framework
{

// Forward Declarations
class dwSoftISPNodeImpl;

///////////////////////////////////////////////////////////////////////////////////////
using dwSoftISPNodeParams = struct dwSoftISPNodeParams
{
    cudaStream_t stream;
    dwRect cropRegion;
    dwSoftISPDemosaicMethod demosaicMethod;
    dwSoftISPDenoiseMethod denoiseMethod;
    dwTonemapMethod tonemapMethod;
    int32_t processMask;
};

///////////////////////////////////////////////////////////////////////////////////////
/**
 * @ingroup dwnodes
 */
class dwSoftISPNode : public ExceptionSafeProcessNode
{
public:
    static constexpr char LOG_TAG[] = "dwSoftISPNode";

    // Port enumeration.
    enum class PortList : uint8_t
    {
        INPUT_IMAGE           = 0,
        OUTPUT_BAYER_IMAGE    = 1,
        OUTPUT_DEMOSAIC_IMAGE = 2,
        OUTPUT_TONEMAP_IMAGE  = 3,
    };

    // Processing Pass enumeration.
    enum class PassList : uint8_t
    {
        PASS_PROCESS_GPU_ASYNC = 0,
        PASS_COUNT
    };

    // Initialization and destruction
    dwSoftISPNode(const dwSoftISPParams& parameters,
                  const dwSoftISPNodeParams& nodeParams,
                  const dwContextHandle_t ctx);

    // Image Properties
    dwStatus getImageProperties(dwImageProperties* prop, dwSoftISPProcessType type);
};

} // namespace framework
} // namespace dw

#endif //DW_FRAMEWORK_SOFTISP_NODE_HPP_
