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
// SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWFRAMEWORK_DWNODES_SENSORS_COMMON_DWSENSORNODE_HPP_
#define DWFRAMEWORK_DWNODES_SENSORS_COMMON_DWSENSORNODE_HPP_

#include <dw/core/container/BaseString.hpp>
#include <dw/rig/Rig.h>
#include <dw/sensors/Sensors.h>
#include <dwcgf/node/Node.hpp>
#include <dwcgf/pass/Pass.hpp>
#include <dwcgf/port/Port.hpp>
#include <dwframework/dwnodes/common/SensorCommonTypes.hpp>

namespace dw
{
namespace framework
{

class dwSensorNode
{
public:
    static std::unique_ptr<Node> create(const dwSensorParams& params,
                                        const dwSALHandle_t sal,
                                        const dwContextHandle_t ctx);
    static std::unique_ptr<Node> create(const char* sensorName,
                                        dwConstRigHandle_t rigHandle,
                                        const dwSALHandle_t sal,
                                        const dwContextHandle_t ctx,
                                        const FixedString<32>& frameSkipMask);
};
}
}
#endif // DWFRAMEWORK_DWNODES_SENSORS_COMMON_DWSENSORNODE_HPP_
