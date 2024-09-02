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
// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWFRAMEWORK_DWNODES_COMMON_SENSORCOMMONIMPL_HPP_
#define DWFRAMEWORK_DWNODES_COMMON_SENSORCOMMONIMPL_HPP_

#include <dwshared/dwfoundation/dw/core/container/BaseString.hpp>

// forward declare
typedef struct dwRigObject const* dwConstRigHandle_t;
typedef struct dwSALObject* dwSALHandle_t;
typedef struct dwSensorObject* dwSensorHandle_t;

namespace dw
{
namespace framework
{

class SensorCommonImpl
{
    using FixedString1024 = dw::core::FixedString<1024>;

public:
    static FixedString1024 adjustPath(const char* paramString, uint32_t sensorId, const char* toFind, dwConstRigHandle_t& rig);
    // TODO(csketch): should not use string
    static dwSensorHandle_t createSensor(std::string sensorName, dwConstRigHandle_t rig, dwSALHandle_t sal);
};

} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_COMMON_SENSORCOMMONIMPL_HPP_
