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
// SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_SELFCALIBRATIONTYPES_HPP_
#define DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_SELFCALIBRATIONTYPES_HPP_

#include <dwcgf/channel/ChannelPacketTypes.hpp>
#include <dwframework/dwnodes/common/SelfCalibrationTypes.hpp>
#include <dw/calibration/engine/common/CalibrationTypesExtra.h>
#include <dw/calibration/engine/common/SelfCalibrationCameraDiagnostics.h>
#include <dw/calibration/engine/common/SelfCalibrationIMUDiagnostics.h>
#include <dw/calibration/engine/common/SelfCalibrationRadarDiagnostics.h>

DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dw::framework::dwCalibratedExtrinsics);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dwCalibratedIMUIntrinsics);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dw::framework::dwCalibratedSteeringProperties);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dw::framework::dwCalibratedWheelRadii);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dwSelfCalibrationCameraDiagnostics);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dwSelfCalibrationImuDiagnostics);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dwSelfCalibrationRadarDiagnostics);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dwCameraTwoViewTransformation);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dwCalibratedSuspensionStateProperties);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dwSelfCalibrationSuspensionStateDiagnostics);

#endif // DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_SELFCALIBRATIONTYPES_HPP_