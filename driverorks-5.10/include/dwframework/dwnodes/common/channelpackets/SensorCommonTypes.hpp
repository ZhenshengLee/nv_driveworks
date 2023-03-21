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

#ifndef DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_SENSORCOMMONTYPES_HPP_
#define DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_SENSORCOMMONTYPES_HPP_

#include <dwframework/dwnodes/common/ChannelPacketImpl.hpp>
#include <dwframework/dwnodes/common/DwavEnums.hpp>
#include <dwframework/dwnodes/common/SensorCommonTypes.hpp>
#include <dwframework/dwnodes/common/factories/impl/ChannelNvSciPacketImpl.hpp>

DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION(dw::framework::SensorServiceNodeRawData, size_t, DW_SENSOR_SERVICE_RAW_DATA);
DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION_SIMPLE(dw::framework::dwFeatureNccScores, DW_FEATURE_NCC_SCORES);
DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION_SIMPLE(dw::framework::dwLatency, DW_LATENCY);
DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION_SIMPLE(dw::framework::dwLidarPacketsArray, DW_LIDAR_PACKETS_ARRAY);
DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION_SIMPLE(dw::framework::dwSensorNodeRawData, DW_SENSOR_NODE_RAW_DATA);
DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION_SIMPLE(dw::framework::dwTraceNodeData, DW_TRACE_NODE_DATA);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dw::framework::dwCameraIntrinsics);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dw::framework::dwCodecMetadata);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dw::framework::dwSensorNodeProperties);

#endif // DWFRAMEWORK_DWNODES_COMMON_CHANNELPACKETS_SENSORCOMMONTYPES_HPP_
