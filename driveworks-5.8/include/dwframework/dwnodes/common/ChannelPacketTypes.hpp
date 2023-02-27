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
// SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_FRAMEWORK_CHANNEL_DW_PACKET_TYPES_HPP_
#define DW_FRAMEWORK_CHANNEL_DW_PACKET_TYPES_HPP_

#include <typeinfo>
#include <cstddef>

#include <dw/core/base/Types.h>
#include <dw/sensors/imu/IMU.h>
#include <dw/sensors/gps/GPS.h>
#include <dw/sensors/lidar/Lidar.h>
#include <dwcgf/Types.hpp>
#include <dw/calibration/cameramodel/CameraModel.h>
#include <dwframework/dwnodes/common/GlobalEgomotionCommonTypes.hpp>
#include <dwframework/dwnodes/common/SelfCalibrationTypes.hpp>
#include <dwframework/dwnodes/common/PointCloudProcessingCommonTypes.hpp>
#include <dwframework/dwnodes/common/SensorCommonTypes.hpp>
#include <dw/image/Image.h>
#include <dw/interop/streamer/ImageStreamer.h>
#include <dw/imageprocessing/features/FeatureList.h>
#include <dw/imageprocessing/filtering/Pyramid.h>
#include <dw/sensors/lidar/Lidar.h>
#include <dw/sensors/radar/Radar.h>
#include <dw/sensors/ultrasonic/Ultrasonic.h>
#include <dw/pointcloudprocessing/pointcloud/PointCloud.h>
#include <dw/egomotion/EgomotionState.h>
#include <dw/egomotion/radar/DopplerMotionEstimator.h>
#include <dwframework/dwnodes/common/PointCloudProcessingCommonTypes.hpp>
#include <dwframework/dwnodes/common/DwRoadCastCommonTypes.hpp>
#include <dw/sensors/Codec.h>
#include <dw/sensors/camera/CodecHeaderVideo.h>

namespace dw
{
namespace framework
{

struct dwFeatureNccScores
{
    float32_t* d_nccScores;
    uint32_t size;
};

struct dwSensorNodeRawData
{
    uint8_t* data;
    size_t size;
};

// might have other field added to it, keep it separate from
// dwSensorNodeRawData for now
struct SensorServiceNodeRawData
{
    uint8_t* data;
    size_t size;
};

struct dwLatency
{
    uint64_t senderTime;
    size_t size;
    uint8_t* data;
};

using dwLidarPacketsArray = struct dwLidarPacketsArray
{
    dwLidarDecodedPacket* packets;
    dwLidarPointXYZI* pointsXYZIArray;
    dwLidarPointRTHI* pointsRTHIArray;
    size_t maxPacketsPerSpin;
    size_t maxPointsPerPacket;
    size_t packetSize;
};

using dwTraceNodeData = struct dwTraceNodeData
{
    size_t maxDataSize;
    size_t dataSize;
    uint8_t* data;
};

using dwCodecMetadata = struct dwCodecMetadata
{
    dwCodecType codecType;
    dwCodecConfigVideo configVideo;
};

} // namespace framework
} // namespace dw

// wraps around base framework macro but allows us to avoid boiler plating of dw::framework::DWChannelPacketTypeID
#define DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION(DATA_TYPE, SPECIMEN_TYPE, ENUM_SPEC) \
    DWFRAMEWORK_DECLARE_PACKET_TYPE_RELATION(DATA_TYPE, SPECIMEN_TYPE, dw::framework::DWChannelPacketTypeID::ENUM_SPEC)

// same as above, but covers simple case where the specimen for data type is data type itself
#define DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION_SIMPLE(DATA_TYPE, ENUM_SPEC) \
    DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION(DATA_TYPE, DATA_TYPE, ENUM_SPEC)

DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION_SIMPLE(dw::framework::dwFeatureNccScores, DW_FEATURE_NCC_SCORES);
DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION_SIMPLE(dwFeatureArray, DW_FEATURE_ARRAY);
DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION_SIMPLE(dw::framework::dwLatency, DW_LATENCY);
DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION_SIMPLE(dwFeatureHistoryArray, DW_FEATURE_HISTORY_ARRAY);
DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION_SIMPLE(dw::framework::dwSensorNodeRawData, DW_SENSOR_NODE_RAW_DATA);
DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION_SIMPLE(dwRadarScan, DW_RADAR_SCAN);
DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION_SIMPLE(dwLidarDecodedPacket, DW_LIDAR_DECODE_PACKET);
DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION_SIMPLE(dwPointCloud, DW_POINT_CLOUD);
DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION_SIMPLE(dw::framework::dwLidarPacketsArray, DW_LIDAR_PACKETS_ARRAY);
DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION_SIMPLE(dw::framework::dwTraceNodeData, DW_TRACE_NODE_DATA);
DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION(dwImageHandle_t, dwImageProperties, DW_IMAGE_HANDLE);
DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION(dwPyramidImage, dwPyramidImageProperties, DW_PYRAMID_IMAGE);
DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION(dwEgomotionStateHandle_t, dwEgomotionStateParams, DW_EGOMOTION_STATE_HANDLE);
DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION(dwCodecPacket, size_t, DW_CODEC_PACKET);
DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION(dw::framework::SensorServiceNodeRawData, size_t, DW_SENSOR_SERVICE_RAW_DATA);

DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dwIMUFrame);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dwIMUFrameNew);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dwVehicleIOState);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dw::framework::CalibratedWheelRadii);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dwCalibratedIMUIntrinsics);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(void*);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dwRadarDopplerMotion);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dwGPSFrame);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dwGPSFrameNew);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dwLidarPose);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dwCalibratedExtrinsics);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dwCalibratedSteeringProperties);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dwRoadCastNodeCalibrationDataArray);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dwGlobalEgomotionState);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dwUltrasonicEnvelope);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dwUltrasonicGroup);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dwUltrasonicMountingPositions);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dwVector2ui);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dwVehicleIOCommand);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dwVehicleIOMiscCommand);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dwVehicleIOSafetyState);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dwVehicleIONonSafetyState);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dwVehicleIOActuationFeedback);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dwCodecMetadata);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dwSensorTsAndID);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dwFThetaCameraConfigNew);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dwRadarProperties);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dw::framework::dwSensorNodeProperties);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(dwCameraIntrinsics);

#endif // DW_FRAMEWORK_CHANNEL_DW_PACKET_TYPES_HPP_
