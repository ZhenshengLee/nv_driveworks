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

#ifndef DWFRAMEWORK_DWNODES_SELFCALIBRATION_DWSELFCALIBRATIONNODE_DWSELFCALIBRATIONNODE_HPP_
#define DWFRAMEWORK_DWNODES_SELFCALIBRATION_DWSELFCALIBRATIONNODE_DWSELFCALIBRATIONNODE_HPP_

#include <dw/calibration/engine/Engine.h>
#include <dw/calibration/engine/camera/CameraParamsExtra.h>
#include <dwcgf/node/Node.hpp>
#include <dwcgf/parameter/ParameterDescriptor.hpp>
#include <dwcgf/pass/PassDescriptor.hpp>
#include <dwcgf/port/Port.hpp>
#include <dwcgf/port/PortDescriptor.hpp>
#include <dwframework/dwnodes/common/channelpackets/DopplerMotionEstimator.hpp>
#include <dwframework/dwnodes/common/channelpackets/DwRoadCastCommonTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/DwframeworkTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/EgomotionState.hpp>
#include <dwframework/dwnodes/common/channelpackets/FeatureList.hpp>
#include <dwframework/dwnodes/common/channelpackets/IMU.hpp>
#include <dwframework/dwnodes/common/channelpackets/PointCloud.hpp>
#include <dwframework/dwnodes/common/channelpackets/PointCloudProcessingCommonTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/Radar.hpp>
#include <dwframework/dwnodes/common/channelpackets/SelfCalibrationTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/SensorCommonTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/VehicleIOLegacyStructures.hpp>
#include <dwframework/dwnodes/common/channelpackets/VehicleIOValStructures.hpp>

namespace dw
{
namespace framework
{

struct dwSelfCalibrationNodeCameraParams
{
    bool sensorEnabledGlobally;
    bool sensorEnabledCalibrationMask;
    uint32_t sensorIndex;
    uint32_t trackerMaxFeatureCount;
    uint32_t trackerMaxHistorySize;
    uint32_t calibrationMethod;
    uint32_t calibrationSignals; // use uint32_t cause enum bitwise-or is not supported by codegen
    dwCameraModelHandle_t cameraHandle;
    dwCameraProperties cameraProps;
    cudaStream_t cudaStream;
};

struct dwSelfCalibrationNodeRadarParams
{
    bool sensorEnabledGlobally;
    bool sensorEnabledCalibrationMask;
    uint32_t sensorIndex;
    uint32_t maxDisplayPoints;
    dwRadarProperties radarProps;
    cudaStream_t cudaStream;
};

struct dwSelfCalibrationNodeLidarParams
{
    bool sensorEnabledGlobally;
    bool sensorEnabledCalibrationMask;
    uint32_t sensorIndex;
    dwLidarProperties lidarProps;
    cudaStream_t cudaStream;
};

struct dwSelfCalibrationNodeParam
{
    dwRigHandle_t rigHandle;
    bool enforceDependencies;

    uint32_t lidarMaxICPIterations;
    uint32_t accumulatorDownsampleFactor;
    bool lidarHandEyeUsesEgomotion;

    bool calibrateVehicle;
    int32_t calibrateOdometryPropertyRadar;
    dwSelfCalibrationNodeCameraParams cameraParams[SELF_CALIBRATION_NODE_MAX_CAMERAS]; // Camera Port Binding index
    dwSelfCalibrationNodeRadarParams radarParams[SELF_CALIBRATION_NODE_MAX_RADARS];    // Radar Port Binding index
    dwSelfCalibrationNodeLidarParams lidarParams[SELF_CALIBRATION_NODE_MAX_LIDARS];    // Lidar Port Binding index
    uint32_t channelFifoSize;
};

/**
 * @ingroup dwnodes
 */
class dwSelfCalibrationNode : public ExceptionSafeProcessNode
{
public:
    static constexpr char LOG_TAG[] = "dwSelfCalibrationNode";

    static constexpr auto describeInputPorts()
    {
        return describePortCollection(
            DW_DESCRIBE_PORT(dwEgomotionStateHandle_t, "EGOMOTION_STATE_ODO_IMU"_sv),
            DW_DESCRIBE_PORT(dwVehicleIOState, "VEHICLE_IO_STATE"_sv),
            DW_DESCRIBE_PORT(dwVehicleIONonSafetyState, "VEHICLE_IO_NON_SAFETY_STATE"_sv),
            DW_DESCRIBE_PORT_ARRAY(dwFeatureHistoryArray, SELF_CALIBRATION_NODE_MAX_CAMERAS, "CAMERA_FEATURE_DETECTION"_sv, PortBinding::OPTIONAL),
            DW_DESCRIBE_PORT_ARRAY(dwTime_t, SELF_CALIBRATION_NODE_MAX_CAMERAS, "CAMERA_TIMESTAMP"_sv, PortBinding::OPTIONAL),
            DW_DESCRIBE_PORT_ARRAY(dwCameraIntrinsics, SELF_CALIBRATION_NODE_MAX_CAMERAS, "CAMERA_INTRINSICS"_sv, PortBinding::OPTIONAL),
            DW_DESCRIBE_PORT_ARRAY(dwRadarDopplerMotion, SELF_CALIBRATION_NODE_MAX_RADARS, "RADAR_DOPPLER_MOTION"_sv, PortBinding::OPTIONAL),
            DW_DESCRIBE_PORT_ARRAY(dwPointCloud, SELF_CALIBRATION_NODE_MAX_LIDARS, "LIDAR_POINT_CLOUD"_sv, PortBinding::OPTIONAL),
            DW_DESCRIBE_PORT_ARRAY(dwLidarPose, SELF_CALIBRATION_NODE_MAX_LIDARS, "LIDAR_POSE"_sv, PortBinding::OPTIONAL));
    };
    static constexpr auto describeOutputPorts()
    {
        return describePortCollection(
            DW_DESCRIBE_PORT_ARRAY(dwCalibratedExtrinsics, SELF_CALIBRATION_NODE_MAX_CAMERAS, "CAMERA_EXTRINSICS"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT_ARRAY(dwCalibratedExtrinsics, SELF_CALIBRATION_NODE_MAX_RADARS, "RADAR_EXTRINSICS"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT_ARRAY(dwCalibratedExtrinsics, SELF_CALIBRATION_NODE_MAX_LIDARS, "LIDAR_EXTRINSICS"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwCalibratedWheelRadii, "WHEEL_RADII"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwCalibratedSteeringProperties, "FRONT_STEERING_OFFSET"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(void*, "MODULE_HANDLE"_sv, PortBinding::REQUIRED));
    };

    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass("SETUP"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("PROCESS_VIO"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("PROCESS_VEHICLE_IO_NON_SAFETY_STATE"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("PROCESS_CAMERA_GPU_ASYNC"_sv, DW_PROCESSOR_TYPE_GPU),
            describePass("PROCESS_CAMERA_CPU_SYNC"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("PROCESS_RADAR"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("PROCESS_LIDAR_GPU_ASYNC"_sv, DW_PROCESSOR_TYPE_GPU),
            describePass("PROCESS_LIDAR_CPU_SYNC"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("TEARDOWN"_sv, DW_PROCESSOR_TYPE_CPU));
    };

    static std::unique_ptr<dwSelfCalibrationNode> create(ParameterProvider& provider);

    static constexpr auto describeParameters()
    {
        return describeConstructorArguments<dwSelfCalibrationNodeParam, dwContextHandle_t>(
            describeConstructorArgument(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwRigHandle_t,
                    &dwSelfCalibrationNodeParam::rigHandle),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "enforceDependencies"_sv,
                    &dwSelfCalibrationNodeParam::enforceDependencies),
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "calibrateVehicle"_sv,
                    &dwSelfCalibrationNodeParam::calibrateVehicle),
                DW_DESCRIBE_PARAMETER(
                    int32_t,
                    "radarSensorWheelCalibration"_sv,
                    &dwSelfCalibrationNodeParam::calibrateOdometryPropertyRadar),

                DW_DESCRIBE_ABSTRACT_ARRAY_PARAMETER(
                    size_t,
                    "cameraSensorIndices"_sv,
                    SELF_CALIBRATION_NODE_MAX_CAMERAS),
                DW_DESCRIBE_ABSTRACT_ARRAY_PARAMETER(
                    size_t,
                    "cameraSensorStreamIndices"_sv,
                    SELF_CALIBRATION_NODE_MAX_CAMERAS),
                DW_DESCRIBE_ABSTRACT_ARRAY_PARAMETER(
                    bool,
                    "cameraEnabledMask"_sv,
                    SELF_CALIBRATION_NODE_MAX_CAMERAS),
                DW_DESCRIBE_ABSTRACT_ARRAY_PARAMETER(
                    uint32_t,
                    "cameraCalibrationMethod"_sv,
                    SELF_CALIBRATION_NODE_MAX_CAMERAS),
                DW_DESCRIBE_ABSTRACT_ARRAY_PARAMETER(
                    uint32_t,
                    "cameraCalibrationSignals"_sv,
                    SELF_CALIBRATION_NODE_MAX_CAMERAS),

                DW_DESCRIBE_ABSTRACT_ARRAY_PARAMETER(
                    size_t,
                    "radarSensorIndices"_sv,
                    SELF_CALIBRATION_NODE_MAX_RADARS),
                DW_DESCRIBE_ABSTRACT_ARRAY_PARAMETER(
                    size_t,
                    "radarSensorStreamIndices"_sv,
                    SELF_CALIBRATION_NODE_MAX_RADARS),
                DW_DESCRIBE_ABSTRACT_ARRAY_PARAMETER(
                    bool,
                    "radarEnabledMask"_sv,
                    SELF_CALIBRATION_NODE_MAX_RADARS),

                DW_DESCRIBE_ABSTRACT_ARRAY_PARAMETER(
                    size_t,
                    "lidarSensorIndices"_sv,
                    SELF_CALIBRATION_NODE_MAX_LIDARS),
                DW_DESCRIBE_ABSTRACT_ARRAY_PARAMETER(
                    size_t,
                    "lidarSensorStreamIndices"_sv,
                    SELF_CALIBRATION_NODE_MAX_LIDARS),
                DW_DESCRIBE_ABSTRACT_ARRAY_PARAMETER(
                    bool,
                    "lidarEnabledMask"_sv,
                    SELF_CALIBRATION_NODE_MAX_LIDARS),

                DW_DESCRIBE_PARAMETER(
                    uint32_t,
                    "channelFifoSize"_sv,
                    &dwSelfCalibrationNodeParam::channelFifoSize)),
            describeConstructorArgument(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwContextHandle_t)));
    }

    dwSelfCalibrationNode(const dwSelfCalibrationNodeParam& param, const dwContextHandle_t ctx);
};
} // namespace framework
} // namespace dw

#endif //DWFRAMEWORK_DWNODES_SELFCALIBRATION_DWSELFCALIBRATIONNODE_DWSELFCALIBRATIONNODE_HPP_
