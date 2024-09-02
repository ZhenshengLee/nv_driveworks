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
// SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_FRAMEWORK_SEMANTICPARAMETERTYPES_HPP_
#define DW_FRAMEWORK_SEMANTICPARAMETERTYPES_HPP_

namespace dw
{
namespace framework
{
namespace semantic_parameter_types
{

struct BasePath
{
};

struct CameraEnabled
{
};

struct CameraExtrinsic
{
};

struct CameraImagePropertiesNativeProcessed
{
};

struct CameraName
{
};

struct CameraNum
{
};

struct Virtual
{
};

struct StandardCameraNum
{
};

struct CameraRigIdx
{
};

struct CameraUid
{
};

struct Output2Rig
{
};

struct ScalingFactor
{
};

struct ScalingFactorInverse
{
};

struct CameraHandle
{
};

struct Camera2OutputTransform
{
};

struct VehicleSensorName
{
};

struct VehicleSensorEnabled
{
};

struct VehicleSensorHandle
{
};

struct GpsName
{
};

struct GpsHandle
{
};

struct ImuName
{
};

struct ImuHandle
{
};

struct UltrasonicName
{
};

struct UltrasonicHandle
{
};

struct LidarEnabled
{
};

struct LidarExtrinsics
{
};

struct LidarName
{
};

struct LidarRigIdx
{
};

struct LidarHandle
{
};

struct LidarNum
{
};

struct LidarEnabledNum
{
};

struct RadarNum
{
};

struct RadarEnabled
{
};

struct RadarExtrinsic
{
};

struct RadarName
{
};

struct RadarRigIdx
{
};

struct RadarHandle
{
};

struct RadarProperties
{
};

struct DwDataPath
{
};

// Semantic for original camera image size
struct OrigImageWidth
{
};

struct OrigImageHeight
{
};

struct ImageWidth
{
};

struct ImageHeight
{
};

struct Sensor2Rig
{
};

struct SensorRigIdx
{
};

struct CameraModelTransform
{
};

struct IspTransformationParams
{
};

struct IspTransformationSpecs
{
};

// Semantic for 2MP Image
struct ImageWidth2Mp
{
};

struct ImageHeight2Mp
{
};

struct CameraIntrinsics2Mp
{
};

struct CameraModelTransform2Mp
{
};

struct Camera2OutputTransform2Mp
{
};

struct IspTransformationParams2Mp
{
};

struct IspTransformationSpecs2Mp
{
};

struct RigLayoutName
{
};

struct RigFileName
{
};

struct RigOutputFileName
{
};

struct CalibrationOutputFileName
{
};

struct CalibrationOverlayFileName
{
};

struct VinOverlayFileName
{
};

struct ConfigOverlayFileName
{
};

struct ConfigOverlayName
{
};

struct ImuEnabled
{
};

struct ImuRigIdx
{
};

struct ConfirmLaneChange
{
};

struct CudlaStream
{
};

struct CupvaStream
{
};

struct NvMediaLdc
{
};

struct NvMedia2D
{
};

struct SemanticSensorName
{
};

// Semantic For Seh Info
struct CameraInstanceId
{
};

struct RadarInstanceId
{
};

struct LidarInstanceId
{
};

struct ImuInstanceId
{
};

struct GpsInstanceId
{
};

struct VehicleSensorInstanceId
{
};

struct UltrasonicInstanceId
{
};

} // namespace semantic_parameter_types
} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_SEMANTICPARAMETERTYPES_HPP_
