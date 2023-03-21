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

#ifndef DW_FRAMEWORK_SEMANTICPARAMETERTYPES_HPP_
#define DW_FRAMEWORK_SEMANTICPARAMETERTYPES_HPP_

namespace dw
{
namespace framework
{
namespace semantic_parameter_types
{

// coverity[autosar_cpp14_a0_1_6_violation]
struct BasePath
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct CameraEnabled
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct CameraExtrinsic
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct CameraName
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct CameraNum
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct Virtual
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct StandardCameraNum
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct CameraRigIdx
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct CameraUid
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct Output2Rig
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct ScalingFactor
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct ScalingFactorInverse
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct CameraHandle
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct MLMCFTrunkEnabled
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct Camera2OutputTransform
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct VehicleSensorName
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct VehicleSensorEnabled
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct VehicleSensorHandle
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct GpsName
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct GpsHandle
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct ImuName
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct ImuHandle
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct UltrasonicName
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct UltrasonicHandle
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct LidarEnabled
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct LidarExtrinsics
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct LidarName
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct LidarRigIdx
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct LidarHandle
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct LidarNum
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct LidarEnabledNum
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct RadarNum
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct RadarEnabled
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct RadarExtrinsic
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct RadarName
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct RadarRigIdx
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct RadarHandle
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct DwDataPath
{
};

// Semantic for original camera image size
// coverity[autosar_cpp14_a0_1_6_violation]
struct OrigImageWidth
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct OrigImageHeight
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct ImageWidth
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct ImageHeight
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct Sensor2Rig
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct SensorRigIdx
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct CameraModelTransform
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct IspTransformationParams
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct IspTransformationSpecs
{
};

// Semantic for 2MP Image
// coverity[autosar_cpp14_a0_1_6_violation]
struct ImageWidth2Mp
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct ImageHeight2Mp
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct CameraIntrinsics2Mp
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct CameraModelTransform2Mp
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct Camera2OutputTransform2Mp
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct IspTransformationParams2Mp
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct IspTransformationSpecs2Mp
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct RigLayoutName
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct MLMCFPrecisionRecallFilepath
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct RigFileName
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct RigOutputFileName
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct CalibrationOutputFileName
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct CalibrationOverlayFileName
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct VinOverlayFileName
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct ConfigOverlayFileName
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct ConfigOverlayName
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct ImuEnabled
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct ImuRigIdx
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct VdcPath
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct ConfirmLaneChange
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct CudlaStream
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct CupvaStream
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct MLMCFTrunkOutputHandleDriving
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct MLMCFTrunkOutputHandleParking
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct MLMCFParamsDriving
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct MLMCFParamsParking
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct MLMCFTrunkOutputPropertiesDriving
{
};

// coverity[autosar_cpp14_a0_1_6_violation]
struct MLMCFTrunkOutputPropertiesParking
{
};
} // namespace semantic_parameter_types
} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_SEMANTICPARAMETERTYPES_HPP_
