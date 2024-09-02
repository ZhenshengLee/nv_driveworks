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
// SPDX-FileCopyrightText: Copyright (c) 2015-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * @file
 * <b>NVIDIA DriveWorks API</b>
 *
 * @b Description: This file is a proxy to include all package declarations. See the
 *                 individual modules for API documentation.
 */

#ifndef DW_DRIVEWORKS_H_
#define DW_DRIVEWORKS_H_

#include <dw/calibration/cameramodel/CameraModel.h>

#include <dw/calibration/engine/camera/CameraParams.h>

#include <dw/calibration/engine/imu/IMUParams.h>

#include <dw/calibration/engine/radar/RadarParams.h>

#include <dw/calibration/engine/Engine.h>

#include <dw/calibration/radarmodel/RadarModel.h>

#include <dw/comms/socketipc/SocketClientServer.h>

#include <dw/control/vehicleio/VehicleIO.h>
#include <dw/control/vehicleio/VehicleIOCapabilities.h>
#include <dw/control/vehicleio/VehicleIOLegacyStructures.h>
#include <dw/control/vehicleio/VehicleIOValStructures.h>
#include <dw/control/vehicleio/drivers/Driver.h>
#include <dw/control/vehicleio/plugins/VehicleIODriver.h>

#include <dw/core/Config.h>
#include <dw/core/ConfigChecks.h>
#include <dw/core/Version.h>
#include <dw/core/base/BasicTypes.h>
#include <dw/core/base/Config.h>
#include <dw/core/base/ErrorDefs.h>
#include <dw/core/base/Exports.h>
#include <dw/core/base/GeoPoints.h>
#include <dw/core/base/GeometricDefs.h>
#include <dw/core/base/GeometricTypes.h>
#include <dw/core/base/MatrixDefs.h>
#include <dw/core/base/MatrixTypes.h>
#include <dw/core/base/Status.h>
#include <dw/core/base/Types.h>
#include <dw/core/base/TypesExtra.h>
#include <dw/core/base/TypesExtraDefs.h>
#include <dw/core/base/Version.h>
#include <dw/core/base/VersionCurrent.h>
#include <dw/core/context/Context.h>
#include <dw/core/context/IOStream.h>
#include <dw/core/context/ObjectExtra.h>
#include <dw/core/health/HealthSignals.h>
#include <dw/core/logger/Logger.h>
#include <dw/core/logger/LoggerDefs.h>
#include <dw/core/memory/Memory.h>
#include <dw/core/platform/GPUProperties.h>
#include <dw/core/signal/MessageMetadata.h>
#include <dw/core/signal/SignalStatus.h>
#include <dw/core/system/NvMedia.h>
#include <dw/core/system/NvMediaExt.h>
#include <dw/core/system/PVA.h>
#include <dw/core/time/Timer.h>

#include <dw/egomotion/base/Egomotion.h>
#include <dw/egomotion/base/EgomotionExtra.h>
#include <dw/egomotion/base/EgomotionState.h>
#include <dw/egomotion/base/EgomotionTypes.h>

#include <dw/egomotion/global/GlobalEgomotion.h>
#include <dw/egomotion/global/GlobalEgomotionSerialization.h>
#include <dw/egomotion/global/GlobalEgomotionTypes.h>

#include <dw/egomotion/radar/DopplerMotionEstimator.h>
#include <dw/egomotion/radar/DopplerMotionEstimatorTypes.h>

#include <dw/egomotion/utils/IMUBiasEstimatorParameters.h>

#include <dw/image/Image.h>

#include <dw/image/ImageSipl.h>

#include <dw/imageprocessing/base/ImageProcessingBase.h>

#include <dw/imageprocessing/featuredetector/FeatureDetector.h>

#include <dw/imageprocessing/features/FeatureDetector.h>
#include <dw/imageprocessing/features/FeatureList.h>

#include <dw/imageprocessing/geometry/imagetransformation/ImageTransformation.h>

#include <dw/imageprocessing/geometry/pose/PnP.h>

#include <dw/imageprocessing/geometry/rectifier/Rectifier.h>

#include <dw/imageprocessing/tracking/featuretracker/FeatureTracker.h>

#include <dw/interop/streamer/ImageStreamer.h>

#include <dw/interop/streamer/TensorStreamer.h>

#include <dw/pointcloudprocessing/pointcloud/PointCloud.h>

#include <dw/rig/CoordinateSystem.h>
#include <dw/rig/Rig.h>
#include <dw/rig/RigTypes.h>
#include <dw/rig/Vehicle.h>

#include <dw/sensors/camera/Camera.h>
#include <dw/sensors/camera/CodecHeaderVideo.h>

#include <dw/sensors/canbus/CAN.h>
#include <dw/sensors/canbus/CANTypes.h>
#include <dw/sensors/canbus/Interpreter.h>
#include <dw/sensors/canbus/VehicleData.h>

#include <dw/sensors/gps/GPS.h>
#include <dw/sensors/gps/GPSFrame.h>

#include <dw/sensors/imu/IMU.h>
#include <dw/sensors/imu/IMUTypes.h>

#include <dw/sensors/radar/Radar.h>
#include <dw/sensors/radar/RadarFullTypes.h>
#include <dw/sensors/radar/RadarSSI.h>
#include <dw/sensors/radar/RadarScan.h>
#include <dw/sensors/radar/RadarTypes.h>

#include <dw/sensors/sensormanager/SensorManager.h>
#include <dw/sensors/sensormanager/SensorManagerConstants.h>

#include <dw/sensors/common/SensorExtras.h>
#include <dw/sensors/common/SensorSerializer.h>
#include <dw/sensors/common/SensorSerializerTypes.h>
#include <dw/sensors/common/SensorStats.h>
#include <dw/sensors/common/SensorTypes.h>
#include <dw/sensors/common/Sensors.h>
#include <dw/sensors/common/SensorsVehicleStateTypes.h>

#include <dw/sensors/codecs/Codec.h>

#include <dw/sensors/codecs/CodecHeader.h>
#include <dw/sensors/codecs/CodecHeaderTypes.h>
#include <dw/sensors/codecs/SensorInfo.h>

#include <dw/sensors/codecs/plugins/CodecHeaderPlugin.h>
#include <dw/sensors/codecs/plugins/DecoderPlugin.h>

#include <dw/sensors/legacy/plugins/SensorCommonPlugin.h>
#include <dw/sensors/legacy/plugins/canbus/CANPlugin.h>
#include <dw/sensors/legacy/plugins/data/DataPlugin.h>
#include <dw/sensors/legacy/plugins/gps/GPSPlugin.h>
#include <dw/sensors/legacy/plugins/imu/IMUPlugin.h>
#include <dw/sensors/legacy/plugins/lidar/LidarDecoder.h>
#include <dw/sensors/legacy/plugins/lidar/LidarPlugin.h>
#include <dw/sensors/legacy/plugins/radar/RadarDecoder.h>
#include <dw/sensors/legacy/plugins/radar/RadarPlugin.h>

#include <dw/calibration/engine/lidar/LidarParams.h>

#include <dw/calibration/engine/stereo/StereoParams.h>

#include <dw/pointcloudprocessing/assembler/PointCloudAssembler.h>

#include <dw/pointcloudprocessing/filter/PointCloudBoxFilter.h>

#include <dw/pointcloudprocessing/icp/PointCloudICP.h>

#include <dw/pointcloudprocessing/lidarpointcloud/LidarPointCloud.h>

#include <dw/pointcloudprocessing/motioncompensator/MotionCompensator.h>

#include <dw/pointcloudprocessing/planeextractor/PointCloudPlaneExtractor.h>

#include <dw/pointcloudprocessing/rangeimagecreator/PointCloudRangeImageCreator.h>

#include <dw/pointcloudprocessing/stitcher/PointCloudStitcher.h>

#include <dw/sensors/lidar/Lidar.h>
#include <dw/sensors/lidar/LidarTypes.h>

#include <dw/pointcloudprocessing/accumulator/PointCloudAccumulator.h>

#include <dw/calibration/engine/vehicle/VehicleParams.h>
#include <dw/calibration/engine/vehicle/VehicleParamsTypes.h>

#include <dw/dnn/DNN.h>
#include <dw/dnn/clusterer/Clusterer.h>
#include <dw/dnn/dataconditioner/DataConditioner.h>
#include <dw/dnn/plugin/DNNPlugin.h>
#include <dw/dnn/tensor/Tensor.h>

#endif // DW_DRIVEWORKS_H_
