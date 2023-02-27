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

#ifndef DW_FRAMEWORK_FEATURE_TRACKER_NODE_H_
#define DW_FRAMEWORK_FEATURE_TRACKER_NODE_H_

#include <dwcgf/node/Node.hpp>
#include <dwcgf/parameter/ParameterDescriptor.hpp>
#include <dwcgf/parameter/SemanticParameterTypes.hpp>
#include <dwcgf/pass/PassDescriptor.hpp>
#include <dwcgf/port/Port.hpp>
#include <dwcgf/port/PortDescriptor.hpp>
/* Need to include the appropriate ChannelPacketTypes.hpp since port initialization requires
   the parameter_trait overrides. Otherwise, it will be considered as a packet of generic type. */
#include <dwframework/dwnodes/common/ChannelPacketTypes.hpp>

#include <dw/imageprocessing/featuredetector/FeatureDetector.h>
#include <dw/imageprocessing/features/FeatureList.h>
#include <dw/imageprocessing/tracking/FeatureTracker_processpipeline.h>

namespace dw
{
namespace framework
{

using dwFeatureTrackerNodeParams = struct dwFeatureTrackerNodeParams
{
    bool initForCamera;
    dwCameraModelHandle_t intrinsic;
    dwTransformation3f extrinsic;

    dwFeature2DTrackerConfig dwTrackerParams;

    cudaStream_t cudaStream;
};

/**
 * @ingroup dwnodes
 */
class dwFeatureTrackerNode : public ExceptionSafeProcessNode
{
public:
    static constexpr char LOG_TAG[] = "dwFeatureTrackerNode";

    static constexpr auto describeInputPorts()
    {
        return describePortCollection(
            DW_DESCRIBE_PORT(dwPyramidImage, "CURRENT_PYRAMID"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwFeatureArray, "DETECTED_FEATURES"_sv, PortBinding::REQUIRED));
    };
    static constexpr auto describeOutputPorts()
    {
        return describePortCollection(
            DW_DESCRIBE_PORT(dwFeatureHistoryArray, "FEATURE_HISTORY_CPU"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwFeatureHistoryArray, "FEATURE_HISTORY_GPU"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwFeatureNccScores, "NCC_SCORES"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(void*, "MODULE_HANDLE"_sv, PortBinding::REQUIRED));
    };

    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass("SETUP"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("TRACK_FEATURE"_sv, DW_PROCESSOR_TYPE_GPU),
            describePass("COMPACT_FEATURE"_sv, DW_PROCESSOR_TYPE_GPU),
            describePass("COPY"_sv, DW_PROCESSOR_TYPE_GPU),
            describePass("WAIT_EVENT"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("TEARDOWN"_sv, DW_PROCESSOR_TYPE_CPU));
    };

    static std::unique_ptr<dwFeatureTrackerNode> create(ParameterProvider& provider);

    static constexpr auto describeParameters()
    {
        return describeConstructorArguments<dwFeatureTrackerNodeParams, dwContextHandle_t>(
            describeConstructorArgument(
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "initForCamera"_sv,
                    &dwFeatureTrackerNodeParams::initForCamera),
                DW_DESCRIBE_INDEX_PARAMETER_WITH_SEMANTIC(
                    dwCameraModelHandle_t,
                    semantic_parameter_types::CameraIntrinsicsYuv,
                    "cameraIndex"_sv,
                    &dwFeatureTrackerNodeParams::intrinsic),
                DW_DESCRIBE_INDEX_PARAMETER_WITH_SEMANTIC(
                    dwTransformation3f,
                    semantic_parameter_types::CameraExtrinsic,
                    "cameraIndex"_sv,
                    &dwFeatureTrackerNodeParams::extrinsic),
                DW_DESCRIBE_PARAMETER(
                    dwFeature2DTrackerAlgorithm,
                    "featureTrackerAlgorithm"_sv,
                    &dwFeatureTrackerNodeParams::dwTrackerParams, &dwFeature2DTrackerConfig::algorithm),
                DW_DESCRIBE_PARAMETER(
                    dwFeature2DDetectorType,
                    "featureDetectorType"_sv,
                    &dwFeatureTrackerNodeParams::dwTrackerParams, &dwFeature2DTrackerConfig::detectorType),
                DW_DESCRIBE_PARAMETER(
                    uint32_t,
                    "maxFeatureCount"_sv,
                    &dwFeatureTrackerNodeParams::dwTrackerParams, &dwFeature2DTrackerConfig::maxFeatureCount),
                DW_DESCRIBE_PARAMETER(
                    uint32_t,
                    "pyramidLevelCount"_sv,
                    &dwFeatureTrackerNodeParams::dwTrackerParams, &dwFeature2DTrackerConfig::pyramidLevelCount),
                DW_DESCRIBE_INDEX_PARAMETER_WITH_SEMANTIC(
                    uint32_t,
                    semantic_parameter_types::ImageWidthYuv,
                    "cameraIndex"_sv,
                    &dwFeatureTrackerNodeParams::dwTrackerParams, &dwFeature2DTrackerConfig::imageWidth),
                DW_DESCRIBE_INDEX_PARAMETER_WITH_SEMANTIC(
                    uint32_t,
                    semantic_parameter_types::ImageHeightYuv,
                    "cameraIndex"_sv,
                    &dwFeatureTrackerNodeParams::dwTrackerParams, &dwFeature2DTrackerConfig::imageHeight),
                DW_DESCRIBE_INDEX_PARAMETER(
                    cudaStream_t,
                    "streamIndex"_sv,
                    &dwFeatureTrackerNodeParams::cudaStream)),
            describeConstructorArgument(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwContextHandle_t)));
    }

    dwFeatureTrackerNode(const dwFeatureTrackerNodeParams& params,
                         const dwContextHandle_t ctx);
};
} // namespace framework
} // namespace dw

#endif //DW_FRAMEWORK_FEATURE_TRACKER_NODE_H_
