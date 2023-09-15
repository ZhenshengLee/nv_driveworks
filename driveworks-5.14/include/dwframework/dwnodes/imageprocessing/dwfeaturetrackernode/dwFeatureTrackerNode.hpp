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
// SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWFRAMEWORK_DWNODES_IMAGEPROCESSING_DWFEATURETRACKERNODE_DWFEATURETRACKERNODE_HPP_
#define DWFRAMEWORK_DWNODES_IMAGEPROCESSING_DWFEATURETRACKERNODE_DWFEATURETRACKERNODE_HPP_

#include <dw/imageprocessing/featuredetector/FeatureDetector.h>
#include <dw/imageprocessing/features/FeatureList.h>
#include <dw/imageprocessing/tracking/featuretracker/FeatureTracker_processpipeline.h>
#include <dwcgf/node/Node.hpp>
#include <dwcgf/parameter/ParameterDescriptor.hpp>
#include <dwcgf/parameter/SemanticParameterTypes.hpp>
#include <dwcgf/pass/PassDescriptor.hpp>
#include <dwcgf/port/Port.hpp>
#include <dwcgf/port/PortDescriptor.hpp>
#include <dwframework/dwnodes/common/channelpackets/DwframeworkTypes.hpp>
#include <dwframework/dwnodes/common/channelpackets/FeatureList.hpp>
#include <dwframework/dwnodes/common/channelpackets/Pyramid.hpp>
#include <dwframework/dwnodes/common/channelpackets/SensorCommonTypes.hpp>

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
    cupvaStream_t cupvaStream;
    cudaStream_t cudaStream;
};

class dwFeatureTrackerNodeBase : public ExceptionSafeProcessNode
{
public:
    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-1984
    static constexpr auto describeInputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwPyramidImage, "CURRENT_PYRAMID"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwFeatureArray, "DETECTED_FEATURES"_sv, PortBinding::REQUIRED));
    };
    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-1984
    static constexpr auto describeOutputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwFeatureHistoryArray, "FEATURE_HISTORY_CPU"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwFeatureHistoryArray, "FEATURE_HISTORY_GPU"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwFeatureHistoryArray, "FEATURE_HISTORY_GPU_INTERNAL"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwFeatureNccScores, "NCC_SCORES"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(void*, "MODULE_HANDLE"_sv, PortBinding::REQUIRED));
    };

    // Inherit constructor from base class
    using ExceptionSafeProcessNode::ExceptionSafeProcessNode;
};

/**
 * @ingroup dwnodes
 */
class dwFeatureTrackerNode : public dwFeatureTrackerNodeBase
{
public:
    // TODO(csketch): FP -- This is used when the logger is called and not just at assignment.
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2980283
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2980283
    static constexpr char8_t LOG_TAG[]{"dwFeatureTrackerNode"};
    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-1984
    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass(StringView{"SETUP"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"TRACK_FEATURE"}, DW_PROCESSOR_TYPE_GPU),
            describePass(StringView{"COMPACT_FEATURE"}, DW_PROCESSOR_TYPE_GPU),
            describePass(StringView{"COPY"}, DW_PROCESSOR_TYPE_GPU),
            describePass(StringView{"WAIT_EVENT"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"TEARDOWN"}, DW_PROCESSOR_TYPE_CPU));
    };

    static std::unique_ptr<dwFeatureTrackerNode> create(ParameterProvider& provider);

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-1984
    static constexpr auto describeParameters()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describeConstructorArguments<dwFeatureTrackerNodeParams, dwContextHandle_t>(
            describeConstructorArgument(
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "initForCamera"_sv,
                    &dwFeatureTrackerNodeParams::initForCamera),
                DW_DESCRIBE_INDEX_PARAMETER(
                    dwCameraModelHandle_t,
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
                    semantic_parameter_types::ImageWidth,
                    "cameraIndex"_sv,
                    &dwFeatureTrackerNodeParams::dwTrackerParams, &dwFeature2DTrackerConfig::imageWidth),
                DW_DESCRIBE_INDEX_PARAMETER_WITH_SEMANTIC(
                    uint32_t,
                    semantic_parameter_types::ImageHeight,
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

#endif // DWFRAMEWORK_DWNODES_IMAGEPROCESSING_DWFEATURETRACKERNODE_DWFEATURETRACKERNODE_HPP_
