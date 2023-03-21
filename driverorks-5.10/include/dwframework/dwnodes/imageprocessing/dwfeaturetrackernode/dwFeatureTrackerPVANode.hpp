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

#ifndef DWFRAMEWORK_DWNODES_IMAGEPROCESSING_DWFEATURETRACKERNODE_DWFEATURETRACKERPVANODE_HPP_
#define DWFRAMEWORK_DWNODES_IMAGEPROCESSING_DWFEATURETRACKERNODE_DWFEATURETRACKERPVANODE_HPP_

#include <dwframework/dwnodes/imageprocessing/dwfeaturetrackernode/dwFeatureTrackerNode.hpp>

namespace dw
{
namespace framework
{

class dwFeatureTrackerPVANode : public dwFeatureTrackerNodeBase
{
public:
    static constexpr char LOG_TAG[] = "dwFeatureTrackerPVANode";
    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass("SETUP"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("PREPARE_DATA"_sv, DW_PROCESSOR_TYPE_GPU),
            describePass("TRACK_FEATURE"_sv, DW_PROCESSOR_TYPE_PVA_0),
            describePass("POST_PROCESS"_sv, DW_PROCESSOR_TYPE_GPU),
            describePass("WAIT_EVENT"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("TEARDOWN"_sv, DW_PROCESSOR_TYPE_CPU));
    };

    static std::unique_ptr<dwFeatureTrackerPVANode> create(ParameterProvider& provider);

    static constexpr auto describeParameters()
    {
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
                DW_DESCRIBE_INDEX_PARAMETER_WITH_SEMANTIC(
                    cupvaStream_t,
                    semantic_parameter_types::CupvaStream,
                    "pvaStreamIndex"_sv,
                    &dwFeatureTrackerNodeParams::cupvaStream),
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

    dwFeatureTrackerPVANode(const dwFeatureTrackerNodeParams& params,
                            const dwContextHandle_t ctx);
};
} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_IMAGEPROCESSING_DWFEATURETRACKERNODE_DWFEATURETRACKERPVANODE_HPP_
