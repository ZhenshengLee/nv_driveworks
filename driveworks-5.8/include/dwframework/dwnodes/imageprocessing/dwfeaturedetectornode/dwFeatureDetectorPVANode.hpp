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

#ifndef DW_FRAMEWORK_FEATURE_DETECTOR_PVA_NODE_H_
#define DW_FRAMEWORK_FEATURE_DETECTOR_PVA_NODE_H_

#include <dwframework/dwnodes/imageprocessing/dwfeaturedetectornode/dwFeatureDetectorNode.hpp>
namespace dw
{
namespace framework
{

class dwFeatureDetectorPVANode : public dwFeatureDetectorNodeBase
{

public:
    static constexpr char LOG_TAG[] = "dwFeatureDetectorPVANode";

    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass("SETUP"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("PREPARE_DATA"_sv, DW_PROCESSOR_TYPE_GPU),
            describePass("DETECT_NEW_FEATURE"_sv, DW_PROCESSOR_TYPE_PVA_0),
            describePass("POST_PROCESS"_sv, DW_PROCESSOR_TYPE_GPU),
            describePass("TEARDOWN"_sv, DW_PROCESSOR_TYPE_CPU));
    };

    static std::unique_ptr<dwFeatureDetectorPVANode> create(ParameterProvider& provider);

    static constexpr auto describeParameters()
    {
        return describeConstructorArguments<dwFeatureDetectorNodeParams, dwFeatureDetectorNodeRuntimeParams, dwContextHandle_t>(
            describeConstructorArgument(
                DW_DESCRIBE_PARAMETER(
                    bool,
                    "initForCamera"_sv,
                    &dwFeatureDetectorNodeParams::initForCamera),
                DW_DESCRIBE_INDEX_PARAMETER_WITH_SEMANTIC(
                    dwCameraModelHandle_t,
                    semantic_parameter_types::CameraIntrinsicsYuv,
                    "cameraIndex"_sv,
                    &dwFeatureDetectorNodeParams::intrinsic),
                DW_DESCRIBE_INDEX_PARAMETER_WITH_SEMANTIC(
                    dwTransformation3f,
                    semantic_parameter_types::CameraExtrinsic,
                    "cameraIndex"_sv,
                    &dwFeatureDetectorNodeParams::extrinsic),
                DW_DESCRIBE_PARAMETER(
                    dwFeature2DDetectorType,
                    "featureDetectorType"_sv,
                    &dwFeatureDetectorNodeParams::dwDetectorParams, &dwFeature2DDetectorConfig::type),
                DW_DESCRIBE_INDEX_PARAMETER_WITH_SEMANTIC(
                    cupvaStream_t,
                    semantic_parameter_types::CupvaStream,
                    "pvaStreamIndex"_sv,
                    &dwFeatureDetectorNodeParams::cupvastream),
                DW_DESCRIBE_INDEX_PARAMETER_WITH_SEMANTIC(
                    uint32_t,
                    semantic_parameter_types::ImageWidthYuv,
                    "cameraIndex"_sv,
                    &dwFeatureDetectorNodeParams::dwDetectorParams, &dwFeature2DDetectorConfig::imageWidth),
                DW_DESCRIBE_INDEX_PARAMETER_WITH_SEMANTIC(
                    uint32_t,
                    semantic_parameter_types::ImageHeightYuv,
                    "cameraIndex"_sv,
                    &dwFeatureDetectorNodeParams::dwDetectorParams, &dwFeature2DDetectorConfig::imageHeight),
                DW_DESCRIBE_PARAMETER(
                    uint32_t,
                    "maxFeatureCount"_sv,
                    &dwFeatureDetectorNodeParams::dwDetectorParams, &dwFeature2DDetectorConfig::maxFeatureCount)),
            describeConstructorArgument(
                DW_DESCRIBE_INDEX_PARAMETER(
                    cudaStream_t,
                    "streamIndex"_sv,
                    &dwFeatureDetectorNodeRuntimeParams::cudaStream)),
            describeConstructorArgument(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwContextHandle_t)));
    }

    dwFeatureDetectorPVANode(const dwFeatureDetectorNodeParams& params,
                             const dwFeatureDetectorNodeRuntimeParams& runtimeParams,
                             const dwContextHandle_t ctx);

    dwStatus setRuntimeParams(const dwFeatureDetectorNodeRuntimeParams& runtimeParams);
};
} // namespace framework
} // namespace dw

#endif //DW_FRAMEWORK_FEATURE_DETECTOR_PVA_NODE_H_
