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

#ifndef DW_FRAMEWORK_FEATURE_DETECTOR_NODE_H_
#define DW_FRAMEWORK_FEATURE_DETECTOR_NODE_H_

#include <dwcgf/node/Node.hpp>
#include <dwcgf/parameter/ParameterDescriptor.hpp>
#include <dwcgf/parameter/SemanticParameterTypes.hpp>
#include <dwcgf/pass/PassDescriptor.hpp>
#include <dwcgf/port/Port.hpp>
#include <dwcgf/port/PortDescriptor.hpp>

#include <dw/imageprocessing/featuredetector/FeatureDetector_processpipeline.h>
#include <dw/imageprocessing/features/FeatureList.h>
/* Need to include the appropriate ChannelPacketTypes.hpp since port initialization requires
   the parameter_trait overrides. Otherwise, it will be considered as a packet of generic type. */
#include <dwframework/dwnodes/common/ChannelPacketTypes.hpp>

namespace dw
{
namespace framework
{

using dwFeatureDetectorNodeRuntimeParams = struct dwFeatureDetectorNodeRuntimeParams
{
    //mask to ignore
    uint8_t* d_mask;
    uint32_t maskStrideBytes;
    uint32_t maskWidth;
    uint32_t maskHeight;
    cudaStream_t cudaStream;
};

using dwFeatureDetectorNodeParams = struct dwFeatureDetectorNodeParams
{
    bool initForCamera;
    dwCameraModelHandle_t intrinsic;
    dwTransformation3f extrinsic;
    cupvaStream_t cupvastream;
    dwFeature2DDetectorConfig dwDetectorParams;
};

class dwFeatureDetectorNodeBase : public ExceptionSafeProcessNode
{
public:
    static constexpr auto describeInputPorts()
    {
        return describePortCollection(
            DW_DESCRIBE_PORT(dwPyramidImage, "PYRAMID"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwFeatureHistoryArray, "PREDICTED_FEATURE"_sv, PortBinding::REQUIRED),
            DW_DESCRIBE_PORT(dwFeatureNccScores, "NCC_SCORES"_sv, PortBinding::REQUIRED));
    };
    static constexpr auto describeOutputPorts()
    {
        return describePortCollection(
            DW_DESCRIBE_PORT(dwFeatureArray, "FEATURE_ARRAY_GPU"_sv, PortBinding::REQUIRED));
    };

    using ExceptionSafeProcessNode::ExceptionSafeProcessNode;
};

class dwFeatureDetectorNode : public dwFeatureDetectorNodeBase
{

public:
    static constexpr char LOG_TAG[] = "dwFeatureDetectorNode";

    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass("SETUP"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("DETECT_NEW_FEATURE"_sv, DW_PROCESSOR_TYPE_GPU),
            describePass("TEARDOWN"_sv, DW_PROCESSOR_TYPE_CPU));
    };

    static std::unique_ptr<dwFeatureDetectorNode> create(ParameterProvider& provider);

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

    dwFeatureDetectorNode(const dwFeatureDetectorNodeParams& params,
                          const dwFeatureDetectorNodeRuntimeParams& runtimeParams,
                          const dwContextHandle_t ctx);

    dwStatus setRuntimeParams(const dwFeatureDetectorNodeRuntimeParams& runtimeParams);
};
} // namespace framework
} // namespace dw

#endif //DW_FRAMEWORK_FEATURE_DETECTOR_NODE_H_
