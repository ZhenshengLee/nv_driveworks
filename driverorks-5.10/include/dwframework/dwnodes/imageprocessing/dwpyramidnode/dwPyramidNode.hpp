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
// SPDX-FileCopyrightText: Copyright (c) 2018-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DWFRAMEWORK_DWNODES_IMAGEPROCESSING_DWPYRAMIDNODE_DWPYRAMIDNODE_HPP_
#define DWFRAMEWORK_DWNODES_IMAGEPROCESSING_DWPYRAMIDNODE_DWPYRAMIDNODE_HPP_

#ifdef DW_SDK_BUILD_PVA
#include <dw/imageprocessing/pyramid/PyramidPVA_processpipeline.h>
#endif
#include <dw/core/system/PVA.h>
#include <dw/imageprocessing/pyramid/Pyramid.h>
#include <dwcgf/node/Node.hpp>
#include <dwcgf/parameter/ParameterDescriptor.hpp>
#include <dwcgf/parameter/SemanticParameterTypes.hpp>
#include <dwcgf/pass/PassDescriptor.hpp>
#include <dwcgf/port/Port.hpp>
#include <dwcgf/port/PortDescriptor.hpp>
#include <dwframework/dwnodes/common/channelpackets/Image.hpp>
#include <dwframework/dwnodes/common/channelpackets/Pyramid.hpp>

namespace dw
{
namespace framework
{

using dwPyramidNodeParams = struct dwPyramidNodeParams
{
    uint32_t width;                                                  ///Width of level 0
    uint32_t height;                                                 ///Height of level 0
    uint32_t levelCount;                                             ///Number of levels in the pyramid
    dwTrivialDataType dataType = dwTrivialDataType::DW_TYPE_FLOAT32; ///Data Type of pyramid
    cupvaStream_t cupvastream;
    cudaStream_t stream;
};

class dwPyramidNodeBase : public ExceptionSafeProcessNode
{
public:
    static constexpr auto describeInputPorts()
    {
        return describePortCollection(
            DW_DESCRIBE_PORT(dwImageHandle_t, "IMAGE"_sv, PortBinding::REQUIRED));
    };
    static constexpr auto describeOutputPorts()
    {
        return describePortCollection(
            DW_DESCRIBE_PORT(dwPyramidImage, "LEVEL_IMAGES"_sv, PortBinding::REQUIRED));
    };

    using ExceptionSafeProcessNode::ExceptionSafeProcessNode;
};

class dwPyramidNode : public dwPyramidNodeBase
{
public:
    static constexpr char LOG_TAG[] = "dwPyramidNode";

    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass("SETUP"_sv, DW_PROCESSOR_TYPE_CPU),
            describePass("BUILD_PYRAMID"_sv, DW_PROCESSOR_TYPE_GPU),
            describePass("TEARDOWN"_sv, DW_PROCESSOR_TYPE_CPU));
    };

    static std::unique_ptr<dwPyramidNode> create(ParameterProvider& provider);

    static constexpr auto describeParameters()
    {
        return describeConstructorArguments<dwPyramidNodeParams, dwContextHandle_t>(
            describeConstructorArgument(
                DW_DESCRIBE_INDEX_PARAMETER_WITH_SEMANTIC(
                    uint32_t,
                    semantic_parameter_types::ImageWidth,
                    "cameraIndex"_sv,
                    &dwPyramidNodeParams::width),
                DW_DESCRIBE_INDEX_PARAMETER_WITH_SEMANTIC(
                    uint32_t,
                    semantic_parameter_types::ImageHeight,
                    "cameraIndex"_sv,
                    &dwPyramidNodeParams::height),
                DW_DESCRIBE_PARAMETER(
                    uint32_t,
                    "pyramidLevelCount"_sv,
                    &dwPyramidNodeParams::levelCount),
                DW_DESCRIBE_INDEX_PARAMETER(
                    cudaStream_t,
                    "streamIndex"_sv,
                    &dwPyramidNodeParams::stream)),
            describeConstructorArgument(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwContextHandle_t)));
    }

    dwPyramidNode(const dwPyramidNodeParams& config,
                  const dwContextHandle_t ctx);
};
} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_IMAGEPROCESSING_DWPYRAMIDNODE_DWPYRAMIDNODE_HPP_
