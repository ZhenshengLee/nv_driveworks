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
// SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    dwTrivialDataType dataType = dwTrivialDataType::DW_TYPE_FLOAT16; ///Data Type of pyramid
    cupvaStream_t cupvastream;
    cudaStream_t stream;
    dwTime_t nvSciChannelWaitTimeUs; /// Wait time for nvSciChannel
    uint32_t rrSehErrorCode;
};

class dwPyramidNodeBase : public ExceptionSafeProcessNode
{
public:
    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeInputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwImageHandle_t, "IMAGE"_sv, PortBinding::REQUIRED));
    };
    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeOutputPorts()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describePortCollection(
            DW_DESCRIBE_PORT(dwPyramidImage, "LEVEL_IMAGES"_sv, PortBinding::REQUIRED));
    };

    using ExceptionSafeProcessNode::ExceptionSafeProcessNode;
};

class dwPyramidNode : public dwPyramidNodeBase
{
public:
    // TODO(csketch): FP -- This is used when the logger is called and not just at assignment.
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2980283
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2980283
    static constexpr char8_t LOG_TAG[]{"dwPyramidNode"};

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describePasses()
    {
        return describePassCollection(
            describePass(StringView{"SETUP"}, DW_PROCESSOR_TYPE_CPU),
            describePass(StringView{"BUILD_PYRAMID"}, DW_PROCESSOR_TYPE_GPU),
            describePass(StringView{"TEARDOWN"}, DW_PROCESSOR_TYPE_CPU));
    };

    static std::unique_ptr<dwPyramidNode> create(ParameterProvider& provider);

    // coverity[autosar_cpp14_a7_1_5_violation] RFD Accepted: TID-2201
    static constexpr auto describeParameters()
    {
        // TODO(csketch): RFD --  user defined literal being interpreted as c style cast.
        // coverity[autosar_cpp14_a5_2_2_violation] RFD Pending: TID-1983
        return describeConstructorArguments(
            describeConstructorArgument<dwPyramidNodeParams>(
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
                DW_DESCRIBE_PARAMETER_WITH_DEFAULT(
                    dwTime_t,
                    "nvSciChannelWaitTimeUs"_sv,
                    0,
                    &dwPyramidNodeParams::nvSciChannelWaitTimeUs),
                DW_DESCRIBE_INDEX_PARAMETER(
                    cudaStream_t,
                    "streamIndex"_sv,
                    &dwPyramidNodeParams::stream)),
            describeConstructorArgument<dwContextHandle_t>(
                DW_DESCRIBE_UNNAMED_PARAMETER(
                    dwContextHandle_t)));
    }

    dwPyramidNode(const dwPyramidNodeParams& config,
                  const dwContextHandle_t ctx);
};
} // namespace framework
} // namespace dw

#endif // DWFRAMEWORK_DWNODES_IMAGEPROCESSING_DWPYRAMIDNODE_DWPYRAMIDNODE_HPP_
