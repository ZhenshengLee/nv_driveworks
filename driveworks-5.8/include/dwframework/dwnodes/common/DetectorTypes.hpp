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

#ifndef DW_FRAMEWORK_DETECTOR_TYPES_HPP_
#define DW_FRAMEWORK_DETECTOR_TYPES_HPP_

#include <dwcgf/enum/EnumDescriptor.hpp>
#include <dw/core/base/Types.h>

namespace dw
{
namespace framework
{

enum class dwFovealMode
{
    OFF,        //FOVEAL MODE IS OFF
    FULL,       //FOVEAL MODE IS FULL TIME
    INTERLEAVED //FOVEAL MODE IS INTERLEAVED - Demosaiced crop and downsampled full view are alternating
};

template <>
struct EnumDescription<dwFovealMode>
{
    static constexpr auto get()
    {
        using EnumT = dwFovealMode;
        return describeEnumeratorCollection<EnumT>(
            DW_DESCRIBE_ENUMERATOR(OFF),
            DW_DESCRIBE_ENUMERATOR(FULL),
            DW_DESCRIBE_ENUMERATOR(INTERLEAVED));
    }
};

/// ROI mode for detector input from original input image.
/// Horizontal rows of the input image may differ based on FPS setting.
/// - 30fps: 1920x1208, use MIDDLE as default
/// - 36fps: 1920x1008, use BOTTOM as default
enum class dwModeDnnROI
{
    TOP,    // DNN ROI is set to the top of the image (cut bottom)
    MIDDLE, // DNN ROI is set to the middle of the image (cut top/bottom half each)
    BOTTOM  // DNN ROI is set to the bottom of the image (cut top)
};

using dwDetectorFovealParams = struct dwDetectorFovealParams
{
    bool fovealEnable;
    dwRect ROI;
    dwVector2f scales;
    dwFovealMode fovealMode;
};

template <>
struct EnumDescription<dwTrivialDataType>
{
    static constexpr auto get()
    {
        return describeEnumeratorCollection<dwTrivialDataType>(
            DW_DESCRIBE_C_ENUMERATOR(DW_TYPE_UNKNOWN),
            DW_DESCRIBE_C_ENUMERATOR(DW_TYPE_BOOL),
            DW_DESCRIBE_C_ENUMERATOR(DW_TYPE_INT8),
            DW_DESCRIBE_C_ENUMERATOR(DW_TYPE_INT16),
            DW_DESCRIBE_C_ENUMERATOR(DW_TYPE_INT32),
            DW_DESCRIBE_C_ENUMERATOR(DW_TYPE_INT64),
            DW_DESCRIBE_C_ENUMERATOR(DW_TYPE_UINT8),
            DW_DESCRIBE_C_ENUMERATOR(DW_TYPE_UINT16),
            DW_DESCRIBE_C_ENUMERATOR(DW_TYPE_UINT32),
            DW_DESCRIBE_C_ENUMERATOR(DW_TYPE_UINT64),
            DW_DESCRIBE_C_ENUMERATOR(DW_TYPE_FLOAT32),
            DW_DESCRIBE_C_ENUMERATOR(DW_TYPE_FLOAT64),
            DW_DESCRIBE_C_ENUMERATOR(DW_TYPE_FLOAT16),
            DW_DESCRIBE_C_ENUMERATOR(DW_TYPE_CHAR8));
    }
};

template <>
struct EnumDescription<dwPrecision>
{
    static constexpr auto get()
    {
        return describeEnumeratorCollection<dwPrecision>(
            DW_DESCRIBE_C_ENUMERATOR(DW_PRECISION_INT8),
            DW_DESCRIBE_C_ENUMERATOR(DW_PRECISION_FP16),
            DW_DESCRIBE_C_ENUMERATOR(DW_PRECISION_FP32),
            DW_DESCRIBE_C_ENUMERATOR(DW_PRECISION_MIXED));
    }
};

template <>
struct EnumDescription<dwProcessorType>
{
    static constexpr auto get()
    {
        return describeEnumeratorCollection<dwProcessorType>(
            DW_DESCRIBE_C_ENUMERATOR(DW_PROCESSOR_TYPE_CPU),
            DW_DESCRIBE_C_ENUMERATOR(DW_PROCESSOR_TYPE_GPU),
            DW_DESCRIBE_C_ENUMERATOR(DW_PROCESSOR_TYPE_DLA_0),
            DW_DESCRIBE_C_ENUMERATOR(DW_PROCESSOR_TYPE_DLA_1),
            DW_DESCRIBE_C_ENUMERATOR(DW_PROCESSOR_TYPE_PVA_0),
            DW_DESCRIBE_C_ENUMERATOR(DW_PROCESSOR_TYPE_PVA_1),
            DW_DESCRIBE_C_ENUMERATOR(DW_PROCESSOR_TYPE_NVENC_0),
            DW_DESCRIBE_C_ENUMERATOR(DW_PROCESSOR_TYPE_NVENC_1),
            DW_DESCRIBE_C_ENUMERATOR(DW_PROCESSOR_TYPE_CUDLA));
    }
};

} // framework
} // dw
#endif // DW_FRAMEWORK_DETECTOR_TYPES_HPP_
