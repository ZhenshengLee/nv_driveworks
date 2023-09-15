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
// SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_FRAMEWORK_DW_ENUM_DESCRIPTIONS_HPP_
#define DW_FRAMEWORK_DW_ENUM_DESCRIPTIONS_HPP_

#include <dwcgf/enum/EnumDescriptor.hpp>
#include <dw/core/base/Types.h>
#include <dw/sensors/Sensors.h>
#include <dw/image/Image.h>
#include <dwframework/dwnodes/common/DetectorTypes.hpp>

namespace dw
{
namespace framework
{

template <>
struct EnumDescription<dwImageType>
{
    static constexpr auto get()
    {
        return describeEnumeratorCollection<dwImageType>(
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_CPU),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_CUDA),
#ifdef VIBRANTE
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_NVMEDIA),
#endif
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_GL));
    }
};

template <>
struct EnumDescription<dwImageFormat>
{
    static constexpr auto get()
    {
        return describeEnumeratorCollection<dwImageFormat>(
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_FORMAT_UNKNOWN),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_FORMAT_R_INT16),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_FORMAT_R_UINT8),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_FORMAT_R_UINT16),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_FORMAT_R_UINT32),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_FORMAT_R_FLOAT16),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_FORMAT_R_FLOAT32),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_FORMAT_RG_INT16),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_FORMAT_RG_UINT8),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_FORMAT_RG_FLOAT32),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_FORMAT_RGB_UINT8),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_FORMAT_RGB_UINT16),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_FORMAT_RGB_FLOAT16),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_FORMAT_RGB_FLOAT32),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_FORMAT_RGBA_UINT8),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_FORMAT_RGBA_UINT16),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_FORMAT_RGBA_FLOAT16),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_FORMAT_RGBA_FLOAT32),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_FORMAT_RGBX_FLOAT16),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_FORMAT_VUYX_UINT8),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_FORMAT_VUYX_UINT16),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_FORMAT_RGB_UINT8_PLANAR),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_FORMAT_RGB_UINT16_PLANAR),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_FORMAT_RGB_FLOAT16_PLANAR),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_FORMAT_RGB_FLOAT32_PLANAR),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_FORMAT_RCB_FLOAT16_PLANAR),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_FORMAT_RCB_FLOAT32_PLANAR),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_FORMAT_RCC_FLOAT16_PLANAR),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_FORMAT_RCC_FLOAT32_PLANAR),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_FORMAT_YUV420_UINT8_PLANAR),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_FORMAT_YUV420_UINT8_SEMIPLANAR),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_FORMAT_YUV420_UINT16_SEMIPLANAR),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_FORMAT_YUV422_UINT8_SEMIPLANAR),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_FORMAT_YUV_UINT8_PLANAR),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_FORMAT_YUV_UINT16_PLANAR),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_FORMAT_RAW_UINT16),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_FORMAT_RAW_FLOAT16));
    }
};

template <>
struct EnumDescription<dwImageMemoryType>
{
    static constexpr auto get()
    {
        return describeEnumeratorCollection<dwImageMemoryType>(
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_MEMORY_TYPE_DEFAULT),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_MEMORY_TYPE_PITCH),
            DW_DESCRIBE_C_ENUMERATOR(DW_IMAGE_MEMORY_TYPE_BLOCK));
    }
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
            DW_DESCRIBE_C_ENUMERATOR(DW_PROCESSOR_TYPE_CUDLA),
            DW_DESCRIBE_C_ENUMERATOR(DW_PROCESSOR_TYPE_VULKAN));
    }
};

template <>
struct EnumDescription<dwSensorType>
{
    static constexpr auto get()
    {
        return describeEnumeratorCollection<dwSensorType>(
            DW_DESCRIBE_C_ENUMERATOR(DW_SENSOR_CAMERA),
            DW_DESCRIBE_C_ENUMERATOR(DW_SENSOR_LIDAR),
            DW_DESCRIBE_C_ENUMERATOR(DW_SENSOR_GPS),
            DW_DESCRIBE_C_ENUMERATOR(DW_SENSOR_IMU),
            DW_DESCRIBE_C_ENUMERATOR(DW_SENSOR_CAN),
            DW_DESCRIBE_C_ENUMERATOR(DW_SENSOR_RADAR),
            DW_DESCRIBE_C_ENUMERATOR(DW_SENSOR_TIME),
            DW_DESCRIBE_C_ENUMERATOR(DW_SENSOR_DATA),
            DW_DESCRIBE_C_ENUMERATOR(DW_SENSOR_ULTRASONIC));
    }
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

} // framework
} // dw
#endif // DW_FRAMEWORK_DW_ENUM_DESCRIPTIONS_HPP_
