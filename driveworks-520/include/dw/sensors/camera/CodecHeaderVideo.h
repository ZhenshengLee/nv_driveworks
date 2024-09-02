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
// SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_SENSORS_CODECS_CAMERA_CODECDATAVIDEO_H_
#define DW_SENSORS_CODECS_CAMERA_CODECDATAVIDEO_H_

#include <dw/core/base/Types.h>
#include <dw/sensors/codecs/Codec.h>

#include "Camera.h"

#ifdef __cplusplus
extern "C" {
#endif

// Video format
typedef enum {
    DW_CODEC_VIDEO_FORMAT_YUV420,
    DW_CODEC_VIDEO_FORMAT_YUV444,
    DW_CODEC_VIDEO_FORMAT_RGB,
    DW_CODEC_VIDEO_FORMAT_RAW,
    DW_CODEC_VIDEO_FORMAT_YUV400,
    DW_CODEC_VIDEO_FORMAT_YUV422,
} dwCodecVideoFormat;

/// SIPL EEPROM information
typedef struct dwCodecCameraSIPLBlobEEPROM
{
    uint8_t sensorName[16];
    uint16_t sensorRevID;
    uint8_t serializerNameID[16];
    uint16_t serializerRevID;
    uint8_t bayerTypeID[4];
    uint8_t scanlineOrderID;
    uint8_t lensManuID[16];
    uint8_t lensNameID[16];
    uint16_t lensFOVID;
    uint16_t imageWidth;
    uint16_t imageHeight;

    // Group: MODULE_MAKER_INTRINSIC
    uint8_t mmIntrinsicModelID;
    float32_t mmIntrinsicCoeff[16];
    uint8_t mmIntrinsicFlagID;

    // Group: CHANNEL_COLOR_RESPONSES
    uint16_t channelLightID;

    // Group: LENS_SHADING
    uint8_t shadingLightID;
    uint8_t shadingDataTypeID[2];

    // Group: FUSE_ID
    uint16_t fuseID;

    // Group: INTRINSIC_PARAMETERS
    uint8_t intrinsicModelID;
    float32_t intrinsicCoeff[16];
    uint8_t intrinsicFlagID;

    // Group: EXTRINSIC_PARAMETERS
    float32_t extrinsicParamsID[6];

    // Group: SECOND_INTRINSIC_PARAMETERS
    uint8_t secIntrinsicModelID;
    float32_t secIntrinsicCoeff[16];
    uint8_t secIntrinsicFlagID;

    // Group: THIRD_INTRINSIC_PARAMETERS
    uint8_t thirdIntrinsicModelID;
    float32_t thirdIntrinsicCoeff[16];
    uint8_t thirdIntrinsicFlagID;

    // Group least important parameters here so that its easy to remove them if needed
    float32_t eflID;
    char8_t iRNameID[16];
    uint16_t iRCutFreqID;
    char8_t modMakerID[16];
    char8_t modNameID[16];
    char8_t modSerialID[16];
    uint64_t assmTimeID;
    uint32_t assmLineID;
    uint16_t channelMeasurementID[4]; // order of channels R, GR, GB, B
    char8_t fuseDataID[16];
    char8_t lensShadingDataID[800];
} dwCodecCameraSIPLBlobEEPROM;

// TODO (nperla): reconsile dwCodecCameraSIPLInfo with the sensor sipl structure
/// SIPL information
typedef struct dwCodecCameraSIPLInfo
{
    char8_t cameraModuleName[64];
    char8_t cameraModuleDescription[128];
    char8_t interface[16];
    uint32_t linkIndex;
    bool cphyMode;
    uint32_t colorFilterArray;
    char8_t serializerName[64];
    char8_t serializerDescription[128];
    uint32_t serializeri2cAddress;
    char8_t sensorParams[512];
    dwCodecCameraSIPLBlobEEPROM eeprom;
} dwCodecCameraSIPLInfo;

/// Required camera metadata for encoding and decoding video
typedef struct dwCodecCameraMetadata
{
    dwCameraType cameraType;
    dwCameraRawFormat rawFormat;
    dwCameraExposureControl exposure;
    dwCameraFOV fov;
    uint32_t sensorRevision;
    uint32_t width;
    uint32_t height;
    uint32_t embeddedTopLines;
    uint32_t embeddedBottomLines;
    uint32_t msbPosition;
    dwCodecCameraSIPLInfo siplInfo;
} dwCodecCameraMetadata;

/// The base configuration for all video encoders and decoders.
typedef struct dwCodecConfigVideo
{
    /// Codec type.
    dwCodecType codec;
    /// Video width.
    uint32_t width;
    /// Video height.
    uint32_t height;
    /// Frame bit depth.
    uint32_t bitDepth;
    /// Frame rate.
    uint32_t frameRate;
    /// Video format.
    dwCodecVideoFormat format;
    /// Required camera metadata for encoding and decoding video.
    dwCodecCameraMetadata cameraMetadata;
    /// Encoder rate control parameters.
    dwEncoderConfigRateControl rateControl;
    /// Codec mime type
    char8_t codecMimeType[DW_MAX_CODEC_MIME_TYPE_LENGTH];
    /// Raw sipl data from the camera
    char8_t rawHeader[DW_MAX_RAW_SIPL_HEADER_LENGTH];
} dwCodecConfigVideo;

#ifdef __cplusplus
}
#endif

#endif //DW_SENSORS_CODECS_CAMERA_CODECDATAVIDEO_H_
