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
// SPDX-FileCopyrightText: Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * @file
 * <b>NVIDIA DriveWorks API: Codec Methods</b>
 *
 * @b Description: This file defines codec methods.
 */

/**
 * @defgroup codecs Codecs
 *
 * @brief Defines the codec types.
 * @{
 */

#ifndef DW_CODECS_CODEC_H_
#define DW_CODECS_CODEC_H_

#include <dw/core/base/Types.h>
#include <dw/sensors/camera/Camera.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Media Type for Codec
typedef enum {
    DW_MEDIA_TYPE_VIDEO      = 0, ///< type of video
    DW_MEDIA_TYPE_LIDAR      = 1, ///< type of lidar
    DW_MEDIA_TYPE_RADAR      = 2, ///< type of radar
    DW_MEDIA_TYPE_IMU        = 3, ///< type of IMU
    DW_MEDIA_TYPE_GPS        = 4, ///< type of GPS
    DW_MEDIA_TYPE_CAN        = 5, ///< type of CAN
    DW_MEDIA_TYPE_DATA       = 6, ///< type of data
    DW_MEDIA_TYPE_TIME       = 7, ///< type of time
    DW_MEDIA_TYPE_ROADCAST   = 8, ///< type of roadcast
    DW_MEDIA_TYPE_ULTRASONIC = 9, ///< type of ultrasonic
    DW_MEDIA_TYPE_COUNT      = 10 ///< type of count
} dwMediaType;

/// Codec Type
typedef enum {
    DW_CODEC_TYPE_INVALID                                                                     = -1,
    DW_CODEC_TYPE_VIDEO_H264 DW_DEPRECATED_ENUM("dwCodecType deprecated")                     = 0,
    DW_CODEC_TYPE_VIDEO_H265 DW_DEPRECATED_ENUM("dwCodecType deprecated")                     = 1,
    DW_CODEC_TYPE_VIDEO_VP9 DW_DEPRECATED_ENUM("dwCodecType deprecated")                      = 2,
    DW_CODEC_TYPE_VIDEO_AV1 DW_DEPRECATED_ENUM("dwCodecType deprecated")                      = 3,
    DW_CODEC_TYPE_VIDEO_LRAW DW_DEPRECATED_ENUM("dwCodecType deprecated")                     = 4,
    DW_CODEC_TYPE_VIDEO_LRAW_V2 DW_DEPRECATED_ENUM("dwCodecType deprecated")                  = 5,
    DW_CODEC_TYPE_VIDEO_XRAW DW_DEPRECATED_ENUM("dwCodecType deprecated")                     = 6,
    DW_CODEC_TYPE_VIDEO_RAW DW_DEPRECATED_ENUM("dwCodecType deprecated")                      = 7,
    DW_CODEC_TYPE_LIDAR_CUSTOM DW_DEPRECATED_ENUM("dwCodecType deprecated")                   = 8,
    DW_CODEC_TYPE_LIDAR_HESAI_P128 DW_DEPRECATED_ENUM("dwCodecType deprecated")               = 9,
    DW_CODEC_TYPE_LIDAR_HESAI_P128_V4P5 DW_DEPRECATED_ENUM("dwCodecType deprecated")          = 10,
    DW_CODEC_TYPE_LIDAR_LUMINAR_H DW_DEPRECATED_ENUM("dwCodecType deprecated")                = 11,
    DW_CODEC_TYPE_LIDAR_OUSTER_OS1 DW_DEPRECATED_ENUM("dwCodecType deprecated")               = 12,
    DW_CODEC_TYPE_LIDAR_OUSTER_OS2_128 DW_DEPRECATED_ENUM("dwCodecType deprecated")           = 13,
    DW_CODEC_TYPE_LIDAR_VELODYNE_HDL32E DW_DEPRECATED_ENUM("dwCodecType deprecated")          = 14,
    DW_CODEC_TYPE_LIDAR_VELODYNE_HDL64E DW_DEPRECATED_ENUM("dwCodecType deprecated")          = 15,
    DW_CODEC_TYPE_LIDAR_VELODYNE_VLP16 DW_DEPRECATED_ENUM("dwCodecType deprecated")           = 16,
    DW_CODEC_TYPE_LIDAR_VELODYNE_VLP16HR DW_DEPRECATED_ENUM("dwCodecType deprecated")         = 17,
    DW_CODEC_TYPE_LIDAR_VELODYNE_VLP32C DW_DEPRECATED_ENUM("dwCodecType deprecated")          = 18,
    DW_CODEC_TYPE_LIDAR_VELODYNE_VLS128 DW_DEPRECATED_ENUM("dwCodecType deprecated")          = 19,
    DW_CODEC_TYPE_RADAR_CONTINENTAL_ARS430 DW_DEPRECATED_ENUM("dwCodecType deprecated")       = 20,
    DW_CODEC_TYPE_RADAR_CONTINENTAL_ARS430CAN DW_DEPRECATED_ENUM("dwCodecType deprecated")    = 21,
    DW_CODEC_TYPE_RADAR_CONTINENTAL_ARS430RDI DW_DEPRECATED_ENUM("dwCodecType deprecated")    = 22,
    DW_CODEC_TYPE_RADAR_CONTINENTAL_ARS430RDI_V2 DW_DEPRECATED_ENUM("dwCodecType deprecated") = 23,
    DW_CODEC_TYPE_RADAR_CUSTOM DW_DEPRECATED_ENUM("dwCodecType deprecated")                   = 24,
    DW_CODEC_TYPE_RADAR_DELPHI DW_DEPRECATED_ENUM("dwCodecType deprecated")                   = 25,
    DW_CODEC_TYPE_IMU_BOSCH DW_DEPRECATED_ENUM("dwCodecType deprecated")                      = 26,
    DW_CODEC_TYPE_IMU_CAN DW_DEPRECATED_ENUM("dwCodecType deprecated")                        = 27,
    DW_CODEC_TYPE_IMU_CONTINENTAL DW_DEPRECATED_ENUM("dwCodecType deprecated")                = 28,
    DW_CODEC_TYPE_IMU_CUSTOM DW_DEPRECATED_ENUM("dwCodecType deprecated")                     = 29,
    DW_CODEC_TYPE_IMU_DATASPEED DW_DEPRECATED_ENUM("dwCodecType deprecated")                  = 30,
    DW_CODEC_TYPE_IMU_NOVATEL_ASCII DW_DEPRECATED_ENUM("dwCodecType deprecated")              = 31,
    DW_CODEC_TYPE_IMU_NOVATEL_BINARY DW_DEPRECATED_ENUM("dwCodecType deprecated")             = 32,
    DW_CODEC_TYPE_IMU_NV_SIM DW_DEPRECATED_ENUM("dwCodecType deprecated")                     = 33,
    DW_CODEC_TYPE_IMU_XSENS_BINARY DW_DEPRECATED_ENUM("dwCodecType deprecated")               = 34,
    DW_CODEC_TYPE_IMU_XSENS_CAN DW_DEPRECATED_ENUM("dwCodecType deprecated")                  = 35,
    DW_CODEC_TYPE_IMU_XSENS_NMEA DW_DEPRECATED_ENUM("dwCodecType deprecated")                 = 36,
    DW_CODEC_TYPE_GPS_CUSTOM DW_DEPRECATED_ENUM("dwCodecType deprecated")                     = 37,
    DW_CODEC_TYPE_GPS_DATASPEED DW_DEPRECATED_ENUM("dwCodecType deprecated")                  = 38,
    DW_CODEC_TYPE_GPS_DW_BINARY DW_DEPRECATED_ENUM("dwCodecType deprecated")                  = 39,
    DW_CODEC_TYPE_GPS_NMEA DW_DEPRECATED_ENUM("dwCodecType deprecated")                       = 40,
    DW_CODEC_TYPE_GPS_NOVATEL DW_DEPRECATED_ENUM("dwCodecType deprecated")                    = 41,
    DW_CODEC_TYPE_GPS_NOVATEL_BINARY DW_DEPRECATED_ENUM("dwCodecType deprecated")             = 42,
    DW_CODEC_TYPE_GPS_NOVATEL_ASCII DW_DEPRECATED_ENUM("dwCodecType deprecated")              = 43,
    DW_CODEC_TYPE_GPS_NV_SIM DW_DEPRECATED_ENUM("dwCodecType deprecated")                     = 44,
    DW_CODEC_TYPE_GPS_UBLOX DW_DEPRECATED_ENUM("dwCodecType deprecated")                      = 45,
    DW_CODEC_TYPE_GPS_XSENS_BINARY DW_DEPRECATED_ENUM("dwCodecType deprecated")               = 46,
    DW_CODEC_TYPE_CAN_DW_BINARY DW_DEPRECATED_ENUM("dwCodecType deprecated")                  = 47,
    DW_CODEC_TYPE_TIMESENSOR_DW_BINARY DW_DEPRECATED_ENUM("dwCodecType deprecated")           = 48,
    DW_CODEC_TYPE_DATA_DW_BINARY DW_DEPRECATED_ENUM("dwCodecType deprecated")                 = 49,
    DW_CODEC_TYPE_ION DW_DEPRECATED_ENUM("dwCodecType deprecated")                            = 50,
    DW_CODEC_TYPE_ULTRASONIC_VALEO_USV DW_DEPRECATED_ENUM("dwCodecType deprecated")           = 51,
    DW_CODEC_TYPE_RADAR_CUSTOM_EX DW_DEPRECATED_ENUM("dwCodecType deprecated")                = 52,
    DW_CODEC_TYPE_LIDAR_CUSTOM_EX DW_DEPRECATED_ENUM("dwCodecType deprecated")                = 53,
    DW_CODEC_TYPE_RADAR_HELLA_ADAS6 DW_DEPRECATED_ENUM("dwCodecType deprecated")              = 54,
    DW_CODEC_TYPE_LIDAR_LUMINAR_IRIS_2129 DW_DEPRECATED_ENUM("dwCodecType deprecated")        = 55,
    DW_CODEC_TYPE_LIDAR_LUMINAR_IRIS_2142 DW_DEPRECATED_ENUM("dwCodecType deprecated")        = 56,
    DW_CODEC_TYPE_VIDEO_JPEG DW_DEPRECATED_ENUM("dwCodecType deprecated")                     = 57,
    DW_CODEC_TYPE_LIDAR_POINT_CLOUD_COMPRESSED DW_DEPRECATED_ENUM("dwCodecType deprecated")   = 58,
    DW_CODEC_TYPE_RADAR_IDC6 DW_DEPRECATED_ENUM("dwCodecType deprecated")                     = 59,
    DW_CODEC_TYPE_IMU_IDC6 DW_DEPRECATED_ENUM("dwCodecType deprecated")                       = 60,
    DW_CODEC_TYPE_GPS_IDC6 DW_DEPRECATED_ENUM("dwCodecType deprecated")                       = 61,
    DW_CODEC_TYPE_ROADCAST_AVMESSAGE DW_DEPRECATED_ENUM("dwCodecType deprecated")             = 62,
    DW_CODEC_TYPE_LIDAR_LUMINAR_SLIMV2 DW_DEPRECATED_ENUM("dwCodecType deprecated")           = 63,
    DW_CODEC_TYPE_USE_CODEC_MIME_TYPE                                                         = 64,
    DW_CODEC_TYPE_COUNT DW_DEPRECATED_ENUM("dwCodecType deprecated")                          = 65,
} dwCodecType;

enum
{
    DW_MAX_CODEC_MIME_TYPE_LENGTH = 64
};

/// Holds codec MIME type string
struct dwCodecMimeType
{
    /// Codec MIME type
    char8_t mime[DW_MAX_CODEC_MIME_TYPE_LENGTH];
};

/// Constant MIME type
typedef struct dwCodecMimeType dwCodecMimeTypeConst_t;

extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_UNK;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_VIDEO_H264;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_VIDEO_H264_ANNEX_B;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_VIDEO_H265;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_VIDEO_VP9;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_VIDEO_AV1;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_VIDEO_LRAW;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_VIDEO_LRAW_V2;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_VIDEO_XRAW;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_VIDEO_RAW;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_LIDAR_CUSTOM;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_LIDAR_HESAI_P128;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_LIDAR_HESAI_P128_V4P5;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_LIDAR_LUMINAR_H;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_LIDAR_OUSTER_OS1;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_LIDAR_OUSTER_OS2_128;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_LIDAR_VELODYNE_HDL32E;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_LIDAR_VELODYNE_HDL64E;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_LIDAR_VELODYNE_VLP16;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_LIDAR_VELODYNE_VLP16HR;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_LIDAR_VELODYNE_VLP32C;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_LIDAR_VELODYNE_VLS128;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_RADAR_CONTINENTAL_ARS430;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_RADAR_CONTINENTAL_ARS430CAN;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_RADAR_CONTINENTAL_ARS430RDI;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_RADAR_CONTINENTAL_ARS430RDI_V2;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_RADAR_CUSTOM;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_RADAR_DELPHI;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_IMU_BOSCH;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_IMU_CAN;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_IMU_CONTINENTAL;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_IMU_CUSTOM;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_IMU_DATASPEED;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_IMU_NOVATEL_ASCII;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_IMU_NOVATEL_BINARY;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_IMU_NV_SIM;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_IMU_XSENS_BINARY;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_IMU_XSENS_CAN;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_IMU_XSENS_NMEA;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_GPS_CUSTOM;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_GPS_DATASPEED;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_GPS_DW_BINARY;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_GPS_NMEA;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_GPS_NOVATEL;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_GPS_NOVATEL_BINARY;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_GPS_NOVATEL_ASCII;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_GPS_NV_SIM;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_GPS_UBLOX;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_GPS_XSENS_BINARY;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_CAN_DW_BINARY;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_TIMESENSOR_DW_BINARY;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_DATA_DW_BINARY;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_ION;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_ULTRASONIC_VALEO_USV;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_ULTRASONIC_VALEO_USV_BSAMPLE;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_RADAR_CUSTOM_EX;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_LIDAR_CUSTOM_EX;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_RADAR_HELLA_ADAS6;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_LIDAR_LUMINAR_IRIS_2129;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_LIDAR_LUMINAR_IRIS_2142;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_VIDEO_JPEG;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_LIDAR_POINT_CLOUD_COMPRESSED;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_RADAR_IDC6;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_IMU_IDC6;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_GPS_IDC6;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_ROADCAST_AVMESSAGE;
extern dwCodecMimeTypeConst_t const DW_CODEC_MIME_TYPE_LIDAR_LUMINAR_SLIMV2;

/// Codec Capability
typedef enum {
    DW_CODEC_CAPABILITY_HARDWARE    = 0,
    DW_CODEC_CAPABILITY_SOFTWARE    = 1,
    DW_CODEC_CAPABILITY_UNSUPPORTED = 2,
} dwCodecCapability;

/// Encoder Rate Control Mode
typedef enum {
    DW_ENCODER_RATE_CONTROL_MODE_CONSTQP = 0, // Constant QP mode
    DW_ENCODER_RATE_CONTROL_MODE_CBR     = 1, // Constant Bitrate mode; constant bitrate throughout all data independently of data complexity
    DW_ENCODER_RATE_CONTROL_MODE_VBR     = 2, // Variable Bitrate mode; aim for target (average) bitrate, but variates bitrate depending on local data complexity
} dwEncoderRateControlMode;

/**
 * @brief Generic struct storing data output from codec.
 * @note The member pts and dts are gotten from function 'getCurrentTime()'.
 *       It has different implements on different platform. More information
 *       see units 'dw_core_time'.
 *
 */
typedef struct dwCodecPacket
{
    /// Any flags necessary for the frame.
    uint64_t flags;

    ///  Presentation time stamp, in microseconds.
    dwTime_t pts;

    ///  Decoding time stamp, in microseconds.
    dwTime_t dts;

    ///  Encoded data.
    uint8_t* data;

    ///  Size of the data pointer in bytes.
    uint64_t dataSizeBytes;

    void* reserved[4];
} dwCodecPacket;

/// Encoder rate control parameters
typedef struct dwEncoderConfigRateControl
{
    uint32_t gopSize;              // number of pictures in one GOP(Group Of Pictures)
    uint32_t bFrames;              // Number of bFrames between two reference frames
    uint32_t pFrames;              // whether there should be p frames; boolean value
    uint32_t quality;              // Target quality, range(0 - 51); used for CONSTQP mode
    uint32_t profile;              // MPEG-4 encoding Profile
    uint32_t level;                // MPEG-4 encoding Level range (1 - 6.2)
    uint64_t averageBitRate;       // target bitrate for VBR and CBR modes
    uint64_t maxBitRate;           // max bitrate for VBR mode
    dwEncoderRateControlMode mode; // rate control mode
} dwEncoderConfigRateControl;

/// NVMedia encoder configs
typedef struct dwEncoderConfigNVMedia
{
    uint8_t encoderInstance; // NVMedia-only
    uint8_t h265UltraFastEncodeEnable;
} dwEncoderConfigNVMedia;

/// Encoder specific configs
typedef struct dwEncoderConfig
{
    dwEncoderConfigRateControl rateControl;
    dwEncoderConfigNVMedia nvMedia;
} dwEncoderConfig;

/// Holds codec MIME type string
struct CodecMimeType
{
    /// Codec MIME type
    char8_t mime[DW_MAX_CODEC_MIME_TYPE_LENGTH];
};

#ifdef __cplusplus
}
#endif

/** @} */
#endif // DW_CODECS_CODEC_H_
