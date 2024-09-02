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
// SPDX-FileCopyrightText: Copyright (c) 2017-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks: Lidar Decoder Plugin Interface</b>
 *
 * @b Description: This file defines the Lidar decoder plugin interface layer.
 */

#ifndef DW_SENSORS_PLUGINS_LIDAR_DECODER_H_
#define DW_SENSORS_PLUGINS_LIDAR_DECODER_H_

/**
 * @defgroup sensor_plugin_lidar_group Lidar Plugin
 * Provides an interface for supporting non-standard Lidar sensors.
 * @ingroup sensor_plugins_group
 *
 * @{
 */

#include <dw/core/base/Types.h>
#include <dw/sensors/lidar/Lidar.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Holds constants for a given lidar.
typedef struct
{
    /// Radar properties.
    dwLidarProperties properties;

    /// Packet header size, in bytes.
    size_t headerSize;

    /// Packet max payload size, in bytes.
    size_t maxPayloadSize;
} _dwLidarDecoder_constants;

/**
 * Initializes the lidar decoder interface.
 *
 * @return DW_FAILURE, DW_SUCCESS
 *
 */
dwStatus _dwLidarDecoder_initialize(const float32_t spinFrequency);

/**
 * Releases the lidar decoder interface.
 *
 * @return DW_FAILURE, DW_SUCCESS
 *
 */
dwStatus _dwLidarDecoder_release();

/**
 * Gets constants associated with this lidar sensor.
 *
 * @param[out] constants Constant parameters for this sensor are written here.
 *
 * @return DW_FAILURE, DW_SUCCESS
 *
 */
dwStatus _dwLidarDecoder_getConstants(_dwLidarDecoder_constants* constants);

/**
 * Decodes a packet from the lidar, given a raw byte array and a specified decoding
 * format.
 *
 * @param[out] output Decoded packet output is written here.
 * @param[in] buffer Byte array containing raw data.
 * @param[in] length Length of the byte array.
 *
 * @return DW_FAILURE, DW_SUCCESS
 *
 */
dwStatus _dwLidarDecoder_decodePacket(dwLidarDecodedPacket* output,
                                      const uint8_t* buffer,
                                      const size_t length);

/**
 * Determines whether a byte array of packet data has a valid lidar signature.
 *
 * @param[in] buffer A pointer to the byte array containing packet data.
 * @param[in] length Length of the buffer in bytes.
 * @param[out] remaining Number of bytes remaining for packet completion.
 *
 * @return DW_SUCCESS if the lidar signature is valid, or DW_FAILURE otherwise.
 *
 */
dwStatus _dwLidarDecoder_synchronize(const uint8_t* buffer,
                                     const size_t length,
                                     size_t* remaining);

/**
 * Determines whether a byte array of packet data is
 * valid. The definition of "valid" here is implementation-specific.
 *
 * @param[in] buffer A pointer to the byte array containing packet data.
 * @param[in] length Length of the buffer in bytes.
 *
 * @return DW_SUCCESS if the packet data is valid, or DW_FAILURE otherwise.
 *
 */
dwStatus _dwLidarDecoder_validatePacket(const uint8_t* buffer,
                                        const size_t length);

/** @} */

#ifdef __cplusplus
}
#endif

#endif
