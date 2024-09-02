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
 * <b>NVIDIA DriveWorks: Radar Decoder Plugin Interface</b>
 *
 * @b Description: This file defines the Radar decoder plugin interface layer.
 */

#ifndef DW_SENSORS_PLUGINS_RADAR_DECODER_H_
#define DW_SENSORS_PLUGINS_RADAR_DECODER_H_

/**
 * @defgroup sensor_plugin_radar_group Radar Plugin
 * Provides an interface for supporting non-standard Radar sensors.
 * @ingroup sensor_plugins_group
 *
 * @{
 */

#include <dw/core/base/Types.h>
#include <dw/sensors/radar/Radar.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Holds constants for a given radar.
typedef struct
{
    /// Radar properties.
    dwRadarProperties properties;

    /// Packet header size (in bytes).
    size_t headerSize;

    /// Packet max payload size (in bytes).
    size_t maxPayloadSize;

    /// Size of the vehicle state message to be sent to the radar.
    size_t vehicleStateSize;

    /// Size of the mount poisition message to be sent to the radar.
    size_t mountSize;
} _dwRadarDecoder_constants;

/**
 * Initializes the radar decoder.
 *
 * @return DW_FAILURE, DW_SUCCESS
 *
 */
dwStatus _dwRadarDecoder_initialize();

/**
 * Releases the radar decoder.
 *
 * @return DW_FAILURE, DW_SUCCESS
 *
 */
dwStatus _dwRadarDecoder_release();

/**
 * Gets constants associated with this radar sensor.
 *
 * @param[out] constants Constant parameters for this sensor are written here.
 *
 * @return DW_FAILURE, DW_SUCCESS
 *
 */
dwStatus _dwRadarDecoder_getConstants(_dwRadarDecoder_constants* constants);

/**
 * Decodes a packet from the radar, given a raw byte array and a specified
 * decoding format.
 *
 * @param[out] output A pointer to a recoded packet output.
 * @param[in] buffer Byte array containing raw.
 * @param[in] length Length of the byte array.
 * @param[in] scanType Type of scan for this packet.
 *
 * @return DW_FAILURE, DW_SUCCESS
 *
 */
dwStatus _dwRadarDecoder_decodePacket(dwRadarScan* output,
                                      const uint8_t* buffer,
                                      const size_t length,
                                      const dwRadarScanType scanType);

/**
 * Determines whether a byte array of packet data has a valid radar signature.
 *
 * @param[in] buffer A pointer to the byte array containing packet data.
 * @param[in] length Length of the buffer in bytes.
 * @param[out] remaining Number of bytes remaining for packet completion.
 *
 * @return DW_SUCCESS if the signature is valid, or DW_FAILURE otherwise.
 *
 */
dwStatus _dwRadarDecoder_synchronize(const uint8_t* buffer,
                                     const size_t length,
                                     size_t* remaining);

/**
 * Determines whether a byte array of packet data is valid.
 * The definition of "valid" here is implementation-specific.
 *
 * @param[in] buffer A pointer to the byte array containing packet data.
 * @param[in] length Length of the buffer in bytes.
 * @param[out] scanType Returns the radar scan type for this packet.
 *
 * @return DW_SUCCESS if the data is valid, or DW_FAILURE otherwise.
 *
 */
dwStatus _dwRadarDecoder_validatePacket(const uint8_t* buffer,
                                        const size_t length,
                                        dwRadarScanType* scanType);

/**
 * Returns the status of the scan.
 * If this function returns true, the complete scan is passed to the user.
 *
 * @param[in] scanType The type of scan for this data.
 * @param[in] buffer A pointer to an array of byte arrays containing packet data
 *                   that potentially belongs to a single scan.
 * @param[in] length A pointer to the array of lengths of the packets.
 * @param[out] numPackets Number of packets in each array.
 * @return @c True if the scan is completed, or @c False otherwise.
 *
 */
bool _dwRadarDecoder_isScanComplete(dwRadarScanType scanType,
                                    const uint8_t** buffer,
                                    size_t* length,
                                    size_t numPackets);

/**
 * Encodes data from a /ref dwRadarVehicleState into a raw byte array.
 * The raw data is in a format expected by the radar.
 *
 * @param[in,out] buffer A pointer to the byte array into which the function
 *                       places encoded data. The caller must deallocate
 *                       the buffer when it is no longer needed.
 * @param[in] maxOutputSize Length of the output buffer, in bytes.
 * @param[in] packet A pointer to the dynamics packet to be encoded.
 *
 * @return DW_FAILURE, DW_SUCCESS
 *
 */
dwStatus _dwRadarDecoder_encodeVehicleState(uint8_t* buffer,
                                            const size_t maxOutputSize,
                                            const dwRadarVehicleState* packet);

/**
 * Encodes data from a /ref dwRadarMountPosition into a raw byte array.
 * The raw data is in a format expected by the radar.
 *
 * @param[in,out] buffer A pointer to the byte array into which the function
 *                       places encoded data. The caller must deallocate
 *                       the buffer when it is no longer needed.
 * @param[in] maxOutputSize Length of the output buffer in bytes.
 * @param[in] packet A pointer to the mounting packet to be encoded.
 *
 * @return DW_FAILURE, DW_SUCCESS
 *
 */
dwStatus _dwRadarDecoder_encodeMountPosition(uint8_t* buffer,
                                             const size_t maxOutputSize,
                                             const dwRadarMountPosition* packet);

/** @} */

#ifdef __cplusplus
}
#endif

#endif
