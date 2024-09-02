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
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Decoder Plugin Interface</b>
 *
 * @b Description: This file defines decoder plugin interface for decoding.
 */

#ifndef DW_SENSORS_CODECS_PLUGINS_DECODERPLUGIN_H_
#define DW_SENSORS_CODECS_PLUGINS_DECODERPLUGIN_H_

#include <dw/core/base/Types.h>
#include <dw/core/base/Status.h>
#include <dw/sensors/codecs/Codec.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Decoder plugin handle.
typedef void* dwDecoderPluginHandle_t;

/**
* Release a decoder managed by the plugin module.
*
* @param[in] handle The handle to a Decoder created previously with the 'dwCodecHeaderPlugin_initializeDecoder' interface.
*
* @return DW_INVALID_ARGUMENT if the handle is NULL or invalid <br>
*         DW_SUCCESS
*/
typedef dwStatus (*dwDecoderPlugin_release)(dwDecoderPluginHandle_t handle);

/**
* Reset a decoder managed by the plugin module.
*
* @param[in] handle The handle to a Decoder created previously with the 'dwCodecHeaderPlugin_initializeDecoder' interface.
*
* @return DW_INVALID_ARGUMENT if the handle is NULL or invalid <br>
*         DW_SUCCESS
*/
typedef dwStatus (*dwDecoderPlugin_reset)(dwDecoderPluginHandle_t handle);

/**
* Decode encoded packet data into a frame.
*
* @param[in] packet The pointer to the encoded packet data.
* @param[in] handle The handle to a Decoder created previously with the 'dwCodecHeaderPlugin_initializeDecoder' interface.
*
* @return DW_INVALID_ARGUMENT if the packet or handle is NULL or invalid <br>
*         DW_BUFFER_FULL if the decoder is unable to accept any more packets <br>
*         DW_SUCCESS
*/
typedef dwStatus (*dwDecoderPlugin_decode)(dwCodecPacket* packet, dwDecoderPluginHandle_t handle);

/**
* Drains previously queued decode requests.
*
* @param[out] data The pointer to the frame data.
* @param[out] dataSize Size of `data`.
* @param[in] handle The handle to a Decoder created previously with the 'dwCodecHeaderPlugin_initializeDecoder' interface.
*
* @return DW_INVALID_ARGUMENT if the packet or handle is NULL or invalid <br>
*         DW_NOT_READY if the enqueued frames require further inputs. <br>
*         DW_NOT_AVAILABLE if there's nothing to be drained.<br>
*         DW_SUCCESS
*/
typedef dwStatus (*dwDecoderPlugin_drain)(void* data, size_t dataSize, dwDecoderPluginHandle_t handle);

/// Function Table exposing decoder plugin functions
typedef struct dwDecoderPluginFunctions
{
    dwDecoderPlugin_release release; //!< Release plugin decoder specified by its handle
    dwDecoderPlugin_reset reset;     //!< Reset plugin decoder specified by its handle
    dwDecoderPlugin_decode decode;   //!< Decode a packet from input
    dwDecoderPlugin_drain drain;     //!< Return a decoded frame
} dwDecoderPluginFunctions;

#ifdef __cplusplus
}
#endif

#endif // DW_SENSORS_CODECS_PLUGINS_DECODERPLUGIN_H_
