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

/**
 * @file
 * <b>NVIDIA DriveWorks API: CodecHeader Plugin Interface</b>
 *
 * @b Description: This file defines codec header plugin interface for encoding and decoding.
 */

#ifndef DW_SENSORS_CODEC_CODECHEADERPLUGIN_H_
#define DW_SENSORS_CODEC_CODECHEADERPLUGIN_H_

#include <dw/core/base/Types.h>
#include <dw/core/base/Status.h>
#include <dw/core/context/Context.h>

#include <dw/sensors/Codec.h>
#include <dw/sensors/codecs/DecoderPlugin.h>

#ifdef __cplusplus
extern "C" {
#endif

/// CodecHeader plugin handle.
typedef void* dwCodecHeaderPluginHandle_t;

/**
* Initialize a new handle to the CodecHeader managed by the plugin module.
*
* @param[out] handle A pointer to CodecHeader handle
* @param[in] codecConfig The config associated with the CodecHeader. Such as dwCodecConfigVideo for video codec types
* @param[in] params Optional params for the CodecHeader plugin
*
* @return DW_INVALID_ARGUMENT if handle is nullptr <br>
*         DW_SUCCESS
*/
typedef dwStatus (*dwCodecHeaderPlugin_initializeHandle)(dwCodecHeaderPluginHandle_t* handle, void* const codecConfig, void* const params);

/**
* Release a CodecHeader managed by the plugin module.
*
* @param[in] handle The handle to a CodecHeader created previously with the 'dwCodecHeaderPlugin_initializeHandle' interface.
*
* @return DW_INVALID_ARGUMENT if the handle is NULL or invalid <br>
*         DW_SUCCESS
*/
typedef dwStatus (*dwCodecHeaderPlugin_release)(dwCodecHeaderPluginHandle_t handle);

/**
* Get the CodecType of the CodecHeader.
*
* @param[out] mimeType The CodecType of given CodecHeader.
* @param[in] size Size of `mimeType`.
* @param[in] handle The handle to a CodecHeader created previously with the 'dwCodecHeaderPlugin_initializeHandle' interface.
*
* @return DW_INVALID_ARGUMENT if the handle is NULL or invalid <br>
*         DW_SUCCESS
*/
typedef dwStatus (*dwCodecHeaderPlugin_getCodecType)(char* const mimeType, const size_t size, dwCodecHeaderPluginHandle_t handle);

/**
* Get the MediaType of the CodecHeader.
*
* @param[out] mediaType The MideaType of given CodecHeader.
* @param[in] handle The handle to a CodecHeader created previously with the 'dwCodecHeaderPlugin_initializeHandle' interface.
*
* @return DW_INVALID_ARGUMENT if the handle is NULL or invalid <br>
*         DW_SUCCESS
*/
typedef dwStatus (*dwCodecHeaderPlugin_getMediaType)(dwMediaType* const mediaType, dwCodecHeaderPluginHandle_t handle);

/**
* Get the max raw packet size of the CodecHeader.
*
* @param[out] size Size of max raw packet.
* @param[in] handle The handle to a CodecHeader created previously with the 'dwCodecHeaderPlugin_initializeHandle' interface.
*
* @return DW_SUCCESS
*/
typedef dwStatus (*dwCodecHeaderPlugin_getMaxRawPacketSize)(size_t* const size, dwCodecHeaderPluginHandle_t handle);

/**
* Get the properties of the CodecHeader.
*
* @param[out] properties The properties of given CodecHeader.
* @param[in] size Size of `properties`.
* @param[in] handle The handle to a CodecHeader created previously with the 'dwCodecHeaderPlugin_initializeHandle' interface.
*
* @return DW_INVALID_ARGUMENT if the handle is NULL or invalid <br>
*         DW_NOT_IMPLEMENTED if the given CodecHeader doesn't have properties. <br>
*         DW_SUCCESS
*/
typedef dwStatus (*dwCodecHeaderPlugin_getProperties)(void* const properties, size_t const size, dwCodecHeaderPluginHandle_t handle);

/**
* Get the config of the CodecHeader.
*
* @param[out] config The codec config of given CodecHeader.
* @param[out] configSize Size of `config`.
* @param[in] handle The handle to a CodecHeader created previously with the 'dwCodecHeaderPlugin_initializeHandle' interface.
*
* @return DW_INVALID_ARGUMENT if the handle is NULL or invalid <br>
*         DW_SUCCESS
*/
typedef dwStatus (*dwCodecHeaderPlugin_getConfig)(void* config, size_t const configSize, dwCodecHeaderPluginHandle_t handle);

/**
* Serializes the codec header
*
* @param[out] dataSize Size of the serialized data
* @param[out] data Pointer to memory block to store the serialized string
* @param[in] maxDataSize Size of the memory block pointed to by data
* @param[in] handle The handle to a CodecHeader created previously with the 'dwCodecHeaderPlugin_initializeHandle' interface.
*
* @return DW_INVALID_ARGUMENT if dataSize, data or handle is nullptr, or if maxDataSize is smaller than the serialized size. <br>
*         DW_SUCCESS
*/
typedef dwStatus (*dwCodecHeaderPlugin_save)(size_t* const dataSize, void* data, size_t const maxDataSize, dwCodecHeaderPluginHandle_t handle);

/**
* Check if given data can be loaded
*
* @param[in] data Pointer to memory block to containing the serialized codec header
* @param[in] dataSize Size of the serialized data
*
* @return DW_INVALID_ARGUMENT data or handle is nullptr<br>
* 		  DW_FAILURE The CodecHeader can't load the given serialized data <br>
*         DW_SUCCESS
*/
typedef dwStatus (*dwCodecHeaderPlugin_canLoad)(void const* const data, size_t const dataSize);

/**
* Load a serialized codec header
*
* @param[out] handle A pointer to CodecHeader handle
* @param[in] data Pointer to memory block to containing the serialized codec header
* @param[in] dataSize Size of the serialized data
*
* @return DW_INVALID_ARGUMENT data or handle is nullptr<br>
*         DW_SUCCESS
*/
typedef dwStatus (*dwCodecHeaderPlugin_load)(dwCodecHeaderPluginHandle_t* handle, void const* const data, size_t const dataSize);

/**
* Initialize a new handle to the Decoder managed by the plugin module.
*
* @param[out] decoderHandle A pointer to Decoder handle
* @param[out] decoderFuncs Function table provided by the Decoder
* @param[in] handle The handle to a CodecHeader created previously with the 'dwCodecHeaderPlugin_initializeHandle' interface.
* @param[in] ctx DW context
*
* @return DW_INVALID_ARGUMENT handle or ctx is nullptr<br>
*         DW_SUCCESS
*/
typedef dwStatus (*dwCodecHeaderPlugin_initializeDecoder)(dwDecoderPluginHandle_t* decoderHandle, dwDecoderPluginFunctions const** decoderFuncs, dwCodecHeaderPluginHandle_t handle, dwContextHandle_t ctx);

/// Function Table exposing plugin functions
typedef struct dwCodecHeaderPluginFunctions
{
    dwCodecHeaderPlugin_initializeHandle initializeHandle;
    dwCodecHeaderPlugin_release release;
    dwCodecHeaderPlugin_getCodecType getCodecType;
    dwCodecHeaderPlugin_getMediaType getMediaType;
    dwCodecHeaderPlugin_getMaxRawPacketSize getMaxRawPacketSize;
    dwCodecHeaderPlugin_getProperties getProperties;
    dwCodecHeaderPlugin_getConfig getConfig;
    dwCodecHeaderPlugin_save save;
    dwCodecHeaderPlugin_canLoad canLoad;
    dwCodecHeaderPlugin_load load;
    dwCodecHeaderPlugin_initializeDecoder initializeDecoder;
} dwCodecHeaderPluginFunctions;

#ifdef __cplusplus
}
#endif

#endif // DW_SENSORS_CODEC_CODECHEADERPLUGIN_H_
