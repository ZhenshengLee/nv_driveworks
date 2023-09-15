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
// SPDX-FileCopyrightText: Copyright (c) 2016-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: CodecHeader Method</b>
 *
 * @b Description: This file defines codec header methods for encoding and decoding.
 */

/**
 * @defgroup codecheader CodecHeader
 * @brief Defines the codecheader types.
 *
 *
 * @{
 */

#ifndef DW_SENSORS_CODEC_CODECHEADER_H_
#define DW_SENSORS_CODEC_CODECHEADER_H_

#include <dw/core/base/Types.h>
#include <dw/core/base/Status.h>
#include <dw/core/context/Context.h>

#include <dw/sensors/Codec.h>
#include <dw/sensors/CodecHeaderPlugin.h>

#ifdef __cplusplus
extern "C" {
#endif

/// CodecHeader handle.
typedef struct dwCodecHeaderObject* dwCodecHeaderHandle_t;

/// CodecHeader handle.
typedef struct dwCodecHeaderObject const* dwCodecHeaderConstHandle_t;

/**
* Create a new codec header with the specified dwCodecType
*
* @param[out] handle the pointer in which the created codec header is stored
* @param[in] type the type of codec header to create
* @param[in] codecConfig the config associated with the type. Such as
*                       dwCodecConfigVideo for video codec types
*
* @return DW_NOT_SUPPORT if type is not a supported type. <br>
*         DW_INVALID_ARGUMENT if handle is nullptr
*         DW_SUCCESS
*
*/
DW_DEPRECATED("dwCodecType deprecated. use dwCodecHeader_createNew instead")
dwStatus dwCodecHeader_create(dwCodecHeaderHandle_t* handle, dwCodecType type, void* codecConfig);

/**
* Create a new codec header with the specified codec MIME type
*
* @param[out] handle the pointer in which the created codec header is stored
* @param[in] codecMimeType the type of codec header to create
* @param[in] codecConfig the config associated with the type. Such as
*                       dwCodecConfigVideo for video codec types
* @param[in] params pointer to additional parameters of associated codec header
* @param[in] context DW context
*
* @return DW_NOT_SUPPORT if type is not a supported type. <br>
*         DW_INVALID_ARGUMENT if handle is nullptr
*         DW_SUCCESS
*
*/
dwStatus dwCodecHeader_createNew(dwCodecHeaderHandle_t* handle, char8_t const* codecMimeType, void* codecConfig, void* params, dwContextHandle_t context);

/**
* Destroy a previously allocated codec header
*
* @param[in] handle the handle to release
*
* @return DW_SUCCESS
*
*/
dwStatus dwCodecHeader_destroy(dwCodecHeaderHandle_t handle);

/**
* Get the specified dwCodecType in a codec header
*
* @param[out] type stored in codec header
* @param[in]  handle the handle of codec header
*
* @return DW_INVALID_ARGUMENT if type or codecHeader is nullptr. <br>
*         DW_SUCCESS
*
*/
DW_DEPRECATED("dwCodecType deprecated. use dwCodecHeader_getCodecTypeNew instead")
dwStatus dwCodecHeader_getCodecType(dwCodecType* const type, dwCodecHeaderConstHandle_t const handle);

/**
* Get the specified codec MIME type in a codec header
*
* @param[out] type stored in codec header
* @param[in]  size the size of the output buffer provided
* @param[in]  handle the handle of codec header
*
* @return DW_INVALID_ARGUMENT if type or codecHeader is nullptr. <br>
*         DW_SUCCESS
*
*/
dwStatus dwCodecHeader_getCodecTypeNew(char8_t* type, const size_t size, dwCodecHeaderConstHandle_t const handle);

/**
* Get the specified dwMediaType in a codec header
*
* @param[out] type stored in codec header
* @param[in]  handle the handle of codec header
*
* @return DW_INVALID_ARGUMENT if type or codecHeader is nullptr. <br>
*         DW_SUCCESS
*
*/
dwStatus dwCodecHeader_getMediaType(dwMediaType* const type, dwCodecHeaderConstHandle_t const handle);

/**
* Get the specified dwCodecConfig* in a codec header
*
* @param[out] config stored in codec header
* @param[in] configSize size of the memory pointed by the passed in config pointer
* @param[in] mediaType media type corresonding to the codecHeader
* @param[in] handle the handle of codec header
*
* @return DW_INVALID_ARGUMENT if config or codecHeader is nullptr, or if configSize mismatches the expected size of specified mediaType. <br>
*         DW_SUCCESS
*
*/
dwStatus dwCodecHeader_getConfig(void* config, size_t configSize, dwMediaType mediaType, dwCodecHeaderConstHandle_t handle);

/**
* Serializes the codec header into JSON string
*
* @param[out] dataSize size of the serialized data
* @param[out] data pointer to memory block to store the serialized string
* @param[in] maxDataSize size of the memory block pointed to by data
* @param[in] handle the handle of codec header
*
* @return DW_INVALID_ARGUMENT if dataSize, data or codecHeader is nullptr, or if maxDataSize is smaller than the serialized size. <br>
*         DW_SUCCESS
*
*/
dwStatus dwCodecHeader_save(size_t* const dataSize, void* data, size_t const maxDataSize, dwCodecHeaderConstHandle_t const handle);

/**
* Load a serialized codec header from JSON string
*
* @param[out] handle the handle of codec to populate
* @param[in] data pointer to memory block to containing the serialized codec header
* @param[in] dataSize size of the serialized data
* @param[in] context DW context
*
* @return DW_INVALID_ARGUMENT if data or codecHeader is nullptr. <br>
*         DW_SUCCESS
*
*/
dwStatus dwCodecHeader_load(dwCodecHeaderHandle_t* const handle, void* const data, size_t const dataSize, dwContextHandle_t context);

/**
* Release resources associated with codec header
*
* @param[in] handle the handle of codec to populate
*
* @return DW_SUCCESS
*
*/
dwStatus dwCodecHeader_release(dwCodecHeaderHandle_t const handle);

/**
* Register CodecHeader plugin with associated codec MIME type
*
* @param[in] codecMimeType the codec type of registered CodecHeader plugin
* @param[in] funcTable pointer to CodecHeader plugin function pointer table
* @param[in] ctx DW context
*
* @return DW_INVALID_ARGUMENT if codecMimeType, funcTable or ctx is nullptr. <br>
*         DW_BUFFER_FULL if no available entry in registration factory
*         DW_SUCCESS
*
*/
dwStatus dwCodecHeader_register(char const* codecMimeType, dwCodecHeaderPluginFunctions const* const funcTable, dwContextHandle_t ctx);

#ifdef __cplusplus
}
#endif

/** @} */
#endif // DW_SENSORS_CODEC_CODECHEADER_H_
