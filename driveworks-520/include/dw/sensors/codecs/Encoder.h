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

/**
 * @file
 * <b>NVIDIA DriveWorks API: Encoder Methods</b>
 *
 * @b Description: This file defines encoder methods.
 */

/**
 * @defgroup encoder Encoder
 *
 * @brief Defines the encoder types.
 * @{
 */
#ifndef DW_SENSORS_CODECS_ENCODER_H
#define DW_SENSORS_CODECS_ENCODER_H

#include <dw/core/base/Exports.h>
#include <dw/core/base/Types.h>
#include <dw/core/context/Context.h>
#include <dw/sensors/codecs/Codec.h>
#include <dw/sensors/codecs/CodecHeader.h>

#ifdef __cplusplus
extern "C" {
#endif

/// \brief Handle representing an encoder.
typedef struct dwCodecEncoderObject* dwEncoderHandle_t;

/**
 * This function initializes the encoder.
 *
 * @param[out] encoder Handle to the encoder.
 * @param[in] config Handle to the codecHeader.
 * @param[in] encoderConfig Pointer to the encoder config.
 * @param[in] context Context handle.
 *
 * @return DW_SUCCESS - If the call is successful. <br>
 *         DW_INVALID_ARGUMENT - If the handle is NULL or invalid.
 */
dwStatus dwEncoder_initialize(dwEncoderHandle_t* encoder,
                              dwCodecHeaderHandle_t config,
                              dwEncoderConfig const* encoderConfig,
                              dwContextHandle_t context);

/**
 * This function releases the encoder.
 *
 * @param[in] encoder Handle to the encoder.
 *
 * @return DW_SUCCESS - If the call is successful. <br>
 *         DW_INVALID_ARGUMENT - If \p handle is nullptr.
 */
dwStatus dwEncoder_release(dwEncoderHandle_t encoder);

/**
 * This function consumes all data from the internal data buffer.
 *
 * @param[in] encoder Handle to the encoder.
 *
 * @return DW_SUCCESS - If the call is successful. <br>
 *         DW_INVALID_ARGUMENT - If \p handle is nullptr.
 */
dwStatus dwEncoder_flush(dwEncoderHandle_t encoder);

/**
 * This function enqueues a codec packet. Output objects are available via dwEncoder_drainPacket.
 *
 * @param[in] data Frame to enqueue to the encoder.
 * @param[in] dataSize Size of the data to encode.
 * @param[in] dataType Type of the data to encode.
 * @param[in] encoder Handle to the encoder.
 *
 * @return DW_SUCCESS - If the output objects(s) are available via drainPacket() <br>
 *         DW_NOT_AVAILABLE - If the codec requires more data. <br>
 *         DW_NOT_READY - If too many objects have been fed in without calling drainPacket(). <br>
 *         DW_FAILURE - In case of an internal failure
 */
dwStatus dwEncoder_encode(void const* data, size_t dataSize, dwMediaType dataType, dwEncoderHandle_t encoder);

/**
 * This function drains previously queued encode requests.
 * Sometimes the encoder doesn't give a packet until few frames(say predictive encoding)
 * in which case the client needs to push more packets before calling drain.
 *
 * @param[out] packet Pointer to the codec packet.
 * @param[in] encoder Handle to the encoder.
 *
 * @return DW_SUCCESS - When a fully realized output object is available <br>
 *         DW_NOT_READY - If there's nothing to be drained.
 */
dwStatus dwEncoder_drainPacket(dwCodecPacket* packet, dwEncoderHandle_t encoder);

/**
 * This function returns the codec packet.
 *
 * @param[in] packet Pointer to the codec packet.
 * @param[in] encoder Handle to the encoder.
 *
 * @return DW_SUCCESS - If the call is successful. <br>
 *         DW_INVALID_ARGUMENT - If the encoder handle is NULL or invalid
 */
dwStatus dwEncoder_returnPacket(dwCodecPacket* packet, dwEncoderHandle_t encoder);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_SENSORS_CODECS_ENCODER_H
