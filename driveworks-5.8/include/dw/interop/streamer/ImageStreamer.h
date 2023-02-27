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
 * <b>NVIDIA DriveWorks API: Image Streamer</b>
 *
 * @b Description: This file defines the image streamer function.
 */

/**
 * @defgroup streamer_group Image Streamer
 * @ingroup image_group
 *
 * @brief Defines the image streamer function, enabling streaming images between different APIs.
 *
 * @{
 *
 */
#ifndef DW_IMAGE_IMAGESTREAMER_H_
#define DW_IMAGE_IMAGESTREAMER_H_

#include <dw/core/base/Config.h>
#include <dw/core/base/Exports.h>
#include <dw/core/context/Context.h>

#include <dw/image/Image.h>

#ifdef __cplusplus
extern "C" {
#endif

/// \ingroup image
typedef struct dwImageStreamerObject* dwImageStreamerHandle_t;

/// \ingroup image

/// Cross-process image streamer operation modes.
typedef enum dwImageStreamerCrossProcessMode {
    DW_IMAGE_STREAMER_CROSS_PROCESS_PRODUCER,
    DW_IMAGE_STREAMER_CROSS_PROCESS_CONSUMER
} dwImageStreamerCrossProcessMode;

/// Sets of parameters for cross-process image streamer creation.
typedef struct dwImageStreamerCrossProcessModeParams
{

    /** Array to additional parameters provided to cross-process image streamer creation.
     * In general, this array has a form of key/value pairs separated by commas,
     * i.e., streamName=nvscistream_0 where nvscistream_0 represents nvscistream endpoint
     * name corresponding to producer or consumer endpoint as per nvsciipc specifications
     * or for deprecated EGL, i.e. mode=socket,file=/tmp/file.socket
     */
    const char* parameters;

    /** Operation mode of the cross-process image streamer.
     */
    dwImageStreamerCrossProcessMode mode;
} dwImageStreamerCrossProcessModeParams;

/**
 * Sets the CUDA stream for CUDA related streaming operations such as
 * post and receive.
 *
 * @note The ownership of the stream remains by the callee.
 *
 * @param[in] stream The CUDA stream to be used. Default is 0.
 * @param[in] streamer A handle to the image streamer.
 *
 * @return DW_INVALID_HANDLE if the given context handle is invalid,i.e null or of wrong type  <br>
 *         DW_NOT_SUPPORTED if the streamer does not work with CUDA images, <br>
 *         or DW_SUCCESS otherwise.
 *
 */
DW_API_PUBLIC
dwStatus dwImageStreamer_setCUDAStream(cudaStream_t stream, dwImageStreamerHandle_t streamer);

/**
 * Get CUDA stream used by the image streamer.
 *
 * @param[out] stream CUDA stream used by the streamer
 * @param[in] streamer A handle to the image streamer.
 *
 * @return DW_INVALID_HANDLE if the given context handle is invalid,i.e null or of wrong type  <br>
 *         DW_NOT_SUPPORTED if the streamer does not work with CUDA images, <br>
 *         or DW_SUCCESS otherwise.
 **/
DW_API_PUBLIC
dwStatus dwImageStreamer_getCUDAStream(cudaStream_t* stream, dwImageStreamerHandle_t streamer);

/**
 * Get image properties of the image received from the streamer.
 *
 * @param[out] props Properties of the resulting images
 * @param[in] streamer A handle to the image streamer.
 *
 * @return DW_INVALID_HANDLE if the given context handle is invalid,i.e null or of wrong type  <br>
 *         DW_SUCCESS if successful.
 *
 * @note An image streamer does not modify the pixel format or pixel type, however, meta information
 *       as well as sizes might differ. For example an image containing embedded lines, will be
 *       most probably unfolded, i.e. the datalines will become part of the image in the new domain.
 */
DW_API_PUBLIC
dwStatus dwImageStreamer_getOutputProperties(dwImageProperties* props, dwImageStreamerHandle_t const streamer);

/**
 * Creates and initializes the image streamer capable of moving images
 * between different API types. For example an image can be moved from
 * NvMedia API into GL textured type or from CPU memory to CUDA.
 *
 * In general an image streamer does not perform any modifications on the format of the stored pixel data.
 * However, in certain cases when going between APIs some meta information might be lost or transformed. Especially
 * if an image contains embedded data lines, in some cases the embedded information is lost, while in others
 * it became part of the image. To find all supported src/dst pairs as well as supported formats
 * refer to the user guide. After successful streamer creation 'dwImageStreamer_getOutputProperties()'
 * will return the expected properties of resulting images.
 *
 * @param[out] streamer A handle to the image streamer (if successfully initialized.)
 * @param[in] from The properties of the source image for streaming.
 * @param[in] to The type of the destination image.
 * @param[in] ctx The DriveWorks context.
 *
 * @return DW_SUCCESS if the translator was created, <br>
 *         DW_INVALID_ARGUMENT if the given image types are invalid or the streamer pointer is null, <br>
 *         DW_INVALID_HANDLE if the given context handle is invalid,i.e null or of wrong type  <br>
 *         DW_NOT_IMPLEMENTED if the desired streamer between given types is not implemented, <br>
 *         DW_NOT_AVAILABLE if the desired streamer between given two types is currently not available
 *                            due to missing resources or non-initialized APIs, <br>
 *         or DW_NOT_SUPPORTED if the desired streamer cannot work with given pixel format or type.
 *
 */
DW_API_PUBLIC
dwStatus dwImageStreamer_initialize(dwImageStreamerHandle_t* streamer,
                                    const dwImageProperties* from, dwImageType to,
                                    dwContextHandle_t ctx);

#ifdef VIBRANTE
/// \ingroup image
/**
 * Creates and initializes the image streamer capable of moving images
 * between different API types across processes using NvSciStream.
 * @note DW_IMAGE_STREAMER_CROSS_PROCESS_CONSUMER must be initialized before
 * DW_IMAGE_STREAMER_CROSS_PROCESS_PRODUCER.
 *
 * @param[out] streamer A handle to the image streamer (if successfully initialized.)
 * @param[in] from The type of the source image and the expected image properties.
 * @param[in] to The type of the destination image.
 * @param[in] params Specifies the parameters for cross-process image streamer creation.
 * @param[in] ctx The DriveWorks context.
 *
 * @return DW_SUCCESS if the translator was created, <br>
 *         DW_INVALID_ARGUMENT if the given image types are invalid or the streamer pointer is null, <br>
 *         DW_INVALID_HANDLE if the given context handle is invalid,i.e null or of wrong type  <br>
 *         DW_NOT_IMPLEMENTED if the desired streamer between given types is not implemented, <br>
 *         DW_NOT_AVAILABLE if the desired streamer between given two types is currently not available
 *                          due to missing resources or non-initialized APIs, <br>
 *         or DW_NOT_SUPPORTED if the desired streamer cannot work with given pixel format or type.
 */
DW_API_PUBLIC
DW_DEPRECATED("The cross process streamer is deprecated and will be removed in the next version. The dwChannel can be used instead for cross process image streaming")
dwStatus dwImageStreamer_initializeCrossProcess(dwImageStreamerHandle_t* streamer,
                                                const dwImageProperties* from, dwImageType to,
                                                dwImageStreamerCrossProcessModeParams params,
                                                dwContextHandle_t ctx);

/**
 * Creates a variant of the cross process streamer based on EGL
 * @note DW_IMAGE_STREAMER_CROSS_PROCESS_CONSUMER must be initialized before
 * DW_IMAGE_STREAMER_CROSS_PROCESS_PRODUCER.
 *
 * @param[out] streamer A handle to the image streamer (if successfully initialized.)
 * @param[in] from The type of the source image and the expected image properties.
 * @param[in] to The type of the destination image.
 * @param[in] params Specifies the parameters for cross-process image streamer creation.
 * @param[in] ctx The DriveWorks context.
 *
 * @return DW_SUCCESS if the translator was created, <br>
 *         DW_INVALID_ARGUMENT if the given image types are invalid or the streamer pointer is null, <br>
 *         DW_INVALID_HANDLE if the given context handle is invalid,i.e null or of wrong type  <br>
 *         DW_NOT_IMPLEMENTED if the desired streamer between given types is not implemented, <br>
 *         DW_NOT_AVAILABLE if the desired streamer between given two types is currently not available
 *                          due to missing resources or non-initialized APIs, <br>
 *         or DW_NOT_SUPPORTED if the desired streamer cannot work with given pixel format or type.
 */
DW_API_PUBLIC
DW_DEPRECATED("The EGL based cross process streamer is deprecated and will be removed in the next version.")
dwStatus dwImageStreamer_initializeCrossProcessEGL(dwImageStreamerHandle_t* streamer,
                                                   const dwImageProperties* from, dwImageType to,
                                                   dwImageStreamerCrossProcessModeParams params,
                                                   dwContextHandle_t ctx);
#endif

/**
 * Sends an image through the streamer acting as the producer.
 *
 * @note The ownership of the image remains by the caller.
 *
 * @param[in] image A dwImageHandle_t to send through the streamer
 * @param[in] streamer A handle to the image streamer.
 *
 * @return DW_INVALID_HANDLE if the given streamer handle (producer) is invalid,i.e. of wrong type  <br>
 *         DW_INVALID_ARGUMENT if the given streamer handle (producer) or the image handle is NULL  <br>
 *         DW_INVALID_HANDLE if the @link dwImageType dwImageType @endlink of the input image doesn't match
 *                           the streamer's <br>
 *         DW_BUSY_WAITING if the image cannot be sent. Non fatal error, it is possible to retry<br>
 *         or DW_SUCCESS otherwise.
 *
 */
DW_API_PUBLIC
dwStatus dwImageStreamer_producerSend(dwImageHandle_t image, dwImageStreamerHandle_t const streamer);

/**
 * The producer streamer waits for the image sent to be returned by the consumer. Waits for a max timeout. Returns
 * a pointer to the dwImageHandle of the returned image, if the pointer passed is not null, otherwise it does not return
 * a pointer
 *
 * @param[out] image A pointer to a dwImageHandle_t returned by the producer
 * @param[in] timeoutUs Timeout in milliseconds before interrupting the waiting and return a timeout error. Image was not returned
 * @param[in] streamer A handle to the image streamer.
 *
 * @return DW_INVALID_ARGUMENT if the given streamer handle (producer) is null  <br>
 *         DW_INVALID_HANDLE if the given streamer handle (producer) is invalid,i.e. of wrong type  <br>
 *         DW_TIME_OUT if times out <br>
 *         or DW_SUCCESS otherwise.
 *
 */
DW_API_PUBLIC
dwStatus dwImageStreamer_producerReturn(dwImageHandle_t* const image, dwTime_t const timeoutUs,
                                        dwImageStreamerHandle_t const streamer);

/**
 * Receive a pointer to a dwImageHandle_t from the streamer, acting as a consumer. Can wait until timeout before
 * failing.
 *
 * @param[out] image A pointer to a dwImageHandle_t sent by the producer, not null if successfully converted.
 * @param[in] timeoutUs Timeout in milliseconds before interrupting the waiting for producer.
 * @param[in] streamer A handle to the image streamer.
 *
 * @return DW_INVALID_ARGUMENT if the given image handle is NULL  <br>
 *         DW_INVALID_HANDLE if the given streamer handle (consumer) is invalid, i.e. null or of wrong type  <br>
 *         DW_TIME_OUT if times out <br>
 *         or DW_SUCCESS otherwise.
 *
 */
DW_API_PUBLIC
dwStatus dwImageStreamer_consumerReceive(dwImageHandle_t* image, dwTime_t const timeoutUs,
                                         dwImageStreamerHandle_t const streamer);

/**
 * Return the received image back to the producer. Ownership is given back and image goes back to null
 *
 * @param[in] image A pointer to the dwImageHandle_t to return to the producer.
 * @param[in] streamer A handle to the image streamer.
 *
 * @return DW_INVALID_HANDLE if the given streamer handle (consumer) is invalid, i.e. null or of wrong type <br>
 *         DW_INTERNAL_ERROR if the image has been received yet or an underlying error depending on the
 *                           producer/consumer combination has occurred<br>
 *         or DW_SUCCESS otherwise.
 *
 */
DW_API_PUBLIC
dwStatus dwImageStreamer_consumerReturn(dwImageHandle_t* image, dwImageStreamerHandle_t const streamer);

/**
 * Releases the image streamer. This releases all memory and closes
 * all related resources. The handle is set to null.
 *
 * @param[in] streamer A handle to the image streamer.
 *
 * @return DW_SUCCESS if successful.
 *
 */
DW_API_PUBLIC
dwStatus dwImageStreamer_release(dwImageStreamerHandle_t streamer);

#ifdef __cplusplus
}
#endif

/** @} */
#endif // DW_IMAGE_IMAGESSTREAMER_H_
