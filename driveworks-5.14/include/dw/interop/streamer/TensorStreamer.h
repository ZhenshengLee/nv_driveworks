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
// SPDX-FileCopyrightText: Copyright (c) 2020-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Tensor Streamer</b>
 *
 * @b Description: This file defines the tensor streamer function.
 */

/**
 * @defgroup tensor_streamer_group Tensor Streamer
 * @ingroup interop_group
 *
 * @brief Defines the tensor streamer function, enabling streaming tensors between different APIs.
 *
 * @{
 *
 */
#ifndef DW_INTEROP_TENSORSTREAMER_H_
#define DW_INTEROP_TENSORSTREAMER_H_

#include <dw/dnn/tensor/Tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/// \ingroup interop
typedef struct dwDNNTensorStreamerObject* dwDNNTensorStreamerHandle_t;

/**
 * Creates and initializes the tensor streamer capable of moving tensors
 * between different API types. For example an tensor can be moved from
 * NvMedia API into GL textured type or from CPU memory to CUDA.
 *
 * @param[out] streamer A handle to the tensor streamer (if successfully initialized.)
 * @param[in] from The properties of the source tensor for streaming.
 * @param[in] to The type of the destination tensor.
 * @param[in] ctx The DriveWorks context.
 *
 * @return DW_SUCCESS if the translator was created, <br>
 *         DW_INVALID_ARGUMENT if the given tensor types are invalid or the streamer pointer is null, <br>
 *         DW_INVALID_HANDLE if the given context handle is invalid,i.e null or of wrong type  <br>
 *         DW_NOT_IMPLEMENTED if the desired streamer between given types is not implemented, <br>
 *         DW_NOT_AVAILABLE if the desired streamer between given two types is currently not available
 *                            due to missing resources or non-initialized APIs, <br>
 *         or DW_NOT_SUPPORTED if the desired streamer cannot work with given pixel format or type.
 *
 */
DW_API_PUBLIC
dwStatus dwDNNTensorStreamer_initialize(dwDNNTensorStreamerHandle_t* streamer,
                                        const dwDNNTensorProperties* from, dwDNNTensorType to,
                                        dwContextHandle_t ctx);

/**
 * Sends an tensor through the streamer acting as the producer.
 *
 * @note The ownership of the tensor remains by the caller.
 *
 * @param[in] tensor A dwTensorHandle_t to send through the streamer
 * @param[in] streamer A handle to the tensor streamer.
 *
 * @return DW_INVALID_HANDLE if the given streamer handle (producer) is invalid,i.e. of wrong type  <br>
 *         DW_INVALID_ARGUMENT if the given streamer handle (producer) or the tensor handle is NULL  <br>
 *         DW_INVALID_HANDLE if the @link dwDNNTensorType dwDNNTensorType @endlink of the input tensor doesn't match
 *                           the streamer's <br>
 *         DW_BUSY_WAITING if the tensor cannot be sent. Non fatal error, it is possible to retry<br>
 *         or DW_SUCCESS otherwise.
 *
 */
DW_API_PUBLIC
dwStatus dwDNNTensorStreamer_producerSend(dwDNNTensorHandle_t tensor, dwDNNTensorStreamerHandle_t streamer);

/**
 * The producer streamer waits for the tensor sent to be returned by the consumer. Waits for a max timeout. Returns
 * a pointer to the dwDNNTensorHandle of the returned tensor, if the pointer passed is not null, otherwise it does not return
 * a pointer
 *
 * @param[out] tensor A pointer to a dwDNNTensorHandle_t returned by the producer
 * @param[in] timeout_us Timeout in milliseconds before interrupting the waiting and return a timeout error.
 * @param[in] streamer A handle to the tensor streamer.
 *
 * @return DW_INVALID_ARGUMENT if the given streamer handle (producer) is null  <br>
 *         DW_INVALID_HANDLE if the given streamer handle (producer) is invalid,i.e. of wrong type  <br>
 *         DW_TIME_OUT if times out <br>
 *         or DW_SUCCESS otherwise.
 *
 */
DW_API_PUBLIC
dwStatus dwDNNTensorStreamer_producerReturn(dwDNNTensorHandle_t* tensor, dwTime_t timeout_us,
                                            dwDNNTensorStreamerHandle_t streamer);

/**
 * Receive a pointer to a dwDNNTensorHandle_t from the streamer, acting as a consumer. Can wait until timeout before
 * failing.
 *
 * @param[out] tensor A pointer to a dwDNNTensorHandle_t sent by the producer, not null if successfully converted.
 * @param[in] timeout_us Timeout in milliseconds before interrupting the waiting for producer.
 * @param[in] streamer A handle to the tensor streamer.
 *
 * @return DW_INVALID_ARGUMENT if the given tensor handle is NULL  <br>
 *         DW_INVALID_HANDLE if the given streamer handle (consumer) is invalid, i.e. null or of wrong type  <br>
 *         DW_TIME_OUT if times out <br>
 *         or DW_SUCCESS otherwise.
 *
 */
DW_API_PUBLIC
dwStatus dwDNNTensorStreamer_consumerReceive(dwDNNTensorHandle_t* tensor, dwTime_t timeout_us,
                                             dwDNNTensorStreamerHandle_t streamer);

/**
 * Return the received tensor back to the producer. Ownership is given back and tensor goes back to null
 *
 * @param[in] tensor A pointer to the dwDNNTensorHandle_t to return to the producer.
 * @param[in] streamer A handle to the tensor streamer.
 *
 * @return DW_INVALID_HANDLE if the given streamer handle (consumer) is invalid, i.e. null or of wrong type <br>
 *         DW_INTERNAL_ERROR if the tensor has been received yet or an underlying error depending on the
 *                           producer/consumer combination has occurred<br>
 *         or DW_SUCCESS otherwise.
 *
 */
DW_API_PUBLIC
dwStatus dwDNNTensorStreamer_consumerReturn(dwDNNTensorHandle_t* tensor, dwDNNTensorStreamerHandle_t streamer);

/**
 * Sets the CUDA stream for CUDA related streaming operations such as
 * post and receive.
 *
 * @note The ownership of the stream remains by the callee.
 *
 * @param[in] stream The CUDA stream to be used. Default is 0.
 * @param[in] streamer A handle to the tensor streamer.
 *
 * @return DW_INVALID_HANDLE if the given context handle is invalid,i.e null or of wrong type  <br>
 *         DW_NOT_SUPPORTED if the streamer does not work with CUDA tensors, <br>
 *         or DW_SUCCESS otherwise.
 *
 */
DW_API_PUBLIC
dwStatus dwDNNTensorStreamer_setCUDAStream(cudaStream_t stream, dwDNNTensorStreamerHandle_t streamer);

/**
 * Get CUDA stream used by the tensor streamer.
 *
 * @param[out] stream CUDA stream used by the streamer
 * @param[in] streamer A handle to the tensor streamer.
 *
 * @return DW_INVALID_HANDLE if the given context handle is invalid,i.e null or of wrong type  <br>
 *         DW_NOT_SUPPORTED if the streamer does not work with CUDA tensors, <br>
 *         or DW_SUCCESS otherwise.
 **/
DW_API_PUBLIC
dwStatus dwDNNTensorStreamer_getCUDAStream(cudaStream_t* stream, dwDNNTensorStreamerHandle_t streamer);

/**
 * Get tensor properties of the tensor received from the streamer.
 *
 * @param[out] props Properties of the resulting tensors.
 * @param[in] streamer A handle to the tensor streamer.
 *
 * @return DW_INVALID_HANDLE if the given context handle is invalid,i.e null or of wrong type  <br>
 *         DW_SUCCESS if successful.
 */
DW_API_PUBLIC
dwStatus dwDNNTensorStreamer_getOutputProperties(dwDNNTensorProperties* props, dwDNNTensorStreamerHandle_t const streamer);

/**
 * Releases the tensor streamer. This releases all memory and closes
 * all related resources. The handle is set to null.
 *
 * @param[in] streamer A handle to the tensor streamer.
 *
 * @return DW_SUCCESS if successful.
 *
 */
DW_API_PUBLIC
dwStatus dwDNNTensorStreamer_release(dwDNNTensorStreamerHandle_t streamer);

#ifdef __cplusplus
}
#endif

/** @} */
#endif // DW_INTEROP_TENSORSTREAMER_H_
