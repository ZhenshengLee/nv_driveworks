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
 * <b>NVIDIA DriveWorks API: Global Egomotion Serialization Methods</b>
 *
 * @b Description: This file defines the serialization API of the global egomotion module
 */

/**
 * @defgroup global_egomotion_serializaation_group Serialization Interface
 * @brief Defines the serialization API of the global egomotion module
 *
 *
 * The typical use case of the serialization API is transmission of the state history across process
 * and/or machine boundaries. The state is transmitted as a variable-size array of state elements
 * and combined into a full state on deserialization.
 *
 * Example:
 *
 * ```
 *    // PRODUCER
 *    {
 *        // buffer that will be transmitted
 *        dwGlobalEgomotionStateElement buffer[CAPACITY];
 *        size_t bufferSize = 0;
 *
 *        // serialize and transmit state
 *        if(dwGlobalEgomotion_serialize(buffer, &bufferSize, CAPACITY, handle) == DW_SUCCESS)
 *        {
 *            // `buffer` now contains `bufferSize` (latest) estimates
 *            // transmit to consumer
 *            ...
 *        }
 *    }
 *
 *    // CONSUMER
 *    {
 *        // initialize deserializer module
 *        dwGlobalEgomotionHandle_t deserializer;
 *        dwGlobalEgomotion_initializeDeserializer(*deserializer, ...);
 *
 *        // dwGlobalEgomotionStateElement buffer[CAPACITY];
 *        size_t bufferSize = 0;
 *
 *        // receive `buffer` and `bufferSize` from producer
 *        ...
 *
 *        // deserialize state
 *        dwGlobalEgomotion_deserialize(buffer, bufferSize, deserializer);
 *
 *        // The `deserializer` handle can now be used with all getter APIs of
 *        // dwGlobalEgomotion (identifiable by the const handle input):
 *
 *        dwGlobalEgomotion_getTimestamp(...);
 *        dwGlobalEgomotion_computeEstimate(...);
 *        dwGlobalEgomotion_getEstimate(...);
 *        ...
 *    }
 * ```
 * @{
 */

#ifndef DW_EGOMOTION_GLOBAL_GLOBALEGOMOTIONSERIALIZER_H_
#define DW_EGOMOTION_GLOBAL_GLOBALEGOMOTIONSERIALIZER_H_

#include <dw/egomotion/global/GlobalEgomotion.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dwGlobalEgomotionSerializer* dwGlobalEgomotionSerializerHandle_t;
typedef struct dwGlobalEgomotionSerializer const* dwGlobalEgomotionSerializerConstHandle_t;

typedef struct dwGlobalEgomotionDeserializer* dwGlobalEgomotionDeserializerHandle_t;
typedef struct dwGlobalEgomotionDeserializer const* dwGlobalEgomotionDeserializerConstHandle_t;

/**
 * @brief Defines global egomotion state element
 */
typedef struct dwGlobalEgomotionStateElement
{
    dwGlobalEgomotionResult estimate;
} dwGlobalEgomotionStateElement;

/**
 * Initializes a global egomotion deserializer object.
 *
 * This object does not support the `dwGlobalEgomotion_addRelativeMotion` and
 * `dwGlobalEgomotion_addGPSMeasurement` APIs.
 *
 * The deserializer is released with `dwGlobalEgomotion_release`.
 *
 * @param[out] handle      A pointer to the handle for the created deserializer object.
 * @param[in]  historySize Size of the internal history, @see dwGlobalEgomotionParameters.historySize.
 *                         The size does not have to match that of the serialized module.
 * @param[in]  ctx         Handle of the context under which the objects are created.
 *
 * @return DW_INVALID_HANDLE - if the provided context handle is invalid <br>
 *         DW_SUCCESS - if initialization succeeded <br>
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
 */
DW_API_PUBLIC
dwStatus dwGlobalEgomotion_initializeDeserializer(dwGlobalEgomotionHandle_t* handle,
                                                  size_t historySize,
                                                  dwContextHandle_t ctx);

/**
 * Serialize the global egomotion state history out into the provided buffer.
 *
 * Fills out the provided buffer with the current global egomotion state history.
 * The most recent state history is serialized, up to the buffer capacity. Filling
 * order is oldest to most recent state history element.
 *
 * Provide a buffer with capacity larger or equal to the `historySize` parameter of
 * the global egomotion module to be serialized in order to ensure the full state
 * history can be stored. If the buffer capacity is too low to fit the state history,
 * the buffer is filled with the latest history elements that fit.
 *
 * @param[out] buffer          Buffer containing serialized state history.
 * @param[out] bufferSize      Number of elements written to `buffer`.
 * @param[in]  bufferCapacity  Capacity of `buffer` in number of elements.
 * @param[in]  handle Handle of a global egomotion object from which to serialize data.
 *
 * @return DW_INVALID_ARGUMENT - `buffer` or `bufferSize` is nullptr. <br>
 *         DW_INVALID_HANDLE   - global egomotion handle is invalid. <br>
 *         DW_NOT_AVAILABLE    - no state history to be serialized. <br>
 *         DW_SUCCESS          - `buffer` has been filled with state history. <br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwGlobalEgomotion_serialize(dwGlobalEgomotionStateElement* buffer, size_t* bufferSize,
                                     size_t const bufferCapacity, dwGlobalEgomotionHandle_t handle);

/**
 * Deserialize the global egomotion state history from the provided buffer.
 *
 * This call replaces any existing state history.
 *
 * @param[in] buffer          Buffer containing serialized state history.
 * @param[in] bufferSize      Number of elements in `buffer`.
 * @param[in] globalEgomotion Handle of a global egomotion object storing the deserialized history.
 *
 * @return DW_INVALID_ARGUMENT - if the provided `buffer` is nullptr or the data is invalid. <br>
 *         DW_INVALID_HANDLE   - if provided deserializer handle is invalid <br>
 *         DW_SUCCESS          - `buffer` has been deserialized into corresponding global egomotion
 *                               object.<br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwGlobalEgomotion_deserialize(dwGlobalEgomotionStateElement* buffer, size_t bufferSize,
                                       dwGlobalEgomotionHandle_t globalEgomotion);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_EGOMOTION_GLOBAL_GLOBALEGOMOTIONSERIALIZER_H_
