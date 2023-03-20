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
// SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Egomotion Producer/State Methods</b>
 *
 * @b Description: This file defines the producer/state API of egomotion module
 *
 */

/**
 * @defgroup egomotion_state_group Producer/State API
 *
 * @brief Defines producer/state related API
 *
 * @{
 *
 * Typical use case of a producer/state API is to (de)serialize a state to other processes/threads. A state
 * will represent anything known to egomotion module up-to the last point of filter update. In most of the cases
 * the state will store internally the history from the egomotion module and provide access to the history
 * to query relative motion and absolute pose estimates.
 *
 * The use-case is usually can be described like this:
 *
 * ```
 *    dwEgomotionHandle_t egomotion = initializeEgomotion();
 *
 *    // get current state estimation
 *    dwConstEgomotionStateHandle_t state;
 *    dwEgomotion_createEmptyState(&state, egomotion);   // create empty state to feed egomotion to
 *    ...
 *    dwEgomotion_getState(state, egomotion); // copy internal state
 *    {
 *        // query state for a relative motion since last request
 *        dwTransformation3f motionLastToNow;
 *        dwEgomotionState_computeRelativeTransformation(&motionLastToNow, getLastTime(), getCurrentTime(), state);
 *        ...
 *
 *        // query state for the latest known absolute estimation
 *        dwEgomotionResult estimation;
 *        dwEgomotionState_getEstimation(&estimation, state);
 *        ...
 *    }
 *    ...
 * ```
 *
 * @ingroup egomotion_group
 */

#ifndef DW_EGOMOTION_PRODUCER_STATE_H_
#define DW_EGOMOTION_PRODUCER_STATE_H_

#include "Egomotion.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dwEgomotionStateObject* dwEgomotionStateHandle_t;
typedef struct dwEgomotionStateObject const* dwConstEgomotionStateHandle_t;

/**
 * @brief Defines egomotion state initialization parameters
 */
typedef struct dwEgomotionStateParams
{
    //! Type of the motion model used to hold data by the state
    dwMotionModel type;

    //! Maximal number of elements to keep in history.
    size_t historySize;
} dwEgomotionStateParams;

/**
 * Create empty state from the given egomotion module. The call is equivalent to calling @ref dwEgomotionState_createEmpty with
 * the right selection of parameters.
 *
 * @param[out] state Handle to be set with pointer to created empty state.
 * @param[in] obj Handle of the motion model.
 *
 * @return DW_INVALID_ARGUMENT - if given state handle is null <br>
 *         DW_INVALID_HANDLE - if context handle is invalid <br>
 *         DW_SUCCESS - if the call was successful. <br>
 *
 * @note Ownership of the state goes back to caller. The state has to be released with @ref dwEgomotionState_release.
 */
DW_API_PUBLIC
dwStatus dwEgomotion_createEmptyState(dwEgomotionStateHandle_t* state, dwEgomotionHandle_t obj);

/**
 * Fills out already preallocated state handle. Given state handle can have a reduced history size.
 * Only as many history elements get copied into the target state as it can fit. A reduced history
 * reduces memory overhead during serialization/deserialization.
 *
 * @param[in] state Preallocated state to be filled with latest state
 * @param[in] obj Egomotion const handle.
 *
 * @return DW_INVALID_ARGUMENT - if given state type does not match that one of the egomotion <br>
 *         DW_NOT_AVAILABLE - if there is no state estimation available on egomotion yet. <br>
 *         DW_INVALID_HANDLE - if the provided egomotion or state handle are invalid. <br>
 *         DW_SUCCESS - if the call was successful. <br>
 */
DW_API_PUBLIC
dwStatus dwEgomotion_getState(dwEgomotionStateHandle_t state, dwEgomotionConstHandle_t obj);

/**
 * Returns the capacity of the history, i.e. same parameter as passed with @ref dwEgomotionParameters.historySize
 * or default value used instead. Returned capacity can be used by @ref dwEgomotionState_createEmpty for initialization
 * of the empty state, if same capacity of the history is desired as egomotion uses internally.
 *
 * @param[out] capacity Capacity to be set with history capacity of given egomotion instance.
 * @param[in] obj Egomotion handle.
 *
 * @return DW_INVALID_ARGUMENT - if the provided pointer is nullptr, or the given handle is invalid. <br>
 *         DW_SUCCESS <br>
 **/
DW_API_PUBLIC
dwStatus dwEgomotion_getHistoryCapacity(size_t* capacity, dwEgomotionConstHandle_t obj);

// --------------------------------------------
//  State
// --------------------------------------------

/**
 * @see dwEgomotion_computeRelativeTransformation
 **/
DW_API_PUBLIC
dwStatus dwEgomotionState_computeRelativeTransformation(dwTransformation3f* poseAtoB,
                                                        dwEgomotionRelativeUncertainty* uncertainty,
                                                        dwTime_t timestamp_a, dwTime_t timestamp_b,
                                                        dwConstEgomotionStateHandle_t obj);

/**
 * @see dwEgomotion_computeBodyTransformation
 **/
DW_API_PUBLIC
dwStatus dwEgomotionState_computeBodyTransformation(dwTransformation3f* const transformationAToB,
                                                    dwEgomotionRelativeUncertainty* const uncertainty,
                                                    dwTime_t const timestamp,
                                                    dwCoordinateSystem const coordinateSystemA,
                                                    dwCoordinateSystem const coordinateSystemB,
                                                    dwConstEgomotionStateHandle_t const obj);

/**
 * @see dwEgomotion_getEstimationTimestamp
 **/
DW_API_PUBLIC
dwStatus dwEgomotionState_getEstimationTimestamp(dwTime_t* timestamp, dwConstEgomotionStateHandle_t obj);

/**
 * @see dwEgomotion_getEstimation
 **/
DW_API_PUBLIC
dwStatus dwEgomotionState_getEstimation(dwEgomotionResult* result, dwConstEgomotionStateHandle_t obj);

/**
 * @see dwEgomotion_getUncertainty
 **/
DW_API_PUBLIC
dwStatus dwEgomotionState_getUncertainty(dwEgomotionUncertainty* result, dwConstEgomotionStateHandle_t obj);

/**
 * @see dwEgomotion_getGyroscopeBias
 **/
DW_API_PUBLIC
dwStatus dwEgomotionState_getGyroscopeBias(dwVector3f* gyroBias, dwConstEgomotionStateHandle_t obj);

/**
 * @see dwEgomotion_getHistorySize
 **/
DW_API_PUBLIC
dwStatus dwEgomotionState_getHistorySize(size_t* num, dwConstEgomotionStateHandle_t obj);

/**
 * @see dwEgomotion_getHistoryCapacity
 **/
DW_API_PUBLIC
dwStatus dwEgomotionState_getHistoryCapacity(size_t* capacity, dwConstEgomotionStateHandle_t obj);

/**
 * @see dwEgomotion_getHistoryElement
 **/
DW_API_PUBLIC
dwStatus dwEgomotionState_getHistoryElement(dwEgomotionResult* pose, dwEgomotionUncertainty* uncertainty,
                                            size_t index, dwConstEgomotionStateHandle_t obj);

/**
 * @see dwEgomotion_getMotionModel
 **/
DW_API_PUBLIC
dwStatus dwEgomotionState_getMotionModel(dwMotionModel* model, dwConstEgomotionStateHandle_t obj);

/**
 * Create empty state for a given motion model type. An empty state can be used to deserialize from a binary
 * buffer into the state. After deserialization state can be queried with state API for it's content.
 * The state can contain internally a history of up-to passed amount of elements.
 *
 * @param[out] state Handle to be set with pointer to created empty state.
 * @param[in] params Parameters to initialize the state
 * @param[in] ctx Handle of the context.
 *
 * @return DW_INVALID_ARGUMENT - if given state handle is null <br>
 *         DW_NOT_SUPPORTED - if given motion model is not supported by the state <br>
 *         DW_INVALID_HANDLE - if context handle is invalid <br>
 *         DW_SUCCESS - if the call was successful. <br>
 *
 * @note Ownership of the state goes back to caller. The state has to be released with @ref dwEgomotionState_release.
 * @note If passed `historySize` is smaller than the egomotion internal history capacity, any retrieval of the state with
 *       @ref dwEgomotion_getState will fill out this state's only with as much data as can fit, dropping oldest entries.
 *       This in turn would mean that calls to @ref dwEgomotionState_computeRelativeTransformation might not succeed if
 *       requested for timestamp outside of the covered history.
 */
DW_API_PUBLIC
dwStatus dwEgomotionState_createEmpty(dwEgomotionStateHandle_t* state, dwEgomotionStateParams params, dwContextHandle_t ctx);

/**
 * Releases the egomotion state previously created with @ref dwEgomotionState_createEmpty.
 *
 * @param[in] state State handle to be released.
 *
 * @return DW_INVALID_HANDLE - if the provided handle is invalid. <br>
 *         DW_SUCCESS - if the call was successful. <br>
 */
DW_API_PUBLIC
dwStatus dwEgomotionState_release(dwEgomotionStateHandle_t state);

/**
 * Get maximal number of bytes required for a buffer to contain serialized state.
 *
 * @param[out] bufferCapacity Capacity to be set with number of bytes required to hold serialized state.
 * @param[in] state State handle to be serialized
 *
 * @return DW_INVALID_ARGUMENT - if the provided @p bufferCapacity is nullptr. <br>
 *         DW_INVALID_HANDLE - if provided handle is invalid <br>
 *         DW_SUCCESS - if the call was successful. <br>
 */
DW_API_PUBLIC
dwStatus dwEgomotionState_getMaxNumBytes(size_t* bufferCapacity, dwConstEgomotionStateHandle_t state);

/**
 * Serialize the state out into the provided buffer.
 *
 * @param[out] numBytes Size to be set with number of bytes written to buffer. The number is always lower than or equal
 *                      @ref dwEgomotionState_getMaxNumBytes.
 * @param[out] buffer Buffer to be filled with serialized state.
 * @param[in] bufferCapacity Capacity of the buffer provided, i.e. must be at least @ref dwEgomotionState_getMaxNumBytes large.
 * @param[in] state State handle to be serialized
 *
 * @return DW_INVALID_ARGUMENT - if the provided pointers are nullptr. <br>
 *         DW_OUT_OF_BOUNDS - if the provided @p bufferCapacity isn't enough to hold the state
 *         DW_INVALID_HANDLE - if provided handle is invalid <br>
 *         DW_SUCCESS - if the call was successful. <br>
 */
DW_API_PUBLIC
dwStatus dwEgomotionState_serialize(size_t* numBytes, uint8_t* buffer, size_t bufferCapacity, dwConstEgomotionStateHandle_t state);

/**
 * Deserialize the state from the provided buffer.
 *
 * @param[in] buffer Buffer containing serialized state
 * @param[in] bufferSize Size of the buffer provided
 * @param[in] state State handle to be deserialized
 *
 * @return DW_INVALID_ARGUMENT - if the provided @p buffer is nullptr. <br>
 *         DW_OUT_OF_BOUNDS - if the provided @p bufferSize isn't enough to hold the state <br>
 *         DW_BUFFER_FULL - if the state's history isn't large enough to hold the deserialized state<br>
 *         DW_INVALID_VERSION - if the version of the data in the stream doesn't match current <br>
 *         DW_INVALID_HANDLE - if provided handle is invalid <br>
 *         DW_SUCCESS - if the call was successful. <br>
 */
DW_API_PUBLIC
dwStatus dwEgomotionState_deserialize(const uint8_t* buffer, size_t bufferSize, dwEgomotionStateHandle_t state);

/**
 * Copy existing egomotion state to preallocated state handle.
 *
 * @param[in] state Handle of pre-allocated egomotion state to be filled with a copy.
 * @param[in] source Handle of the existing egomotion state to be copied.
 *
 * @return DW_INVALID_HANDLE - if the provided egomotion state handle are invalid. <br>
 *         DW_NOT_AVAILABLE - if there not state estimation available on egomotion yet. <br>
 *         DW_SUCCESS - if the call was successful. <br>
 *
 */
DW_API_PUBLIC
dwStatus dwEgomotionState_copy(dwEgomotionStateHandle_t state, dwConstEgomotionStateHandle_t source);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_EGOMOTION_PRODUCER_STATE_H_
