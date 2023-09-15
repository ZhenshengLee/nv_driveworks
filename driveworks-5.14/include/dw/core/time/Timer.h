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
// SPDX-FileCopyrightText: Copyright (c) 2017-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Timer</b>
 *
 * @b Description: This file defines the timer interface.
 */

/**
 * @defgroup timer_group Timer
 * @ingroup core_group
 *
 * @brief Defines the methods for the timer interface
 * @{
 */

#ifndef DW_CORE_TIMER_H_
#define DW_CORE_TIMER_H_

#include <dw/core/base/Exports.h>
#include <dw/core/base/Types.h>
#include <dw/core/base/Status.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dwTimeSourceObject* dwTimeSourceHandle_t;
typedef struct dwTimeSourceObject const* dwConstTimeSourceHandle_t;

typedef struct dwTimerObject* dwTimerHandle_t;
typedef struct dwTimerObject const* dwConstTimerHandle_t;
typedef void (*dwTimerWork)(void* ptr);

/**
 * @brief Creates and initializes a DW Timer.
 * This method creates a timer instance and registers it with the
 * primary time source for the context.
 *
 * @param[out] timer A pointer to the timer handle
 * @param[in] timerName Name of the timer, which will be associated with the
 *                      created thread. Note maximal length is 16 (incl. null character). Any length above will be cut.
 * @param[in] source Specifies the handle to the time source which is used as time provider.
 *
 * @return DW_INVALID_ARGUMENT if pointer to the timer handle is NULL. Provide a valid timer pointer.<br>
 *         DW_INVALID_HANDLE if provided source is NULL or invalid. Provide a valid source pointer.<br>
 *         DW_SUCCESS if initialization is done without any error.
 */
DW_API_PUBLIC
dwStatus dwTimer_initializeFromSource(dwTimerHandle_t* const timer, char8_t const* const timerName,
                                      dwTimeSourceHandle_t const source);

/**
 * @brief Release the timer instance
 *
 * @param[in] timer The timer handle
 *
 * @return DW_INVALID_ARGUMENT if pointer to the timer handle is NULL. Provide a valid timer pointer.<br>
 *         DW_INVALID_HANDLE if provided timer handle is invalid. Provide a valid timer handle.<br>
 *         DW_SUCCESS if timer released without any error.
 */
DW_API_PUBLIC
dwStatus dwTimer_release(dwTimerHandle_t const timer);

/**
 * @brief Synchronously cancels all scheduled work associated with this timer.
 *        This call will allow pending work to complete for cancelling the job.
 *
 * @param[in] timer A handle to the timer
 *
 * @return DW_INVALID_HANDLE if provided context handle is invalid, i.e. null or of wrong type. Provide a valid timer pointer.<br>
 *         DW_NOT_SUPPORTED if the timer does not support this feature.<br>
 *         DW_SUCCESS if timer cancelled without any error.
 */
DW_API_PUBLIC
dwStatus dwTimer_cancelSync(dwTimerHandle_t const timer);

/**
 * @brief Asynchronously cancels all scheduled work associated with this timer
 *
 * @param[in] timer A handle to the timer
 *
 * @return DW_INVALID_HANDLE if provided context handle is invalid, i.e. null or of wrong type. Provide a valid timer pointer.<br>
 *         DW_NOT_SUPPORTED if the timer does not support this feature.<br>
 *         DW_SUCCESS if timer cancelled without any error.
 */
DW_API_PUBLIC
dwStatus dwTimer_cancelAsync(dwTimerHandle_t const timer);

/**
 * @brief Scheduled a task to be run at a future time (non-recurring)
 *
 * @param[in] task Function pointer for the task to be executed
 * @param[in] clientData Pointer to data to be passed back to task
 * @param[in] startTime Time when the task should be activated
 * @param[in] timer Handle to the timer object
 *
 * @return DW_INVALID_HANDLE if provided timer handle is invalid, i.e. null or of wrong type. Provide a valid timer pointer.<br>
 *         DW_CALL_NOT_ALLOWED if the task has been already scheduled.<br>
 *         DW_NOT_SUPPORTED if the timer does not support this feature.<br>
 *         DW_FAILURE if the task scheduling failed.<br>
 *         DW_SUCCESS if timer scheduled without any error.
 */
DW_API_PUBLIC
dwStatus dwTimer_scheduleTaskOneShot(dwTimerWork const task, void* const clientData,
                                     dwTime_t const startTime, dwTimerHandle_t const timer);

/**
 * @brief Scheduled a task to be run at a future time (recurring)
 *
 * @param[in] task Function pointer for the task to be executed
 * @param[in] clientData Pointer to data to be passed back to task
 * @param[in] startTime Time when the task should be activated
 * @param[in] period Period at which to fire the task
 * @param[in] timer Handle to the timer object
 *
 * @return DW_INVALID_HANDLE if provided timer handle is invalid, i.e. null or of wrong type. Provide a valid timer pointer.<br>
 *         DW_CALL_NOT_ALLOWED if the task has been already scheduled.<br>
 *         DW_NOT_SUPPORTED if the timer does not support this feature.<br>
 *         DW_FAILURE if the task scheduling failed.<br>
 *         DW_SUCCESS if timer scheduled without any error.
 */
DW_API_PUBLIC
dwStatus dwTimer_scheduleTaskRecurring(dwTimerWork const task, void* const clientData,
                                       dwTime_t const startTime, dwTime_t const period, dwTimerHandle_t const timer);

#ifdef __cplusplus
}
#endif

/** @} */
#endif // DW_CORE_TIMER_H_
