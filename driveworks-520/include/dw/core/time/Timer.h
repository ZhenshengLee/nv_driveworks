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

/// Handle represents a TimeSource object.
typedef struct dwTimeSourceObject* dwTimeSourceHandle_t;
/// Handle represents a const TimeSource object.
typedef struct dwTimeSourceObject const* dwConstTimeSourceHandle_t;

/// Handle represents a Timer object.
typedef struct dwTimerObject* dwTimerHandle_t;
/// Handle represents a const Timer object.
typedef struct dwTimerObject const* dwConstTimerHandle_t;
/// dwTimerWork is a timer schedule task callback function, input void* ptr passes the extra data to the task
typedef void (*dwTimerWork)(void* ptr);

/**
 * @brief Creates a new DW timer. The new timer is initially disarmed.
 * This call registers the timer with the primary time source of the context.
 *
 * @param[out] timer A pointer to the timer handle
 * @param[in] timerName Name of the timer, Note maximal length is 16 (incl. null character). Names longer than 16 characters will be truncated.
 * @param[in] source Specifies the handle to the time source which is used as time provider.
 * @note @a source is gotten from dwContext_getTimeSource, which depends on the platform users use.
 * @return DW_INVALID_ARGUMENT if pointer to the timer handle is NULL.<br>
 *         DW_INVALID_HANDLE if provided source is NULL or invalid.<br>
 *         DW_SUCCESS if initialization is done without any error.<br>
 *         DW_FAILURE failed to create a timer from system.<br>
 *         DW_BUFFER_FULL number of timer created has exceeded the max number of timer(Now is 512).
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
 */
DW_API_PUBLIC
dwStatus dwTimer_initializeFromSource(dwTimerHandle_t* const timer, char8_t const* const timerName,
                                      dwTimeSourceHandle_t const source);

/**
 * @brief Release the timer instance.
 *        The timer will be stopped if it's started.
 *
 * @param[in] timer The timer handle
 * @return DW_INVALID_ARGUMENT if pointer to the timer handle is NULL.<br>
 *         DW_SUCCESS if timer released without any error.
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwTimer_release(dwTimerHandle_t const timer);

/**
 * @brief Synchronously cancels all scheduled work associated with this timer. Synchronously means this call will wait until the task is canceled. 
 *        This call will allow pending work to complete before cancelling the job.
 *        This call will clear the all the time value assigned to timer. And then stop the timer.
 *
 * @param[in] timer A handle to the timer
 * @return DW_INVALID_HANDLE if provided context handle is invalid, i.e. null or of wrong type.<br>
 *         DW_SUCCESS if timer cancelled without any error.<br>
 *         DW_FAILURE if clear time value assigned to @a timer fails.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwTimer_cancelSync(dwTimerHandle_t const timer);

/**
 * @brief Asynchronously cancels all scheduled work associated with this timer. Asynchronously means this call will not wait the cancel operation finished.
 *        This call will allow pending work to complete before cancelling the job.
 *        This call will clear the all the time value assigned to timer. And then stop the timer.
 *
 * @param[in] timer A handle to the timer
 *
 * @return DW_INVALID_HANDLE if provided context handle is invalid, i.e. null or of wrong type.<br>
 *         DW_SUCCESS if timer cancelled without any error.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
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
 * @note @a startTime should be at least 1 microsecond into the future. If users set 0, this call will set this value to 1 microsecond into the future.
 * @return DW_INVALID_HANDLE if provided timer handle is invalid, i.e. null or of wrong type.
 *         DW_CALL_NOT_ALLOWED if the task has been already scheduled.<br>
 *         DW_NOT_SUPPORTED if the timer does not support this feature.<br>
 *         DW_FAILURE if the task scheduling failed.<br>
 *         DW_SUCCESS if timer scheduled without any error.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwTimer_scheduleTaskOneShot(dwTimerWork const task, void* const clientData,
                                     dwTime_t const startTime, dwTimerHandle_t const timer);

/**
 * @brief Scheduled a task to be run at a future time (recurring)
 *
 * @param[in] task Function pointer for the task to be executed
 * @param[in] clientData Pointer to data to be passed back to task
 * @param[in] startTime Time when the task should be activated, in Us.
 * @param[in] period Period at which to fire the task, in Us.
 * @param[in] timer Handle to the timer object
 *
 * @note @a period should be greater than 0 if users want to schedule recurring task. If set as 0, it will schedule one-shot task as dwTimer_scheduleTaskOneShot().
 * @note @a startTime If the start time is less or equal to current time, timer will start at current time.
 * @return DW_INVALID_HANDLE if provided timer handle is invalid, i.e. null or of wrong type.
 *         DW_CALL_NOT_ALLOWED if the task has been already scheduled.<br>
 *         DW_NOT_SUPPORTED if the timer does not support this feature.<br>
 *         DW_FAILURE if the task scheduling failed.<br>
 *         DW_SUCCESS if timer scheduled without any error.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwTimer_scheduleTaskRecurring(dwTimerWork const task, void* const clientData,
                                       dwTime_t const startTime, dwTime_t const period, dwTimerHandle_t const timer);

#ifdef __cplusplus
}
#endif

/** @} */
#endif // DW_CORE_TIMER_H_
