/*
 * Copyright (c) 2019-2023 NVIDIA CORPORATION.  All Rights Reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */

/**
 * \file stm_manager.h
 *
 * \brief STM Runtime
 *
 * Overview:
 * STM is a co-operative, non-preemptive, static scheduling framework for real-time systems.
 * The STM runtime is a library that executes a schedule produced offline by the STM compiler, enforcing
 * data dependency ordering and control flow ordering across engines on a Tegra SoC.
 *
 * STM Manager:
 * Manager correspond to a single process controlling schedule execution. Manager is expected to launch and connect with
 * Master before any Client connect to Master.
 *
 * STM Master:
 * Alongside STM clients, STM master expects a Manager process to connect with it. STM Manager will block at the call 
 * to stmScheduleManagerInit() until the STM master process has been launched.
 **/

#ifndef STM_MANAGER_H_
#define STM_MANAGER_H_

#if __GNUC__ >= 4
#define STM_API __attribute__((visibility("default")))
#endif

#include <stdint.h>
#include "stm_error.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Initialize STM schedule manager context.
 *
 *  @remark This API must be called before any other STM Schedule Management APIs. This API will block until the STM master
 *  process is started. Manager's context should be cleaned up with stmScheduleManagerExit() after STM has completed its execution.
 *  There is only one schedule manager in the system
 *  @return stmErrorCode_t, the completion code of the operation:
 *  - STM_ERROR_INSUFFICIENT_MEMORY : failure of memory allocation of STM.
 *  - STM_ERROR_BAD_VALUE : unsupported value for discriminator.
 *  - STM_ERROR_NOT_SUPPORTED : connection to master not enabled. ignore this error for dual SOC case and soc is secondary
 *  - STM_ERROR_GENERIC : Failed to receive messages from master.
 */
STM_API stmErrorCode_t stmScheduleManagerInit(const char* scheduleManagerName);

/** @brief Initialize STM schedule manager context with a discriminator.
 *
 *  @remark This API must be called before any other STM Schedule Management APIs. This API will block until the STM master
 *  process with this \p discriminator is started. Manager's context should be cleaned up with stmScheduleManagerExit()
 *  after STM has completed its execution.
 *  Manager must call at most one of the two: stmScheduleManagerInit() or stmScheduleManagerInitWithDiscriminator(),
 *  if both are called single or multiple times, it will cause undefined behavior.
 *  There is only one schedule manager in the system. All negative values of \p discriminator are equivalent and the same
 *  as calling stmScheduleManagerInit()
 *  @return stmErrorCode_t, the completion code of the operation:
 *  - STM_ERROR_INSUFFICIENT_MEMORY : failure of memory allocation of STM.
 *  - STM_ERROR_BAD_VALUE : unsupported value for discriminator.
 *  - STM_ERROR_NOT_SUPPORTED : connection to master not enabled. ignore this error for dual SOC case and soc is secondary
 *  - STM_ERROR_GENERIC : Failed to receive messages from master.
 */
STM_API stmErrorCode_t stmScheduleManagerInitWithDiscriminator(const char* scheduleManagerName, int32_t discriminator);
/** @brief Cleans up STM schedule Manager context. No STM APIs can be called after this.
 *
 *  @remark This API can only be called once per call to stmScheduleManagerInit(); doing so multiple times will cause undefined
 *  behavior.
 *  @return stmErrorCode_t, the completion code of the operation:
 *  - STM_ERROR_NOT_INITIALIZED if not init api was called or called but failed.
 *  - STM_ERROR_GENERIC if fail to release some OS resources. check debug logs for further information
 */
STM_API stmErrorCode_t stmScheduleManagerExit(void);

/** @brief Schedule Management API for starting Schedule Execution.
 *
 *  @remark This API must be called between stmScheduleManagerInit()/stmScheduleManagerInitWithDiscriminator() and stmScheduleManagerExit()
 *  @return stmErrorCode_t, the completion code of the operation:
 *  - STM_ERROR_NOT_INITIALIZED if not init api was called or called but failed.
 *  - STM_ERROR_BAD_STATE_TRANSITION : If current state is not READY.
 *  - STM_ERROR_BAD_VALUE : if format of state messages are incorrect.
 *  - STM_ERROR_GENERIC : if message send/receive fails.
 */
STM_API stmErrorCode_t stmStartSchedule(uint16_t scheduleId);

/** @brief Schedule Management API for stopping Schedule Execution.
 *
 *  @remark This API must be called between stmScheduleManagerInit()/stmScheduleManagerInitWithDiscriminator() and stmScheduleManagerExit()
 *  @return stmErrorCode_t, the completion code of the operation:
 *  - STM_ERROR_NOT_INITIALIZED if not init api was called or called but failed.
 *  - STM_ERROR_BAD_STATE_TRANSITION : If current state is not RUNNING.
 *  - STM_ERROR_BAD_VALUE : if format of state messages are incorrect.
 *  - STM_ERROR_INSUFFICIENT_MEMORY : if malloc failure happened.
 *  - STM_ERROR_GENERIC : if message send/receive fails.
 */
STM_API stmErrorCode_t stmStopSchedule(uint16_t scheduleId);

/** @brief Schedule Management API for stopping current Schedule Execution and starting a new one in a per-hyperepoch
 *  fashion to minimize deadtime.
 *
 *  @remark This API must be called between stmScheduleManagerInit()/stmScheduleManagerInitWithDiscriminator() and stmScheduleManagerExit()
 *  The schedules corresponding to startScheduleId and stopScheduleId must have the same number of hyperepochs and the same resource
 *  partitioning across hyperepochs.
 *  @return stmErrorCode_t, the completion code of the operation:
 *  - STM_ERROR_NOT_INITIALIZED if not init api was called or called but failed.
 *  - STM_ERROR_BAD_STATE_TRANSITION : If current state is not READY
 *  - STM_ERROR_BAD_VALUE : if format of state messages are incorrect.
 *  - STM_ERROR_INSUFFICIENT_MEMORY : if malloc failure happened.
 *  - STM_ERROR_GENERIC : if message send/receive fails.
 */
STM_API stmErrorCode_t stmRollOverSchedule(uint16_t startScheduleId, uint16_t stopScheduleId);

#ifdef __cplusplus
}
#endif

#endif //STM_MANAGER_H_
