/*
 * Copyright (c) 2021-2022 NVIDIA CORPORATION.  All Rights Reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */

/**
 * \file stm_error.h
 *
 * \brief Return/error codes for all STM functions.
 *
 * This enumeration contains unique return/error codes to identify the
 * source of a failure. Some errors have direct correspondence to standard
 * errno.h codes, indicated [IN BRACKETS], and may result from failures in
 * lower level system calls. Others indicate failures specific to misuse
 * of STM library functions.
 * 
 **/

#ifndef STM_ERROR_H_
#define STM_ERROR_H_

#if __GNUC__ >= 4
#define STM_API __attribute__((visibility("default")))
#endif

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief Return/error codes for all STM functions.
 *
 * This enumeration contains unique return/error codes to identify the
 * source of a failure. Some errors have direct correspondence to standard
 * errno.h codes, indicated [IN BRACKETS], and may result from failures in
 * lower level system calls. Others indicate failures specific to misuse
 * of STM library functions.
 */
typedef enum stmErrorCode_t {
    /** [EOK] No error */
    STM_SUCCESS = 0,
    /** STM client context not initialized through previous call to stmClientInit() */
    STM_ERROR_NOT_INITIALIZED,
    /** Invalid parameter value */
    STM_ERROR_BAD_PARAMETER,
    /** Operation not supported */
    STM_ERROR_NOT_SUPPORTED,
    /** Operation timeout */
    STM_ERROR_TIMEOUT,
    /** Lookup failed in STM */
    STM_ERROR_NOT_FOUND,
    /** STM in invalid state */
    STM_ERROR_INVALID_STATE,
    /** Generic error; no other information */
    STM_ERROR_GENERIC,
    /** Invalid value in STM */
    STM_ERROR_BAD_VALUE,
    /** NvSciSync operation in STM returned an error */
    STM_ERROR_NVSCISYNC,
    /** Synchronization operation in STM returned an error */
    STM_ERROR_SYNC,
    /** NvSciSync operation in STM timed out */
    STM_ERROR_NVSCISYNC_TIMEOUT,
    /** CUDA operation in STM returned an error */
    STM_ERROR_CUDA,
    /** [ENOMEM] Not enough memory */
    STM_ERROR_INSUFFICIENT_MEMORY,
    /** NVMEDIA_DLA operation in STM returned an error */
    STM_ERROR_NVMEDIA_DLA,
    /** CUDLA operation in STM returned an error */
    STM_ERROR_CUDLA,
    /** CUPVA operation in STM returned an error */
    STM_ERROR_PVA,
    /** [EAGAIN] mq_send to non-blocking queue failed because it was full; increase logging frequency */
    STM_ERROR_MQ_FULL,
    /* Error in establishing TCP scoket connection */
    STM_ERROR_CONNECTION,
    /* Error in Unknown clock source setup */
    STM_ERROR_UNKNOWN_CLOCK,
    /* Error in reading timestamp */
    STM_ERROR_FAILED_READING_TIMESTAMPS,
    /* Error unknown scheduling state  */
    STM_ERROR_UNKNOWN_STATE,
    /* Error state transition not possible between current and next state */
    STM_ERROR_BAD_STATE_TRANSITION,
    /* Error schedule id invalid */
    STM_ERROR_SCHEDULE_ID_INVALID,
    STM_NUM_ERROR_CODES,
} stmErrorCode_t;

extern const char* stmErrorCodeNames[];

/**
 * \brief Convert STM error code to string representation.
 *
 * @param[in] err stmErrorCode_t error.
 * @return String representation of stmErrorCode_t.
 */
static inline const char* stmErrorCodeToString(stmErrorCode_t err)
{
    return stmErrorCodeNames[(uint32_t)err];
}

#ifdef __cplusplus
}
#endif

#endif //STM_ERROR_H_
