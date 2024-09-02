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
// SPDX-FileCopyrightText: Copyright (c) 2016-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Core Dynamic Memory</b>
 *
 * @b Description: Processes wide dynamic memory API.
 */

/**
 * @defgroup core_memory_group Core Dynamic Memory
 * @brief Processes wide dynamic memory API.
 *
 * @{
 * @ingroup core_group
 */

#ifndef DW_CORE_DYNAMIC_MEMORY_H_
#define DW_CORE_DYNAMIC_MEMORY_H_

#include <stddef.h>
#include <stdbool.h>

#ifdef DW_EXPORTS
#define DWALLOC_API_PUBLIC __attribute__((visibility("default")))
#else
#define DWALLOC_API_PUBLIC
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Error to be reported through error callback
 **/
typedef enum dwDynamicMemoryError {
    DW_DYNAMIC_MEMORY_SUCCESS = 0,       //!< no error
    DW_DYNAMIC_MEMORY_BAD_ALLOC,         //!< Memory allocation failed, probably not enough memory
    DW_DYNAMIC_MEMORY_ALLOC_NOT_ALLOWED, //!< Allocation is currently not allowed, because in runtime mode
    DW_DYNAMIC_MEMORY_FREE_NOT_ALLOWED   //!< Memory release is currently not allowed, because in runtime mode
} dwDynamicMemoryError;

typedef void* (*dwDynamicMemoryMallocCallback)(size_t sizeInByte, void* userData);
typedef void (*dwDynamicMemoryFreeCallback)(void* addr, void* userData);
typedef void (*dwDynamicMemoryErrorCallback)(dwDynamicMemoryError error, size_t lastRequestedSizeInByte, void* userData);

/**
 * Initialize dwDynamicMemory with user-defined callback for user space memory allocations. If initialization
 * is not performed or if user provides NULL pointers for the callbacks then posix_memalign()
 * will be used for heap allocation and free() for deallocation.
 *
 * The mallocCallback must always return 16-byte aligned memory.
 *
 * @param[in] mallocCallback functions which will be used for memory allocation. NULL for CRT implementation
 * @param[in] freeCallback functions which will be used for memory deallocation. NULL for CRT implementation
 * @param[in] userData general pointer to something understandable by mallocCallback or freeCallback
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
 **/
DWALLOC_API_PUBLIC
void dwDynamicMemory_initialize(dwDynamicMemoryMallocCallback mallocCallback,
                                dwDynamicMemoryFreeCallback freeCallback,
                                void* const userData);

/**
 * Set error callback to be executed on an allocation error.
 *
 * @param[in] errorCallback Callback to be called on any error.
 * @param[in] userData User pointer to return back through the error callback
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DWALLOC_API_PUBLIC
void dwDynamicMemory_setErrorCallback(dwDynamicMemoryErrorCallback errorCallback, void* const userData);

/**
 * Get callbacks and user-defined general pointer previously passed in dwDynamicMemory_initialize. Optionally
 * pass NULL if not interested in retrieving the pointers
 *
 * @param[out] mallocCallback returned by pointer address of the mallocCallback.
 * @param[out] freeCallback returned by pointer address of the freeCallback.
 * @param[out] userData returned by pointer value of previously passed userData during initialization.
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DWALLOC_API_PUBLIC
void dwDynamicMemory_getCallbacks(dwDynamicMemoryMallocCallback* const mallocCallback,
                                  dwDynamicMemoryFreeCallback* const freeCallback,
                                  void** const userData);

/**
 * Context runtime mode
 */

typedef enum dwRuntimeMode {
    DW_MODE_INIT    = 0x0000, ///< Heap memory allocations and deallocations are allowed. Default
    DW_MODE_RUNTIME = 0x0001, ///< Heap memory allocations are not allowed
    DW_MODE_CLEANUP = 0x0002, ///< Heap memory allocations and deallocations are allowed
    DW_MODE_COUNT   = 3
} dwRuntimeMode;

/**
 * Switch runtime mode for all Driveworks SDK. A runtime mode is activated after full initialization phase.
 * Switching to runtime mode allows context to perform checks if for example heap allocations
 * are still hapenning. Runtime mode does not have any effect on the performance of the SDK modules,
 * it can, however, used for debugging purpose.
 *
 * @param[in] newMode mode that you want to setup
 *
 * @return false if given enum is ininvalid otherwise true
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DWALLOC_API_PUBLIC
bool dwDynamicMemory_setRuntimeMode(dwRuntimeMode const newMode);

/**
 * Return currently selected runtime mode.
 *
 * @param[out] mode Pointer to mode where result to be returned
 *
 * @return false if given pointer is null otherwise true
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DWALLOC_API_PUBLIC
bool dwDynamicMemory_getRuntimeMode(dwRuntimeMode* const mode);

/**
 * Release dwDynamicMemory. After this call all following requests for memory allocation will use the CRT
 * heap allocation mechanisms.
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: Yes
 */
DWALLOC_API_PUBLIC
void dwDynamicMemory_release(void);

/**
 * Allocate chunk of memory using allocator passed through `dwDynamicMemory_initialize()`.
 * If not enough memory is available for allocation, nullptr is returned;
 *
 * @param[in] size Size in bytes of memory chunk to allocate.
 *
 * @return Pointer to the begnning of the memory block or nullptr if there is not enough space to allocate the requested chunk.
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
 */
DWALLOC_API_PUBLIC
void* dwDynamicMemory_malloc(size_t const size);

/**
 * Release memory chunk previously allocated with `dwDynamicMemory_malloc()`.
 *
 * @param[in] ptr Pointer to the memory to release.
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: Yes
 */
DWALLOC_API_PUBLIC
void dwDynamicMemory_free(void* const ptr);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_CORE_DYNAMIC_MEMORY_H_
