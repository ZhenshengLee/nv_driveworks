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
 * <b>NVIDIA DriveWorks API: Core Methods</b>
 *
 * @b Description: This file defines the core methods of the SDK.
 */

/**
 * @defgroup core_group Core Interface
 *
 * @brief Defines the core methods of the SDK.
 *
 * Unless explicitly specified, all errors returned by DriveWorks APIs are non recoverable and the user application should transition to fail safe mode.
 * In addition, any error code not described in this documentation should be consider as fatal and the user application should also transition to fail safe mode.
 *
 */

#ifndef DW_CORE_CONTEXT_H_
#define DW_CORE_CONTEXT_H_

#include <dw/core/base/Config.h>
#include <dw/core/base/Exports.h>
#include <dw/core/base/Types.h>
#include <dw/core/base/Version.h>
#include <dw/core/base/Status.h>
#include <dw/core/time/Timer.h>
#include <dw/core/platform/GPUProperties.h>

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <nvscibuf.h>
#include <nvscisync.h>

// type definitions for CUDA structs
typedef struct cudaDeviceProp cudaDeviceProp;
typedef enum cudaDeviceAttr cudaDeviceAttr;
typedef enum cudaTextureAddressMode cudaTextureAddressMode;

// Forward declares from EGL
typedef void* EGLDisplay;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup core_context_group Core Context
 * @ingroup core_group
 * Defines the core Context methods of the SDK.
 *
 * @{
 */

/// Context handle.
// coverity[misra_c_2012_rule_1_1_violation]
typedef struct dwContextObject* dwContextHandle_t;
typedef struct dwContextObject const* dwConstContextHandle_t;

/// The Generic Module Object Handle.
typedef struct dwModuleObject* dwModuleHandle_t;
typedef struct dwModuleObject const* dwConstModuleHandle_t;

/**
 * A set of parameters that is passed to the SDK to create the context.
 */
typedef struct dwContextParameters
{

#ifdef DW_USE_EGL
    /// EGL display to be used or EGL_NO_DISPLAY if SDK should handle EGL context
    EGLDisplay eglDisplay;

    /**
     * if true will skip EGL initialization in the context
     * @note Without proper EGL display some SDK modules will stop working properly.
     */
    bool skipEglInit;
#endif

    /// Path where all DriveWorks related data required during runtime are stored.
    /// If path is set to NULL, then a data folder is looked for in typical
    /// install locations in relation to the current driveworks library install
    /// location.
    const char8_t* dataPath;

    /**
     * if true will skip CUDA initialization in the context
     * @note Without CUDA some SDK modules will stop working properly.
     */
    bool skipCudaInit;

    /**
     * if true, PVA platform will be enabled.
     * @note Without PVA some SDK modules will stop working properly.
     * @note There can be maximum 4 SDK contexts with PVA running simultaneously.
     * @note This parameter only has effect on platforms containing PVA, it is otherwise ignored.
     */
    bool enablePVA;

    /**
     *  if true, Cuda task-graph will be used where possible.
     */
    bool enableCudaTaskGraph;

    /**
     * if true, the context's time source will be a virtual clock.
     * This clock must be advanced manually, see {@link dwContext_advanceTime}
     */
    bool useVirtualTime;
} dwContextParameters;

/// FileStream function pointers handle
typedef void* dwCustomizedFileHandle;

/**
 * Data structure representing a customized FileStream that can be passed in.
 */
typedef struct dwCustomizedFileFunctions
{
    /// Function pointer: Close the file stream
    /// @param hnd File handle
    void (*close)(dwCustomizedFileHandle hnd);

    /// Function pointer: Write raw data to the file stream from an input buffer and return the number of bytes written.
    /// @param hnd File handle
    /// @param ptr Pointer to the input data buffer
    /// @param size The size of the buffer
    /// @return the number of bytes written
    size_t (*write)(dwCustomizedFileHandle hnd, const void* ptr, size_t size);

    /// Function pointer: Read raw data from the file stream and return the number of bytes read.
    /// @param hnd File handle
    /// @param ptr Pointer to the buffer to contain the raw data
    /// @return the number of bytes read
    size_t (*read)(dwCustomizedFileHandle hnd, void* ptr, size_t size);

    /// Function pointer: Read a string from the file stream.
    /// @param hnd File handle
    /// @param ptr Pointer to the input data buffer
    /// @return true if the string has been read succesfully, otherwise false.
    bool (*getString)(dwCustomizedFileHandle hnd, char8_t* ptr, size_t size);

    /// Function pointer: Returns the total number of bytes in the file stream.
    /// @param hnd File handle
    /// @return the file length
    size_t (*size)(dwCustomizedFileHandle hnd);

    /// Function pointer: Returns the current position inside the file stream
    /// @param hnd File handle
    /// @return the position (in bytes) of the stream
    size_t (*getPosition)(dwCustomizedFileHandle hnd);

    /// Function pointer: Set the desired position inside the file stream
    /// @param hnd File handle
    /// @param size The position (in bytes) to set
    void (*setPosition)(dwCustomizedFileHandle hnd, size_t size);

    /// Function pointer: Check if the end of file flag is set for the file stream
    /// @param hnd File handle
    /// @return true if EOF flag is set, false otherwise
    bool (*eof)(dwCustomizedFileHandle hnd);

    /// Function pointer: Check if any I/O error occurred during OPEN/RD/WR operations
    /// @param hnd File handle
    /// @return true if any error occurred, false otherwise
    bool (*error)(dwCustomizedFileHandle hnd);

    /// Function pointer: Synchronizes the file stream, waiting that all requested bytes have been read.
    /// @param hnd File handle
    void (*flush)(dwCustomizedFileHandle hnd);

    /// Function pointer: Open the file stream and returns the handle to the file.
    /// @param ptr1 Path of the file to open
    /// @param ptr2 Open mode "READ", "WRITE" or "R/W"
    /// @return the handle of the opened file
    dwCustomizedFileHandle (*open)(const char8_t* ptr1, const char8_t* ptr2);

} dwCustomizedFileFunctions;

/**
 * @brief Defines PTP synchronization status
 */
typedef enum {
    //! PTP synchronization is not started
    DW_PTP_STATUS_NO_INIT_SYNC = 0,

    //! No error
    DW_PTP_STATUS_NO_ERROR,

    //! PTP synchronization lost
    DW_PTP_STATUS_LOST_SYNC,

    //! PTP internal error
    DW_PTP_STATUS_INTERNAL_ERROR,
} dwPTPStatus;

/**
* Set the customized fileStream function pointers in context
*
* @param[in] context handle
* @param[in] fileFunctionPtr dwCustomizedFileFunctions pointer to be passed into context
*
* @return DW_INVALID_ARGUMENT if given context handle or fileFunctionPtr is NULL. <br>
          DW_INVALID_HANDLE   if given context handle is invalid. <br>
*         DW_SUCCESS if no error occurs.
*
*/
DW_API_PUBLIC
dwStatus dwContext_setCustomFileFunctions(dwContextHandle_t const context, dwCustomizedFileFunctions* const fileFunctionPtr);

/**
 * Retrieves the last error encountered. When a DriveWorks function reports an error, this
 * error is stored internally. It can later be retrieved by this function. After calling this
 * function, the error is reset and future calls will return 'DW_SUCCESS' until another error is
 * encountered.
 *
 * Error information is thread local. Each thread has its own last error and an error encountered
 * in one thread will not be visible in another thread.
 *
 * @param[out] errorMsg A pointer to a pointer to a string containing a description of the last error encountered.
 *                      Can be NULL to just reset last error.
 *
 * @return The code of the last error encountered, or DW_SUCCESS if no error is recorded.
 */
DW_API_PUBLIC
dwStatus dwGetLastError(char8_t const** const errorMsg);

/**
 * Creates and initializes an SDK context. The context is required
 * for initialization of the SDK modules.
 *
 *
 * @param[out] context A pointer to the context handler is returned here.
 * @param[in] headerVersion Specifies current driveworks API version (usually DW_VERSION).
 * @param[in] params A pointer with a set of parameters to create the SDK.
 *            Can be NULL for a default set of parameters.
 *
 * @return DW_INVALID_VERSION if the DW version does not match the version of the library.<br>
 *         DW_INVALID_ARGUMENT if the input context pointer is null.<br>
 *         DW_INTERNAL_ERROR if SDK could not to detect platform or if the offset between realtime and monotonic clocks is too large
 *                           and cannot be represented in microseconds.<br>
 *         DW_CANNOT_CREATE_OBJECT if the system call clock_gettime() or EGL display initialization fails.<br>
 *         DW_BUFFER_FULL if not enough memory for initialization is available.<br>
 *         DW_OUT_OF_BOUNDS if not enough memory for initialization is available.<br>
 *         DW_FAILURE if initialization fails.<br>
 *         DW_CUDA_ERROR if CUDA initialization fails.<br>
 *         DW_NOT_SUPPORTED if an available GPU is not supported.<br>
 *         DW_NOT_AVAILABLE if Deep Learning Accelerator (DLA) initialization fails or<br>
 *                          if Programmable Vision Accelerator (PVA) initialization fails or<br>
 *                          if EGL display initialization fails.<br>
 *         DW_INVALID_HANDLE if context initialization fails.<br>
 *         DW_SUCCESS if no error occurs.
 *
 */
DW_API_PUBLIC
dwStatus dwInitialize(dwContextHandle_t* const context, dwVersion const headerVersion, dwContextParameters const* const params);

/**
 * Releases the context. Any module created with this context must be released
 * before the context can be released.
 *
 * @note This method renders the context handle unusable.
 *
 * @param[in] context The context to be released.
 *
 * @return DW_INVALID_HANDLE if provided context is NULL or invalid. <br>
 *         DW_SUCCESS if no error occurs.
 *
 */
DW_API_PUBLIC
dwStatus dwRelease(dwContextHandle_t const context);

/**
 * Returns the current timestamp. Timestamps from the same context are guaranteed to be in sync.
 * The returned time represents the absolute time as received through system time source.
 * On POSIX based systems, the time is measured using CLOCK_MONOTONIC time source.
 * On Windows based systems, the returned time is relative to the epoch, i.e., 31std dec, 1969.
 *
 * @note An offset is calculated and added to CLOCK_MONOTONIC, to ensure the time has an epoch base.
 *
 * @param[out] time A pointer to the return time to a given location, in [us].
 * @param[in] context Specifies the context.
 *
 * @return DW_INVALID_ARGUMENT if given context handle or time is NULL. <br>
 *         DW_INVALID_HANDLE if given context handle is not valid. <br>
 *         DW_SUCCESS if no error occurs.
 */
DW_API_PUBLIC
dwStatus dwContext_getCurrentTime(dwTime_t* const time, dwContextHandle_t const context);

/**
 * @brief Retrieve time source used in the context.
 *
 * @param[out] source Hadle to the time source used by the context
 * @param[in] context Specifies the context.
 *
 * @return DW_INVALID_ARGUMENT if given context handle or source is null. <br>
 *         DW_INVALID_HANDLE if given context handle is not valid. <br>
 *         DW_SUCCESS if no error occurs.
 *
 * @note Ownership of the time source remains by the context.
 */
DW_API_PUBLIC
dwStatus dwContext_getTimeSource(dwTimeSourceHandle_t* const source, dwContextHandle_t const context);

/**
 * Check if the used time source inside the context is synchronized over PTP.
 *
 * PTP synchronization is available starting from PDK 4.1.6.4 and provides a solution to synchronize
 * multiple NVIDIA DRIVE boxes to a common clock base. All sensor readings as well as the
 * 'dwContext_getCurrentTime()' method will be based on the same time source.
 *
 * @param[out] flag Return true if PTP synchronized time is used.
 * @param[in] context Specifies the context.
 *
 * @return DW_INVALID_ARGUMENT if given context handle or time is NULL. <br>
 *         DW_INVALID_HANDLE if given context handle is not valid. <br>
 *         DW_NOT_SUPPORTED if the method is called on a system where PTP synchronized time is not supported, 
 *                          i.e. PDK < 4.1.6.4 or x86 based host. <br>
 *         DW_SUCCESS if no error occurs.
 *
 * @note Synchronized time is only available if PTP daemon has been identified during context creation.
 * @note PTP time base is not guaranteed to be epoch based.
 * @note It might happen at run-time that the PTP daemon is not responding anymore and PTP synchronization
 *       will be lost. In such case DW context will continue to count based on the last synced PTP time.
 **/
DW_API_PUBLIC
dwStatus dwContext_isTimePTPSynchronized(bool* const flag, dwContextHandle_t const context); // clang-tidy NOLINT(readability-non-const-parameter);

/**
 * Get the PTP synchronization status of time source inside the context.
 *
 * @param[out] status Return PTP synchronization status.
 * @param[in] context Specifies the context.
 *
 * @return DW_INVALID_ARGUMENT if given context handle or time is NULL. <br>
 *         DW_INVALID_HANDLE if given context handle is not valid. <br>
 *         DW_NOT_SUPPORTED if the method is called on a system where PTP synchronized time is not supported,
 *                          i.e. PDK < 4.1.6.4 or x86 based host. <br>
 *         DW_SUCCESS if no error occurs.
 **/
DW_API_PUBLIC
dwStatus dwContext_getTimePTPSynchronizationStatus(dwPTPStatus* const status, dwContextHandle_t const context); // clang-tidy NOLINT(readability-non-const-parameter);

/**
 * Advances the virtual time to newTime.
 * @return DW_INVALID_ARGUMENT if the given context is NULL. <br>
 *         DW_INVALID_HANDLE if given context handle is not valid. <br>
 *         DW_CALL_NOT_ALLOWED if the given context is not using a virtual time source. <br>
 *         DW_SUCCESS if no error occurs.
 **/
DW_API_PUBLIC
dwStatus dwContext_advanceTime(dwTime_t const newTime, dwContextHandle_t const context);

/**
 * Selects a GPU device, if available. Note that the selected gpu is valid for the current thread
 *
 * @param[in] deviceNumber The number of GPU device.
 * @param[in] context Specifies the context handle.
 *
 * @return DW_INVALID_ARGUMENT if context handle is NULL or the selected device doesn't exist or
 *                             if the input device number does not match to the number of available devices. <br>
 *         DW_INVALID_HANDLE if given context handle is not valid. <br>
 *         DW_NOT_AVAILABLE if no GPU devices are available. <br>
 *         DW_CUDA_ERROR in case of an underlying CUDA failure. <br>
 *         DW_SUCCESS if no error occurs.
 */
DW_API_PUBLIC
dwStatus dwContext_selectGPUDevice(int32_t const deviceNumber, dwContextHandle_t const context);

/**
 * Returns the currently selected GPU device. If no device is selected, will return -1.
 *
 * @param[out] deviceNumber The number of GPU device.
 * @param[in] context Specifies the context handle.
 *
 * @return DW_INVALID_ARGUMENT if context handle is NULL. <br>
 *         DW_INVALID_HANDLE if given context handle is not valid. <br>
 *         DW_SUCCESS if no error occurs.
 */
DW_API_PUBLIC
dwStatus dwContext_getGPUDeviceCurrent(int32_t* const deviceNumber, dwContextHandle_t const context);

/**
 * Get the available GPU devices count.
 *
 * @param[out] count The number of GPU devices available.
 * @param[in] context Specifies the context handle.
 *
 * @return DW_INVALID_ARGUMENT if context handle is NULL. <br>
 *         DW_INVALID_HANDLE if given context handle is not valid. <br>
 *         DW_SUCCESS if no error occurs.
 */
DW_API_PUBLIC
dwStatus dwContext_getGPUCount(int32_t* const count, dwContextHandle_t const context);

/**
 * Returns the properties for the specific CUDA device.
 *
 * @param[out] properties A struct containing the properties.
 * @param[in] deviceNum Specifies the device number.
 * @param[in] context Specifies the context handle.
 *
 * @return DW_INVALID_ARGUMENT if value pointer or context handle is NULL or deviceNum doesn't exist or
 *                             if the input device number does not match to the number of available devices. <br>
 *         DW_INVALID_HANDLE if given context handle is not valid. <br>
 *         DW_CUDA_ERROR in case of an underlying CUDA failure. <br>
 *         DW_NOT_AVAILABLE if no GPU devices are available. <br>
 *         DW_SUCCESS if no error occurs.
 */
DW_API_PUBLIC
dwStatus dwContext_getGPUProperties(dwGPUDeviceProperties* const properties, int32_t const deviceNum,
                                    dwContextHandle_t const context);

/**
 * Returns the value of the selected CUDA attribute for the specific CUDA device.
 *
 * @param[out] value Integer representing the value of the requested attribute.
 * @param[in] attribute Specifies the attribute requested.
 * @param[in] deviceNum Specifies the device number.
 * @param[in] context Specifies the context handle.
 *
 * @return DW_INVALID_ARGUMENT if value pointer or context handle is NULL or context is NULL or
 *                             if the input device number does not match to the number of available devices or
 *                             if the selected device doesn't exist. <br>
 *         DW_INVALID_HANDLE if given context handle is not valid. <br>
 *         DW_CUDA_ERROR in case of an underlying CUDA failure. <br>
 *         DW_NOT_AVAILABLE if no GPU devices are available. <br>
 *         DW_SUCCESS if no error occurs.
 */
DW_API_PUBLIC
dwStatus dwContext_getGPUAttribute(int32_t* const value, cudaDeviceAttr const attribute, int32_t const deviceNum,
                                   dwContextHandle_t const context);

/**
 * Returns the architecture for the currently selected CUDA device.
 *
 * @param[out] architecture A string containing the architecture.
 * @param[in] context Specifies the context handle.
 *
 * @return DW_INVALID_ARGUMENT if value pointer or context handle is NULL. <br>
 *         DW_INVALID_HANDLE if given context handle is not valid. <br>
 *         DW_SUCCESS if no error occurs.
 */
DW_API_PUBLIC
dwStatus dwContext_getGPUArchitecture(char8_t const** const architecture, dwContextHandle_t const context);

/**
 * Returns Driver and Runtime API version of CUDA on the current machine.
 *
 * @param[out] driverVersion Driver version.
 * @param[out] apiVersion Runtime API version.
 * @param[in] context Specifies the context handle.
 *
 * @return DW_INVALID_ARGUMENT if context handle, driverVersion or apiVersion are NULL. <br>
 *         DW_INVALID_HANDLE if given context handle is not valid. <br>
 *         DW_CUDA_ERROR in case of an underlying CUDA failure. <br>
 *         DW_SUCCESS if no error occurs.
 */
DW_API_PUBLIC
dwStatus dwContext_getCUDAProperties(int32_t* const driverVersion, int32_t* const apiVersion, dwContextHandle_t const context);

/**
 * Returns the device type of the input GPU number.
 *
 * @param[out] deviceType The type of GPU device.
 * @param[in] deviceNum Specifies the device number.
 * @param[in] context Specifies the context handle.
 *
 * @return DW_INVALID_ARGUMENT if context handle is NULL or number is invalid or
 *                             if the input device number does not match to the number of available devices. <br>
 *         DW_INVALID_HANDLE if given context handle is not valid. <br>
 *         DW_SUCCESS if no error occurs.
 */
DW_API_PUBLIC
dwStatus dwContext_getGPUDeviceType(dwGPUDeviceType* const deviceType, int32_t const deviceNum, dwContextHandle_t const context);

#ifdef DW_USE_EGL
/**
 * Sets EGLDisplay to be used within the SDK context.
 * You must call this method before any EGL/GL related methods are used.
 *
 * @param[in] display Specifies the EGLDisplay handle.
 * @param[in] context Specifies the context handle.
 *
 * @return DW_INVALID_ARGUMENT if given context handle is NULL or display is EGL_NO_DISPLAY. <br>
 *         DW_INVALID_HANDLE if given context handle is not valid. <br>
 *         DW_SUCCESS if no error occurs.
 *
 * @note The EGL display remains under the callee's ownership. Ensure that the EGL
 *       context is valid until 'dwRelease()' is finished.
 */
DW_API_PUBLIC
dwStatus dwContext_setEGLDisplay(EGLDisplay display, dwContextHandle_t context);

/**
 * Get EGLDisplay currently in use within the SDK context.
 *
 * @param[out] display EGLDisplay handle in use by SDK.
 * @param[in] context Specifies the context handle.
 *
 * @return DW_INVALID_ARGUMENT if given context handle or display pointer is NULL. <br>
 *         DW_INVALID_HANDLE if given context handle is not valid. <br>
 *         DW_SUCCESS if no error occurs.
 *
 * @note Context has per default already a valid EGL display associated. It will be released when
 *       replaced with the provided display
 **/
DW_API_PUBLIC
dwStatus dwContext_getEGLDisplay(EGLDisplay* display, dwContextHandle_t context);
#endif

/**
 * Get the available DLA engines count.
 *
 * @param[out] count The number of DLA engines available.
 * @param[in] context Specifies the context handle.
 *
 * @return DW_INVALID_ARGUMENT if context handle or descriptor pointer is NULL. <br>
 *         DW_INVALID_HANDLE if given context handle is not valid. <br>
 *         DW_SUCCESS if no error occurs.
 */
DW_API_PUBLIC
dwStatus dwContext_getDLAEngineCount(int32_t* const count, dwContextHandle_t const context);

/**
 * Gets the initial data path of the library that contains the driveworks context.
 *
 * Note: Later added data paths are not returned here.
 *
 * @param[out] dataPath Path to the Driveworks data.
 * @param[in] context Specifies the context handle.
 *
 * @return DW_INVALID_ARGUMENT if context handle or descriptor pointer is NULL. <br>
 *         DW_INVALID_HANDLE if given context handle is not valid. <br>
 *         DW_SUCCESS if no error occurs.
 */
DW_API_PUBLIC
dwStatus dwContext_getDataPath(char8_t const** const dataPath, dwContextHandle_t const context);

/**
 * Check if in virtual time mode.
 *
 * @param[out] useVirtualTime boolean for if virtual time is used.
 * @param[in] ctx Specifies the context handle.
 *
 * @return DW_INVALID_ARGUMENT if context handle or descriptor pointer is NULL. <br>
 *         DW_INVALID_HANDLE if given context handle is not valid. <br>
 *         DW_SUCCESS if no error occurs.
 */
DW_API_PUBLIC
dwStatus dwContext_getUseVirtualTime(bool* const useVirtualTime, dwContextHandle_t const ctx);

/**
 * Get the NvSciSync module
 *
 * @param[out] mod The nvscisync module
 * @param[in] context Specifies the context handle.
 *
 * @return DW_INVALID_ARGUMENT if mod is NULL. <br>
 *         DW_SUCCESS if module has been successfully updated with NvSciSyncModule owned by DriveWorks.
 * 
 * @note the NvSciSync module is owned by the DriveWorks context and shall not be closed by the application.
 */
DW_API_PUBLIC
dwStatus dwContext_getNvSciSyncModule(NvSciSyncModule* mod,
                                      dwContextHandle_t const context);

/**
 * Get the nvscibuf module
 *
 * @param[out] mod The nvscibuf module
 * @param[in] context Specifies the context handle.
 *
 * @return DW_INVALID_ARGUMENT if mod is NULL.
 *         DW_SUCCESS if module has been successfully updated with NvSciBufModule owned by DriveWorks.
 * 
 * @note the nvscibuf module is owned by the DriveWorks context and shall not be closed by the application.
 */
DW_API_PUBLIC
dwStatus dwContext_getNvSciBufModule(NvSciBufModule* mod,
                                     dwContextHandle_t const context);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_CORE_CONTEXT_H_
