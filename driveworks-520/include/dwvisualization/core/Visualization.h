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
// SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Visualization Methods</b>
 *
 * @b Description: This file defines the visualization methods of the SDK.
 */

/**
 * @defgroup vizualization_group Visualization Interface
 *
 * @brief Defines the visualization methods of the SDK.
 *
 */

#ifndef DWVISUALIZATION_CONTEXT_H_
#define DWVISUALIZATION_CONTEXT_H_

#include "Exports.h"

#include <dw/core/context/Context.h>

// Visualization SDK depends on GL library
#include <dwvisualization/gl/GL.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declares from EGL
typedef void* EGLDisplay;

/**
 * @defgroup core_context_group Core Context
 * @ingroup core_group
 * Defines the core Context methods of the SDK.
 * @{
 */

/// Context handle.
typedef struct dwVisualizationContextObject* dwVisualizationContextHandle_t;
typedef struct dwVisualizationContextObject const* dwVisualizationConstContextHandle_t;
/**
 * A set of parameters that is passed to the visualization
 */
typedef struct dwVisualizationParameters
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
} dwVisualizationParameters;

/**
 * Creates and initializes a Visualization SDK context. The context is required
 * for initialization of the Visualization SDK modules.
 *
 * @param[out] vizContext A pointer to the context handler is returned here.
 * @param[in] ctx Handle to existing DW SDK context, created with dwInitialize.
 *
 * @return DW_INVALID_VERSION - If provided context' version does not match the version of the library. <br>
 *         DW_INVALID_ARGUMENT - If provided, context pointer is null. <br>
 *         DW_SUCCESS
 *
 * @note Visualization SDK depends on GL components, hence an automatic initialization with `dwGLInitialize` will happen
 */
DW_VIZ_API_PUBLIC
dwStatus dwVisualizationInitialize(dwVisualizationContextHandle_t* vizContext, dwContextHandle_t ctx);

/**
 * Creates and initializes a Visualization SDK context. The context is required
 * for initialization of the Visualization SDK modules.
 *
 * @param[out] vizContext A pointer to the context handler is returned here.
 * @param[in] params A set of parameters
 * @param[in] ctx Handle to existing DW SDK context, created with dwInitialize.
 *
 * @return DW_INVALID_VERSION - If provided context' version does not match the version of the library. <br>
 *         DW_INVALID_ARGUMENT - If provided, context pointer is null. <br>
 *         DW_SUCCESS
 *
 * @note Visualization SDK depends on GL components, hence an automatic initialization with `dwGLInitialize` will happen
 */
DW_VIZ_API_PUBLIC
dwStatus dwVisualizationInitializeWithParams(dwVisualizationContextHandle_t* vizContext, dwVisualizationParameters params, dwContextHandle_t ctx);

/**
 * Releases the context. Any module created with this context must be released
 * before the context can be released.
 *
 * @note This method renders the context handle unusable.
 *
 * @param[in] context The context to be released.
 *
 * @return DW_INVALID_ARGUMENT - If provided context is NULL. <br>
 *         DW_SUCCESS
 */
DW_VIZ_API_PUBLIC
dwStatus dwVisualizationRelease(dwVisualizationContextHandle_t context);

/**
 * Return DW context associated with this visualization library.
 *
 * @param[out] dwctx Context of the DW SDK as given through `dwVisualizationInitialize()`
 * @param[in] context Handle to existing dwVisualization context, created with dwInitialize.
 *
 * @return DW_INVALID_ARGUMENT - If provided, context pointer is null. <br>
 *         DW_SUCCESS
 */
DW_VIZ_API_PUBLIC
dwStatus dwVisualizationGetDWContext(dwContextHandle_t* dwctx, dwVisualizationContextHandle_t context);

#ifdef DW_USE_EGL
/**
 * Sets EGLDisplay to be used within the visualization context.
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
dwStatus dwVisualization_setEGLDisplay(EGLDisplay display, dwVisualizationContextHandle_t context);

/**
 * Get EGLDisplay currently in use within the visualization context.
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
dwStatus dwVisualization_getEGLDisplay(EGLDisplay* display, dwVisualizationContextHandle_t context);
#endif

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_VISUALIZATION_CONTEXT_H_
