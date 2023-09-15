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
// SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Perspective-n-point Methods</b>
 *
 * @b Description: This file defines methods to solve the perspective-n-points (PnP) problem.
 */

/**
 * @defgroup pnp_group PnP Interface
 *
 * @brief Defines the PnP module.
 *
 * @{
 */

#ifndef DW_IMAGEPROCESSING_GEOMETRY_PNP_H_
#define DW_IMAGEPROCESSING_GEOMETRY_PNP_H_

#include <dw/core/context/Context.h>
#include <dw/core/base/Types.h>

#ifdef __cplusplus
extern "C" {
#endif

/** A pointer to the handle representing a PnP solver.
 * This object allows you to solve perspective-n-points problems, i.e. estimating camera pose from 2D-3D correspondences.
 */
typedef struct dwPnPObject* dwPnPHandle_t;

/**
 * Initializes a PnP solver.
 *
 * @param[out] obj A pointer to the PnP handle for the created module.
 * @param[in] ransacIterations The number of P3P ransac iterations to run when solving a PnP pose.
 * @param[in] optimizerIterations The number of non-linear optimization iterations to run when solving a PnP pose.
 * @param[in] ctx Specifies the handler to the context to create the rectifier.
 *
 * @return DW_INVALID_ARGUMENT - if the PnP handle is NULL<br>
 *         DW_SUCCESS
 */
DW_API_PUBLIC
dwStatus dwPnP_initialize(dwPnPHandle_t* obj,
                          size_t ransacIterations,
                          size_t optimizerIterations,
                          dwContextHandle_t ctx);

/**
 * Resets the PnP solver.
 *
 * @param[in] obj Specifies the solver to reset.
 *
 * @return DW_SUCCESS <br>
 *         DW_INVALID_HANDLE - If the given handle is invalid,i.e. null or of wrong type  <br>
 */
DW_API_PUBLIC
dwStatus dwPnP_reset(dwPnPHandle_t obj);

/**
 * @brief Defines the maximum number of points that can be processed by the PnP solver
 */
#define DW_PNP_MAX_POINT_COUNT 128

/**
 * Estimates the worldToCamera pose based on optical ray to 3D world point correspondences. The rays can be
 * obtained by applying the camera model to 2D pixel positions. The 3D world points come from known world structure.
 *
 * @param[out] worldToCamera The estimated pose
 * @param[in] matchCount The number of ray to point correspondences.
 * @param[in] rays The optical rays. Their norm is expected to be one.
 * @param[in] worldPoints The 3D world points that correspond to the rays provided.
 * @param[in] obj A pointer to the PnP handle for the created module.
 *
 * @return DW_SUCCESS <br>
 *         DW_INVALID_HANDLE - If the given handle is invalid,i.e. null or of wrong type  <br>
 *         DW_INVALID_ARGUMENT - If any arguments is null, or too few (minimum three are required), or too many points (more than DW_PNP_MAX_POINT_COUNT) are provided <br>
 */
DW_API_PUBLIC
dwStatus dwPnP_solve(dwTransformation3f* worldToCamera,
                     size_t matchCount,
                     const dwVector3f* rays,
                     const dwVector3f* worldPoints,
                     dwPnPHandle_t obj);

/**
 * Releases the PnP solver.
 *
 * @param[in] obj The object handle to release.
 *
 * @return DW_SUCCESS <br>
 *         DW_INVALID_HANDLE - If the given handle is invalid,i.e. null or of wrong type  <br>
 *
 * @note This method renders the handle unusable.
 */
DW_API_PUBLIC
dwStatus dwPnP_release(dwPnPHandle_t obj);

#ifdef __cplusplus
}
#endif
/** @} */
#endif
