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
// SPDX-FileCopyrightText: Copyright (c) 2018-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Clusterer Methods</b>
 *
 * @b Description: This file defines Clusterer methods.
 */

/**
 * @defgroup clusterer_group Clusterer Interface
 *
 * @brief Defines the Clusterer module for performing DBSCAN clusterer on bounding boxes.
 *
 * @{
 */

#ifndef DW_CLUSTERER_H_
#define DW_CLUSTERER_H_

#include <dw/core/context/Context.h>
#include <dw/core/base/Types.h>
#include <dw/image/Image.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dwClustererParams
{
    /// Maximum distance from the core box to be considered within a region. Default value is 0.4.
    float32_t epsilon;
    /// Minimum number of samples required to form a dense region. minSamples and minSumOfConfidences are
    /// checked conjunctively. Default value is 3.
    uint32_t minSamples;
    /// Minimum sum of weights required to form a dense region. minSamples and minSumOfWeights are
    /// checked conjunctively. Default value is 0.
    float32_t minSumOfWeights;
    /// Maximum number of samples that will be given as input. Default value is 100.
    uint32_t maxSampleCount;
} dwClustererParams;

/**
 * @brief Handle to a Clusterer.
 */
typedef struct dwClustererObject* dwClustererHandle_t;

/**
 * @brief Initializes Clusterer parameters with default values.
 *
 * @param[out] clustererParams Clusterer parameters.
 *
 * @return DW_INVALID_ARGUMENT if parameters are NULL.<br>
 *         DW_SUCCESS
 */
DW_API_PUBLIC
dwStatus dwClusterer_initParams(dwClustererParams* const clustererParams);

/**
  * Initializes a Clusterer module.
  *
  * @param[out] obj A pointer to the Clusterer handle for the created module.
  * @param[in] clustererParams Clusterer parameters.
  * @param[in] ctx Specifies the handle to the context under which the Clusterer module is created.
  *
  * @note Clusterer parameters must be initialized using dwClusterer_initParams
  * before modifying.
  *
  * @return DW_INVALID_ARGUMENT if clusterer handle is NULL or clustererParams is invalid.<br>
  *         DW_SUCCESS
  */
DW_API_PUBLIC
dwStatus dwClusterer_initialize(dwClustererHandle_t* const obj, dwClustererParams const* const clustererParams,
                                dwContextHandle_t const ctx);

/**
 * Resets the Clusterer module.
 *
 * @param[in] obj Specifies the Clusterer handle to reset.
 *
 * @return DW_INVALID_ARGUMENT if clusterer handle is NULL.<br>
 *         DW_SUCCESS.
 */
DW_API_PUBLIC
dwStatus dwClusterer_reset(dwClustererHandle_t const obj);

/**
 * Releases the Clusterer module.
 *
 * @param[in] obj Specifies the Clusterer handle to release.
 *
 * @return DW_INVALID_ARGUMENT if clusterer handle is NULL.<br>
 *         DW_SUCCESS.
 *
 * @note This method renders the handle unusable.
 */
DW_API_PUBLIC
dwStatus dwClusterer_release(dwClustererHandle_t const obj);

/**
 * Runs DBScan clusterer on given bounding boxes and returns labels for each bounding box in the same order.
 * @param[in] obj Specifies the Clusterer handle.
 *
 * @note This method requires that the input and output have already been provided by calling
 * bindInput() and bindOutput().
 *
 * @return DW_INVALID_ARGUMENT if clusterer handle is NULL.<br>
 *         DW_SUCCESS
 */
DW_API_PUBLIC
dwStatus dwClusterer_process(dwClustererHandle_t const obj);

/**
 * Binds the input for clusterer.
 * @param[in] boxes Pointer to list where boxes are stored.
 * @param[in] weights Pointer to list where corresponding weights are stored.
 * @param[in] boxesCount Pointer to the number of input boxes
 * @param[in] obj Specifies the Clusterer handle.
 * @return DW_INVALID_ARGUMENT if any of the arguments is NULL
 *         DW_SUCCESS
 */
DW_API_PUBLIC
dwStatus dwClusterer_bindInput(dwRectf const* const* const boxes, float32_t const* const* const weights,
                               uint32_t const* const boxesCount, dwClustererHandle_t const obj);

/**
 * Bind the ouput of the clusterer to list of cluster labels
 * @param clusterLabels Pointer to a list to store cluster label for each input bounding box at the same index.
 * Label values are in [-1, clusterCount) where -1 means the corresponding sample does not belong to any cluster.
 * @param clusterLabelsCount Pointer to store the number of cluster labels.
 * @param clusterCount Pointer to store the number of clusters.
 * @param obj Specifies the Clusterer handle.
 * @return DW_INVALID_ARGUMENT if any of the arguments is NULL
 *         DW_SUCCESS
 */
DW_API_PUBLIC
dwStatus dwClusterer_bindOutput(int32_t** const clusterLabels, uint32_t* const clusterLabelsCount,
                                uint32_t* const clusterCount, dwClustererHandle_t const obj);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_CLUSTERER_H_
