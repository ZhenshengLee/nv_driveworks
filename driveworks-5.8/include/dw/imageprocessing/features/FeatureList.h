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
// SPDX-FileCopyrightText: Copyright (c) 2018-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Feature Array and Feature History Array</b>
 *
 * @b Description: This file defines the 2d feature array.
 */

/**
 * @defgroup featureArray_group Feature Array Interface
 *
 * @brief Defines the feature array used by detector and tracker.
 *
 * @{
 */

#ifndef DW_IMAGEPROCESSING_FEATURES_FEATURELIST_H_
#define DW_IMAGEPROCESSING_FEATURES_FEATURELIST_H_

#include <dw/core/base/Config.h>
#include <dw/core/base/Exports.h>
#include <dw/core/context/Context.h>
#include <dw/core/base/Types.h>
#include <dw/image/Image.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Defines the possible status of a feature.
 */
typedef enum dwFeature2DStatus {
    /**
     * A feature with this entry is garbage.
     */
    DW_FEATURE2D_STATUS_INVALID = 0,

    /**
     * The feature was just detected in the current frame.
     */
    DW_FEATURE2D_STATUS_DETECTED,

    /**
    * The feature was successfully tracked in the current frame.
    **/
    DW_FEATURE2D_STATUS_TRACKED,

    DW_FEATURE2D_STATUS_TYPE_COUNT
} dwFeature2DStatus;

/**
* Holds pointers to the data exposed by a feature2d list.
* The pointers can point to GPU or CPU memory.
*
* The @a locationHistory field is a 2D structure that stores the pixel
* positions for each feature at each time instant.
* It is a column-major matrix, where the column is the time-index and the row
* is the feature-index.
* For example:
* @code
*    [x_00,y_00],[x_10,y_10],...,[x_N0,y_N0],[x_01,y_01],...,[x_N1,y_N1],...,[x_NH,y_NH]
* @endcode
*
* Where:
* * N = max feature count and
* * H = location history size.
*
* To avoid unnecessary memory copies, the time-index is treated as a circular buffer.
* The feature2d list has a global "currentTimeIdx" that wraps around the location history size.
* The following example shows how to access the locations for the current and previous frames.
* @code
*   dwVector2f currentLocation = locationHistory[currentTimeIdx * maxFeatureCount + featureIdx];
*   uint32_t previousTimeIdx = (currentTimeIdx + 1) % historyCapacity;
*   dwVector2f currentLocation = locationHistory[previousTimeIdx * maxFeatureCount + featureIdx];
* @endcode
*
* The @a ages array holds how many frames each feature has been tracked.
* This value can be larger than @a historyCapacity.
* Therefore when you want to access locationHistory you need to do check it, for example,
* @code
*   uint32_t validAge = min(age, historyCapacity);
*   for (uint32_t historyIdx = 0; historyIdx < validAge; historyIdx++) {
*      access to locationHistory
*   }
* @endcode
*/

typedef struct dwFeatureHistoryArray
{
    dwFeature2DStatus* statuses; /**< Status of each feature. 1D array of size maxFeatures.  */
    uint32_t* ages;              /**< Age of each feature. 1D array of size maxFeatures. */
    float32_t* scales;           /**< Scale change for each feature. 1D array of size maxFeatures. */
    uint32_t* ids;               /**< Id of each feature. 1D array of size maxFeatures. */
    uint32_t* newToOldMap;       /**< New to old index map, 1D array of size maxFeatures. See more details in `dwFeature2DTracker_trackFeatures`. */
    dwVector2f* locationHistory; /**< Location history of feature points. 2D array of size maxFeatures*maxHistory. */
    uint32_t* featureCount;      /**< Total number of feature points. Single value. */
    uint32_t* validTrackedCount; /**< Valid tracked features from last frame. Single value. */

    uint32_t currentTimeIdx; /**< Index that points to the latest feature records */
    uint32_t maxHistory;     /**< Max feature history size */
    uint32_t maxFeatures;    /**< Max number of features in one timeIdx */

    uint8_t* data; /**< Pointer to the raw data address*/
    size_t bytes;  /**< Bytes of raw data*/

    dwMemoryType memoryType; /**< Where feature array is located, GPU, CPU or pinned memory. */
} dwFeatureHistoryArray;

typedef struct dwFeatureArray
{
    dwFeature2DStatus* statuses; /**< Status of each feature. 1D array of size maxFeatures.  */
    uint32_t* ages;              /**< Age of each feature. 1D array of size maxFeatures. */
    float32_t* scales;           /**< Scale change for each feature. 1D array of size maxFeatures. */
    uint32_t* ids;               /**< Id of each feature. 1D array of size maxFeatures. */
    uint32_t* newToOldMap;       /**< New to old index map, 1D array of size maxFeatures. See more details in `dwFeature2DTracker_trackFeatures`. */
    dwVector2f* locations;       /**< Location of feature points. 2D array of size maxFeatures */
    uint32_t* featureCount;      /**< Total number of feature points. Single value. */
    uint32_t* validTrackedCount; /**< Valid tracked features from last frame. Single value. */

    uint32_t timeIdx;     /**< Time index, 0 means latest, N means N frames earlier to latest*/
    uint32_t maxFeatures; /**< Max number of features. */

    dwMemoryType memoryType; /**< Where feature array is located, GPU, CPU or pinned memory. */
} dwFeatureArray;

typedef struct dwFeatureDescriptorArray
{
    dwFeature2DStatus* statuses; /**< Status of each feature. 1D array of size maxFeatures.  */
    dwVector2f* locations;       /**< Location of feature points. 2D array of size maxFeatures. */
    uint8_t* descriptors;        /**< Descriptor of feature points. 1D array of size maxFeatures*descriptorDimension. */
    uint32_t* featureCount;      /**< Total number of feature points. Single value. */

    dwTrivialDataType dataType; /**< Descriptor data type dimension. */
    uint32_t dimension;         /**< Descriptor dimension. */
    uint32_t maxFeatures;       /**< Max number of features. */

    uint8_t* data; /**< Pointer to the raw data address*/
    size_t bytes;  /**< Bytes of raw data*/

    dwMemoryType memoryType; /**< Where feature array is located, GPU, CPU or pinned memory. */
} dwFeatureDescriptorArray;
///////////////////////////////////////////////////////////////////////
// dwFeature2DList
/**
 * Creates and initializes a feature array.
 *
 * @param[out] featureArray pointer to the dwFeatureArray is returned here.
 * @param[in] maxFeatureCount maximum number of features that the feature array can have.
 * @param[in] memoryType DW_FEATURE2D_MEMORY_TYPE_CUDA for CUDA array, <br>
 *                       DW_FEATURE2D_MEMORY_TYPE_CPU for CPU array, <br>
 *                       DW_FEATURE2D_MEMORY_TYPE_PINNED for pinned memory
 * @param[in] context handle to the context under which it is created.
 *
 * @return DW_INVALID_ARGUMENT if feature arry or context are NULL. <br>
 *                             if maxFeatureCount is 0. <br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwFeatureArray_create(dwFeatureArray* featureArray,
                               const uint32_t maxFeatureCount,
                               const dwMemoryType memoryType,
                               dwContextHandle_t context);

/**
 * Destroys the featureArray and frees any memory created by dwFeatureArray_create().
 *
 * @param[in] featureArray feature array to be destroyed.
 *
 * @return DW_INVALID_ARGUMENT if featureArray contains invalid pointers. <br>
 *         DW_SUCCESS otherwise. <br>
 *
 * @note dwFeatureArray got by dwFeatureHistoryArray_get*() API doesn't need to be destroyed by this API,
 * all resource will be freed when calling dwFeatureHistoryArray_destroy().
 */
DW_API_PUBLIC
dwStatus dwFeatureArray_destroy(dwFeatureArray featureArray);

/**
 * Resets the feature array.
 * Sets the feature count back to zero.
 *
 * @param[in] featureArray feature array to be reset.
 * @param[in] stream CUDA stream used to reset the feature array
 *
 * @return DW_INVALID_ARGUMENT if featureArray.featureCount is NULL. <br>
 *         DW_SUCCESS otherwise. <br>
 */
DW_API_PUBLIC
dwStatus dwFeatureArray_reset(dwFeatureArray* featureArray,
                              cudaStream_t stream);

/**
 * Deep copy all contents from `srcFeatures` to `dstFeatures`
 * @param[out] dstFeatures `dwFeatureArray` to copy to
 * @param[in] srcFeatures `dwFeatureArray` to copy from
 * @param[in] stream Working cuda stream
 * @return DW_INVALID_ARGUMENT if `dstFeatures` or `srcFeatures` is NULL. <br>
 *                             if `dstFeatures.maxFeatures != srcFeatures.maxFeatures`. <br>
 *         A specific CUDA error in case of an underlying cuda failure.<br>
 *         DW_SUCCESS otherwise. <br>
 *
 * @note If the copy function is run on a non-zero stream, it's asynchronous calling, need do
 * stream sync or event sync to make sure the copy is done.
 */
DW_API_PUBLIC
dwStatus dwFeatureArray_copyAsync(dwFeatureArray* dstFeatures,
                                  const dwFeatureArray* srcFeatures,
                                  cudaStream_t stream);

/**
 * Creates and initializes a feature history array.
 *
 * @param[out] featureHistoryArray pointer to the dwFeatureHistoryArray is returned here.
 * @param[in] maxFeatureCount maximum number of features that each time slice can have.
 * @param[in] maxHistoryCapacity maximum length of history in feature history array. There'll be maxFeatureCount*maxHistoryCapacity features totally.
 * @param[in] memoryType DW_FEATURE2D_MEMORY_TYPE_CUDA for CUDA array, <br>
 *                       DW_FEATURE2D_MEMORY_TYPE_CPU for CPU array, <br>
 *                       DW_FEATURE2D_MEMORY_TYPE_PINNED for pinned memory
 * @param[in] context handle to the context under which it is created.
 * @return DW_INVALID_ARGUMENT if feature arry or context are NULL. <br>
 *                             if maxFeatureCount or maxHistoryCapacity is 0. <br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwFeatureHistoryArray_create(dwFeatureHistoryArray* featureHistoryArray,
                                      const uint32_t maxFeatureCount,
                                      const uint32_t maxHistoryCapacity,
                                      const dwMemoryType memoryType,
                                      dwContextHandle_t context);

/**
 * Destroys the featureHistoryArray and frees any memory created by dwFeatureHistoryArray_create().
 *
 * @param[in] featureHistoryArray feature history array to be destroyed.
 *
 * @return DW_INVALID_ARGUMENT if featureHistoryArray contains invalid pointers. <br>
 *         DW_SUCCESS otherwise. <br>
 */
DW_API_PUBLIC
dwStatus dwFeatureHistoryArray_destroy(dwFeatureHistoryArray featureHistoryArray);

/**
 * Resets the feature history array.
 * Sets the feature count back to zero.
 *
 * @param[in] featureHistoryArray feature history array to be reset.
 * @param[in] stream CUDA stream used to reset the feature history array
 *
 * @return DW_INVALID_ARGUMENT if featureHistoryArray.featureCount is NULL. <br>
 *         DW_SUCCESS otherwise. <br>
 */
DW_API_PUBLIC
dwStatus dwFeatureHistoryArray_reset(dwFeatureHistoryArray* featureHistoryArray,
                                     cudaStream_t stream);

/**
 * Get the latest feature snapshot(arraySize = maxFeatureCount) from history.
 *
 * @param[out] featureArray Snapshot of latest time in feature history.
 * @param[in] featureHistoryArray Complete feature history.
 *
 * @return DW_INVALID_ARGUMENT if featureArray or featureHistoryArray is NULL. <br>
 *         DW_SUCCESS otherwise. <br>
 *
 * @note featureArray got by this API DOES NOT need to be freed by dwFeatureArray_destroy().
 */
DW_API_PUBLIC
dwStatus dwFeatureHistoryArray_getCurrent(dwFeatureArray* featureArray,
                                          dwFeatureHistoryArray const* featureHistoryArray);

/**
 * Get the feature snapshot(arraySize = maxFeatureCount) 1 frame previous to current time.
 * For N valid features in current time T, this API returns the snapshot of those N features
 * in T - 1. For a given feature F, ages(F) <= 1 means feature F is invalid in T - 1.
 *
 * @param[out] featureArray Snapshot of the timeIdx that is 1 frame previous to current in feature history.
 * @param[in] featureHistoryArray Complete feature history.
 *
 * @return DW_INVALID_ARGUMENT if featureArray or featureHistoryArray is NULL. <br>
 *         DW_SUCCESS otherwise. <br>
 *
 * @note featureArray got by this API DOES NOT need to be freed by dwFeatureArray_destroy()
 */
DW_API_PUBLIC
dwStatus dwFeatureHistoryArray_getPrevious(dwFeatureArray* featureArray,
                                           dwFeatureHistoryArray const* featureHistoryArray);

/**
 * Get the feature snapshot(arraySize = maxFeatureCount) historyIdx-th frame earlier.
 * For N valid features in current time T, this API returns the snapshot of those N features
 * in T - historyIdx. For a given feature F, ages(F) <= historyIdx means feature F is invalid
 * in T - historyIdx.
 *
 * @param[out] featureArray Snapshot of the timeIdx that is historyIdx frame previous to current in feature history.
 * @param[in] historyIdx Time index that need to be backtraced.
 * @param[in] featureHistoryArray Complete feature history.
 *
 * @return DW_INVALID_ARGUMENT if featureArray or featureHistoryArray is NULL <br>
 *         DW_SUCCESS otherwise <br>
 *
 * @note featureArray got by this API DOES NOT need to be freed by dwFeatureArray_destroy().
 */
DW_API_PUBLIC
dwStatus dwFeatureHistoryArray_get(dwFeatureArray* featureArray,
                                   const uint32_t historyIdx,
                                   dwFeatureHistoryArray const* featureHistoryArray);

/**
 * Deep copy all contents from `srcFeatureHistory` to `dstFeatureHistory`
 * @param[out] dstFeatureHistory `dwFeatureHistoryArray` to copy to
 * @param[in] srcFeatureHistory `dwFeatureHistoryArray` to copy from
 * @param[in] stream Working cuda stream
 *
 * @return DW_INVALID_ARGUMENT if `dstFeatureHistory` or `srcFeatureHistory` is NULL. <br>
 *                             if `dstFeatureHistory.bytes != srcFeaturesHistory.bytes`. <br>
 *         A specific CUDA error in case of an underlying cuda failure.<br>
 *         DW_SUCCESS otherwise. <br>
 *
 * @note If the copy function is run on a non-zero stream, it's asynchronous calling, need do
 * stream sync or event sync to make sure the copy is done.
 */
DW_API_PUBLIC
dwStatus dwFeatureHistoryArray_copyAsync(dwFeatureHistoryArray* dstFeatureHistory,
                                         const dwFeatureHistoryArray* srcFeatureHistory,
                                         cudaStream_t stream);
/**
 * Merge `newToOldMap` from 2 inputs.
 * i.e. There're 3 working stages: A->B->C, `newToOldMapC2B` gives the mapping of C->B,
 * while `newToOldMapB2A` gives the mapping of B->A. The final C->A mapping will be computed
 * and output to `newToOldMapC2A`
 *
 * @param[out] d_newToOldMapC2A A GPU pointer to the output merged newToOldMap
 * @param[in] d_newToOldMapC2B A GPU pointer to newToOldMap from stage C->B
 * @param[in] d_newToOldMapB2A A GPU pointer to newToOldMap from stage B->A
 * @param[in] d_newToOldMapCount A GPU pointer to valid number of entries in `newToOldMapC2B`
 * @param[in] maxFeatureCount max number of features for `newToOldMapC2A`, `newToOldMapC2B` and `newToOldMapB2A`
 * @param[in] stream Working cuda stream
 *
 * @return DW_INVALID_ARGUMENT if any of the input argument is NULL. <br>
 *         DW_SUCCESS otherwise. <br>
 */
DW_API_PUBLIC
dwStatus dwFeature_mergeNewToOldMap(uint32_t* d_newToOldMapC2A,
                                    const uint32_t* d_newToOldMapC2B,
                                    const uint32_t* d_newToOldMapB2A,
                                    const uint32_t* d_newToOldMapCount,
                                    const uint32_t maxFeatureCount,
                                    cudaStream_t stream);

/**
 * Creates descriptor array
 * @param[out] descriptorArray output pointer returned for descriptors
 * @param[in] dataType data type of descriptor
 * @param[in] dimension dimension of descriptor
 * @param[in] maxFeatureCount maximum feature count for the descriptor array
 * @param[in] memoryType DW_MEMORY_TYPE_CUDA for CUDA array, <br>
 *                       DW_MEMORY_TYPE_CPU for CPU array, <br>
 *                       DW_MEMORY_TYPE_PINNED for pinned memory
 * @param[in] context handle to the context under which it is created.
 *
 * @return DW_INVALID_ARGUMENT if descriptor array or context are NULL. <br>
 *                             if maxFeatureCount is 0. <br>
 *         DW_SUCCESS otherwise.<br>
 */
DW_API_PUBLIC
dwStatus dwFeatureDescriptorArray_create(dwFeatureDescriptorArray* descriptorArray,
                                         dwTrivialDataType const dataType,
                                         uint32_t const dimension,
                                         uint32_t const maxFeatureCount,
                                         dwMemoryType const memoryType,
                                         dwContextHandle_t context);

/**
 * Destroys the descriptor array and frees any memory created by dwDescriptorArray_create()
 * @param[in] descriptorArray dwFeatureDescriptorArray to be destroyed
 *
 * @return DW_INVALID_ARGUMENT if featureArray contains invalid pointers. <br>
 *         DW_SUCCESS otherwise. <br>
 *
 */
DW_API_PUBLIC
dwStatus dwFeatureDescriptorArray_destroy(dwFeatureDescriptorArray const* descriptorArray);

/**
 * Deep copy all contents from srcDescriptors to dstDescriptors. Not to use for CPU to CPU copy.
 * @param[out] dstDescriptors dwFeatureDescriptorArray to copy to
 * @param[in] srcDescriptors dwFeatureDescriptorArray to copy from
 * @param[in] stream working cuda stream
 * @return DW_INVALID_ARGUMENT if `dstDescriptors` or `srcDescriptors` is NULL. <br>
 *                             if `dstDescriptors.maxFeatures != srcDescriptors.maxFeatures`. <br>
 *         A specific CUDA error in case of an underlying cuda failure.<br>
 *         DW_SUCCESS otherwise. <br>
 *
 * @note If the copy function is run on a non-zero stream, it's asynchronous calling, need do
 * stream sync or event sync to make sure the copy is done.
 */
DW_API_PUBLIC
dwStatus dwFeatureDescriptorArray_copyAsync(dwFeatureDescriptorArray* dstDescriptors,
                                            dwFeatureDescriptorArray const* srcDescriptors,
                                            cudaStream_t stream);

/**
 * Deep copy all contents from srcDescriptors to dstDescriptors. Only for CPU to CPU copy.
 * @param[out] dstDescriptors dwFeatureDescriptorArray to copy to
 * @param[in] srcDescriptors dwFeatureDescriptorArray to copy from
 * @return DW_INVALID_ARGUMENT if `dstDescriptors` or `srcDescriptors` is NULL. <br>
 *                             if `dstDescriptors.maxFeatures != srcDescriptors.maxFeatures`. <br>
 *         DW_SUCCESS otherwise. <br>
 *
 */
DW_API_PUBLIC
dwStatus dwFeatureDescriptorArray_copy(dwFeatureDescriptorArray* dstDescriptors,
                                       dwFeatureDescriptorArray const* srcDescriptors);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_IMAGEPROCESSING_FEATURES_FEATURELIST_H_
