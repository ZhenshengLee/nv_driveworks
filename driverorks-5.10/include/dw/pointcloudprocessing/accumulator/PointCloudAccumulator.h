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
 * <b>NVIDIA DriveWorks API: Point Cloud Accumulator</b>
 *
 * @b Description: This file defines API of point cloud accumulator module.
 */

/**
 * @defgroup pointcloudaccumulator_group Point Cloud Accumulator
 * @ingroup pointcloudprocessing_group
 *
 * @brief Defines datatypes and functions to accumulate cloud of points.
 *
 * @{
 */

#ifndef DW_POINTCLOUDPROCESSING_POINTCLOUDACCUMULATOR_H_
#define DW_POINTCLOUDPROCESSING_POINTCLOUDACCUMULATOR_H_

#include <dw/core/base/Types.h>
#include <dw/pointcloudprocessing/pointcloud/PointCloud.h>
#include <dw/egomotion/Egomotion.h>
#include <dw/sensors/lidar/Lidar.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct dwPointCloudAccumulatorObject* dwPointCloudAccumulatorHandle_t;

/**
 * @brief Defines point cloud accumulator parameters
 *
 * @note This parameter data struct can be set to collect subsets of the original raw data.
 * For rotational lidar, accumulator sets rotational angle of interest and Euclidean distance
 * of interest to collect. User can also specify `filterWindowSize` to pick one data point in the window.
 * The variable `filterWindowSize` is expected to be the power of 2 whose exponent ranges from 0 to 4.
 * It reduces the jittering of the Lidar beam firing in the horizontal direction when window size is larger than 1.
 */
typedef struct
{
    dwMemoryType memoryType; //!< The module will process lidar packets and output to cuda memory <br>
                             //!< if `memoryType = DW_MEMORY_CUDA`. The module will process <br>
                             //!< lidar packets and output to cpu memory if `memoryType = DW_MEMORY_CPU`

    uint32_t outputFormats; //!< Combination of desired dwPointCloudFormat flags

    uint32_t filterWindowSize;  //!< The horizontal smoothing filter window size
    float32_t minAngleDegree;   //!< Starting angle in degree
    float32_t maxAngleDegree;   //!< Ending angle in degree
    float32_t minDistanceMeter; //!< Starting distance in meter
    float32_t maxDistanceMeter; //!< Ending distance in degree

    bool enableMotionCompensation; //!< Setting it to true will correct the distortions caused by lidar sensor motion
    bool organized;                //!< If user sets it to true, the module will process the lidar packets such that
                                   //!< the output data is aligned on 3D grid
    bool enableZeroCrossDetection; //!< If set to true end of spin is detected based on angle of incoming points
                                   //!< otherwise on number of incoming packets
    bool outputInRigCoordinates;   //!< If true output points are in rig coordinates

    dwEgomotionConstHandle_t egomotion;      //!< Handle to egomotion module
    dwTransformation3f sensorTransformation; //!< Transformation aligns the lidar sensor with the platform that produces the egomotion

} dwPointCloudAccumulatorParams;

/**
 * @brief Defines timestamp range of a point cloud
 *
 * @note The structure is used as an optional output of point cloud accumulator.
 * Contains timestamp range in both host ans sensor domains.
 */
typedef struct dwPointCloudTimestampRange
{
    dwTime_t hostStartTimestamp;
    dwTime_t hostEndTimestamp;

    dwTime_t sensorStartTimestamp;
    dwTime_t sensorEndTimestamp;

} dwPointCloudTimestampRange;

/**
 * @brief Initializes point cloud accumulator
 * @param[out] obj                 Pointer to point cloud accumulator handle
 * @param[in]  accumulationParams  Pointer to point cloud accumulator parameters
 * @param[in]  lidarProperties     Pointer to lidar properties
 * @param[in]  ctx                 Handle to the context
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT
 */
DW_API_PUBLIC
dwStatus dwPointCloudAccumulator_initialize(dwPointCloudAccumulatorHandle_t* obj,
                                            const dwPointCloudAccumulatorParams* accumulationParams,
                                            const dwLidarProperties* lidarProperties,
                                            dwContextHandle_t ctx);
/**
 * @brief Resets point cloud accumulator
 * @param[in] obj  Handle to point cloud accumulator
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT
 */
DW_API_PUBLIC
dwStatus dwPointCloudAccumulator_reset(dwPointCloudAccumulatorHandle_t obj);

/**
 * @brief Releases point cloud accumulator
 * @param[in] obj Handle to point cloud accumulator
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT
 */
DW_API_PUBLIC
dwStatus dwPointCloudAccumulator_release(dwPointCloudAccumulatorHandle_t obj);

/**
 * @brief Gets default point cloud accumulator parameters
 * @param[out] params  Pointer to point cloud parameters
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT
 */
DW_API_PUBLIC
dwStatus dwPointCloudAccumulator_getDefaultParams(dwPointCloudAccumulatorParams* params);

/**
 * @brief Sets lidar to rig transformation
 * @param[in] transformation A pointer to the transform
 * @param[in] obj            Handle to point cloud accumulator
 * @return DW_SUCCESS
 */
DW_API_PUBLIC
dwStatus dwPointCloudAccumulator_setLidarToRigTransformation(const dwTransformation3f* transformation,
                                                             dwPointCloudAccumulatorHandle_t obj);

/**
 * @brief Sets CUDA stream of point cloud accumulator
 * @param[in] stream  Handle to CUDA stream
 * @param[in] obj     Handle to point cloud accumulator
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT<br>
 *         DW_CALL_NOT_ALLOWED   - point cloud accumulator is initialized to process host memory
 */
DW_API_PUBLIC
dwStatus dwPointCloudAccumulator_setCUDAStream(const cudaStream_t stream,
                                               dwPointCloudAccumulatorHandle_t obj);
/**
 * @brief Gets CUDA stream of point cloud accumulator
 * @param[out] stream  Pointer to CUDA stream handle
 * @param[in]  obj     Handle to point cloud accumulator
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT<br>
 *         DW_CALL_NOT_ALLOWED   - point cloud accumulator is initialized to process host memory
 */
DW_API_PUBLIC
dwStatus dwPointCloudAccumulator_getCUDAStream(cudaStream_t* stream,
                                               dwPointCloudAccumulatorHandle_t obj);

/**
 * @brief Pushes lidar packet to point cloud accumulator
 * @param[in] packet  Pointer to decoded lidar packet
 * @param[in] obj     Handle to point cloud accumulator
 * @return DW_SUCCESS           - if successfully added packet<br>
 *         DW_INTERNAL_ERROR    - the values included in decoded lidar packet
 *                              do not match the ones in the lidar properties <br>
 *         DW_INVALID_ARGUMENT
 */
DW_API_PUBLIC
dwStatus dwPointCloudAccumulator_addLidarPacket(const dwLidarDecodedPacket* packet,
                                                dwPointCloudAccumulatorHandle_t obj);
/**
* @brief Indicate that enough data has been collected to perform full combination
*
* When this method returns true, a call to `dwPointCloudAccumulator_process()`
* can be used to retrieve a solution.
*
* @param[out] isReady If `true` we have enough packets to full fill the selcted strategy
* @param[in]  obj     Handle to point cloud accumulator
*
* @return DW_INVALID_HANDLE - if given handle is invalid<br>
*         DW_INVALID_ARGUMENT - if `isReady` is nullptr<br>
*         DW_SUCCESS
*/
DW_API_PUBLIC
dwStatus dwPointCloudAccumulator_isReady(bool* isReady, dwPointCloudAccumulatorHandle_t obj);

/**
 * @brief Binds output point cloud buffer
 * @param[out] pointCloud  Pointer to output buffer
 * @param[in]  obj         Handle to point cloud accumulator
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT   - the memory type in the output buffer does not match the one
 *                               specified in `dwPointCloudAccumulatorParams`. <br>
 *                               Or user did not allocate memory. Please call `dwPointCloud_createBuffer` in advance
 *                               to allocate proper memory storage. <br>
 */
DW_API_PUBLIC
dwStatus dwPointCloudAccumulator_bindOutput(dwPointCloud* pointCloud,
                                            dwPointCloudAccumulatorHandle_t obj);

/**
 * @brief Binds output timestamp range
 * @param[out] timestampRange Timestamp range of a output point cloud
 * @param[in]  obj         Handle to point cloud accumulator
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT
 */
DW_API_PUBLIC
dwStatus dwPointCloudAccumulator_bindOutputTimestamps(dwPointCloudTimestampRange* timestampRange,
                                                      dwPointCloudAccumulatorHandle_t obj);

/**
 * @brief Accumulates lidar packets and stores the results to the output buffer
 * @param[in] obj Handle to point cloud accumulator
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT
 * @note Upon successful execution, the function will modify the following variables in output buffer
 *         -# `organized` will be set to the same value specified in `dwPointCloudAccumulatorParams`
 *         -# `size` will be non-zero value denotes the current accumulated points in the memory.
 */
DW_API_PUBLIC
dwStatus dwPointCloudAccumulator_process(dwPointCloudAccumulatorHandle_t obj);

/**
 * Gets sweeps/spins size
 * @param[out] size Sweep size, x is width and y is height
 * @param[in] obj Handle to the Point Cloud Accumulator module
 * @return DW_INVALID_HANDLE   if given handle is invalid, i.e. null or of wrong type  <br/>
 *         DW_INVALID_ARGUMENT if given arguments are invalid <br/>
 *         DW_SUCCESS<br/>
 *
 * @note User can call this function once Point Cloud Accumulator is initialized
 */
DW_API_PUBLIC
dwStatus dwPointCloudAccumulator_getSweepSize(dwVector2ui* size, dwPointCloudAccumulatorHandle_t obj);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_POINTCLOUDPROCESSING_POINTCLOUDACCUMULATOR_H_
