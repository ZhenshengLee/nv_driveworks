/////////////////////////////////////////////////////////////////////////////////////////
// This code contains NVIDIA Confidential Information and is disclosed
// under the Mutual Non-Disclosure Agreement.
//
// Notice
// ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS" NVIDIA MAKES
// NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//
// NVIDIA Corporation assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of NVIDIA Corporation. No third party distribution is allowed unless
// expressly authorized by NVIDIA.  Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// NVIDIA Corporation products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// NVIDIA Corporation.
//
// Copyright (c) 2021-2022 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

/**
 * @file
 * <b>NVIDIA DriveWorks API: Point Cloud Processing</b>
 *
 * @b Description: This file defines API of point cloud processing module
 */

/**
 * @defgroup pointcloudprocessing_group Point Cloud Processing Interface
 *
 * @brief Defines point cloud assembling module
 *
 * @{
 */

#ifndef DW_POINTCLOUDPROCESSING_POINTCLOUDASSEMBLER_H_
#define DW_POINTCLOUDPROCESSING_POINTCLOUDASSEMBLER_H_

#include <dw/pointcloudprocessing/pointcloud/LidarPointCloud.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dwPointCloudAssemblerObject* dwPointCloudAssemblerHandle_t;
typedef struct dwPointCloudAssemblerObject const* dwPointCloudAssemblerConstHandle_t;

/**
 * @brief Initialization parameters
 *
*/
typedef struct
{
    /** If set to true, assembling to GPU memory */
    bool enableCuda;

    /** Layers and aux channels mapping */
    dwLidarPointCloudMapping mapping;

} dwPointCloudAssemblerParams;

/**
 * @brief Get default initialization parameters for specified lidar device
 *
 * @param[out] params              Initialization parameters
 * @param[in]  lidarProperties     Lidar properties
 *
 * @return DW_INVALID_ARGUMENT - if one of the given pointers is NULL<br>
 *         DW_SUCCESS
 */
DW_API_PUBLIC
dwStatus dwPointCloudAssembler_getDefaultParams(dwPointCloudAssemblerParams* const params,
                                                dwLidarProperties const* const lidarProperties);

/**
 * @brief Initialize point cloud assembler module
 *
 * @param[out] obj                 Pointer to point cloud assembler handle
 * @param[in]  params              Initialization parameters
 * @param[in]  lidarProperties     Input lidar properties
 * @param[in]  ctx                 Handle to the context
 *
 * @return DW_INVALID_ARGUMENT - if one of the given pointers is NULL<br>
 *         DW_SUCCESS
 */
DW_API_PUBLIC
dwStatus dwPointCloudAssembler_initialize(dwPointCloudAssemblerHandle_t* const obj,
                                          dwPointCloudAssemblerParams const* const params,
                                          dwLidarProperties const* const lidarProperties,
                                          dwContextHandle_t const ctx);
/**
 * @brief Reset point cloud assembler.
 *
 * @param[in] obj  Module handle
 *
 * @return DW_INVALID_HANDLE - if given handle is invalid<br>
 *         DW_SUCCESS
 */
DW_API_PUBLIC
dwStatus dwPointCloudAssembler_reset(dwPointCloudAssemblerHandle_t const obj);

/**
 * @brief Release point cloud assembler
 *
 * @param[in] obj Module handle
 *
 * @return DW_INVALID_HANDLE - if given handle is invalid<br>
 *         DW_SUCCESS
 */
DW_API_PUBLIC
dwStatus dwPointCloudAssembler_release(dwPointCloudAssemblerHandle_t const obj);

/**
 * @brief Set CUDA stream of point cloud assembler
 *
 * @param[in] stream  Handle to CUDA stream
 * @param[in] obj     Module handle
 *
 * @return DW_INVALID_HANDLE - if given handle is invalid<br>
 *         DW_CALL_NOT_ALLOWED   - point cloud assembler is initialized to process host memory <br>
 *         DW_SUCCESS
 */
DW_API_PUBLIC
dwStatus dwPointCloudAssembler_setCUDAStream(cudaStream_t const stream,
                                             dwPointCloudAssemblerHandle_t const obj);
/**
 * @brief Get CUDA stream of point cloud assembler
 *
 * @param[out] stream  Pointer to CUDA stream handle
 * @param[in]  obj     Module handle
 *
 * @return DW_INVALID_HANDLE - if given handle is invalid<br>
 *         DW_CALL_NOT_ALLOWED   - point cloud assembler is initialized to process host memory<br>
 *         DW_SUCCESS
 */
DW_API_PUBLIC
dwStatus dwPointCloudAssembler_getCUDAStream(cudaStream_t* const stream,
                                             dwPointCloudAssemblerConstHandle_t const obj);

/**
 * @brief Bind output point cloud
 *
 * @param[out] pointCloud                 Lidar point cloud
 * @param[in]  obj                        Module handle
 * @return DW_INVALID_HANDLE     - if given handle is invalid<br>
 *         DW_INVALID_ARGUMENT   - if trying to bind GPU point cloud when cuda is not
 *                                 is not enabled at initialization<br>
 *         DW_SUCCESS
 *
 * @note Output points are either XYZI or RTHI points depending on the format of bound point cloud.
 */
DW_API_PUBLIC
dwStatus dwPointCloudAssembler_bindOutput(dwPointCloud* const pointCloud,
                                          dwPointCloudAssemblerHandle_t const obj);

/**
 * @brief Bind output lidar specific point cloud
 *
 * @param[out] pointCloud          Lidar point cloud
 * @param[in]  obj                 Module handle
 * @return DW_INVALID_HANDLE     - if given handle is invalid<br>
 *         DW_INVALID_ARGUMENT   - if trying to bind GPU point cloud when cuda is not
 *                                 is not enabled at initialization<br>
 *         DW_SUCCESS
 */
DW_API_PUBLIC
dwStatus dwPointCloudAssembler_bindLidarPointCloud(dwLidarPointCloud* const pointCloud,
                                                   dwPointCloudAssemblerHandle_t const obj);

/**
 * @brief Push lidar packet to point cloud assembler
 * @param[in] packet  Pointer to decoded lidar packet
 * @param[in] obj     Module handle
 * @return DW_SUCCESS           - if successfully added packet<br>
 *         DW_INVALID_ARGUMENT  - the values included in decoded lidar packet
 *                              do not match the ones in the lidar properties <br>
 */
DW_API_PUBLIC
dwStatus dwPointCloudAssembler_addLidarPacket(dwLidarDecodedPacket const* const packet,
                                              dwPointCloudAssemblerHandle_t const obj);

/**
 * @brief Indicate that lidar frame has been accumulated
 *
 * @param[out] isReady If `true` frame has been accumulated
 * @param[in]  obj     Module handle
 *
 * @return DW_INVALID_HANDLE - if given handle is invalid<br>
 *         DW_INVALID_ARGUMENT - if provided pointer to a packet is nullptr<br>
 *         DW_SUCCESS
 */
DW_API_PUBLIC
dwStatus dwPointCloudAssembler_isReady(bool* const isReady, dwPointCloudAssemblerConstHandle_t const obj);

/**
 * @brief Perform processing of accumulated data.
 *
 * @param[in] obj Module handle
 *
 * @return DW_INVALID_HANDLE   - if given handle is invalid<br>
 *         DW_NOT_READY        - if accumulation is not complete, i.e.
 *                               `dwPointCloudAssembler_isReady` returns false
 *         DW_CALL_NOT_ALLOWED - if output is not bound
 *         DW_SUCCESS
 *
 * @note The function return the most recent full spin in output point cloud.
 *
 * Example:
 *
 * Arrived packets  | First process call | Second process call
 * 0,1,2*,3,4*      | {2,3}              | DW_NOT_READY
 * 0,1,2*,3,4*,5    | {2,3}              | DW_NOT_READY
 * 0,1,2*,3,4*,5,6* | {4,5}              | DW_NOT_READY
 *
 * where numbers denote packet ids and (*) denotes if the packet has zero crossing
 * flag set to true.
 */
DW_API_PUBLIC
dwStatus dwPointCloudAssembler_process(dwPointCloudAssemblerHandle_t const obj);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_POINTCLOUDPROCESSING_POINTCLOUDASSEMBLER_H_