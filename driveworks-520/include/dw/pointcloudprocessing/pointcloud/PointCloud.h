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
// SPDX-FileCopyrightText: Copyright (c) 2018-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Point Cloud Processing</b>
 *
 * @b Description: This file defines API of point cloud processing module
 */

/**
 * @defgroup pointcloudprocessing_group Point Cloud Processing Interface
 *
 * @brief Defines point cloud processing datatypes and memory handling functions
 *
 * @{
 */

#ifndef DW_POINTCLOUDPROCESSING_POINTCLOUD_POINTCLOUD_H_
#define DW_POINTCLOUDPROCESSING_POINTCLOUD_POINTCLOUD_H_

#include <dw/core/base/Types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Defines point format
 */

typedef enum {
    DW_POINTCLOUD_FORMAT_XYZI = 0, //!< Cartesian 3D coordinate + intensity
    DW_POINTCLOUD_FORMAT_RTHI = 1  //!< Polar 3D coordinate + intensity
} dwPointCloudFormat;

/**
* @brief Struct holding information about a single point cloud layer
*/
typedef struct
{
    uint32_t layerIdx;  //!< Index of a layer within a point cloud
    uint32_t size;      //!< Number of points in a layer (@note: points are not compactified, that means there are at most maxPointsPerLayer entries whereby some of them are (0,0,0,0) indicating an invalid point)
    const void* points; //!< Pointer to the start of the data for this layer

} dwPointCloudLayer;

/**
* @brief Struct holding information about aux channel
*/
typedef struct
{
    uint32_t elementSize; //!< Size in bytes of an element in the aux buffer
    uint32_t channelIdx;  //!< Channel index the aux data refers to
    uint32_t size;        //!< Number of elements in a channel

    /// Pointer to the start of the data for this channel.
    /// Number of elements is the size of the point cloud.
    /// To access channel[C] for point I, one would do pc.auxChannels[C].data[I]
    void* data;

} dwPointCloudAuxChannel;

#define DW_POINT_CLOUD_MAX_LAYERS 16U
#define DW_POINT_CLOUD_MAX_AUX_CHANNELS 16U

/**
 * @brief Defines point cloud data structure
 */
typedef struct
{
    dwMemoryType type;         //!< Defines type of a memory GPU or CPU
    dwPointCloudFormat format; //!< Format of a buffer
    bool organized;            //!< Flag indicating the data is ordered on the 3D grid

    dwTime_t timestamp; //!< Time when the point cloud capturing is finished

    void* points;      //!< Pointer to memory buffer of all points. Not compact, but contiguous with invalid points in between layers.
    uint32_t capacity; //!< numLayers * maxPointsPerLayer
    uint32_t size;     //!< Number of points in the cloud including all layers

    dwTime_t hostStartTimestamp; //!< Point cloud start timestamp
    dwTime_t hostEndTimestamp;   //!< Point cloud end timestamp

    uint32_t maxPointsPerLayer; //!< Maximum number of points in layer (use even number of points to benefit from aligned memory access per layer if each point is 16 bytes long)
    uint32_t numLayers;         //!< Number of layers in a point cloud
    dwPointCloudLayer layers[DW_POINT_CLOUD_MAX_LAYERS];

    uint32_t numAuxChannels; //!< Number of aux channels in a point cloud
    dwPointCloudAuxChannel auxChannels[DW_POINT_CLOUD_MAX_AUX_CHANNELS];

} dwPointCloud;

/**
 * @brief Allocates memory for point cloud data structure
 * @param[out] buffer Pointer to `dwPointCloud`
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT
 *
 * @note User must specify the followings for the argument `buffer`:
 *           -# The member variable `capacity` to indicate the memory allocation size
 *           -# The member variable `type` to indicate the memory type, either host memory or device memory
 *
 * @note Upon successful return, this API will set `size` to zero to indicate it contains no element in the memory
 *       By default this API will set `organized` to false
 *
 * @note This API does not add any information on point cloud layers and auxiliary channels.
 *       Upon successful return `numLayers` and `numAuxChannels`are both set to zero.
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: No
 */
DW_API_PUBLIC
dwStatus dwPointCloud_createBuffer(dwPointCloud* buffer);

/**
 * @brief Destroys allocated memory for point cloud data structure
 * @param[out] buffer Pointer to `dwPointCloud`
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT
 *
 * @par API Group
 * - Init: Yes
 * - Runtime: No
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwPointCloud_destroyBuffer(dwPointCloud* buffer);

/**
* @brief Create point cloud with layers and aux channel information.
*
* @param[out] pointCloud Point cloud to be created
* @param[in] format Point cloud format XYZI or RTHI
* @param[in] memoryType Point cloud memory type, CUDA, CPU or PINNED
* @param[in] maxPointsPerLayer Maximum number of points in single layer
* @param[in] numRequestedLayers Number of layer in a point cloud
* @param[in] auxChannelsElemSize An array of auxiliary channel elements sizes
* @param[in] numRequestedAuxChannels Number of requested aux channels
*
*
* @return DW_SUCCESS<br>
*         DW_INVALID_ARGUMENT If provided pointer to point cloud is null
* @par API Group
* - Init: Yes
* - Runtime: No
* - De-Init: No
*/
DW_API_PUBLIC
dwStatus dwPointCloud_create(dwPointCloud* pointCloud,
                             dwPointCloudFormat const format,
                             dwMemoryType const memoryType,
                             uint32_t const maxPointsPerLayer,
                             uint32_t const numRequestedLayers,
                             uint32_t const* auxChannelsElemSize,
                             uint32_t const numRequestedAuxChannels);

/**
* @brief Destroy point cloud buffers.
*
* @param[in] pointCloud A pointer to point cloud
*
* @note The call releases all the memory associated with point cloud.
*
* @return DW_SUCCESS<br>
*         DW_INVALID_HANDLE
* @par API Group
* - Init: Yes
* - Runtime: No
* - De-Init: Yes
*/
DW_API_PUBLIC
dwStatus dwPointCloud_destroy(dwPointCloud* pointCloud);

/**
 * @brief Get the size of the point cloud data type
 *
 * @param[out] size Number of bytes in a point
 *
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT If provided pointer to size is null
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwPointCloud_getPointStride(uint32_t* size);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_POINTCLOUDPROCESSING_POINTCLOUD_POINTCLOUD_H_
