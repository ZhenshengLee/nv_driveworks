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
// Copyright (c) 2022-2023 NVIDIA Corporation. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Lidar Point Cloud Processing</b>
 *
 * @b Description: This file defines API of lidar point cloud processing module
 */

/**
 * @defgroup pointcloudprocessing_group Point Cloud Processing Interface
 *
 * @brief Defines lidar point cloud structure
 *
 * @{
 */

#ifndef DW_POINTCLOUDPROCESSING_LIDARPOINTCLOUD_LIDARPOINTCLOUD_H_
#define DW_POINTCLOUDPROCESSING_LIDARPOINTCLOUD_LIDARPOINTCLOUD_H_

#include <dw/pointcloudprocessing/pointcloud/PointCloud.h>
#include <dw/sensors/lidar/Lidar.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Defines point cloud coordinate reference frame
 */
typedef enum {
    DW_POINTCLOUD_REFERENCE_FRAME_SENSOR = 0, //!< Coordinate frame with the sensor at the origin
    DW_POINTCLOUD_REFERENCE_FRAME_RIG    = 1, //!< Coordinate frame with the ego vehicle at the origin
    DW_POINTCLOUD_REFERENCE_FRAME_CUSTOM = 2  //!< Custom coordinate reference frame
} dwPointCloudReferenceFrame;

/**
 * @brief Declares motion compensation traits of the point cloud
 */
typedef struct
{
    bool compensated;               //!< True if this pointcloud has been motion compensated
    dwTime_t compensationTimestamp; //!< Motion compensation reference timestamp
} dwLidarMotionCompensationInfo;

/**
 * @brief Struct indicating layer and aux channel mapping
 *
*/
typedef struct
{
    /** Number of layers */
    uint32_t numLayers;
    /** Map a return type to a layer, i.e. put into layer i a return type from SAL indicated by layerType[i] */
    dwLidarReturnType layerType[DW_POINT_CLOUD_MAX_LAYERS];

    uint32_t numAuxChannels;
    /** Map aux channel from SAL to aux channel of a point cloud */
    dwLidarAuxDataType auxType[DW_POINT_CLOUD_MAX_AUX_CHANNELS];
} dwLidarPointCloudMapping;

/**
 * @brief Number of elements in user buffer
 */
#define DW_LIDAR_POINT_CLOUD_USER_DATA_SIZE 8

/**
 * @brief Defines a LIDAR-specific point cloud data structure
 */
typedef struct dwLidarPointCloud
{
    /** User defined data */
    uint32_t userData[DW_LIDAR_POINT_CLOUD_USER_DATA_SIZE];
    /** Wrapped point cloud */
    dwPointCloud pointCloud;
    /** Mapping of returns and aux channels */
    dwLidarPointCloudMapping mapping;
    /** Coordinate reference frame for the data in this pointcloud */
    dwPointCloudReferenceFrame coordinateFrame;
    /** Motion compensation information */
    dwLidarMotionCompensationInfo motionCompensation;
} dwLidarPointCloud;

/**
* @brief Create lidar specific point cloud
*
* @param[out] lidarPointCloud Point cloud to be created
* @param[in] format Point cloud format XYZI or RTHI
* @param[in] memoryType Point cloud memory type, CUDA, CPU or PINNED
* @param[in] maxPointsPerReturn Maximum number of points in single lidar return
* @param[in] mapping Return and aux data mapping
*
* @return DW_SUCCESS<br>
*         DW_INVALID_ARGUMENT If provided pointer to point cloud is null
* @par API Group
* - Init: Yes
* - Runtime: No
* - De-Init: No
*/
DW_API_PUBLIC
dwStatus dwLidarPointCloud_create(dwLidarPointCloud* lidarPointCloud,
                                  dwPointCloudFormat const format,
                                  dwMemoryType const memoryType,
                                  uint32_t const maxPointsPerReturn,
                                  dwLidarPointCloudMapping const* mapping);

/**
* @brief Destroy lidar specific point cloud.
*
* @param[in] lidarPointCloud A pointer to point cloud
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
dwStatus dwLidarPointCloud_destroy(dwLidarPointCloud* lidarPointCloud);

/**
 * @brief Get the size of the lidar point cloud data type
 * @param[out] size number of bytes in a point
 * @param[in] format describes the values used to represent a point in a point cloud
 *
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT If provided pointer is null or provided enum is invalid
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwLidarPointCloud_getLidarPointStride(uint32_t* size, dwPointCloudFormat const format);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_POINTCLOUDPROCESSING_LIDARPOINTCLOUD_LIDARPOINTCLOUD_H_
