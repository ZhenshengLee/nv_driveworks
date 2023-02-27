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
 * <b>NVIDIA DriveWorks API: Point Cloud Range Image Creator</b>
 *
 * @b Description: This file defines API of point cloud range image creator module
 */

/**
 * @defgroup pointcloudrangeimagecreator_group Point Cloud Range Image Creator
 * @ingroup pointcloudprocessing_group
 *
 * @brief Defines module to produce range image via spherical projection of the point cloud
 *
 * @{
 */

#ifndef DW_POINTCLOUDPROCESSING_POINTCLOUDRANGEIMAGECREATOR_H_
#define DW_POINTCLOUDPROCESSING_POINTCLOUDRANGEIMAGECREATOR_H_

#include <dw/core/base/Types.h>
#include <dw/core/context/Context.h>
#include <dw/pointcloudprocessing/pointcloud/PointCloud.h>
#include <dw/image/Image.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dwPointCloudRangeImageCreatorObject* dwPointCloudRangeImageCreatorHandle_t;
typedef const struct dwPointCloudRangeImageCreatorObject* dwConstPointCloudRangeImageCreatorHandle_t;

/// Definition of the image type of Lidar cylindrical projection image.
typedef enum {
    DW_POINT_CLOUD_IMAGE_TYPE_DISTANCE  = 0, //!< R_FLOAT32 image where each pixel is the 3D distance in XYZ space
    DW_POINT_CLOUD_IMAGE_TYPE_INTENSITY = 1, //!< R_FLOAT32 image where each pixel is the Lidar intensity
    DW_POINT_CLOUD_IMAGE_TYPE_2D_GRID   = 2, //!< RGBA_FLOAT32 image where each pixel is a tuple of 3D Lidar coordinate and intensity
} dwPointCloudRangeImageType;

/**
 * @brief Defines range image clipping parameters
 */
typedef struct
{
    float32_t farDist;  //!< Maximum distance
    float32_t nearDist; //!< Minimum distance

    float32_t minElevationRadians; //!< Mimimum pitch angle
    float32_t maxElevationRadians; //!< Maximum pitch angle
    float32_t minAzimuthRadians;   //!< Mimimum yaw angle
    float32_t maxAzimuthRadians;   //!< Maximum yaw angle

    dwOrientedBoundingBox3f orientedBoundingBox; //!< Bounding box identifying clipping planes
} dwPointCloudRangeImageClippingParams;

/**
 * @brief Defines point cloud range image creator parameters
 */
typedef struct
{
    dwMemoryType memoryType;                             //!< Memory type, CUDA or CPU
    uint32_t maxInputPoints;                             //!< Maximum number of point in input point cloud
    uint32_t width;                                      //!< Output image width
    uint32_t height;                                     //!< Output image height
    dwPointCloudRangeImageType type;                     //!< Range image type
    dwTransformation3f transformation;                   //!< Transformation applied to input points, if 0, identity will be used
    dwPointCloudRangeImageClippingParams clippingParams; //!< Clipping parameters
} dwPointCloudRangeImageCreatorParams;

/**
 * @brief Initializes range image creator
 *
 * @param[out] obj     Pointer to range image creator handle
 * @param[in]  params  Pointer to range image creator parameters
 * @param[in]  ctx     Handle to the context
 *
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT   If given parameter pointer is invalid<br>
 *         DW_INVALID_HANDLE     If given handle is valid<br>
 */
DW_API_PUBLIC
dwStatus dwPCRangeImageCreator_initialize(dwPointCloudRangeImageCreatorHandle_t* const obj,
                                          dwPointCloudRangeImageCreatorParams const* const params,
                                          dwContextHandle_t const ctx);

/**
 * @brief Resets range image creator
 *
 * @param[in] obj Handle to range image creator
 *
 * @return DW_SUCCESS<br>
 *         DW_INVALID_HANDLE     If given handle is valid<br>
 */
DW_API_PUBLIC
dwStatus dwPCRangeImageCreator_reset(dwPointCloudRangeImageCreatorHandle_t const obj);

/**
 * @brief Releases range image creator
 *
 * @param[in] obj Handle to range image creator
 *
 * @return DW_SUCCESS<br>
 *         DW_INVALID_HANDLE     If given handle is valid<br>
 */
DW_API_PUBLIC
dwStatus dwPCRangeImageCreator_release(dwPointCloudRangeImageCreatorHandle_t const obj);

/**
 * @brief Gets default range image creator parameters
 *
 * @param[out] params Pointer to range image creator parameters
 *
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT if given param pointer is invalid
 */
DW_API_PUBLIC
dwStatus dwPCRangeImageCreator_getDefaultParams(dwPointCloudRangeImageCreatorParams* const params);

/**
 * @brief Gets CUDA stream of range image creator
 *
 * @param[out] stream  Pointer to CUDA stream handle
 * @param[in]  obj     Handle to range image creator
 *
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT if given stream pointer is invalid<br>
 *         DW_INVALID_HANDLE     If given handle is valid<br>
 */
DW_API_PUBLIC
dwStatus dwPCRangeImageCreator_getCUDAStream(cudaStream_t* const stream,
                                             dwConstPointCloudRangeImageCreatorHandle_t const obj);

/**
 * @brief Sets CUDA stream of range image creator
 *
 * @param[in] stream  Handle to CUDA stream
 * @param[in] obj     Handle to range image creator
 *
 * @return DW_SUCCESS<br>
 *         DW_INVALID_HANDLE     If given handle is valid<br>
 */
DW_API_PUBLIC
dwStatus dwPCRangeImageCreator_setCUDAStream(cudaStream_t const stream,
                                             dwPointCloudRangeImageCreatorHandle_t const obj);

/**
 * @brief Get properties of an image to bind as an output
 *
 * @param[out] imageProperties Pointer to image properties
 * @param[in] obj Handle to range image creator
 *
 * @return DW_SUCCESS<br>
 *         DW_INVALID_HANDLE     If given handle is valid<br>
 *         DW_INVALID_ARGUMENT   If imageProperties in null
 */
DW_API_PUBLIC
dwStatus dwPCRangeImageCreator_getImageProperties(dwImageProperties* const imageProperties,
                                                  dwConstPointCloudRangeImageCreatorHandle_t const obj);

/**
 * @brief Binds input point cloud to range image creator
 *
 * @param[in] pointCloud  Pointer to input buffer
 * @param[in] obj         Handle to range image creator
 *
 * @return DW_SUCCESS<br>
 *         DW_INVALID_HANDLE     If neither of given handle is valid<br>
 *         DW_INVALID_ARGUMENT   If point cloud input is nullptr
 */
DW_API_PUBLIC
dwStatus dwPCRangeImageCreator_bindInput(dwPointCloud const* const pointCloud,
                                         dwPointCloudRangeImageCreatorHandle_t const obj);
/**
 * @brief Binds output range image to range image creator
 *
 * @param[in] image       Handle to output range image
 * @param[in]  obj         Handle to range image creator
 *
 * @return DW_SUCCESS<br>
 *         DW_INVALID_HANDLE     If neither of given handle is valid<br>
 *         DW_INVALID_ARGUMENT   If image is nullptr, image properties are not correct
 *                               or memory type of input/output is inconsistent.
 */
DW_API_PUBLIC
dwStatus dwPCRangeImageCreator_bindOutput(dwImageHandle_t const image,
                                          dwPointCloudRangeImageCreatorHandle_t const obj);
/**
 * @brief Binds output point cloud to range image creator
 *
 * @param[in] pointCloud  Pointer to output point cloud. If null provided, any currently bound point cloud will be unbound
 * @param[in]  obj         Handle to range image creator
 *
 * @return DW_SUCCESS<br>
 *         DW_INVALID_HANDLE     If given handle is not valid<br>
 *         DW_INVALID_ARGUMENT   If memory type or size of input/output is inconsistent.
 *
 * @note Output point cloud is optional. It may be used in couple with `DW_POINT_CLOUD_IMAGE_TYPE_DISTANCE` or
 *       `DW_POINT_CLOUD_IMAGE_TYPE_INTENSITY` image to organize points of input point cloud. Makes sense for
 *       unorganized input only.
 */
DW_API_PUBLIC
dwStatus dwPCRangeImageCreator_bindPointCloudOutput(dwPointCloud* const pointCloud,
                                                    dwPointCloudRangeImageCreatorHandle_t const obj);

/**
 * Organizes input point cloud and projects on the spherical coordinate to form a range image.
 * If bound point cloud is already organized, the data is simply copied into output range image.
 *
 * @param[in] obj Handle to range image creator
 *
 * @return DW_SUCCESS<br>
 *         DW_INVALID_HANDLE    If given handle is not valid<br>
 *         DW_CALL_NOT_ALLOWED  If no input/output buffer is bound
 */
DW_API_PUBLIC
dwStatus dwPCRangeImageCreator_process(dwPointCloudRangeImageCreatorHandle_t const obj);

/////////////////////////////////////////////////////////////////////////////////////////
// DEPRECATED API FUNCTIONS

/**
 * @brief Initializes range image creator
 *
 * @param[out] obj     Pointer to range image creator handle
 * @param[in]  params  Pointer to range image creator parameters
 * @param[in]  ctx     Handle to the context
 *
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT   If given parameter pointer is invalid<br>
 *         DW_INVALID_HANDLE     If given handle is valid<br>
 */
DW_API_PUBLIC
DW_DEPRECATED("dwPointCloudRangeImageCreator_initialize() is renamed / deprecated and will be removed in the next major release,"
              " use dwPCRangeImageCreator_initialize() instead")
// coverity[misra_c_2012_rule_5_1_violation] Deprecated API
dwStatus dwPointCloudRangeImageCreator_initialize(dwPointCloudRangeImageCreatorHandle_t* const obj,
                                                  dwPointCloudRangeImageCreatorParams const* const params,
                                                  dwContextHandle_t const ctx);

/**
 * @brief Resets range image creator
 *
 * @param[in] obj Handle to range image creator
 *
 * @return DW_SUCCESS<br>
 *         DW_INVALID_HANDLE     If given handle is valid<br>
 */
DW_API_PUBLIC
DW_DEPRECATED("dwPointCloudRangeImageCreator_reset() is renamed / deprecated and will be removed in the next major release,"
              " use dwPCRangeImageCreator_reset() instead")
// coverity[misra_c_2012_rule_5_1_violation] Deprecated API
dwStatus dwPointCloudRangeImageCreator_reset(dwPointCloudRangeImageCreatorHandle_t const obj);

/**
 * @brief Releases range image creator
 *
 * @param[in] obj Handle to range image creator
 *
 * @return DW_SUCCESS<br>
 *         DW_INVALID_HANDLE     If given handle is valid<br>
 */
DW_API_PUBLIC
DW_DEPRECATED("dwPointCloudRangeImageCreator_release() is renamed / deprecated and will be removed in the next major release,"
              " use dwPCRangeImageCreator_release() instead")
// coverity[misra_c_2012_rule_5_1_violation] Deprecated API
dwStatus dwPointCloudRangeImageCreator_release(dwPointCloudRangeImageCreatorHandle_t const obj);

/**
 * @brief Gets default range image creator parameters
 *
 * @param[out] params Pointer to range image creator parameters
 *
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT if given param pointer is invalid
 */
DW_API_PUBLIC
DW_DEPRECATED("dwPointCloudRangeImageCreator_getDefaultParams() is renamed / deprecated and will be removed in the next major release,"
              " use dwPCRangeImageCreator_getDefaultParams() instead")
// coverity[misra_c_2012_rule_5_1_violation] Deprecated API
dwStatus dwPointCloudRangeImageCreator_getDefaultParams(dwPointCloudRangeImageCreatorParams* const params);

/**
 * @brief Gets CUDA stream of range image creator
 *
 * @param[out] stream  Pointer to CUDA stream handle
 * @param[in]  obj     Handle to range image creator
 *
 * @return DW_SUCCESS<br>
 *         DW_INVALID_ARGUMENT if given stream pointer is invalid<br>
 *         DW_INVALID_HANDLE     If given handle is valid<br>
 */
DW_API_PUBLIC
DW_DEPRECATED("dwPointCloudRangeImageCreator_getCUDAStream() is renamed / deprecated and will be removed in the next major release,"
              " use dwPCRangeImageCreator_getCUDAStream() instead")
// coverity[misra_c_2012_rule_5_1_violation] Deprecated API
dwStatus dwPointCloudRangeImageCreator_getCUDAStream(cudaStream_t* const stream,
                                                     dwConstPointCloudRangeImageCreatorHandle_t const obj);

/**
 * @brief Sets CUDA stream of range image creator
 *
 * @param[in] stream  Handle to CUDA stream
 * @param[in] obj     Handle to range image creator
 *
 * @return DW_SUCCESS<br>
 *         DW_INVALID_HANDLE     If given handle is valid<br>
 */
DW_API_PUBLIC
DW_DEPRECATED("dwPointCloudRangeImageCreator_setCUDAStream() is renamed / deprecated and will be removed in the next major release,"
              " use dwPCRangeImageCreator_setCUDAStream() instead")
// coverity[misra_c_2012_rule_5_1_violation] Deprecated API
dwStatus dwPointCloudRangeImageCreator_setCUDAStream(cudaStream_t const stream,
                                                     dwPointCloudRangeImageCreatorHandle_t const obj);

/**
 * @brief Get properties of an image to bind as an output
 *
 * @param[out] imageProperties Pointer to image properties
 * @param[in] obj Handle to range image creator
 *
 * @return DW_SUCCESS<br>
 *         DW_INVALID_HANDLE     If given handle is valid<br>
 *         DW_INVALID_ARGUMENT   If imageProperties in null
 */
DW_API_PUBLIC
DW_DEPRECATED("dwPointCloudRangeImageCreator_getImageProperties() is renamed / deprecated and will be removed in the next major release,"
              " use dwPCRangeImageCreator_getImageProperties() instead")
// coverity[misra_c_2012_rule_5_1_violation] Deprecated API
dwStatus dwPointCloudRangeImageCreator_getImageProperties(dwImageProperties* const imageProperties,
                                                          dwConstPointCloudRangeImageCreatorHandle_t const obj);

/**
 * @brief Binds input point cloud to range image creator
 *
 * @param[in] pointCloud  Pointer to input buffer
 * @param[in] obj         Handle to range image creator
 *
 * @return DW_SUCCESS<br>
 *         DW_INVALID_HANDLE     If neither of given handle is valid<br>
 *         DW_INVALID_ARGUMENT   If point cloud input is nullptr
 */
DW_API_PUBLIC
DW_DEPRECATED("dwPointCloudRangeImageCreator_bindInput() is renamed / deprecated and will be removed in the next major release,"
              " use dwPCRangeImageCreator_bindInput() instead")
// coverity[misra_c_2012_rule_5_1_violation] Deprecated API
dwStatus dwPointCloudRangeImageCreator_bindInput(dwPointCloud const* const pointCloud,
                                                 dwPointCloudRangeImageCreatorHandle_t const obj);
/**
 * @brief Binds output range image to range image creator
 *
 * @param[in] image       Handle to output range image
 * @param[in]  obj         Handle to range image creator
 *
 * @return DW_SUCCESS<br>
 *         DW_INVALID_HANDLE     If neither of given handle is valid<br>
 *         DW_INVALID_ARGUMENT   If image is nullptr, image properties are not correct
 *                               or memory type of input/output is inconsistent.
 */
DW_API_PUBLIC
DW_DEPRECATED("dwPointCloudRangeImageCreator_bindOutput() is renamed / deprecated and will be removed in the next major release,"
              " use dwPCRangeImageCreator_bindOutput() instead")
// coverity[misra_c_2012_rule_5_1_violation] Deprecated API
dwStatus dwPointCloudRangeImageCreator_bindOutput(dwImageHandle_t const image,
                                                  dwPointCloudRangeImageCreatorHandle_t const obj);
/**
 * @brief Binds output point cloud to range image creator
 *
 * @param[in] pointCloud  Pointer to output point cloud. If null provided, any currently bound point cloud will be unbound
 * @param[in]  obj         Handle to range image creator
 *
 * @return DW_SUCCESS<br>
 *         DW_INVALID_HANDLE     If given handle is not valid<br>
 *         DW_INVALID_ARGUMENT   If memory type or size of input/output is inconsistent.
 *
 * @note Output point cloud is optional. It may be used in couple with `DW_POINT_CLOUD_IMAGE_TYPE_DISTANCE` or
 *       `DW_POINT_CLOUD_IMAGE_TYPE_INTENSITY` image to organize points of input point cloud. Makes sense for
 *       unorganized input only.
 */
DW_API_PUBLIC
DW_DEPRECATED("dwPointCloudRangeImageCreator_bindOutputPointCloud() is renamed / deprecated and will be removed in the next major release,"
              " use dwPCRangeImageCreator_bindPointCloudOutput() instead")
// coverity[misra_c_2012_rule_5_1_violation] Deprecated API
dwStatus dwPointCloudRangeImageCreator_bindOutputPointCloud(dwPointCloud* const pointCloud,
                                                            dwPointCloudRangeImageCreatorHandle_t const obj);

/**
 * Organizes input point cloud and projects on the spherical coordinate to form a range image.
 * If bound point cloud is already organized, the data is simply copied into output range image.
 *
 * @param[in] obj Handle to range image creator
 *
 * @return DW_SUCCESS<br>
 *         DW_INVALID_HANDLE    If given handle is not valid<br>
 *         DW_CALL_NOT_ALLOWED  If no input/output buffer is bound
 */
DW_API_PUBLIC
DW_DEPRECATED("dwPointCloudRangeImageCreator_process() is renamed / deprecated and will be removed in the next major release,"
              " use dwPCRangeImageCreator_process() instead")
// coverity[misra_c_2012_rule_5_1_violation] Deprecated API
dwStatus dwPointCloudRangeImageCreator_process(dwPointCloudRangeImageCreatorHandle_t const obj);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_POINTCLOUDPROCESSING_POINTCLOUDRANGEIMAGECREATOR_H_
