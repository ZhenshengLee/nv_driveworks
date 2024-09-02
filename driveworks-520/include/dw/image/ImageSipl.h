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
// SPDX-FileCopyrightText: Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Extra Image Functionality</b>
 *
 * @b Description: This file extra methods for the image interface
 *
 */

/**
 * @defgroup image_group_ext Image Extra Interface
 *
 * @brief Defines extra methods for image interface.
 *
 * @{
 */

#ifndef DW_IMAGE_IMAGE_SIPL_H_
#define DW_IMAGE_IMAGE_SIPL_H_

#include <dw/image/Image.h>

#include <NvSIPLClient.hpp>
#include <NvSIPLISPStat.hpp>
typedef nvsipl::NvSiplISPBadPixelStats NvMediaISPBadPixelStats;
typedef nvsipl::NvSiplISPBadPixelStatsData NvMediaISPBadPixelStatsData;
typedef nvsipl::NvSiplISPLocalAvgClipStats NvMediaISPLocalAvgClipStats;
typedef nvsipl::NvSiplISPLocalAvgClipStatsData NvMediaISPLocalAvgClipStatsData;
typedef nvsipl::NvSiplISPHistogramStatsData NvMediaISPHistogramStatsData;
typedef nvsipl::NvSiplISPHistogramStats NvMediaISPHistogramStats;

#ifdef __cplusplus
extern "C" {
#endif

/// Number of available ISP Units
#define DW_IMAGE_NUM_ISP_UNITS 2
#define DW_LUMINANCE_NUM_ISP_UNITS 1
#define DW_ISP_MAX_LAC_ROI_WINDOWS (NVSIPL_ISP_MAX_LAC_ROI_WINDOWS)
#define DW_ISP_MAX_LAC_ROI (NVSIPL_ISP_MAX_LAC_ROI)

/// SIPL meta information stored with each image
typedef struct
{
    // Only the first three params from NvMediaISPLocalAvgClipStatsData are needed for luminance per
    // https://nvbugs/3808315

    /* Holds number of windows horizontally in one region of interest.
     */
    uint32_t numWindowsH;
    /**
     * Holds number of windows vertically in one region of interest.
     */
    uint32_t numWindowsV;
    /**
     * Holds average pixel value for each color component in each window in
     * RGGB/RCCB/RCCC order.
     */
    float32_t average[DW_ISP_MAX_LAC_ROI_WINDOWS][DW_ISP_MAX_COLOR_COMPONENT];

} dwImageNvMediaLuminanceROIData;

typedef struct
{
    dwImageNvMediaLuminanceROIData data[DW_ISP_MAX_LAC_ROI];
} dwImageNvMediaLuminanceStatsData;

typedef struct
{
    /// Holds the number of the exposures
    uint32_t numExposures;

    /// Holds sensor exposure info including exposure times and gains
    DevBlkCDIExposure sensorExpInfo;

    /// Holds the sensor white balance info
    DevBlkCDIWhiteBalance sensorWBInfo;

    /// Holds the sensor illumination info
    DevBlkCDIIllumination illuminationInfo;

    /// Holds a flag indicating if the ISP histogram statistics are valid
    bool histValid[DW_IMAGE_NUM_ISP_UNITS];

    /// Holds the ISP histogram statistics
    NvMediaISPHistogramStatsData histogramStats[DW_IMAGE_NUM_ISP_UNITS];

    /// Holds the ISP histogram settings
    NvMediaISPHistogramStats histogramSettings[DW_IMAGE_NUM_ISP_UNITS];

    /// Holds a flag indicating if the ISP bad pixel statistics are valid
    bool badPixelStatsValid;

    /// Holds the ISP bad pixel statistics for the previous ISP output frame
    NvMediaISPBadPixelStatsData badPixelStats;

    /// Holds the ISP bad pixel settings for the previous ISP output frame
    NvMediaISPBadPixelStats badPixelSettings;

    /// Holds a flag indicating if the ISP Local Average and Clipped statistics are valid
    bool localAvgClipStatsValid[DW_LUMINANCE_NUM_ISP_UNITS];

    /// Holds the ISP Local Average and Clipped statistics for the previous ISP output frame
    //  This is primarily being used to store the luminance data as per https://nvbugs/3808315
    dwImageNvMediaLuminanceStatsData localAvgClipStats[DW_LUMINANCE_NUM_ISP_UNITS];

    /// Holds the ISP Local Average and Clipped settings for the previous ISP output frame
    NvMediaISPLocalAvgClipStats localAvgClipSettings[DW_IMAGE_NUM_ISP_UNITS];
} dwImageNvSIPLMetadata;

/**
 * Retrieves the SIPL metadata of a dwImageHandle_t
 *
 * @param[out] metadata A pointer to the SIPL metadata
 * @param[in] image A handle to the image
 *
 * @return DW_SUCCESS, <br>
 *         DW_INVALID_ARGUMENT if the given metadata pointer is null, <br>
 *         DW_INVALID_HANDLE if the given image handle is invalid,i.e null or of wrong type  <br>
 * @par API Group
 * - Init: Yes
 * - Runtime: Yes
 * - De-Init: Yes
 */
DW_API_PUBLIC
dwStatus dwImage_getNvSIPLMetadata(dwImageNvSIPLMetadata* metadata, dwConstImageHandle_t image);

/**
 * Sets the SIPL metadata of a dwImageHandle_t
 *
 * @param[out] metadata A pointer to the SIPL metadata
 * @param[in] image A handle to the image
 *
 * @return DW_SUCCESS, <br>
 *         DW_INVALID_ARGUMENT if the given metadata pointer is null, <br>
 *         DW_INVALID_HANDLE if the given image handle is invalid,i.e null or of wrong type  <br>
 */
DW_API_PUBLIC
dwStatus dwImage_setNvSIPLMetadata(dwImageNvSIPLMetadata const* const metadata, dwImageHandle_t const image);

#ifdef __cplusplus
}
#endif

/** @} */
#endif // DW_IMAGE_IMAGE_SIPL_H_