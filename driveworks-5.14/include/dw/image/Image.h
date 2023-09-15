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
// SPDX-FileCopyrightText: Copyright (c) 2016-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: Image Conversion and Streaming Functionality</b>
 *
 * @b Description: This file defines methods for image conversion.
 */

/**
 * @defgroup image_group Image Interface
 *
 * @brief Defines image abstractions, and streamer and format conversion APIs.
 *
 * @brief Unless explicitly specified, all errors returned by DW APIs are non recoverable and the user application should transition to fail safe mode.
 * In addition, any error code not described in this documentation should be consider as fatal and the user application should also transition to fail safe mode.
 *
 *
 * @{
 */

#ifndef DW_IMAGE_IMAGE_H_
#define DW_IMAGE_IMAGE_H_

#include <dw/core/base/Config.h>
#include <dw/core/base/Exports.h>
#include <dw/core/base/Types.h>
#include <dw/core/base/Status.h>
#include <dw/core/context/Context.h>

#if (defined(__cplusplus) && (defined(LINUX) || defined(VIBRANTE)))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif

#if (defined(__cplusplus) && (defined(LINUX) || defined(VIBRANTE)))
#pragma GCC diagnostic pop
#endif

#include <sys/time.h>
#ifdef PRE_NVSCI
#include <nvmedia_image.h>
#include <nvmedia_isp_stat.h>
#else
//#include <NvSIPLISPStat.hpp>
#endif //#ifdef PRE_NVSCI
#include <nvscibuf.h>

#ifdef DW_SDK_BUILD_PVA
#include "cupva_host_wrapper.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Enum representing a sync type. Currently it represents synchronization methods for NvSci based buffers with
 * HW engines like GPU or Drive platform VIC
 */
typedef enum dwSyncType {
    DW_SYNC_TYPE_WAITER,   // The sync will wait for a signaler
    DW_SYNC_TYPE_SIGNALER, // The sync will notify the waiters
} dwSyncType;

#define DW_MAX_IMAGE_PLANES 3
#define DW_ISP_MAX_COLOR_COMPONENT (4U)
/// should be the same as the DEVBLK_CDI_MAX_NUM_TEMPERATURES in NvSIPLCDICommon.h of NvSIPL
#define DW_MAX_NUM_TEMPERATURES 4
#define DW_IMAGE_NUM_SPLINE_COMPONENTS 18U

/// Specifies the image type.
typedef enum dwImageType {
    DW_IMAGE_CPU     = 0,
    DW_IMAGE_CUDA    = 2,
    DW_IMAGE_NVMEDIA = 3,
    /// This type is provided here for completeness only. It will not function on any image API as provided
    /// by this SDK. For DW_IMAGE_GL images to function properly APIs from dwgl library are required.
    DW_IMAGE_GL = 0xFF
} dwImageType;

typedef struct dwImageObject* dwImageHandle_t;
typedef struct dwImageObject const* dwConstImageHandle_t;

/// dwImage Allocation Attributes List
typedef struct dwImageAllocationAttrList* dwImageAllocationAttrListHandle_t;

/// Specifies a pool of images
typedef struct dwImagePool
{
    /// Pointer to image handles
    dwImageHandle_t* images;
    /// Number of images in the pool.
    size_t imageCount;
} dwImagePool;

/// Format of the image represented as DW_IMAGE_FORMAT_COLORSPACE(_PIXELTYPE)(_PIXELORDER)
/// @note The default memory allocation is via NvSciBuf but certain legacy formats are not compatible as listed below. These formats will continue to use API specific memory allocation based on dwImageType
typedef enum dwImageFormat {
    /// Normal formats
    DW_IMAGE_FORMAT_UNKNOWN = 0,

    // Interleaved formats
    DW_IMAGE_FORMAT_R_INT16 = 900,

    DW_IMAGE_FORMAT_R_UINT8 = 1000,
    DW_IMAGE_FORMAT_R_UINT16,
    DW_IMAGE_FORMAT_R_UINT32,
    DW_IMAGE_FORMAT_R_FLOAT16,
    /// Not backed by NvSci
    DW_IMAGE_FORMAT_R_FLOAT32,

    DW_IMAGE_FORMAT_RG_INT16,
    /// Not backed by NvSci
    DW_IMAGE_FORMAT_RG_UINT8,
    /// Not backed by NvSci
    DW_IMAGE_FORMAT_RG_FLOAT32,

    /// Not backed by NvSci
    DW_IMAGE_FORMAT_RGB_UINT8,
    /// Not backed by NvSci
    DW_IMAGE_FORMAT_RGB_UINT16,
    /// Not backed by NvSci
    DW_IMAGE_FORMAT_RGB_FLOAT16,
    /// Not backed by NvSci
    DW_IMAGE_FORMAT_RGB_FLOAT32,

    DW_IMAGE_FORMAT_RGBA_UINT8,
    DW_IMAGE_FORMAT_RGBA_UINT16,
    DW_IMAGE_FORMAT_RGBA_FLOAT16,
    /// Not backed by NvSci
    DW_IMAGE_FORMAT_RGBA_FLOAT32,

    DW_IMAGE_FORMAT_RGBX_FLOAT16 = 1200,

    // interleaved YUV444 format
    DW_IMAGE_FORMAT_VUYX_UINT8,
    DW_IMAGE_FORMAT_VUYX_UINT16,

    // Planar formats
    /// Not backed by NvSci
    DW_IMAGE_FORMAT_RGB_UINT8_PLANAR = 2000,
    DW_IMAGE_FORMAT_RGB_UINT16_PLANAR,
    /// Not backed by NvSci
    DW_IMAGE_FORMAT_RGB_FLOAT16_PLANAR,
    /// Not backed by NvSci
    DW_IMAGE_FORMAT_RGB_FLOAT32_PLANAR,

    /// Not backed by NvSci
    DW_IMAGE_FORMAT_RCB_FLOAT16_PLANAR,
    /// Not backed by NvSci
    DW_IMAGE_FORMAT_RCB_FLOAT32_PLANAR,

    /// Not backed by NvSci
    DW_IMAGE_FORMAT_RCC_FLOAT16_PLANAR,
    /// Not backed by NvSci
    DW_IMAGE_FORMAT_RCC_FLOAT32_PLANAR,

    /// YUV encoding formats from camera
    DW_IMAGE_FORMAT_YUV420_UINT8_PLANAR = 3000,
    DW_IMAGE_FORMAT_YUV420_UINT8_SEMIPLANAR,
    DW_IMAGE_FORMAT_YUV420_UINT16_SEMIPLANAR,
    DW_IMAGE_FORMAT_YUV422_UINT8_SEMIPLANAR = 3100,

    // planar YUV 444
    DW_IMAGE_FORMAT_YUV_UINT8_PLANAR,
    DW_IMAGE_FORMAT_YUV_UINT16_PLANAR,
    DW_IMAGE_FORMAT_YUV422_UINT8_PACKED = 3200,
    /// RAW
    /// for images directly from sensory
    DW_IMAGE_FORMAT_RAW_UINT16 = 4000,
    /// for debayered images
    DW_IMAGE_FORMAT_RAW_FLOAT16,

} dwImageFormat;

/// Specifies memory type layout.
typedef enum {
    /// the default memory layout for a given image type, can be either pitch or block
    DW_IMAGE_MEMORY_TYPE_DEFAULT = 0,
    /// pitch linear memory layout
    DW_IMAGE_MEMORY_TYPE_PITCH = 1,
    /// block memory layout
    DW_IMAGE_MEMORY_TYPE_BLOCK = 2,

} dwImageMemoryType;

/// \brief Container for data lines from the camera.
typedef struct dwImageDataLines
{
    /// Number of bytes for each line
    uint32_t bytesPerLine;

    /// this defines the number of rows before and after the image
    dwVector2ui embeddedDataSize;

    /// pointer to the beginning of top lines
    uint8_t* topLineData;
    /// pointer to the beginning of bottom lines
    uint8_t* bottomLineData;
} dwImageDataLines;

/// Flags defining the meta information available in an image
typedef enum {

    /// If an image was extracted from a camera, additional embedded data lines might be provided
    /// The data lines are stored before and after the actual image content in memory
    /// Note manually setting this flag will influence the image creation by allocating the specified number of datalines in dwImageDatalines
    DW_IMAGE_FLAGS_EMBEDDED_LINES = (1 << 2),

    /// Image contains valid sensor settings information, such as exposure, gain, whitebalance, etc.
    DW_IMAGE_FLAGS_SENSOR_SETTINGS = (1 << 3),

    /// Image contains valid frame sequence number
    DW_IMAGE_FLAGS_FRAME_SEQUENCE_NUMBER = (1 << 4),

    /// By default CUDA images are created in vidmem on DGPU, this flag forces CUDA image to sysmem
    /// Note manually setting this flag will influence the image allocation as per description
    DW_IMAGE_FLAGS_SYSMEM = (1 << 5),

    /// Image contains details of raw order descriptor. This is valid only if the image has format DW_IMAGE_FORMAT_RAW_X.
    /// Note manually setting this flag will influence the image creation, selecting the raw order described for a DW_IMAGE_FORMAT_RAW
    DW_IMAGE_FLAGS_HAS_RAW_ORDER_DESCRIPTOR = (1 << 6),

    /// Holds a flag to determine whether or not the control info is valid. If no ISP processing occurs this value is false.
    DW_IMAGE_FLAGS_CONTROLINFO = (1 << 7),

    /// Holds a flag to indicating if the luminance is calibrated
    DW_IMAGE_FLAGS_LUMINANCE_CALIBRATED = (1 << 8),

    /// Holds the total white balance gains, which includes both sensor channel and ISP gains
    DW_IMAGE_FLAGS_TOTAL_WHITE_BALANCE_GAIN = (1 << 9),

    /// Image contains valid global tone map block
    DW_IMAGE_FLAGS_GTM_SPLINE_INFO = (1 << 10),

    /// Image contains valid sensor temperature info
    DW_IMAGE_FLAGS_SENSOR_TEMPERATURE = (1 << 11),

    /// Image contains NvSci surface based attributes
    DW_IMAGE_FLAGS_NVSCI_SURF_ATTR = (1 << 12),

    /// Image maps pointer to CUPVA
    DW_IMAGE_FLAGS_MAPS_CUPVA = (1 << 13),
} dwImageMetaDataFlags;

typedef struct dwImageSplineControlPoint
{
    /// Holds X coordinate of the control point.
    float32_t x;
    /// Holds Y coordinate of the control point.
    float32_t y;
    /// Holds slope of the spline curve at the control point.
    float64_t slope;
} dwImageSplineControlPoint;

// This is the max number of exposures supported currently
#define DW_DEVBLK_CDI_MAX_EXPOSURES 4

typedef struct dwExposureDuration
{
    /// Specifies the  exposure duration (microsecond)
    /// Each row of the image is captured using DW_DEVBLK_CDI_MAX_EXPOSURES
    /// of HDR exposures which are constant per row within the image.
    float32_t data[DW_DEVBLK_CDI_MAX_EXPOSURES];
} dwExposureDuration;

typedef struct dwSensorTemperature
{
    /// Holds the number of active temperatures. Valid only if DW_IMAGE_FLAGS_SENSOR_TEMPERATURE is enabled.
    /// Value range: [1, DW_MAX_NUM_TEMPERATURES].
    /// DW_MAX_NUM_TEMPERATURES is same as DEVBLK_CDI_MAX_NUM_TEMPERATURES which is defined in NvSIPLCDICommon.h of NvSIPL.
    /// The value is equal to the number of the temperature sensors in the specific image sensor being used.
    uint8_t numTemperatures;
    /// Holds the values of active sensor temperatures in degrees Celsius. Valid only if DW_IMAGE_FLAGS_SENSOR_TEMPERATURE is enabled.
    /// Indexes are in the range [0, numTemperatures - 1]. These values are obtained from the mapped registers of image sensor by NvSIPL and
    /// the detailed implementation is depending on the camera module and camera driver.
    float32_t dataCelsius[DW_MAX_NUM_TEMPERATURES];
} dwSensorTemperature;

/// Defines the length(M) of a MxM luminance calibration matrix.
#define DW_LUMINANCE_CALIB_MATRIX_SIZE 4
/// Sensor statistics associated with the image
typedef struct dwImageSensorStatistics
{
    /// Specifies the  exposure duration (microsecond)
    dwExposureDuration exposureDurationUs;

    /// Specifies the  exposure time (microsecond)
    /// This variable is deprecated and will be removed in the next major release. Please use exposureDurationUs in this struct
    float32_t exposureTime;

    /// Specifies the  analog Gain
    float32_t analogGain;

    /// Specifies the  conversion Gain
    float32_t conversionGain;

    /// Specifies the digital Gain
    float32_t digitalGain;

    /// Specifies the  sensor white balance gains : R(0) G1(1) G2(2) B(3)
    float32_t wbGain[4];

    /// Holds power factor for isp statistics compression. Valid range: [0.5, 1.0]
    float32_t alpha;

    /**
     * (note: parameter to be deprecated starting 6.0.7.0)
     * Holds a luminance calibration factor ( K / f^2 ) for luminance calibrated sensors.
     *
     * Definition:
     *
     *    N = K * (t*S / (f^2)) * L
     *    N: pixel value (e.g. RGGB channel average value)
     *    K: calibration constant
     *    t: sensor exposure time in seconds (if number of exposures > 1, use long exposure time value)
     *    S: sensor gain (if number of exposures > 1, use long exposure gain value)
     *    f: aperture number (f-stop)
     *    L: Luminance
     *
     *    Let luminanceCalibrationFactor = K / (f^2) = N / (t*S*L)
     */
    float64_t luminanceCalibrationFactor DW_DEPRECATED("This structure variable is deprecated and will be removed in the next major release. Please use the new variable in luminanceCalibrationMatrix in this struct");

    /**
     * Holds the luminance calibration matrix for the sensor.
     * @li Supported values [1E-12, 1E12]
     */
    float64_t luminanceCalibrationMatrix[DW_LUMINANCE_CALIB_MATRIX_SIZE][DW_LUMINANCE_CALIB_MATRIX_SIZE];

    /// Holds the total white balance gains, which includes both sensor channel and ISP gains.
    float32_t autoAwbGain[DW_ISP_MAX_COLOR_COMPONENT];

    /// Holds the correlated color temperature.
    float32_t cct;

    /// Holds the scene brightness key.
    float32_t brightnessKey;

    /// Holds the midtone value of the raw image.
    float32_t rawImageMidTone;

    /// Holds the values of active sensor temperatures in degrees Celsius. Valid only if DW_IMAGE_FLAGS_SENSOR_TEMPERATURE is enabled.
    /// Indexes are in the range [0, numTemperatures - 1]. These values are obtained from the mapped registers of image sensor by NvSIPL and
    /// the detailed implemetation is depending on the camera module and camera driver.
    dwSensorTemperature sensorTemperature;
} dwImageSensorStatistics;

typedef struct dwImageRawProperties
{
    /// Specifies the  msb of the pixel data
    uint32_t msbPosition;

    /// Describes the raw order and used when explicitly creating a DW_IMAGE_NVMEDIA if DW_IMAGE_FLAGS_HAS_RAW_ORDER_DESCRIPTOR is enabled
    /// This is represented by the define integers of the form:
    /// 1 - NVM_SURF_ATTR_COMPONENT_ORDER_XXX in nvmedia_surface.h (PDK < 6.0.4.0)
    /// 2 - NVSIPL_PIXEL_ORDER_XXX in NvSIPLCapStructs.h (PDK >= 6.0.4.0)
    /// If the flag is not specified, this value is ignored and a default RGGB order will be chosen
    uint32_t rawFormatDescriptor;

    /// Describes the raw pixel depth, specified like the above format descriptor by
    /// 1 - NVM_SURF_ATTR_BITS_PER_COMPONENT_XX (PDK < 6.0.4.0)
    /// 2 - integer bit depth (PDK >= 6.0.4.0)
    uint32_t rawBitDataType;

    /// Describes if raw order alignment is left justified(msb alligned) or right justified (lsb allignment)
    /// and used when explicitly creating a DW_IMAGE_NVMEDIA if DW_IMAGE_FLAGS_HAS_RAW_ORDER_DESCRIPTOR is enabled
    /// If the flag is not specified, this value is false and a default msb alligned order will be chosen.
    bool rawFormatRJ;
} dwImageRawProperties;

/// \brief Image timestamps
typedef struct dwImageTimestamps
{
    /// Rolling shutter timestamp fields:
    /// startOfFrameTimestampUs and endOfFrameTimestampUs
    /// can be used to compute precise per row timestamps for rolling shutter
    /// correction for use in sensor fusion.
    /// Specifies the time, in microseconds, when the first row of the image was fully
    /// exposed, i.e., the time at the end of the first row’s exposure.
    dwTime_t sofTimestampUs;
    /// This time indicates the middle of the image frame exposure.
    /// Note that this is the recommended timestamp to use for sensor fusion
    /// without rolling shutter compensation, i.e., the whole image frame is
    /// treated as a single time event.
    /// For the most precise sensor fusion, rolling shutter compensation is necessary
    dwTime_t moeTimestampUs;
    /// Specifies the time, in microseconds, when the last row of the image was fully
    /// exposed, i.e., the time at the end of the last row’s exposure.
    dwTime_t eofTimestampUs;
    /// Sub exposure start timestamps in microseconds
    dwTime_t subExposureStartTimestampUs[DW_DEVBLK_CDI_MAX_EXPOSURES];
} dwImageTimestamps;

/// Additional meta information stored with each image
typedef struct dwImageMetaData
{
    /// combination of multiple flags 'dwImageMetaDataFlags' defining which of the meta fields are valid
    uint32_t flags;

    /// Specifies image timestamps
    dwImageTimestamps imageTimestamps;

    /// Specifies the  exposure time (microsecond)
    float32_t exposureTime
        DW_DEPRECATED("This structure variable is deprecated and will be removed in the next major release. Please use the new variable in sensorStatistics in this struct");

    /// Specifies the  analog Gain
    float32_t analogGain
        DW_DEPRECATED("This structure variable is deprecated and will be removed in the next major release. Please use the new variable in sensorStatistics in this struct");

    /// Specifies the  conversion Gain
    float32_t conversionGain
        DW_DEPRECATED("This structure variable is deprecated and will be removed in the next major release. Please use the new variable in sensorStatistics in this struct");

    /// Specifies the  digital Gain
    float32_t digitalGain
        DW_DEPRECATED("This structure variable is deprecated and will be removed in the next major release. Please use the new variable in sensorStatistics in this struct");

    /// Specifies the  sensor white balance gains : R(0) G1(1) G2(2) B(3)
    float32_t wbGain[4] DW_DEPRECATED("This structure variable is deprecated and will be removed in the next major release. Please use the new variable in sensorStatistics in this struct");

    /// Specifies the  msb of the pixel data
    uint32_t msbPosition
        DW_DEPRECATED("This structure variable is deprecated and will be removed in the next major release. Please use the new variable in rawProperties in this struct");

    /// Holds a frame sequence number, that is, a monotonically increasing frame counter.
    uint32_t frameSequenceNumber;

    /**
     * embedded data lines. This is where the meta data defined above are parsed from
     * A RAW image coming from a sensor often contains this information arranged as extra lines before and
     * after the image. The full image is stored in a buffer structured as
     * [TOP DATA][IMAGE][BOTTOM DATA]
     * (top * W) + (H * W) + (bott * W)
     */
    dwImageDataLines dataLines;

    // Holds properties unique to raw images
    dwImageRawProperties rawProperties;

    // All paramters from here are read-only

    /// Holds info on sensor statistics at the time of the image capture
    dwImageSensorStatistics sensorStatistics;

    /// Holds the global tonemap block, containing a set of spline control points
    dwImageSplineControlPoint gtmSplineControlPoint[DW_IMAGE_NUM_SPLINE_COMPONENTS];

    /// Allocation attributes used by internal drivers
    dwImageAllocationAttrListHandle_t allocAttrs;
} dwImageMetaData;

/// Defines the properties of the image.
typedef struct dwImageProperties
{
    /// Specifies the type of image.
    dwImageType type;
    /// Specifies the width of the image in pixels.
    uint32_t width;
    /// Specifies the height of the image in pixels.
    uint32_t height;
    /// Specifies the format of the image.
    dwImageFormat format;
    /// additional meta information stored with the image. Not all images might provide it
    dwImageMetaData meta;
    /// Memory layout type
    dwImageMemoryType memoryLayout;
} dwImageProperties;

/// Defines a CPU-based image.
typedef struct dwImageCPU
{
    /// Specifies the properites of the image.
    dwImageProperties prop;
    /// Specifies the pitch of the image in bytes.
    size_t pitch[DW_MAX_IMAGE_PLANES];
    /// Specifies the raw image data.
    uint8_t* data[DW_MAX_IMAGE_PLANES];
    /// Specifies the time  in microseconds from system time epoch, when the image content was updated
    /// (ie EOF GMSL acquisition, USB camera processed frame, copied over from converted image etc...)
    dwTime_t timestamp_us;
} dwImageCPU;

/// Defines a CUDA image.
typedef struct dwImageCUDA
{
    /// Defines the properties of the image.
    dwImageProperties prop;
    /// Defines the pitch of each plane in bytes.
    size_t pitch[DW_MAX_IMAGE_PLANES]; // pitch in bytes
    /// Holds the pointer to the image planes.
    void* dptr[DW_MAX_IMAGE_PLANES];
    /// Holds the CUDA image plane data.
    cudaArray_t array[DW_MAX_IMAGE_PLANES];
    /// Specifies the time  in microseconds from system time epoch, when the image content was updated
    /// (ie EOF GMSL acquisition, USB camera processed frame, copied over from converted image etc...)
    dwTime_t timestamp_us;
} dwImageCUDA;

/// Defines an NvMedia image.
typedef struct dwImageNvMedia
{
    /// Holds image properties.
    dwImageProperties prop;
#ifdef PRE_NVSCI
    /// Holds the pointer to the NvMedia image.
    NvMediaImage* img
        DW_DEPRECATED("NvMediaImage is deprecated and will be removed in the next major release.");
#endif
    /// Holds the pointer to the NvSciBufObj image
    NvSciBufObj imgBuf;
    /// Specifies the time  in microseconds from system time epoch, when the image content was updated
    /// (ie EOF GMSL acquisition, USB camera processed frame, copied over from converted image etc...)
    dwTime_t timestamp_us;
} dwImageNvMedia;

/**
 * Creates and allocates resources for a dwImageHandle_t based on the properties passed as input.
 *
 * @param[out] image A handle to the image
 * @param[in] properties The image properties.
 * @param[in] ctx The DriveWorks context.
 *
 * @return DW_SUCCESS if the image was created, <br>
 *         DW_INVALID_ARGUMENT if the given image types are invalid or the streamer pointer is null, <br>
 *         DW_INVALID_HANDLE if the given context handle is invalid, i.e null or of wrong type  <br>
 */
DW_API_PUBLIC
dwStatus dwImage_create(dwImageHandle_t* const image,
                        dwImageProperties properties,
                        dwContextHandle_t const ctx);

/**
 * Creates a dwImageHandle_t based on the properties passed and binds a memory buffer provided by the application. Valid only for types
 * of DW_IMAGE_CPU and DW_IMAGE_CUDA with DW_IMAGE_LAYOUT_PITCH (or DEFAULT) layout.
 *
 * @param[out] image A handle to the image
 * @param[in] properties The image properties.
 * @param[in] buffersIn An array of pointers to individual buffers.
 * @param[in] pitches An array of pitches for each buffer.
 * @param[in] bufferCount The number of buffers (maximum is DW_MAX_IMAGE_PLANES).
 * @param[in] ctx The DriveWorks context.
 *
 * @return DW_SUCCESS if the image was created, <br>
 *         DW_INVALID_ARGUMENT if the given image properties are invalid or the image pointer is null or the buffer count is not equal to plane count  <br>
 *         DW_INVALID_HANDLE if the given context handle is invalid, i.e null or of wrong type  <br>
 *         DW_NOT_SUPPORTED if given image type is not supported (expecting only DW_IMAGE_CPU an DW_IMAGE_CUDA)
 */
DW_API_PUBLIC
dwStatus dwImage_createAndBindBuffer(dwImageHandle_t* const image,
                                     dwImageProperties properties,
                                     void* const buffersIn[DW_MAX_IMAGE_PLANES],
                                     size_t const pitches[DW_MAX_IMAGE_PLANES], size_t const bufferCount,
                                     dwContextHandle_t const ctx);

/**
 * Creates a dwImageHandle_t based on the properties passed and binds a cudaArray_t to it. Valid only for types
 * of DW_IMAGE_CUDA with DW_IMAGE_LAYOUT_BLOCK layout
 *
 * @param[out] image A handle to the image
 * @param[in] properties The image properties.
 * @param[in] buffers An array of pointers to cudaArray_t.
 * @param[in] bufferCount The number of buffers (maximum is DW_MAX_IMAGE_PLANES).
 * @param[in] ctx The DriveWorks context.
 *
 * @return DW_SUCCESS if the image was created, <br>
 *         DW_INVALID_ARGUMENT if the given image types are invalid or the buffer is null, <br>
 *         DW_INVALID_HANDLE if the given context handle is invalid, i.e null or of wrong type  <br>
 */
DW_API_PUBLIC
dwStatus dwImage_createAndBindCUDAArray(dwImageHandle_t* const image,
                                        dwImageProperties properties,
                                        cudaArray_t const buffers[DW_MAX_IMAGE_PLANES], size_t const bufferCount,
                                        dwContextHandle_t const ctx);

/**
 * Destroys the image handle and frees any memory created by dwImage_create(). If the image was created with
 * createAndBindX, it will unbind the memory and delete the handle without freeing the buffers
 *
 * @param[in] image A handle to the image
 *
 * @return DW_SUCCESS if the image was destroyed, <br>
 *         DW_INVALID_HANDLE if the given image handle is invalid, i.e null or of wrong type  <br>
 */
DW_API_PUBLIC
dwStatus dwImage_destroy(dwImageHandle_t const image);

// specifiers

/**
 * Retrieves the properties of a dwImageHandle_t
 *
 * @param[out] properties A pointer to the properties
 * @param[in] image A handle to the image
 *
 * @return DW_SUCCESS, <br>
 *         DW_INVALID_HANDLE if the given image handle is invalid, i.e null or of wrong type <br>
 */
DW_API_PUBLIC
dwStatus dwImage_getProperties(dwImageProperties* const properties, dwConstImageHandle_t const image);

/**
 * Retrieves the timestamp of acquisition of a dwImageHandle_t
 *
 * @param[out] timestamp A pointer to the timestamp
 * @param[in] image A handle to the image
 *
 * @return DW_SUCCESS, <br>
 *         DW_INVALID_HANDLE if the given image handle is invalid,i.e null or of wrong type  <br>
 */
DW_API_PUBLIC
dwStatus dwImage_getTimestamp(dwTime_t* const timestamp, dwConstImageHandle_t const image);

/**
 * Sets the timestamp of a dwImageHandle_t
 *
 * @param[out] timestamp Timestamp to be set
 * @param[in] image A handle to the image
 *
 * @return DW_SUCCESS, <br>
 *         DW_INVALID_HANDLE if the given image handle is invalid,i.e null or of wrong type  <br>
 */
DW_API_PUBLIC
dwStatus dwImage_setTimestamp(dwTime_t const timestamp, dwImageHandle_t const image);

/**
 * Retrieves the metadata of a dwImageHandle_t
 *
 * @param[out] metaData A pointer to the metadata
 * @param[in] image A handle to the image
 *
 * @return DW_SUCCESS, <br>
 *         DW_INVALID_ARGUMENT if the given metadaa pointer is null, <br>
 *         DW_INVALID_HANDLE if the given image handle is invalid,i.e null or of wrong type  <br>
 */
DW_API_PUBLIC
dwStatus dwImage_getMetaData(dwImageMetaData* const metaData, dwConstImageHandle_t const image);

/**
 * Sets the metadata of a dwImageHandle_t
 *
 * @param[in] metaData A pointer to the metadata
 * @param[in] image A handle to the image
 *
 * @return DW_SUCCESS, <br>
 *         DW_INVALID_HANDLE if the given image handle is invalid, i.e null or of wrong type <br>
 */
DW_API_PUBLIC
dwStatus dwImage_setMetaData(dwImageMetaData const* const metaData, dwImageHandle_t const image);

/**
 * Retrieves the dwImageCPU of a dwImageHandle_t. The image must have been created as a DW_IMAGE_CPU
 * type.
 *
 * @param[out] imageCPU A pointer to the dwImageCPU pointer
 * @param[in] image A handle to the image
 *
 * @return DW_SUCCESS if the dwImageCPU is successfully retrieved, <br>
 *         DW_INVALID_ARGUMENT if the given image pointer is null, <br>
 *         DW_INVALID_HANDLE if the given image handle is invalid, i.e null or of wrong type  <br>
 */
DW_API_PUBLIC
dwStatus dwImage_getCPU(dwImageCPU** const imageCPU, dwImageHandle_t const image);

/**
 * Retrieves the dwImageCUDA of a dwImageHandle_t. The image must have been created as a DW_IMAGE_CUDA
 * type.
 *
 * @param[out] imageCUDA A pointer to the dwImageCUDA pointer
 * @param[in] image A handle to the image
 *
 * @return DW_SUCCESS if the dwImageCUDA is successfully retrieved, <br>
 *         DW_INVALID_ARGUMENT if the given image pointer or image handle is null, <br>
 */
DW_API_PUBLIC
dwStatus dwImage_getCUDA(dwImageCUDA** const imageCUDA, dwImageHandle_t const image);

/**
 * Retrieves the dwImageNvMedia of a dwImageHandle_t. The image must have been created as a DW_IMAGE_NVEMDIA
 * type.
 *
 * @param[out] imageNvMedia A pointer to the dwImageNvMedia pointer
 * @param[in] image A handle to the image
 *
 * @return DW_SUCCESS if the dwImageNvMedia is successfully retrieved, <br>
 *         DW_INVALID_ARGUMENT if the given image pointer is null, <br>
 *         DW_INVALID_HANDLE if the given image handle is invalid,null or of wrong type <br>
 */
DW_API_PUBLIC
dwStatus dwImage_getNvMedia(dwImageNvMedia** imageNvMedia, dwImageHandle_t image);

/**
 * Retrieves dwTrivialDataType associated with a specific format
 *
 * @param[out] type The datatype
 * @param[in] format The format
 *
 * @return DW_SUCCESS <br>
 *         DW_INVALID_ARGUMENT if the given pointer is null, <br>
 */
DW_API_PUBLIC
dwStatus dwImage_getPixelType(dwTrivialDataType* const type, dwImageFormat const format);

/**
 * Retrieves number of planes of the image format
 *
 * @param[out] planeCount The plane count
 * @param[in] format The format
 *
 * @return DW_SUCCESS <br>
 *         DW_INVALID_ARGUMENT if the given pointer is null, <br>
 */
DW_API_PUBLIC
dwStatus dwImage_getPlaneCount(size_t* const planeCount, dwImageFormat const format);

/**
 * Converts CUDA or NvMedia images by copying into an output image, following the properties in the output
 * image. The output image must have been memory allocated (see 'dwImage_create'). Note that both images must
 * be of the same type. If the properties and memory layout of both images are the same, an identical copy of
 * the input image onto the output image is performed. The sizes must match (for conversion with resize, see
 * dwImageTransformation under dw/imageprocessing/geometry). The conversion is performed, for both image types
 * using DW's GPU kernel implementations. For use of the VIC engine via NvMedia2D, refer to dwImageTransformation.
 *
 * @param[out] output A pointer to the output image.
 * @param[in] input A pointer to the input image.
 * @param[in] context The sdk context.
 *
 * @return DW_INVALID_ARGUMENT if the provided pointers are null, or the provided images cannot be used.<br>
 *         DW_BAD_CAST if the image is not a cuda/nvmedia image<br>
 *         DW_CUDA_ERROR if the CUDA conversion fails, it is possible to recover this situation by switching to DW_IMAGE_NVMEDIA as input images<br>
 *         or DW_SUCCESS otherwise.
 *
 */
DW_API_PUBLIC
dwStatus dwImage_copyConvert(dwImageHandle_t const output, dwConstImageHandle_t const input, dwContextHandle_t const context);

/**
 * Converts CUDA or NvMedia images by copying into an output image, following the properties in the output
 * image. The output image must have been memory allocated (see 'dwImage_create'). Note that both images must
 * be of the same type. If the properties and memory layout of both images are the same, an identical copy of
 * the input image onto the output image is performed. The sizes must match (for conversion with resize, see
 * dwImageTransformation under dw/imageprocessing/geometry). The conversion is performed, for both image types
 * using DW's GPU kernel implementations. For use of the VIC engine via NvMedia2D, refer to dwImageTransformation.
 *
 * @param[out] output A pointer to the output image.
 * @param[in] input A pointer to the input image.
 * @param[in] stream The CUDA stream for executing the conversion kernel. Note this is ignored in case of NvMedia
 * @param[in] context The sdk context.
 *
 * @return DW_INVALID_ARGUMENT if the provided pointers are null, <br>
 *         DW_BAD_CAST if the image is not a cuda/nvmedia image<br>
 *         DW_CUDA_ERROR if the CUDA conversion fails, <br>
 *         or DW_SUCCESS otherwise.
 *
 */
DW_API_PUBLIC
dwStatus dwImage_copyConvertAsync(dwImageHandle_t const output, dwConstImageHandle_t const input, cudaStream_t const stream, dwContextHandle_t const context);

/**
 * Returns a specific plane of a CUDA image as its own single-plane CUDA image.
 * No data is copied, only the pointers are duplicated.
 *
 * @param[out] planeImage 'dwImageCUDA' to be filled with a single plane from 'srcImage'
 * @param[in] srcImage Source image to extract plane from
 * @param[in] planeIdx Index of the plane to extract
 *
 * @return DW_INVALID_ARGUMENT - if planeImage or srcImage are NULL, or if planeIdx is invalid<br>
 *         DW_SUCCESS
 *
 **/
DW_API_PUBLIC dwStatus dwImageCUDA_getPlaneAsImage(dwImageCUDA* const planeImage,
                                                   dwImageCUDA const* const srcImage,
                                                   uint32_t const planeIdx);

/**
 * Returns a dwImageCUDA that is mapped to a region of interest in the data of the srcImg. NOTE: only
 * single plane images are supported.
 *
 * @param[out] dstImg A pointer to the dwImageCUDA containing the roi. NOTE: the properties will reflect the
 *                    size of the roi. (NOTE: this is not a copy)
 * @param[in] srcImg A pointer to the source dwImageCUDA.
 * @param[in] roi A dwRect specifying the coordinates of the region of interest.
 *
 * @return DW_SUCCESS, DW_INVALID_ARGUMENT
 *
 **/
DW_API_PUBLIC dwStatus dwImageCUDA_mapToROI(dwImageCUDA* const dstImg, dwImageCUDA const* const srcImg, dwRect const roi);

/**
 * Returns the expected data layout of an image given its properties.
 * The byte size for plane i's row is:
 *
 *     elementSize*planeChannelCount[i]*planeWidth[i]
 *
 * @param[out] elementSize Size in bytes of the pixel type
 * @param[out] planeCount Number of planes
 * @param[out] planeChannelCount  Number of channels within each plane
 * @param[out] planeSize Size of each plane
 * @param[in] prop A pointer to the image properties to compute data layout from.
 *
 * @return DW_INVALID_ARGUMENT - if channel count or plane size or properties are invalid or NULL<br>
 *         DW_SUCCESS
 */

DW_API_PUBLIC dwStatus dwImage_getDataLayout(size_t* const elementSize,
                                             size_t* const planeCount,
                                             uint32_t planeChannelCount[DW_MAX_IMAGE_PLANES],
                                             dwVector2ui planeSize[DW_MAX_IMAGE_PLANES],
                                             dwImageProperties const* const prop);

#ifdef DW_SDK_BUILD_PVA
/**
 * Returns the pointer mapped to cupva, available only if DW_IMAGE_FLAGS_MAPS_CUPVA  flag is passed as part of the metadata and the underlying image is NvSci compatible
 *
 * @param[out] ptr pointer to cpva mapping
 * @param[in] image handle to image
 *
 * @return DW_INVALID_ARGUMENT - if channel count or plane size or properties are invalid or NULL<br>
 *         DW_FAILURE - if current setup returns unexpected errors <br>
 *         DW_SUCCESS
 */

DW_API_PUBLIC dwStatus dwImage_getCUPVA(void** ptr, dwImageHandle_t image);
#endif

#ifdef __cplusplus
}
#endif

/** @} */
#endif // DW_IMAGE_IMAGE_H_
