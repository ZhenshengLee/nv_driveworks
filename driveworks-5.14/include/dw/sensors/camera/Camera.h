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
 * <b>NVIDIA DriveWorks API: Cameras</b>
 *
 * @b Description: This file defines camera methods.
 */

/**
 * @defgroup camera_group Camera Sensor
 * @ingroup sensors_group
 *
 * @brief Defines the camera module, which provides access to a virtual camera reading
 * reading the data from a video file or a real camera connected to different sockets.
 *
 * @brief Unless explicitly specified, all errors returned by DW APIs are non recoverable and the user application should transition to fail safe mode.
 * In addition, any error code not described in this documentation should be consider as fatal and the user application should also transition to fail safe mode.
 *
 * @{
 */

#ifndef DW_SENSORS_CAMERA_CAMERA_H_
#define DW_SENSORS_CAMERA_CAMERA_H_

#include <dw/core/base/Config.h>
#include <dw/core/base/Types.h>
#include <dw/sensors/Sensors.h>
#include <dw/image/Image.h>

#include <nvscisync.h>
#include <nvscibuf.h>

// Forward declares from NvMedia
typedef void NvMediaIPPManager;

#ifdef __cplusplus
extern "C" {
#endif

/// Number of available ISP Units
#define DW_CAMERA_NUM_ISP_UNITS 2U

/// \brief Handle to captured frame.
typedef struct dwCameraFrame* dwCameraFrameHandle_t;

/// \brief Output types supported by the camera.
/// DW_CAMERA_OUTPUT_NATIVE_* types return the image directly from the layer underneath as it is represented
/// in system memory, while for non native output types images are converted and streamed through ad hoc streamers.
typedef enum dwCameraOutputType {
    /// processed images (usually be YUV420 planar or RGB planar)
    DW_CAMERA_OUTPUT_NATIVE_PROCESSED = 1 << 0,
    /// raw image
    DW_CAMERA_OUTPUT_NATIVE_RAW = 1 << 1,
    /// for processed images<br>
    /// RGBA image supported in all processed use cases
    DW_CAMERA_OUTPUT_CUDA_RGBA_UINT8 = 1 << 2,
    /// simple yuv420 output, supported in most use cases (see doc)
    DW_CAMERA_OUTPUT_CUDA_YUV420_UINT8_PLANAR = 1 << 3,
    /// for raw images
    DW_CAMERA_OUTPUT_CUDA_RAW_UINT16 = 1 << 4,
    /// other YUV processed outputs (see devguide)
    DW_CAMERA_OUTPUT_CUDA_YUV420_UINT16_SEMIPLANAR = 1 << 5,
    /// other YUV processed outputs (see devguide)
    DW_CAMERA_OUTPUT_CUDA_YUV444_UINT8_PLANAR = 1 << 6,
    /// other YUV processed outputs (see devguide)
    DW_CAMERA_OUTPUT_CUDA_YUV444_UINT16_PLANAR = 1 << 7,
    /// simple yuv420 semiplanar output, supported in most use cases (see doc)
    DW_CAMERA_OUTPUT_CUDA_YUV420_UINT8_SEMIPLANAR = 1 << 8,
    /// processed images from ISP1 output (usually be YUV420 planar or RGB planar)
    DW_CAMERA_OUTPUT_NATIVE_PROCESSED1 = 1 << 9,
    /// processed images from ISP2 output (RGB-FP16)
    DW_CAMERA_OUTPUT_NATIVE_PROCESSED2 = 1 << 10,
} dwCameraOutputType;

/// \brief Raw encoding formats pixel-order.
typedef enum dwCameraRawFormat {
    /// Format unsopported.
    DW_CAMERA_RAW_FORMAT_UNKNOWN = 0,

    /// Format:RGGB.
    DW_CAMERA_RAW_FORMAT_RGGB = 7000,
    /// Format:BGGR.
    DW_CAMERA_RAW_FORMAT_BGGR = 7100,
    /// Format:GRBG.
    DW_CAMERA_RAW_FORMAT_GRBG = 7200,
    /// Format:GBRG.
    DW_CAMERA_RAW_FORMAT_GBRG = 7300,

    /// Format:RCCB.
    DW_CAMERA_RAW_FORMAT_RCCB = 8000,
    /// Format:BCCR.
    DW_CAMERA_RAW_FORMAT_BCCR = 8100,
    /// Format:CRBC.
    DW_CAMERA_RAW_FORMAT_CRBC = 8200,
    /// Format:CBRC.
    DW_CAMERA_RAW_FORMAT_CBRC = 8300,

    /// Format:RCCC.
    DW_CAMERA_RAW_FORMAT_RCCC = 9000,
    /// Format:CRCC.
    DW_CAMERA_RAW_FORMAT_CRCC = 9100,
    /// Format:CCRC.
    DW_CAMERA_RAW_FORMAT_CCRC = 9200,
    /// Format:CCCR.
    DW_CAMERA_RAW_FORMAT_CCCR = 9300,

    /// Format:CCCC.
    DW_CAMERA_RAW_FORMAT_CCCC = 10000,

    /// Format:TOF.
    DW_CAMERA_RAW_FORMAT_TOF = 20000
} dwCameraRawFormat;

/// \brief Enum of available camera sensors
typedef enum dwCameraType {
    DW_CAMERA_GENERIC      = 0,  /*!< Generic video source, e.g. video file with raw or processed data */
    DW_CAMERA_GMSL_AR0231  = 3,  /*!< GMSL AR0231 camera (rev-7) */
    DW_CAMERA_USB_GENERIC  = 4,  /*!< Generic USB camera */
    DW_CAMERA_GMSL_AR0144  = 5,  /*!< GMSL AR0144 camera */
    DW_CAMERA_GMSL_AR0138  = 6,  /*!< GMSL AR0138 camera */
    DW_CAMERA_GMSL_AR0220  = 7,  /*!< GMSL AR0220 camera */
    DW_CAMERA_GMSL_AR0820  = 8,  /*!< GMSL AR0820 camera */
    DW_CAMERA_GMSL_MN34906 = 9,  /*!< GMSL MN34906 camera */
    DW_CAMERA_GMSL_OV2311  = 10, /*!< GMSL OV2311 camera */
    DW_CAMERA_GMSL_IMX390  = 11, /*!< GMSL IMX390 camera */
    DW_CAMERA_USB_KINECT   = 12, /*!< USB Kinect camera */
    DW_CAMERA_GMSL_CUSTOM  = 99  /*!< GMSL custom camera defined using NvSIPL*/
} dwCameraType;

/// \brief Enum of exposure control types
typedef enum dwCameraExposureControl {
    /// No exposure control
    DW_CAMERA_EXPOSURE_NONE,
    /// Unknown exposure control
    DW_CAMERA_EXPOSURE_UNKNOWN,
    /// Default exposure control
    DW_CAMERA_EXPOSURE_AE,
    /// Exposure control with bracketed auto exposure
    DW_CAMERA_EXPOSURE_BAE,
    /// Exposure control using user's custom callback
    DW_CAMERA_EXPOSURE_CUSTOM
} dwCameraExposureControl;

/// \brief Enum of available FOV in degrees for camera lenses
typedef enum dwCameraFOV {
    /// FOV for camera lenses is not supported.
    DW_CAMERA_FOV_UNKNOWN = 0,
    /// FOV for camera lenses is 30 degrees.
    DW_CAMERA_FOV_30 = 30,
    /// FOV for camera lenses is 48 degrees.
    DW_CAMERA_FOV_48 = 48,
    /// FOV for camera lenses is 50 degrees.
    DW_CAMERA_FOV_50 = 50,
    /// FOV for camera lenses is 55 degrees.
    DW_CAMERA_FOV_55 = 55,
    /// FOV for camera lenses is 60 degrees.
    DW_CAMERA_FOV_60 = 60,
    /// FOV for camera lenses is 65 degrees.
    DW_CAMERA_FOV_65 = 65,
    /// FOV for camera lenses is 90 degrees.
    DW_CAMERA_FOV_90 = 90,
    /// FOV for camera lenses is 100 degrees.
    DW_CAMERA_FOV_100 = 100,
    /// FOV for camera lenses is 110 degrees.
    DW_CAMERA_FOV_110 = 110,
    /// FOV for camera lenses is 120 degrees.
    DW_CAMERA_FOV_120 = 120,
    /// FOV for camera lenses is 185 degrees.
    DW_CAMERA_FOV_185 = 185
} dwCameraFOV;

/// \brief Enum of available SIPL interface provider types
typedef enum dwCameraSIPLInterfaceProviderType {
    /// Provider type: Deserializer.
    DW_SIPL_CAMERA_INTERFACE_PROVIDER_TYPE_DESERIALIZER,
    /// Provider type: Module.
    DW_SIPL_CAMERA_INTERFACE_PROVIDER_TYPE_MODULE,
    /// Provider type: Count.
    DW_SIPL_CAMERA_INTERFACE_PROVIDER_TYPE_COUNT
} dwSIPLCameraInterfaceProviderType;

/// \brief ISP types supported by the camera.
typedef enum dwCameraISPType {
    /// Supports YUV420 planar or RGB planar
    DW_CAMERA_ISP0 = 0,
    /// Similar to ISP0
    DW_CAMERA_ISP1,
    /// Supports RGB FP16
    DW_CAMERA_ISP2,
    DW_CAMERA_MAX_ISP_COUNT,
    // Only to be used for raw
    DW_CAMERA_ISP_UNKNOWN,
} dwCameraISPType;

/// \brief Camera Properties
typedef struct dwCameraProperties
{
    dwCameraType cameraType;          /*!< Type of the camera */
    dwCameraRawFormat rawFormat;      /*!< Raw bayer pattern*/
    dwCameraExposureControl exposure; /*!< Exposure control*/
    float32_t framerate;              /*!< Framerate in Hz */
    dwCameraFOV fov;                  /*!< FOV of the lens */
    dwVector2ui resolution;           /*!< Physical resolution of the camera sensor */
    int32_t outputTypes;              /*!< Output types referring list of available 'dwCameraOutputType' */
    uint32_t siblings;                /*!< Number of sibling frames */
    uint32_t revision;                /*!< Revision of the camera (0 if info is not available) */
    uint32_t imageBitDepth;           /*!< Bit depth of image */
    bool isSimulated;                 /*!< Camera is being simulated */
} dwCameraProperties;

/// \brief Enum of available SIPL event notification type (Copy from nvsipl::NvSIPLPipelineNotifier::NotificationType).
typedef enum dwCameraSIPLNotificationData {
    /// Pipeline event, indicates ICP processing is finished.
    DW_NOTIF_INFO_ICP_PROCESSING_DONE = 0,
    /// Pipeline event, indicates ISP processing is finished.
    DW_NOTIF_INFO_ISP_PROCESSING_DONE = 1,
    /// Pipeline event, indicates auto control processing is finished.
    DW_NOTIF_INFO_ACP_PROCESSING_DONE = 2,
    /// Pipeline event, indicates CDI processing is finished.
    DW_NOTIF_INFO_CDI_PROCESSING_DONE = 3,
    /// Pipeline event, indicates image authentication success.
    DW_NOTIF_INFO_ICP_AUTH_SUCCESS = 4,
    /// Pipeline event, indicates pipeline was forced to drop a frame due to a slow consumer or system issues.
    DW_NOTIF_WARN_ICP_FRAME_DROP = 100,
    /// Pipeline event, indicates a discontinuity was detected in parsed embedded data frame sequence number.
    DW_NOTIF_WARN_ICP_FRAME_DISCONTINUITY = 101,
    /// Pipeline event, indicates occurrence of timeout while capturing.
    DW_NOTIF_WARN_ICP_CAPTURE_TIMEOUT = 102,
    /// Pipeline event, indicates ICP bad input stream.
    DW_NOTIF_ERROR_ICP_BAD_INPUT_STREAM = 200,
    /// Pipeline event, indicates ICP capture failure.
    DW_NOTIF_ERROR_ICP_CAPTURE_FAILURE = 201,
    /// Pipeline event, indicates embedded data parsing failure.
    DW_NOTIF_ERROR_ICP_EMB_DATA_PARSE_FAILURE = 202,
    /// Pipeline event, indicates ISP processing failure.
    DW_NOTIF_ERROR_ISP_PROCESSING_FAILURE = 203,
    /// Pipeline event, indicates auto control processing failure.
    DW_NOTIF_ERROR_ACP_PROCESSING_FAILURE = 204,
    /// Pipeline event, indicates CDI set sensor control failure.
    DW_NOTIF_ERROR_CDI_SET_SENSOR_CTRL_FAILURE = 205,
    /// Device block event, indicates a deserializer link error. Deprecated in the future
    DW_NOTIF_ERROR_DESER_LINK_FAILURE = 206,
    /// Device block event, indicates a deserializer failure.
    DW_NOTIF_ERROR_DESERIALIZER_FAILURE = 207,
    /// Device block event, indicates a serializer failure.
    DW_NOTIF_ERROR_SERIALIZER_FAILURE = 208,
    /// Device block event, indicates a sensor failure.
    DW_NOTIF_ERROR_SENSOR_FAILURE = 209,
    /// Indicates image authentication failure.
    DW_NOTIF_ERROR_ICP_AUTH_FAILURE = 211,
    /// Pipeline and device block event, indicates an unexpected internal failure.
    DW_NOTIF_ERROR_INTERNAL_FAILURE = 300,
} dwCameraSIPLNotificationData;

/// \brief defines camera events exposed by dwCamera
typedef enum dwCameraEvent {
    /// Pipeline event, indicates a discontinuity was detected in parsed embedded data frame sequence number.
    DW_CAMERA_EVENT_WARN_ICP_FRAME_DISCONTINUITY = 4,
    /// Pipeline event, indicates occurrence of timeout while capturing.
    DW_CAMERA_EVENT_WARN_ICP_CAPTURE_TIMEOUT,
    /// Pipeline event, indicates ICP bad input stream.
    DW_CAMERA_EVENT_ERROR_ICP_BAD_INPUT_STREAM,
    /// Pipeline event, indicates ICP capture failure.
    DW_CAMERA_EVENT_ERROR_ICP_CAPTURE_FAILURE,
    /// Pipeline event, indicates embedded data parsing failure.
    DW_CAMERA_EVENT_ERROR_ICP_EMB_DATA_PARSE_FAILURE,
    /// Pipeline event, indicates ISP processing failure.
    DW_CAMERA_EVENT_ERROR_ISP_PROCESSING_FAILURE,
    /// Pipeline event, indicates auto control processing failure.
    DW_CAMERA_EVENT_ERROR_ACP_PROCESSING_FAILURE,
    /// Pipeline event, indicates CDI set sensor control failure.
    DW_CAMERA_EVENT_ERROR_CDI_SET_SENSOR_CTRL_FAILURE,
    /// Device block event, indicates a deserializer link error. Deprecated in the future
    DW_CAMERA_EVENT_ERROR_DESER_LINK_FAILURE,
    /// Device block event, indicates a deserializer failure.
    DW_CAMERA_EVENT_ERROR_DESERIALIZER_FAILURE,
    /// Device block event, indicates a serializer failure.
    DW_CAMERA_EVENT_ERROR_SERIALIZER_FAILURE,
    /// Device block event, indicates a sensor failure.
    DW_CAMERA_EVENT_ERROR_SENSOR_FAILURE,
    /// Pipeline and device block event, indicates an unexpected internal failure.
    DW_CAMERA_EVENT_ERROR_INTERNAL_FAILURE,
    /// SAL event to signal frame overrun
    DW_CAMERA_EVENT_ERROR_FRAME_OVERRUN,
    /// SAL event to signal frame sequence counter error
    DW_CAMERA_EVENT_ERROR_FRAME_COUNTER,
} dwCameraEvent;

/// \brief maximal error id reported by the camera module via module health service
#define DW_CAMERA_ERROR_ID_MAX DW_NOTIF_ERROR_INTERNAL_FAILURE

/// \brief Indicates the maximum number of gpio indices.
#define DW_CAMERA_MAX_DEVICE_GPIOS 8U

/// \brief NotificationData from SIPL
typedef struct dwCameraNotificationData
{
    /// Holds the notification event type.
    dwCameraSIPLNotificationData eNotifyType;
    /// Holds the ID of each camera sensor
    uint32_t uIndex;
    /// Holds the device block link mask.
    uint8_t uLinkMask;
    /// Holds a sequence number of a captured frame.
    uint64_t frameSeqNumber;
    /// Holds the TSC timestamp of the frame capture.
    uint64_t frameCaptureTSC;
    /// Holds the GPIO indices.
    uint32_t gpioIdxs[DW_CAMERA_MAX_DEVICE_GPIOS];
    /// Holds the number of GPIO indices in the array.
    uint32_t numGpioIdxs;
} dwCameraNotificationData;

/// \brief Struct of the detailed error info from SIPL
typedef struct dwCameraSIPLEErrorDetails
{
    /// Pointer to buffer which is filled by driver with error information.
    /// Note: DO NOT delete/free this pointer. This is managed by DW
    uint8_t const* errorBuffer;
    /// Holds size of error written to the buffer, filled by driver.
    size_t sizeWritten;
} dwCameraSIPLEErrorDetails;

/// \brief Indicates the maximum number of camera modules per device block.
#define DW_CAMERA_MAX_CAMERAMODULES_PER_BLOCK 4U

/// \brief Notification Data from SIPL
typedef struct dwCameraSIPLNotification
{
    /// NotificationData from SIPL, pipeline & device block event
    dwCameraNotificationData data;

    /// Error info for deserializer, valid only for device block event
    dwCameraSIPLEErrorDetails deserializerErrorInfo;
    /// Set to true if remote (serializer) error detected, valid only for device block event
    bool isRemoteError;
    /// Store link mask for link error state, valid only for device block event
    /// (1 in index position indicates error, all 0 means no link error detected).
    uint8_t linkErrorMask;

    /// Number of the camera modules, valid only for device block event
    uint32_t numCameraModules;
    /// Error info for serializer, valid only for device block event
    dwCameraSIPLEErrorDetails serializerErrorInfoList[DW_CAMERA_MAX_CAMERAMODULES_PER_BLOCK];
    /// Error info for sensor, valid only for device block event
    dwCameraSIPLEErrorDetails sensorErrorInfoList[DW_CAMERA_MAX_CAMERAMODULES_PER_BLOCK];
} dwCameraSIPLNotification;

/// \brief Function type of the camera error event handling
typedef void (*dwCameraCallback)(dwCameraSIPLNotification* notification, dwSensorHandle_t sensor);

/// \brief Defines Ellipse Properties for Override Histogram Statistics
typedef struct dwCameraISPEllipse
{
    /// Holds center of the ellipse.
    dwVector2f center;
    /// Holds horizontal axis of the ellipse.
    uint32_t horizontalAxis;
    /// Holds vertical axis of the ellipse.
    uint32_t verticalAxis;
    ///  Holds angle of the ellipse horizontal
    float32_t angle;
} dwCameraISPEllipse;

typedef struct
{
    /// Holds width of the window in pixels.
    uint32_t width;
    /// Holds height of the window in pixels.
    uint32_t height;
    /// Holds number of windows horizontally.
    uint32_t numWindowsH;
    /// Holds number of windows vertically.
    uint32_t numWindowsV;
    /// Holds the distance between the left edge of one window and a horizontally adjacent window.
    uint32_t horizontalInterval;
    /// Holds the distance between the top edge of one window and a vertically adjacent window.
    uint32_t verticalInterval;
    /// Holds the position of the top left pixel in the top left window.
    dwVector2i startOffset;
} dwCameraISPStatisticsWindows;

typedef struct
{
    /**
     * Holds a Boolean to enable the bad pixel statistics block.
     * \note Bad Pixel Correction must also be enabled to get bad pixel
     *  statistics.
     */
    bool enable;
    /**
     * Holds rectangular mask for excluding pixel outside a specified area.
     *
     * Coordinates of the image's top left and bottom right points are (0, 0)
     * and(width, height), respectively. Either set the rectangle's dimensions
     ( to 0 or set the rectangle to include the full image with no rectangular
     * mask.
     *
     * Supported values: Rectangle must be within the input image and must
     *  be a valid rectangle ((right > left) && (bottom > top)). The minimum
     *  supported rectangular mask size is 4x4.
     * Constraints: All left, top, bottom, and right coordinates must be even.
     */
    dwRect rectangularMask;
} dwCameraISPBadPixelStats;

/// \brief  SIPL ISP Histogram Statistics Override Params
typedef struct dwCameraISPHistogramStatsOverride
{
    /// Holds a Boolean to enable histogram statistics Control block.
    bool enable;
    /**
     * Holds offset to be applied to input data prior to bin mapping.
     * Supported values: [-2.0, 2.0]
     */
    float32_t offset;
    /**
     * Holds bin index specifying different zones in the histogram. Each zone
     * can have a different number of bins.
     * Supported values: [1, 255]
     */
    uint8_t knees[8];
    /**
     * Holds range of the pixel values to be considered for each
     * zone. The whole pixel range is divided into NVSIPL_ISP_HIST_KNEE_POINTS
     * zones.
     * Supported values: [0, 21]
     */
    uint8_t ranges[8];
    /**
     * Holds a rectangular mask for excluding pixels outside a specified area.
     *
     * The coordinates of image top left and bottom right points are (0, 0) and
     * (width, height), respectively.
     * A rectangle ((right &gt; left) && (bottom &gt; top)), or must be
     * ((0,0), (0,0)) to disable the rectangular mask.
     *
     * The rectangle settings(x0, y0, x1, y1) must follow the constraints listed below:
     * - (x0 >= 0) and (y0 >= 0)
     * - x0 and x1 should be even
     * - (x1 <= image width) and (y1 <= image height)
     * - rectangle width(x1 - x0) >= 2 and height(y1 - y0) >= 2
     */
    dwRect rectangularMask;
    /// Holds a Boolean to enable an elliptical mask for excluding pixels outside a specified area.
    bool ellipticalMaskEnable;
    /// Holds an elliptical mask for excluding pixels outside a specified area.
    dwCameraISPEllipse ellipticalMask;
    /// boolean flag to disable lens shading compensation for histogram statistics block
    bool disableLensShadingCorrection;
} dwCameraISPHistogramStatsOverride;

typedef struct
{
    /// Holds a Boolean to enable the local average and clip statistics block.
    bool enable;
    /**
     * Holds minimum value of pixels in RGGB/RCCB/RCCC order.
     * Supported values: [0.0, 1.0]
     */
    float32_t min[DW_ISP_MAX_COLOR_COMPONENT];
    /**
     * Holds maximum value of pixels in RGGB/RCCB/RCCC order.
     * Supported values: [0.0, 1.0], max >= min
     */
    float32_t max[DW_ISP_MAX_COLOR_COMPONENT];
    ///  Holds a Boolean to enable an individual region of interest.
    bool roiEnable[DW_ISP_MAX_COLOR_COMPONENT];
    /**
     * Holds local average and clip windows for each region of interest.
     * Supported values for width of the window: [2, 256] and must be an even number
     * Supported values for height of the window: [2, 256]
     * Supported values for number of the windows horizontally: [1, 32]
     * Supported values for number of the windows vertically: [1, 32]
     * Supported values for horizontal interval between windows: [max(4, window width), image ROI width]
     *  and must be an even number
     * Supported values for vertical interval between windows: [max(2, window height), image ROI height]
     * Supported values for X coordinate of start offset: [0, image ROI width-3] and must be an even number
     * Supported values for Y coordinate of start offset: [0, image ROI height-3]
     * startOffset.x + horizontalInterval * (numWindowH - 1) + winWidth <= image ROI width
     * startOffset.y + veritcallInterval * (numWindowV - 1) + winHeight <= image ROI height
     */
    dwCameraISPStatisticsWindows windows[DW_ISP_MAX_COLOR_COMPONENT];
    /**
     * Holds a Boolean to enable an elliptical mask for excluding pixels
     * outside a specified area for each region of interest.
     */
    bool ellipticalMaskEnable[DW_ISP_MAX_COLOR_COMPONENT];
    /**
     * Holds an elliptical mask for excluding pixels outside specified area.
     *
     * Coordinates of the image's top left and bottom right points are (0, 0)
     *  and (width, height), respectively.
     *
     * Supported values for X coordinate of the center: [0, input width]
     * Supported values for Y coordinate of the center: [0, input height]
     * Supported values for horizontal axis: [16, 2 x input width]
     * Supported values for vertical axis: [16, 2 x input height]
     * @note Supported values for angle: [0.0, 360.0]
     */
    dwCameraISPEllipse ellipticalMask;
} dwCameraISPLocalAvgClipStats;

/**
 * @brief ISP Override Statistics Settings
 * @note: ISP histStats[0] and lacStats[0] statistics are consumed by internal
 * algorithms to generate new sensor and ISP settings. Incorrect usage or
 * disabling these statistics blocks would result in failure or
 * image quality degradation. Refer to SIPL documentation for more details.
 **/
typedef struct dwCameraIspStatsOverrideSetting
{
    /// boolean flag to enable histogram statistics settings override
    bool enableHistStatsOverride[DW_CAMERA_NUM_ISP_UNITS];
    ///  Structure containing override settings for histogram statistics block
    dwCameraISPHistogramStatsOverride histStats[DW_CAMERA_NUM_ISP_UNITS];
    /// boolean flag to enable local average clip statistics settings override
    bool enableLacStatsOverride[DW_CAMERA_NUM_ISP_UNITS];
    ///  Structure containing override settings for local average clip statistics block
    dwCameraISPLocalAvgClipStats lacStats[DW_CAMERA_NUM_ISP_UNITS];
    /// boolean flag to enable bad pixel statistics settings override
    bool enableBpStatsOverride;
    /// Structure containing override settings for bad pixel statistics block
    dwCameraISPBadPixelStats bpStats;
} dwCameraIspStatsOverrideSetting;

/**
* Reads a frame handle from the camera sensor. The reading is a blocking call. It will block the thread within a timeout[us]
* With the frame handle, the associated data can be queried from the sensor. Available data is configured
* during sensor creation.
*
* @param[out] frameHandle A pointer to a handle to a frame read from the camera. With the handle, different
*              data can be queried. The frame handle must be returned to be put back into the internal pool.
* @param[in] timeoutUs Timeout in microseconds to wait for a new frame. Special values: DW_TIMEOUT_INFINITE - to wait infinitly.  Zero - means polling of internal queue.
* @param[in] sensor Sensor handle of the camera previously created with 'dwSAL_createSensor()'.
*
* @return DW_CUDA_ERROR - if the underlying camera driver had a CUDA error. <br>
*         DW_NVMEDIA_ERROR - if underlying camera driver had an NvMedia error. <br>
*         DW_INVALID_HANDLE - if given sensor handle is not valid. <br>
*         DW_INVALID_ARGUMENT - if given frame pointer is NULL <br>
*         DW_NOT_IMPLEMENTED - if the method for this image type is not implemented by given camera. <br>
*         DW_TIME_OUT - if no frame could be acquired within given time interval.  <br>
*                       This is a recoverable error if the following behavior is observed: the expected rate <br>
*                       at which frames are returned is a single frame every 1/fps seconds, therefore a timeout <br>
*                       should be set such as to cover one frame instance. Frames in a live camera are not returned <br>
*                       consistently therefore a time delta duration is to be expected close to the rate (ie +-10%). <br>
*                       If the total timeouts that have been accummulated after all retries amount to 3 frames interval <br>
*                       the camera is to be considered unresponsive.<br>
*         DW_NOT_AVAILABLE - if sensor has not been started or data is not available in polling mode. <br>
*         DW_NOT_READY - if sensor is stopped or not started or has started but hasn't begun acquiring frames after the specified timeout.
*                        If the sensor is in a stopped or not started state, the sensor should be started with `dwSensor_start`. After that
*                        The status will be returned until the first frame acquired. If this doesn't happen within 2 seconds, the sensor is to be considered
*                        unresponsive. <br>
*         DW_SAL_SENSOR_ERROR - if there was an i/o error. <br>
*         DW_END_OF_STREAM - if end of stream reached. <br>
*         DW_BUFFER_FULL - if there are no more available frames to be read. To recover, return frames to free buffer space<br>
*         DW_SUCCESS -if call is successful.
*
*/
DW_API_PUBLIC
dwStatus dwSensorCamera_readFrame(dwCameraFrameHandle_t* const frameHandle,
                                  dwTime_t const timeoutUs, dwSensorHandle_t const sensor);

/**
* Returns a frame to the camera after it has been consumed. All data associated with this
* handle is invalid after the handle has been returned.
*
* @param[in] frameHandle Handle previously read from the camera to be returned to the pool.
*
* @return DW_CUDA_ERROR - if the underlying camera driver had a CUDA error. <br>
*         DW_NVMEDIA_ERROR - if underlying camera driver had NvMedia error. <br>
*         DW_INVALID_ARGUMENT - if given handle is NULL <br>
*         DW_INVALID_HANDLE - if given handle is not valid. <br>
*         DW_CALL_NOT_ALLOWED - if the sensor this frame camera from has already been released. <br>
*         DW_SUCCESS - if call is successful.
*
**/
DW_API_PUBLIC
dwStatus dwSensorCamera_returnFrame(dwCameraFrameHandle_t* const frameHandle);

/**
* Gets the output image/s image in a format specified by the output type. Depending on the type requested,
* conversion and streaming handled by the camera implicitly might be required. The call is blocking
* NOTE: the underlying resources are still in the frame handle and the image returned is intended not to be modified.
*       For this reason, any modifications to this 'dwImageHandle_t' or 'dwImageCPU', 'dwImageCUDA', 'dwImageGL' or
*       'dwImageNvMedia' returned by 'dwImage_getCPU' ('dwImage_getCUDA', 'dwImage_getGL' and 'dwImage_getNvMedia')
*       will result in undefined behavior.
*
* @param[out] image Handle to the image received by the camera
* @param[in] type Ouptut type of the image. This is represented by a limited useful number of options which can be chosen at runtime
* @param[in] frame Camera frame handle of the captured frame
*
* @return DW_CUDA_ERROR - if the underlying camera driver had a CUDA error. <br>
*                         In such case it is not possible to recover a DW_CAMERA_OUTOUT_CUDA_X output, however it is  <br>
*                         possible to successfully request a DW_CAMERA_NATIVE_X output and work with it <br>
*         DW_NVMEDIA_ERROR - if underlying camera driver had an NvMedia error. <br>
*         DW_INVALID_HANDLE - if given camera frame handle is not valid. <br>
*         DW_INVALID_ARGUMENT - if given image handle is null. <br>
*         DW_NOT_IMPLEMENTED - if the method for this image type is not implemented by given camera. <br>
*         DW_SUCCESS - if call is successful.
*
*/
DW_API_PUBLIC
dwStatus dwSensorCamera_getImage(dwImageHandle_t* const image, dwCameraOutputType const type,
                                 dwCameraFrameHandle_t const frame);

/**
* Gets the output image/s image in a format specified by the output type. Depending on the type requested,
* conversion and streaming handled by the camera implicitly might be required, which happens on the cudaStream
* specified at 'dwSensorCamera_setCUDAStream()'
* NOTE: the underlying resources are still in the frame handle and the image returned is intended not to be modified.
*       For this reason, any modifications to this 'dwImageHandle_t' or 'dwImageCPU', 'dwImageCUDA', 'dwImageGL' or
*       'dwImageNvMedia' returned by 'dwImage_getCPU' ('dwImage_getCUDA', 'dwImage_getGL' and 'dwImage_getNvMedia')
*       will result in undefined behavior.
*
* @param[out] image Handle to the image received by the camera
* @param[in] type Ouptut type of the image. This is represented by a limited useful number of options which can be chosen at runtime
* @param[in] frame Camera frame handle of the captured frame
*
* @return DW_CUDA_ERROR - if the underlying camera driver had a CUDA error. <br>
*                         In such case it is not possible to recover a DW_CAMERA_OUTOUT_CUDA_X output, however it is  <br>
*                         possible to successfully request a DW_CAMERA_NATIVE_X output and work with it <br>
*         DW_NVMEDIA_ERROR - if underlying camera driver had an NvMedia error. <br>
*         DW_INVALID_HANDLE - if given handle is not valid. <br>
*         DW_NOT_IMPLEMENTED - if the method for this image type is not implemented by given camera. <br>
*         DW_SUCCESS - if call is successful.
*
*/
DW_API_PUBLIC
dwStatus dwSensorCamera_getImageAsync(dwImageHandle_t* const image, dwCameraOutputType const type,
                                      dwCameraFrameHandle_t const frame);

/**
* Sets a pool of image to be used as output by the camera layer. If this is called, the default pool is not allocated.
* The pool's type (raw/isp0/isp1/isp2) is deduced automatically by the format of the image. All images in the pool must match in properties
* If the size of the pool mismatches fifo-size, the fifo-size will be overridden.
*
* @param[in] imagePool Handle to the dwImagePool, ownership remains of the creator of the pool, the camera will be using the images as outputs during capture and processing
* @param[in] sensor Camera sensor handle
*
* @return DW_NVMEDIA_ERROR - if underlying camera driver had an NvMedia error. <br>
*         DW_INVALID_HANDLE - if given handle is not valid. <br>
*         DW_NOT_IMPLEMENTED - if the method for this image type is not implemented by given camera. <br>
*         DW_SUCCESS - if call is successful.
*
*/
DW_API_PUBLIC
dwStatus dwSensorCamera_setImagePool(dwImagePool imagePool, dwSensorHandle_t const sensor);

/**
 * Gets the NvMediaIPPManager used for GMSL camera IPP setup and event callback.
 *
 * @param[out] manager A pointer to the NvMediaIPPManager instance created by the sensor.
 * @param[in] sensor Sensor handle of the camera sensor previously created with
 *            dwSAL_createSensor().
 *
 * @return DW_NOT_AVAILABLE if NvMedia is not available. <br>
 *         DW_INVALID_ARGUMENT if given sensor handle or output pointer are NULL<br>
 *         DW_SUCCESS - if call is successful.
 *
 * @note The ownership of the NvMedia IPP manager remains with the sensor.
 */
DW_API_PUBLIC
dwStatus dwSensorCamera_getNvMediaIPPManager(NvMediaIPPManager** const manager, dwSensorHandle_t const sensor);

/**
 * Append the allocation attribute such that images allocated by the application and given to the camera via
 * dwSensorCamera_setImagePool() can be imported into the underlying driver.
 * This API is used to append the underlying driver's allocation attributes to the image properties.
 * @param[inout] imgProps Image properties
 * @param[in] outputType Ouptut type of the camera.
 * @param[in] sensor Sensor handle of the camera sensor previously created with
 *            dwSAL_createSensor().
 * @note The given imgProps should be compatible with that returned by
 *       dwSensorCamra_getImageProperties API.
 * @note The imgProps are read and used to generate the allocation attributes
 *       needed by the driver. The allocation attributes are stored back into
 *       imgProps.meta.allocAttrs. Applications do not need to free or alter the
 *       imgProps.meta.allocAttrs in any way. The imgProps.meta.allocAttrs are only used
 *       by DriveWorks as needed when the given imgProps are used to allocate dwImages.
 *       If the application alters the imgProps after calling this API, the
 *       imgProps.meta.allocAttrs may no longer be applicable to the imgProps and calls related
 *       to allocating images will fail.
 * @note if imgProps.meta.allocAttrs does not have allocated Memory, this would be allocated by
 *       DW and will be owned by DW context until context is destroyed
 *       and should be used wisely as it the space is limited.
 * @note Must be called after dwSAL_start().
 *
 * @return DW_NVMEDIA_ERROR - if underlying camera driver had an NvMedia error. <br>
 *         DW_INVALID_HANDLE - if given handle is not valid. <br>
 *         DW_NOT_IMPLEMENTED - if the method for this image type is not implemented by given camera. <br>
 *         DW_SUCCESS - if call is successful.
 *
 */
DW_API_PUBLIC
dwStatus dwSensorCamera_appendAllocationAttributes(dwImageProperties* const imgProps,
                                                   dwCameraOutputType const outputType,
                                                   dwSensorHandle_t const sensor);

/**
* Gets information about the camera sensor.
*
* @param[out] properties A pointer to the properties of the camera.
* @param[in] sensor Sensor handle of the camera sensor previously created with
*            dwSAL_createSensor().
*
* @return DW_INVALID_ARGUMENT: if the sensor handle is null <br>
*         DW_INVALID_HANDLE: if the sensor handle is not a camera <br>
*         DW_SUCCESS - if call is successful.
*/
DW_API_PUBLIC
dwStatus dwSensorCamera_getSensorProperties(dwCameraProperties* const properties,
                                            dwSensorHandle_t const sensor);

/**
* Gets number of supported capture modes.
*
* @param[out] numModes A pointer to the number of available capture modes.
* @param[in] sensor Sensor handle of the camera sensor previously created with
*            dwSAL_createSensor().
*
* @return DW_INVALID_ARGUMENT: if the sensor handle is null <br>
*         DW_INVALID_HANDLE: if the sensor handle is not a camera <br>
*         DW_SUCCESS: if call is successful.
*/
DW_API_PUBLIC
dwStatus dwSensorCamera_getNumSupportedCaptureModes(uint32_t* const numModes,
                                                    dwSensorHandle_t const sensor);

/**
* Gets capture modes by specified index.
*
* @param[out] captureMode A pointer to available capture mode.
* @param[in] modeIdx Index of a mode to retrieve.
* @param[in] sensor Sensor handle of the camera sensor previously created with
*            dwSAL_createSensor().
*
* @return DW_INVALID_ARGUMENT: if the sensor handle is null <br>
*         DW_INVALID_HANDLE: if the sensor handle is not a camera <br>
*         DW_SUCCESS: if call is successful.
*/
DW_API_PUBLIC
dwStatus dwSensorCamera_getSupportedCaptureMode(dwCameraProperties* const captureMode,
                                                uint32_t const modeIdx,
                                                dwSensorHandle_t const sensor);

/**
* Gets information about the image properties for a given 'dwCameraImageOutputType'.
*
* @param[out] imageProperties A pointer to image properties of the frames captured by the camera.
* @param[in] outputType Format of the output image to get the properties of
* @param[in] sensor Sensor handle of the camera sensor previously created with
*            dwSAL_createSensor().
*
* @note The dimensions of the returned properties corresponds to the dimension of the image returned through
*       'dwSensorCamera_getImage*' methods
*
* @return DW_INVALID_ARGUMENT: if the sensor handle is null <br>
*         DW_INVALID_HANDLE: if the sensor handle is not a camera <br>
*         DW_NOT_AVAILABLE: when the setup of the sensor is incompatible with the output type requested <br>
*         DW_SUCCESS: if call is successful.
*/
DW_API_PUBLIC
dwStatus dwSensorCamera_getImageProperties(dwImageProperties* const imageProperties,
                                           dwCameraOutputType const outputType,
                                           dwSensorHandle_t const sensor);

/**
 * Sets the CUDA stream used by getImageAsync during internal cuda related operations
 * Cuda stream is a bunch of asynchronous Cuda operations executed on the device in the order that the host code calls.
 * @param[in] stream The CUDA stream to use.
 * @param[in] sensor A pointer to the camera handle that is updated.
 *
* @return DW_INVALID_ARGUMENT: if the sensor handle is null <br>
*         DW_INVALID_HANDLE: if the sensor handle is not a camera <br>
*         DW_CUDA_ERROR: when the cudaStream cannot be set <br>
*         DW_SUCCESS: if call is successful.
 */
DW_API_PUBLIC
dwStatus dwSensorCamera_setCUDAStream(cudaStream_t const stream, dwSensorHandle_t const sensor);

/**
 * Gets the CUDA stream used.
 *
 * @param[out] stream Returns the CUDA stream in sensor.
 * @param[in] sensor A pointer to the camera handle that is updated.
 *
* @return DW_INVALID_ARGUMENT: if the sensor handle or cudaStream is null <br>
*         DW_INVALID_HANDLE: if the sensor handle is not a camera <br>
*         DW_SUCCESS: if call is successful.
 */
DW_API_PUBLIC
dwStatus dwSensorCamera_getCUDAStream(cudaStream_t* const stream, dwSensorHandle_t const sensor);

/**
* Gets the timestamp of the current camera frame.
*
* @param[out] timestamp The timestamp of the current camera frame.
* @param[in] frameHandle Handle to a captured frame.
*
* @return DW_INVALID_HANDLE - if given handle is not valid. <br>
*         DW_SUCCESS - if call is successful.
*
**/
DW_API_PUBLIC
dwStatus dwSensorCamera_getTimestamp(dwTime_t* const timestamp, dwCameraFrameHandle_t const frameHandle);

/**
* Gets the timestamps of the current camera frame.
*
* @param[out] imageTimestamps The timestamps of the current camera frame.
* @param[in] frameHandle Handle to a captured frame.
*
* @return DW_INVALID_HANDLE - if given handle is not valid. <br>
*         DW_INVALID_ARGUMENT - if given imageTimestamps point is null
*         DW_INVALID_ARGUMENT - if given frameHandle is null 
*         DW_SUCCESS
*
**/
DW_API_PUBLIC
dwStatus dwSensorCamera_getImageTimestamps(dwImageTimestamps* const imageTimestamps, dwCameraFrameHandle_t const frameHandle);

/**
* Gets SIPL Interface provider for a custom camera sensor.
*
* @param[out] interfaceProvider nvsipl::IInterfaceProvider*& for given sensor
* @param[in] sensor Handle to sensor
* @param[in] type type of interface provider
*
* @return DW_INVALID_HANDLE - if given handle is not valid. <br>
*         DW_NOT_SUPPORTED  - if the APIis not supported on platform
*         DW_SUCCESS - if call is successful.
**/
DW_API_PUBLIC
dwStatus dwSensorCamera_getSIPLInterfaceProvider(void** const interfaceProvider, dwSensorHandle_t const sensor, dwSIPLCameraInterfaceProviderType const type);

/**
 * Read data associated with a parameter stored on the EEPROM device and write to the provided buffer.
 * If the parameter is not present, does not contain valid data, or is corrupted, then this API
 * call will fail, and no data will be written to the provided buffer.
 * Currently doesn't support virtual sensors.
 *
 * Note: This reads from the copy of the EEPROM data in local memory, so this can be done,
 *       even if the ISC device is in a state where it cannot be read from.
 *
 * @param[in] paramId The ID of the parameter to be read
 * @param[out] buffer A pointer to the buffer that the data is to be read to.
 *                    It must be at least 'size' bytes long.
 * @param[in] size The number of bytes that are to be read from the parameter. This must be greater
 *                 than zero and less than or equal to the maximum size of the parameter.
 * @param[in] sensor A handle to the camera GMSL
 *
 * @return DW_SUCCESS if call is successful, <br>
 *         DW_NVMEDIA_ERROR If the read is unsuccessful due to nvmedia, <br>
 *         DW_NOT_SUPPORTED for unsupported platforms, <br>
 *         DW_INVALID_HANDLE if the given sensor handle is invalid,i.e null or of wrong type  <br>
 */
DW_API_PUBLIC
dwStatus dwSensorCamera_readEEPROM(uint32_t const paramId, void* const buffer, uint32_t const size, dwSensorHandle_t const sensor);

/**
 * Get EOF fence of the current camera frame according to the type of dwCameraOutputType.
 *
 * @param[out] syncFence The sync fence of the frame
 * @param[in] outputType The output type
 * @param[in] frameHandle Handle to the camera frame

 **/
DW_API_PUBLIC
dwStatus dwSensorCamera_getEOFFence(NvSciSyncFence* syncFence, dwCameraOutputType outputType, dwCameraFrameHandle_t const frameHandle);

/**
 * Fill the sync attributes for the camera pipeline to signal EOF fences. Note that multiple calls on the same syncAttrList will append the same attributes.
 *
 * @param[out] syncAttrList The sync attributes list to be filled
 * @param[in] outputType The output type
 * @param[in] sensor The sensor handle
 **/
DW_API_PUBLIC
DW_DEPRECATED("dwSensorCamera_fillSyncAttributes() is deprecated and will be removed in the next major release,"
              " use dwSensorCamera_fillSyncAttributesNew() instead")
dwStatus dwSensorCamera_fillSyncAttributes(NvSciSyncAttrList syncAttrList, dwCameraOutputType outputType, dwSensorHandle_t sensor);

/**
 * Set the sync obj to which the camera pipeline will signal EOF fences. The sync object is not reference counted
 *
 * @param[in] syncObj The sync object
 * @param[in] outputType The output type
 * @param[in] sensor The sensor handle
 **/
DW_API_PUBLIC
DW_DEPRECATED("dwSensorCamera_setSyncObject() is deprecated and will be removed in the next major release,"
              " use dwSensorCamera_setSyncObjectNew() instead")
dwStatus dwSensorCamera_setSyncObject(NvSciSyncObj syncObj, dwCameraOutputType outputType, dwSensorHandle_t sensor);

/**
 * Set the sync obj to which the camera pipeline will signal EOF fences. The sync object is not reference counted
 *
 * @param[in] syncObj The sync object
 * @param[in] syncType The sync type
 * @param[in] outputType The output type
 * @param[in] sensor The sensor handle
 **/
DW_API_PUBLIC
dwStatus dwSensorCamera_setSyncObjectNew(NvSciSyncObj syncObj, dwSyncType syncType, dwCameraOutputType outputType, dwSensorHandle_t sensor);

/**
 * Fill the sync attributes for the camera pipeline to signal EOF fences. Note that multiple calls on the same syncAttrList will append the same attributes.
 *
 * @param[out] syncAttrList The sync attributes list to be filled
 * @param[in] syncType The sync type
 * @param[in] outputType The output type
 * @param[in] sensor The sensor handle
 **/
DW_API_PUBLIC
// coverity[misra_c_2012_rule_5_1_violation] RFD Pending: TID-2085 Deprecated API
dwStatus dwSensorCamera_fillSyncAttributesNew(NvSciSyncAttrList syncAttrList, dwSyncType syncType, dwCameraOutputType outputType, dwSensorHandle_t sensor);

/**
 * Set array of prefences and a type of dwCameraOutputType so that camera waits on those fences before the use of that output
 *
 * @param[in] syncFences Array of prefences
 * @param[in] count Prefence count
 * @param[in] outputType The output type
 * @param[in] frameHandle Handle to the camera frame

 **/
DW_API_PUBLIC
dwStatus dwSensorCamera_addPreFenceArray(NvSciSyncFence* syncFences, uint32_t count, dwCameraOutputType outputType, dwCameraFrameHandle_t const frameHandle);

/**
 * Set the Camera Error Handling callbacks.
 * If receiving the device block error event, the blkCallback will be invoked.
 * If receiving the pipeline error event, the lineCallback will be invoked.
 *
 * @note
 * 1. At least one of the blkCallback and lineCallback should not be "nullptr" \n
 * 2. The callback functions need to return in a very short time and they shouldn't block the caller thread
 *
 * @param[in] blkCallback device block error handling function. Set to nullptr if not used
 * @param[in] lineCallback pipeline error handling function. Set to nullptr if not used
 * @param[in] sensor A handle to the camera GMSL
 *
 * @return DW_NOT_SUPPORTED If current SDK/PDK version is not supported
 *         DW_SUCCESS if call is successful.
 */
DW_API_PUBLIC
dwStatus dwSensorCamera_setEventCallback(dwCameraCallback blkCallback, dwCameraCallback lineCallback, dwSensorHandle_t sensor);

/**
 * Disable the camera link
 * This method should only be called after dwSensor_start() and before dwSensor_stop().
 *
 * @param[in] sensor A handle to the camera GMSL
 *
 * @return DW_NOT_SUPPORTED If current SDK/PDK version is not supported
 *         DW_INTERNAL_ERROR If the disableLink failed
 *         DW_SUCCESS if call is successful.
 */
DW_API_PUBLIC
dwStatus dwSensorCamera_disableLink(dwSensorHandle_t const sensor);

/**
 * Enable the camera link
 * This method enables a given link and, if reset is asserted, reconfigures
 * the camera module to restablish the link.
 *
 * This method should only be called after dwSensor_start() and before dwSensor_stop().
 * Please note, it is not necessary to call dwSensorCamera_enableLink() after dwSensor_start().
 * This API is used to enable the link again after the link has been disabled by dwSensorCamera_disableLink().
 * It is called if the error occurs on initializing.
 * @param[in] sensor A handle to the camera GMSL
 * @param[in] resetModule If true, reconfigure the camera module before enabling the link.
 *
 * @return DW_NOT_SUPPORTED If current SDK/PDK version is not supported
 *         DW_INTERNAL_ERROR If the enableLink failed
 *         DW_SUCCESS if call is successful.
 */
DW_API_PUBLIC
dwStatus dwSensorCamera_enableLink(dwSensorHandle_t const sensor, bool const resetModule);

/**
 * Overrides ISP statistics (Histogram, Local Average Clip and bad pixel) settings
 *
 * @note The ISP Statistics updates the platform configuration for entire image Stream.
 * @note Must be called before dwSAL_start().
 *
 * @return DW_NVMEDIA_ERROR - if underlying camera driver had an NvMedia error. <br>
 *         DW_INVALID_HANDLE - if given handle is not valid. <br>
 *         DW_NOT_IMPLEMENTED - if the method for this image type is not implemented by given camera. <br>
 *         DW_CALL_NOT_ALLOWED - if the method is called before dwAL_start. <br>
 *         DW_SUCCESS - if call is successful.
 *
 * @param[out] overrideISPStats A pointer to override ISP statistics settings
 * @param[in] sensor The sensor handle.
 **/
DW_API_PUBLIC
dwStatus dwSensorCamera_setImageMetaDataStats(dwCameraIspStatsOverrideSetting const* overrideISPStats, dwSensorHandle_t sensor);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_SENSORS_CAMERA_CAMERA_H_
