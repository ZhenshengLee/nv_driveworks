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
// SPDX-FileCopyrightText: Copyright (c) 2015-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "SimpleCamera.hpp"

#include <chrono>
#include <stdexcept>
#include <thread>

#include <framework/Checks.hpp>
#include <framework/Log.hpp>
#include <framework/Checks.hpp>
#include <dwshared/dwfoundation/dw/core/container/BaseString.hpp>

namespace dw_samples
{
namespace common
{

using FixedString32                         = dw::core::FixedString<32>;
using FixedString512                        = dw::core::FixedString<512>;
static constexpr uint32_t ICP_PIPELINE_IDX  = 0;
static constexpr uint32_t ISP_PIPELINE_IDX  = 1;
static constexpr uint32_t ISP1_PIPELINE_IDX = 2;
static constexpr uint32_t ISP2_PIPELINE_IDX = 3;

SimpleCamera::SimpleCamera(dwSALHandle_t sal,
                           dwContextHandle_t ctx)
    : m_ctx(ctx)
    , m_sal(sal)
    , m_pendingFrame(nullptr)
    , m_started(false)
    , m_bSetImagePool(false)
    , m_useRaw(false)
    , m_useProcessed(false)
    , m_useProcessed1(false)
    , m_useProcessed2(false)
{
}

SimpleCamera::SimpleCamera(const dwSensorParams& params,
                           dwSALHandle_t sal,
                           dwContextHandle_t ctx,
                           dwCameraOutputType outputType,
                           bool bSetImagePool)
    : SimpleCamera(sal, ctx)
{
    dwImageProperties imageProperties;
    m_bSetImagePool = bSetImagePool;

    createSensor(imageProperties, params, outputType);

    m_framePipeline.reset(new CameraFramePipeline(imageProperties, outputType, ctx));
}

SimpleCamera::SimpleCamera(const dwImageProperties& outputProperties,
                           const dwSensorParams& params,
                           dwSALHandle_t sal,
                           dwContextHandle_t ctx,
                           dwCameraOutputType outputType)
    : SimpleCamera(params, sal, ctx, outputType)
{
    m_framePipeline->setOutputProperties(outputProperties);
}

SimpleCamera::~SimpleCamera()
{
    if (m_pendingFrame)
    {
        releaseFrame();
    }

    if ((m_bSetImagePool == true) && (m_fifoSize > 0))
    {
        for (uint32_t pipelineIdx = 0; pipelineIdx < MAX_PIPELINE_COUNT; pipelineIdx++)
        {
            for (uint32_t imageIdx = 0; imageIdx < m_fifoSize; imageIdx++)
            {
                if (m_imagesForImagePool[pipelineIdx][imageIdx] != nullptr)
                {
                    dwImage_destroy(m_imagesForImagePool[pipelineIdx][imageIdx]);
                }
            }
        }

        if (m_imgTrans != nullptr)
        {
            dwImageTransformation_release(m_imgTrans);
        }
    }

    if (m_sensor)
    {
        if (m_started)
            dwSensor_stop(m_sensor);
        dwSAL_releaseSensor(m_sensor);
    }
}

void SimpleCamera::setImagePool(uint32_t imagePoolIndex, dwCameraOutputType outputType)
{
    dwImagePool pool{};

    std::cout << "Setting Image Pool for CameraOutputType: " << outputType;

    for (uint32_t i = 0; i < m_fifoSize; i++)
    {
        dwImageProperties imgProps{};

        // Gets information about the image properties for a given 'dwCameraImageOutputType'.
        CHECK_DW_ERROR(dwSensorCamera_getImageProperties(&imgProps, outputType, m_sensor));

        CHECK_DW_ERROR(dwImageTransformation_appendAllocationAttributes(&imgProps, m_imgTrans));

        // Append the NvSci allocation attribute of Camera Module.
        CHECK_DW_ERROR(dwSensorCamera_appendAllocationAttributes(&imgProps, outputType, m_sensor));

        CHECK_DW_ERROR(dwImage_create(&m_imagesForImagePool[imagePoolIndex][i], imgProps, m_ctx));
    }

    // pass the frames to the camera.
    pool.imageCount = m_fifoSize;
    pool.images     = &m_imagesForImagePool[imagePoolIndex][0];

    CHECK_DW_ERROR(dwSensorCamera_setImagePool(pool, m_sensor));
}

/**
     * @brief Helper method to set and allocate and exchange buffers and synchronization objects across Camera Sensor and external modules/applications.
     */
void SimpleCamera::setNvSciAttributesHelper()
{
    std::cout << "Main: Setting Image Pool." << std::endl;

    // NvSciBuf
    if (m_useRaw)
    {
        setImagePool(ICP_PIPELINE_IDX, dwCameraOutputType::DW_CAMERA_OUTPUT_NATIVE_RAW);
    }

    if (m_useProcessed)
    {
        // Create images to set the ImagePool for ISP0 Pipeline.
        setImagePool(ISP_PIPELINE_IDX, dwCameraOutputType::DW_CAMERA_OUTPUT_NATIVE_PROCESSED);
    }

    if (m_useProcessed1)
    {
        // Create images to set the ImagePool for ISP1 Pipeline.
        setImagePool(ISP1_PIPELINE_IDX, dwCameraOutputType::DW_CAMERA_OUTPUT_NATIVE_PROCESSED1);
    }

    if (m_useProcessed2)
    {
        // Create images to set the ImagePool for ISP2 Pipeline.
        setImagePool(ISP2_PIPELINE_IDX, dwCameraOutputType::DW_CAMERA_OUTPUT_NATIVE_PROCESSED2);
    }
}

std::string SimpleCamera::getFlagValue(const std::string& input, const std::string& flag)
{
    std::stringstream ss(input);
    std::string token;

    while (std::getline(ss, token, ','))
    {
        size_t pos = token.find('=');
        if (pos != std::string::npos)
        {
            std::string currentFlag = token.substr(0, pos);
            if (currentFlag == flag)
            {
                return token.substr(pos + 1);
            }
        }
    }

    return ""; // Flag not found
}

void SimpleCamera::createSensor(dwImageProperties& imageProps, const dwSensorParams& params, dwCameraOutputType outputType)
{
    CHECK_DW_ERROR(dwSAL_createSensor(&m_sensor, params, m_sal));
    CHECK_DW_ERROR(dwSensorCamera_getSensorProperties(&m_cameraProperties, m_sensor));
    CHECK_DW_ERROR(dwSensorCamera_getImageProperties(&imageProps, outputType, m_sensor));

    auto pixelFormat = getFlagValue(std::string(params.parameters), std::string("output-format"));

    if (pixelFormat.length() > 0)
    {
        if (pixelFormat.find("processed") != FixedString32::NPOS)
        {
            m_useProcessed = true;
        }
        if (pixelFormat.find("processed1") != FixedString32::NPOS)
        {
            m_useProcessed1 = true;
        }
        if (pixelFormat.find("processed2") != FixedString32::NPOS)
        {
            m_useProcessed2 = true;
        }
        if (pixelFormat.find("raw") != FixedString32::NPOS)
        {
            m_useRaw = true;
        }
    }
    else
    {
        std::cerr << "No output-format specified. Specify processing pipelines to use in rig-params" << std::endl;
        m_useProcessed  = false;
        m_useProcessed1 = false;
        m_useProcessed2 = false;
        m_useRaw        = false;
        return;
    }

    auto fifoSizeStr = getFlagValue(std::string(params.parameters), std::string("fifo-size"));
    if (fifoSizeStr.length() > 0)
    {
        m_fifoSize = std::stoi(fifoSizeStr);
    }

    if (m_fifoSize == 0)
    {
        // Minimum Recommended FifoSize
        m_fifoSize = 4;
    }

    if (m_bSetImagePool)
    {
        CHECK_DW_ERROR(dwSAL_start(m_sal));

        // Setup transformation Module
        dwImageTransformationParameters imageTrnsformationParams{};
        CHECK_DW_ERROR(dwImageTransformation_initialize(&m_imgTrans, imageTrnsformationParams, m_ctx));

        setNvSciAttributesHelper();
    }

    log("SimpleCamera: Camera image: %ux%u\n", imageProps.width, imageProps.height);
}

void SimpleCamera::setOutputProperties(const dwImageProperties& outputProperties)
{
    m_framePipeline->setOutputProperties(outputProperties);
}

void SimpleCamera::getFrameTimestamp(dwTime_t* timestamp)
{
    dwSensorCamera_getTimestamp(timestamp, m_pendingFrame);
}

dwImageHandle_t SimpleCamera::readFrame(dwTime_t timeout)
{
    if (!m_started)
    {
        CHECK_DW_ERROR(dwSensor_start(m_sensor));
        m_started = true;
    }

    if (m_pendingFrame)
        releaseFrame();

    dwStatus status = dwSensorCamera_readFrame(&m_pendingFrame, timeout, m_sensor);

    if (status == DW_END_OF_STREAM)
    {
        log("SimpleCamera: Camera reached end of stream.\n");
        return nullptr;
    }
    else if (status == DW_NOT_READY)
    {
        while (status == DW_NOT_READY)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            status = dwSensorCamera_readFrame(&m_pendingFrame, timeout, m_sensor);
        }
    }
    else if (status == DW_TIME_OUT)
    {
        throw std::runtime_error("Reading from camera timed out - please increase timeout");
    }
    else if (status != DW_SUCCESS)
    {
        throw std::runtime_error("Error reading from camera");
    }

    m_framePipeline->processFrame(m_pendingFrame);
    if (isGLOutputEnabled())
    {
        m_imageRgbaGL = m_streamerGL->post(m_framePipeline->getFrameRgba());
    }

    return m_framePipeline->getFrame();
}

dwImageHandle_t SimpleCamera::readCameraFrame(dwCameraFrameHandle_t& cameraFrame, dwTime_t timeout)
{
    dwImageHandle_t imageHandle;
    if (!m_started)
    {
        CHECK_DW_ERROR(dwSensor_start(m_sensor));
        m_started = true;
    }

    dwStatus status = dwSensorCamera_readFrame(&m_pendingFrame, timeout, m_sensor);

    if (status == DW_END_OF_STREAM)
    {
        log("SimpleCamera: Camera reached end of stream.\n");
        return nullptr;
    }
    else if (status == DW_NOT_READY)
    {
        while (status == DW_NOT_READY)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            status = dwSensorCamera_readFrame(&m_pendingFrame, timeout, m_sensor);
        }
    }
    else if (status == DW_TIME_OUT)
    {
        throw std::runtime_error("Reading from camera timed out - please increase timeout");
    }
    else if (status != DW_SUCCESS)
    {
        throw std::runtime_error("Error reading from camera");
    }

    if (status == DW_SUCCESS)
    {
        status = dwSensorCamera_getImage(&imageHandle, DW_CAMERA_OUTPUT_NATIVE_PROCESSED, m_pendingFrame);
        if (status != DW_SUCCESS)
        {
            throw std::runtime_error("Camera frame with invalid handle");
        }
    }

    m_framePipeline->processFrame(m_pendingFrame);
    if (isGLOutputEnabled())
    {
        m_imageRgbaGL = m_streamerGL->post(m_framePipeline->getFrameRgba());
    }

    cameraFrame = m_pendingFrame;

    return m_framePipeline->getFrame();
}

bool SimpleCamera::enableSeeking(size_t& frameCount, dwTime_t& startTimestamp, dwTime_t& endTimestamp)
{
    // Try to get seek range of a sensor
    dwStatus res = dwSensor_getSeekRange(&frameCount, &startTimestamp, &endTimestamp, m_sensor);
    if (res != DW_SUCCESS)
    {
        // Seek table has not been created. Trying to create here.
        res = dwSensor_createSeekTable(m_sensor);
        if (res != DW_SUCCESS)
        {
            logError("SimpleCamera: Error creating index table: %s. Seeking is not available.\n", dwGetStatusName(res));
            return false;
        }

        CHECK_DW_ERROR_MSG(dwSensor_getSeekRange(&frameCount, &startTimestamp, &endTimestamp, m_sensor),
                           "Cannot obtain seek range from the camera.");
    }

    return true;
}

void SimpleCamera::seekToTime(dwTime_t timestamp)
{
    dwStatus res = dwSensor_seekToTime(timestamp, m_sensor);

    if (res != DW_SUCCESS)
    {
        logError("SimpleCamera: seek to time failed with %s.\n", dwGetStatusName(res));
    }
}

void SimpleCamera::seekToFrame(size_t frameIdx)
{
    dwStatus res = dwSensor_seekToEvent(frameIdx, m_sensor);

    if (res != DW_SUCCESS)
    {
        logError("SimpleCamera: seek to frame failed with %s.\n", dwGetStatusName(res));
    }
}

void SimpleCamera::getCurrFrameIdx(size_t* frameIdx)
{
    dwStatus res = dwSensor_getCurrentSeekPosition(frameIdx, m_sensor);
    if (res != DW_SUCCESS)
    {
        logError("SimpleCamera: get current frame index with %s.\n", dwGetStatusName(res));
    }
}

dwImageHandle_t SimpleCamera::getFrameRgbaGL()
{
    if (!isGLOutputEnabled())
    {
        logWarn("SimpleCamera: GL output is not enabled. Did you forget to call enableGLOutput()?\n");
    }
    return m_imageRgbaGL;
}

void SimpleCamera::releaseFrame()
{
    if (m_pendingFrame)
    {
        CHECK_DW_ERROR(dwSensorCamera_returnFrame(&m_pendingFrame));
        m_pendingFrame = nullptr;
    }
}

void SimpleCamera::resetCamera()
{
    m_framePipeline->reset();
    releaseFrame();
    CHECK_DW_ERROR(dwSensor_reset(m_sensor));
}

void SimpleCamera::enableGLOutput()
{
    if (!m_framePipeline->isRgbaOutputEnabled())
    {
        m_framePipeline->enableRgbaOutput();
    }

    dwImageProperties props = m_framePipeline->getRgbaOutputProperties();
    props.type              = DW_IMAGE_CUDA;
    m_streamerGL.reset(new SimpleImageStreamerGL<>(props, 60000, m_ctx));
}
}
}
