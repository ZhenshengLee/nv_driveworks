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
// SPDX-FileCopyrightText: Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Samples
#include <framework/DriveWorksSample.hpp>
#include <framework/SimpleStreamer.hpp>
#include <framework/SimpleCamera.hpp>
#include <framework/Checks.hpp>

// Core
#include <dw/core/context/Context.h>
#include <dw/core/logger/Logger.h>
#include <dw/core/base/Version.h>

// HAL
#include <dw/sensors/Sensors.h>

// DNN
#include <dw/dnn/DNN.h>
#include <dw/interop/streamer/TensorStreamer.h>

// Renderer
#include <dwvisualization/core/RenderEngine.h>

using namespace dw_samples::common;

class DNNTensorSample : public DriveWorksSample
{
private:
    // ------------------------------------------------
    // Driveworks Context and SAL
    // ------------------------------------------------
    dwContextHandle_t m_sdk              = DW_NULL_HANDLE;
    dwVisualizationContextHandle_t m_viz = DW_NULL_HANDLE;
    dwSALHandle_t m_sal                  = DW_NULL_HANDLE;

    // ------------------------------------------------
    // DNN
    // ------------------------------------------------
    typedef std::pair<dwRectf, float32_t> BBoxConf;
    static constexpr float32_t COVERAGE_THRESHOLD       = 0.6f;
    static constexpr uint32_t NUM_OUTPUT_TENSORS        = 2U;
    const uint32_t m_maxDetections                      = 1000U;
    const float32_t m_nonMaxSuppressionOverlapThreshold = 0.5;

    dwDNNHandle_t m_dnn                         = DW_NULL_HANDLE;
    dwDataConditionerHandle_t m_dataConditioner = DW_NULL_HANDLE;
    dwDNNTensorStreamerHandle_t m_dnnOutputStreamers[NUM_OUTPUT_TENSORS];
    dwDNNTensorHandle_t m_dnnInput;
    dwDNNTensorHandle_t m_dnnOutputsDevice[NUM_OUTPUT_TENSORS];
    std::vector<dwBox2D> m_detectedBoxList;
    std::vector<dwRectf> m_detectedBoxListFloat;

    uint32_t m_cellSize = 1U;
    uint32_t m_cvgIdx;
    uint32_t m_bboxIdx;

    dwRect m_detectionRegion;

    // ------------------------------------------------
    // Renderer
    // ------------------------------------------------
    dwRenderEngineHandle_t m_renderEngine = DW_NULL_HANDLE;
    dwImageHandle_t m_imageRGBA;
    std::unique_ptr<SimpleImageStreamerGL<>> m_streamerCUDA2GL;
    cudaStream_t m_cudaStream = 0;

    // ------------------------------------------------
    // Camera
    // ------------------------------------------------
    std::unique_ptr<SimpleCamera> m_camera;
    dwImageGL* m_imgGl;
    dwImageProperties m_rcbProperties;

    // image width and height
    uint32_t m_imageWidth;
    uint32_t m_imageHeight;
    bool m_isRaw = false;

public:
    /// -----------------------------
    /// Initialize application
    /// -----------------------------
    DNNTensorSample(const ProgramArguments& args)
        : DriveWorksSample(args) {}

    /// -----------------------------
    /// Initialize modules
    /// -----------------------------
    bool onInitialize() override
    {
        // -----------------------------------------
        // Initialize DriveWorks SDK context and SAL
        // -----------------------------------------
        {
            // initialize logger to print verbose message on console in color
            CHECK_DW_ERROR(dwLogger_initialize(getConsoleLoggerCallback(true)));
            CHECK_DW_ERROR(dwLogger_setLogLevel(DW_LOG_VERBOSE));

            // initialize SDK context, using data folder
            dwContextParameters sdkParams = {};

#ifdef VIBRANTE
            sdkParams.eglDisplay = getEGLDisplay();
#endif

            CHECK_DW_ERROR(dwInitialize(&m_sdk, DW_VERSION, &sdkParams));
            CHECK_DW_ERROR(dwSAL_initialize(&m_sal, m_sdk));
        }

        //------------------------------------------------------------------------------
        // initialize Renderer
        //------------------------------------------------------------------------------
        {
            CHECK_DW_ERROR(dwVisualizationInitialize(&m_viz, m_sdk));

            // Setup render engine
            dwRenderEngineParams params{};
            CHECK_DW_ERROR(dwRenderEngine_initDefaultParams(&params, getWindowWidth(), getWindowHeight()));
            params.defaultTile.lineWidth = 0.2f;
            params.defaultTile.font      = DW_RENDER_ENGINE_FONT_VERDANA_20;
            CHECK_DW_ERROR(dwRenderEngine_initialize(&m_renderEngine, &params, m_viz));
        }

        //------------------------------------------------------------------------------
        // initialize Sensors
        //------------------------------------------------------------------------------
        {
            dwSensorParams params{};
            std::string parameterString;
            {
#ifdef VIBRANTE
                if (getArgument("input-type").compare("camera") == 0)
                {
                    parameterString = "camera-type=" + getArgument("camera-type");
                    parameterString += ",camera-group=" + getArgument("camera-group");
                    parameterString += ",serialize=false,camera-count=4";
                    std::string cameraMask[4] = {"0001", "0010", "0100", "1000"};
                    uint32_t cameraIdx        = std::stoi(getArgument("camera-index"));
                    if (cameraIdx < 0 || cameraIdx > 3)
                    {
                        std::cerr << "Error: camera index must be 0, 1, 2 or 3" << std::endl;
                        return false;
                    }
                    parameterString += ",camera-mask=" + cameraMask[cameraIdx];

                    params.parameters = parameterString.c_str();
                    params.protocol   = "camera.gmsl";
                }
                else
#endif
                {
                    parameterString   = getArgs().parameterString();
                    params.parameters = parameterString.c_str();
                    params.protocol   = "camera.virtual";

                    std::string videoFormat = getArgument("video");
                }

                {
                    m_camera.reset(new SimpleCamera(params, m_sal, m_sdk));
                    dwImageProperties outputProperties = m_camera->getOutputProperties();
                    outputProperties.type              = DW_IMAGE_CUDA;
                    m_camera->setOutputProperties(outputProperties);
                }
            }

            if (m_camera == nullptr)
            {
                logError("Camera could not be created\n");
                return false;
            }

            std::cout << "Camera image with " << m_camera->getCameraProperties().resolution.x << "x"
                      << m_camera->getCameraProperties().resolution.y << " at "
                      << m_camera->getCameraProperties().framerate << " FPS" << std::endl;

            dwImageProperties displayProperties = m_camera->getOutputProperties();
            displayProperties.format            = DW_IMAGE_FORMAT_RGBA_UINT8;

            CHECK_DW_ERROR(dwImage_create(&m_imageRGBA, displayProperties, m_sdk));

            m_streamerCUDA2GL.reset(new SimpleImageStreamerGL<>(displayProperties, 1000, m_sdk));

            m_rcbProperties = m_camera->getOutputProperties();

            m_imageWidth  = displayProperties.width;
            m_imageHeight = displayProperties.height;
        }

        //------------------------------------------------------------------------------
        // initialize DNN
        //------------------------------------------------------------------------------
        {
            // If not specified, load the correct network based on platform
            std::string tensorRTModel = getArgument("tensorRT_model");
            if (tensorRTModel.empty())
            {
                tensorRTModel = dw_samples::SamplesDataPath::get() + "/samples/detector/";
                tensorRTModel += getPlatformPrefix();
                tensorRTModel += "/tensorRT_model.bin";
            }

            // Initialize DNN from a TensorRT file
            CHECK_DW_ERROR(dwDNN_initializeTensorRTFromFile(&m_dnn, tensorRTModel.c_str(), nullptr,
                                                            DW_PROCESSOR_TYPE_GPU, m_sdk));

            CHECK_DW_ERROR(dwDNN_setCUDAStream(m_cudaStream, m_dnn));

            // Get input and output dimensions
            dwDNNTensorProperties inputProps;
            dwDNNTensorProperties outputProps[NUM_OUTPUT_TENSORS];

            // Allocate input tensor
            CHECK_DW_ERROR(dwDNN_getInputTensorProperties(&inputProps, 0U, m_dnn));
            CHECK_DW_ERROR(dwDNNTensor_create(&m_dnnInput, &inputProps, m_sdk));

            // Allocate outputs
            for (uint32_t outputIdx = 0U; outputIdx < NUM_OUTPUT_TENSORS; ++outputIdx)
            {
                CHECK_DW_ERROR(dwDNN_getOutputTensorProperties(&outputProps[outputIdx], outputIdx, m_dnn));
                // Allocate device tensors
                CHECK_DW_ERROR(dwDNNTensor_create(&m_dnnOutputsDevice[outputIdx], &outputProps[outputIdx], m_sdk));
                // Allocate host tensors
                dwDNNTensorProperties hostProps = outputProps[outputIdx];
                hostProps.tensorType            = DW_DNN_TENSOR_TYPE_CPU;

                // Allocate streamer
                CHECK_DW_ERROR(dwDNNTensorStreamer_initialize(&m_dnnOutputStreamers[outputIdx],
                                                              &outputProps[outputIdx],
                                                              hostProps.tensorType, m_sdk));
            }

            // Get coverage and bounding box blob indices
            const char* coverageBlobName    = "coverage_sig";
            const char* boundingBoxBlobName = "bbox_regressor";
            CHECK_DW_ERROR(dwDNN_getOutputIndex(&m_cvgIdx, coverageBlobName, m_dnn));
            CHECK_DW_ERROR(dwDNN_getOutputIndex(&m_bboxIdx, boundingBoxBlobName, m_dnn));

            // Get metadata from DNN module
            // DNN loads metadata automatically from json file stored next to the dnn model,
            // with the same name but additional .json extension if present.
            // Otherwise, the metadata will be filled with default values and the dataconditioner parameters
            // should be filled manually.
            dwDNNMetaData metadata;
            CHECK_DW_ERROR(dwDNN_getMetaData(&metadata, m_dnn));

            // Initialie data conditioner
            CHECK_DW_ERROR(dwDataConditioner_initializeFromTensorProperties(&m_dataConditioner, &inputProps, 1U,
                                                                            &metadata.dataConditionerParams, m_cudaStream,
                                                                            m_sdk));

            // Reserve space for detected objects
            m_detectedBoxList.reserve(m_maxDetections);
            m_detectedBoxListFloat.reserve(m_maxDetections);

            // Detection region
            m_detectionRegion.width = std::min(static_cast<uint32_t>(inputProps.dimensionSize[0]),
                                               m_imageWidth);
            m_detectionRegion.height = std::min(static_cast<uint32_t>(inputProps.dimensionSize[1]),
                                                m_imageHeight);
            m_detectionRegion.x = (m_imageWidth - m_detectionRegion.width) / 2;
            m_detectionRegion.y = (m_imageHeight - m_detectionRegion.height) / 2;

            // Compute a pixel (cell) in output in relation to a pixel in input of the network
            uint32_t gridW = outputProps[0].dimensionSize[0];
            m_cellSize     = inputProps.dimensionSize[0] / gridW;
        }

        return true;
    }

    ///------------------------------------------------------------------------------
    /// Main processing of the sample
    ///     - collect sensor frame
    ///     - run detection and tracking
    ///------------------------------------------------------------------------------
    void onProcess() override
    {
        // read from camera
        dwImageCUDA* yuvImage = nullptr;
        getNextFrame(&yuvImage, &m_imgGl);
        std::this_thread::yield();
        while (yuvImage == nullptr)
        {
            onReset();

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            getNextFrame(&yuvImage, &m_imgGl);
        }

        // Run data conditioner to prepare input for the network
        CHECK_DW_ERROR(dwDataConditioner_prepareData(m_dnnInput, &m_imageRGBA, 1, &m_detectionRegion,
                                                     cudaAddressModeClamp, m_dataConditioner));

        // Run DNN on the output of data conditioner
        dwConstDNNTensorHandle_t inputs[1U] = {m_dnnInput};
        CHECK_DW_ERROR(dwDNN_infer(m_dnnOutputsDevice, NUM_OUTPUT_TENSORS, inputs, 1U, m_dnn));

        // Stream output to host
        dwDNNTensorHandle_t outputsHost[NUM_OUTPUT_TENSORS];
        for (uint32_t outputIdx = 0U; outputIdx < NUM_OUTPUT_TENSORS; ++outputIdx)
        {
            dwDNNTensorStreamerHandle_t streamer = m_dnnOutputStreamers[outputIdx];
            CHECK_DW_ERROR(dwDNNTensorStreamer_producerSend(m_dnnOutputsDevice[outputIdx],
                                                            streamer));

            CHECK_DW_ERROR(dwDNNTensorStreamer_consumerReceive(&outputsHost[outputIdx], 1000,
                                                               streamer));
        }

        // Interpret output blobs to extract detected boxes
        interpretOutput(outputsHost[m_cvgIdx], outputsHost[m_bboxIdx],
                        &m_detectionRegion);

        // Return streamed outputs
        for (uint32_t outputIdx = 0U; outputIdx < NUM_OUTPUT_TENSORS; ++outputIdx)
        {
            dwDNNTensorStreamerHandle_t streamer = m_dnnOutputStreamers[outputIdx];
            CHECK_DW_ERROR(dwDNNTensorStreamer_consumerReturn(&outputsHost[outputIdx], streamer));

            CHECK_DW_ERROR(dwDNNTensorStreamer_producerReturn(nullptr, 1000, streamer));
        }
    }

    ///------------------------------------------------------------------------------
    /// Render sample output on screen
    ///     - render video
    ///     - render boxes with labels
    ///------------------------------------------------------------------------------
    void onRender() override
    {
        CHECK_DW_ERROR(dwRenderEngine_setTile(0, m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_resetTile(m_renderEngine));

        dwVector2f range{};
        range.x = m_imgGl->prop.width;
        range.y = m_imgGl->prop.height;
        CHECK_DW_ERROR(dwRenderEngine_setCoordinateRange2D(range, m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_renderImage2D(m_imgGl, {0.0f, 0.0f, range.x, range.y}, m_renderEngine));

        // Render detection region
        dwRectf detectionRegionFloat;
        detectionRegionFloat.x      = m_detectionRegion.x;
        detectionRegionFloat.y      = m_detectionRegion.y;
        detectionRegionFloat.width  = m_detectionRegion.width;
        detectionRegionFloat.height = m_detectionRegion.height;

        CHECK_DW_ERROR(dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_YELLOW, m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_BOXES_2D, &detectionRegionFloat,
                                             sizeof(dwRectf), 0, 1, m_renderEngine));

        CHECK_DW_ERROR(dwRenderEngine_setColor(DW_RENDER_ENGINE_COLOR_RED, m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_setLineWidth(2.0f, m_renderEngine));
        CHECK_DW_ERROR(dwRenderEngine_render(DW_RENDER_ENGINE_PRIMITIVE_TYPE_BOXES_2D,
                                             m_detectedBoxListFloat.data(), sizeof(dwRectf), 0,
                                             m_detectedBoxListFloat.size(), m_renderEngine));
    }

    ///------------------------------------------------------------------------------
    /// Free up used memory here
    ///------------------------------------------------------------------------------
    void onRelease() override
    {
        m_streamerCUDA2GL.reset();
        m_camera.reset();

        if (m_imageRGBA)
        {
            CHECK_DW_ERROR(dwImage_destroy(m_imageRGBA));
        }

        if (m_dnnInput)
        {
            CHECK_DW_ERROR(dwDNNTensor_destroy(m_dnnInput));
        }

        for (uint32_t outputIdx = 0U; outputIdx < NUM_OUTPUT_TENSORS; ++outputIdx)
        {
            if (m_dnnOutputsDevice[outputIdx])
            {
                CHECK_DW_ERROR(dwDNNTensor_destroy(m_dnnOutputsDevice[outputIdx]));
            }
            if (m_dnnOutputStreamers[outputIdx])
            {
                CHECK_DW_ERROR(dwDNNTensorStreamer_release(m_dnnOutputStreamers[outputIdx]));
            }
        }

        // Release detector
        CHECK_DW_ERROR(dwDNN_release(m_dnn));
        // Release data conditioner
        CHECK_DW_ERROR(dwDataConditioner_release(m_dataConditioner));
        // Release render engine
        CHECK_DW_ERROR(dwRenderEngine_release(m_renderEngine));

        // Release SDK
        CHECK_DW_ERROR(dwSAL_release(m_sal));
        CHECK_DW_ERROR(dwVisualizationRelease(m_viz));
        CHECK_DW_ERROR(dwRelease(m_sdk));
    }

    ///------------------------------------------------------------------------------
    /// Reset tracker and detector
    ///------------------------------------------------------------------------------
    void onReset() override
    {
        CHECK_DW_ERROR(dwDNN_reset(m_dnn));
        CHECK_DW_ERROR(dwDataConditioner_reset(m_dataConditioner));
    }

    ///------------------------------------------------------------------------------
    /// Change renderer properties when main rendering window is resized
    ///------------------------------------------------------------------------------
    void onResizeWindow(int width, int height) override
    {
        {
            CHECK_DW_ERROR(dwRenderEngine_reset(m_renderEngine));
            dwRectf rect;
            rect.width  = width;
            rect.height = height;
            rect.x      = 0;
            rect.y      = 0;
            CHECK_DW_ERROR(dwRenderEngine_setBounds(rect, m_renderEngine));
        }
    }

private:
    //------------------------------------------------------------------------------
    void getNextFrame(dwImageCUDA** nextFrameCUDA, dwImageGL** nextFrameGL)
    {
        dwImageHandle_t nextFrame = m_camera->readFrame();
        if (nextFrame == nullptr)
        {
            m_camera->resetCamera();
        }
        else
        {
            dwImage_getCUDA(nextFrameCUDA, nextFrame);
            CHECK_DW_ERROR(dwImage_copyConvert(m_imageRGBA, nextFrame, m_sdk));
            dwImageHandle_t frameGL = m_streamerCUDA2GL->post(m_imageRGBA);
            dwImage_getGL(nextFrameGL, frameGL);
        }
    }

    //------------------------------------------------------------------------------
    void interpretOutput(dwDNNTensorHandle_t outConf, dwDNNTensorHandle_t outBBox, const dwRect* const roi)
    {
        // Clear detection list
        m_detectedBoxList.clear();
        m_detectedBoxListFloat.clear();

        uint32_t numBBoxes = 0U;

        void* tmpConf;
        CHECK_DW_ERROR(dwDNNTensor_lock(&tmpConf, outConf));
        float32_t* confData = reinterpret_cast<float32_t*>(tmpConf);

        void* bboxData;
        CHECK_DW_ERROR(dwDNNTensor_lock(&bboxData, outBBox));

        size_t offsetY;
        size_t strideY;
        size_t height;
        uint32_t indices[4] = {0, 0, 0, 0};
        CHECK_DW_ERROR(dwDNNTensor_getLayoutView(&offsetY, &strideY, &height, indices, 4U, 1U, outConf));

        for (uint32_t gridY = 0U; gridY < height; ++gridY)
        {
            size_t offsetX;
            size_t strideX;
            size_t width;
            uint32_t subIndices[4] = {0, gridY, 0, 0};
            CHECK_DW_ERROR(dwDNNTensor_getLayoutView(&offsetX, &strideX, &width,
                                                     subIndices, 4U, 0U, outConf));
            for (uint32_t gridX = 0U; gridX < width; gridX += strideX)
            {
                float32_t conf = confData[offsetX + gridX * strideX];
                if (conf > COVERAGE_THRESHOLD && numBBoxes < m_maxDetections)
                {
                    // This is a detection!
                    float32_t imageX = (float32_t)gridX * (float32_t)m_cellSize;
                    float32_t imageY = (float32_t)gridY * (float32_t)m_cellSize;

                    size_t bboxOffset;
                    size_t bboxStride;
                    size_t numDimensions;
                    uint32_t bboxIndices[4] = {gridX, gridY, 0, 0};
                    CHECK_DW_ERROR(dwDNNTensor_getLayoutView(&bboxOffset, &bboxStride,
                                                             &numDimensions,
                                                             bboxIndices, 4U, 2U, outBBox));

                    float32_t* bboxOut = &(reinterpret_cast<float32_t*>(bboxData)[bboxOffset + 0 * bboxStride]);

                    float32_t boxX1;
                    float32_t boxY1;
                    float32_t boxX2;
                    float32_t boxY2;

                    dwDataConditioner_outputPositionToInput(&boxX1, &boxY1, bboxOut[0 * bboxStride] + imageX,
                                                            bboxOut[1 * bboxStride] + imageY, roi,
                                                            m_dataConditioner);
                    dwDataConditioner_outputPositionToInput(&boxX2, &boxY2,
                                                            bboxOut[2 * bboxStride] + imageX,
                                                            bboxOut[3 * bboxStride] + imageY, roi,
                                                            m_dataConditioner);
                    dwRectf bboxFloat{boxX1, boxY1, boxX2 - boxX1, boxY2 - boxY1};
                    dwBox2D bbox;
                    bbox.width  = static_cast<int32_t>(std::round(bboxFloat.width));
                    bbox.height = static_cast<int32_t>(std::round(bboxFloat.height));
                    bbox.x      = static_cast<int32_t>(std::round(bboxFloat.x));
                    bbox.y      = static_cast<int32_t>(std::round(bboxFloat.y));

                    m_detectedBoxList.push_back(bbox);
                    m_detectedBoxListFloat.push_back(bboxFloat);
                    numBBoxes++;
                }
            }
        }

        CHECK_DW_ERROR(dwDNNTensor_unlock(outConf));
        CHECK_DW_ERROR(dwDNNTensor_unlock(outBBox));
    }

    //------------------------------------------------------------------------------
    float32_t overlap(const dwRectf& boxA, const dwRectf& boxB)
    {

        int32_t overlapWidth = std::min(boxA.x + boxA.width,
                                        boxB.x + boxB.width) -
                               std::max(boxA.x, boxB.x);
        int32_t overlapHeight = std::min(boxA.y + boxA.height,
                                         boxB.y + boxB.height) -
                                std::max(boxA.y, boxB.y);

        return (overlapWidth < 0 || overlapHeight < 0) ? 0.0f : (overlapWidth * overlapHeight);
    }

    // ------------------------------------------------
    std::string getPlatformPrefix()
    {
        static const int32_t CUDA_AMPERE_MAJOR_COMPUTE_CAPABILITY           = 8;
        static const int32_t CUDA_TURING_VOLTA_MAJOR_COMPUTE_CAPABILITY     = 7;
        static const int32_t CUDA_VOLTA_DISCRETE_MINOR_COMPUTE_CAPABILITY   = 0;
        static const int32_t CUDA_VOLTA_INTEGRATED_MINOR_COMPUTE_CAPABILITY = 2;
        static const int32_t CUDA_TURING_DISCRETE_MINOR_COMPUTE_CAPABILITY  = 5;

        std::string path;
        int32_t currentGPU;
        cudaDeviceProp gpuProp{};

        CHECK_DW_ERROR(dwContext_getGPUDeviceCurrent(&currentGPU, m_sdk));
        CHECK_DW_ERROR(dwContext_getGPUProperties(&gpuProp, currentGPU, m_sdk));

        if (gpuProp.major == CUDA_AMPERE_MAJOR_COMPUTE_CAPABILITY)
        {
            if (gpuProp.integrated)
            {
                path = "ampere-integrated";
            }
            else
            {
                path = "ampere-discrete";
            }
        }
        else if (gpuProp.major == CUDA_TURING_VOLTA_MAJOR_COMPUTE_CAPABILITY)
        {
            if (gpuProp.minor == CUDA_TURING_DISCRETE_MINOR_COMPUTE_CAPABILITY)
            {
                path = "turing";
            }
            else if (gpuProp.minor == CUDA_VOLTA_INTEGRATED_MINOR_COMPUTE_CAPABILITY)
            {
                path = "volta-integrated";
            }
            else if (gpuProp.minor == CUDA_VOLTA_DISCRETE_MINOR_COMPUTE_CAPABILITY)
            {
                path = "volta-discrete";
            }
        }
        else
        {
            path = "pascal";
        }

        return path;
    }
};

int main(int argc, const char** argv)
{
    // -------------------
    // define all arguments used by the application
    ProgramArguments args(argc, argv,
                          {
#ifdef VIBRANTE
                              ProgramArguments::Option_t("camera-type", "ar0231-rccb-bae-sf3324", "camera gmsl type (see sample_sensors_info for all available camera types on this platform)"),
                              ProgramArguments::Option_t("camera-group", "a", "input port"),
                              ProgramArguments::Option_t("camera-index", "0", "camera index within the camera-group 0-3"),
                              ProgramArguments::Option_t("input-type", "video", "input type either video or camera"),
#endif
                              ProgramArguments::Option_t("video", (dw_samples::SamplesDataPath::get() + "/samples/sfm/triangulation/video_0.h264").c_str(), "path to video"),
                              ProgramArguments::Option_t("tensorRT_model", "", (std::string("path to TensorRT model file. By default: ") + dw_samples::SamplesDataPath::get() + "/samples/detector/<gpu-architecture>/tensorRT_model.bin").c_str())},
                          "DNN Tensor sample which detects and tracks cars.");

    DNNTensorSample app(args);
    app.initializeWindow("DNN Tensor Sample", 1280, 800, args.enabled("offscreen"));

    if (!args.enabled("offscreen"))
        app.setProcessRate(30);

    return app.run();
}
