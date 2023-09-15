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
// SPDX-FileCopyrightText: Copyright (c) 2016-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma GCC diagnostic ignored "-Wold-style-cast"

#include <signal.h>
#include <cstring> // for memset
#include <iostream>

#include "SampleFramework.hpp"

#include "WindowGLFW.hpp"

void (*gUserKeyPressCallback)(int) = 0;
ProgramArguments gArguments;
WindowBase* gWindow = nullptr;
bool gRun           = false;
bool gPause         = false;

void sig_int_handler(int sig)
{
    (void)sig;

    gRun = false;
}

void keyPressCallback(int key, int /*scancode*/, int action, int /*mods*/)
{
    if (action == GLFW_PRESS)
    {
        // stop application
        if (key == GLFW_KEY_ESCAPE)
            gRun = false;
        else if (key == GLFW_KEY_SPACE)
            gPause = !gPause;
    }

    if (gUserKeyPressCallback)
    {
        gUserKeyPressCallback(key);
    }
}

void drawBoxes(const std::vector<dwBox2D>& boxes, const std::vector<uint32_t>* boxIds,
               float32_t normalizationWidth, float32_t normalizationHeight,
               dwRenderBufferHandle_t renderBuffer, dwRendererHandle_t renderer)
{
    if (boxes.size() == 0)
        return;

    bool renderText = boxIds && boxIds->size() == boxes.size();
    dwRenderer_setFont(DW_RENDER_FONT_VERDANA_20, renderer);
    char idString[64];

    float32_t* coords     = nullptr;
    uint32_t maxVertices  = 0;
    uint32_t vertexStride = 0;
    dwRenderBuffer_map(&coords, &maxVertices, &vertexStride, renderBuffer);

    uint32_t n_boxes = static_cast<uint32_t>(boxes.size());
    uint32_t n_verts = 8 * n_boxes;
    if (n_verts > maxVertices)
    {
        n_boxes = maxVertices / 8;
        n_verts = 8 * n_boxes;
    }

    dwRect screenRectangle;
    dwRenderer_getRect(&screenRectangle, renderer);
    float32_t screenWidth  = static_cast<float32_t>(screenRectangle.width);
    float32_t screenHeight = static_cast<float32_t>(screenRectangle.height);

    for (uint32_t i = 0U; i < n_boxes; ++i)
    {
        // transform pixel coords into rendered rectangle
        float32_t x_start = static_cast<float32_t>(boxes[i].x) - 0.5f;
        float32_t y_start = static_cast<float32_t>(boxes[i].y) - 0.5f;
        float32_t x_end   = static_cast<float32_t>(boxes[i].width) + x_start;
        float32_t y_end   = static_cast<float32_t>(boxes[i].height) + y_start;

        coords[0] = x_start;
        coords[1] = y_start;
        coords += vertexStride;

        coords[0] = x_start;
        coords[1] = y_end;
        coords += vertexStride;

        coords[0] = x_start;
        coords[1] = y_end;
        coords += vertexStride;

        coords[0] = x_end;
        coords[1] = y_end;
        coords += vertexStride;

        coords[0] = x_end;
        coords[1] = y_end;
        coords += vertexStride;

        coords[0] = x_end;
        coords[1] = y_start;
        coords += vertexStride;

        coords[0] = x_end;
        coords[1] = y_start;
        coords += vertexStride;

        coords[0] = x_start;
        coords[1] = y_start;
        coords += vertexStride;

        if (renderText)
        {
#ifndef _MSC_VER
            sprintf(idString, "%u", (*boxIds)[i]);
#else
            sprintf_s(idString, "%u", (*boxIds)[i]);
#endif
            dwRenderer_renderText(static_cast<int32_t>((x_start * screenWidth) / normalizationWidth),
                                  screenRectangle.height -
                                      static_cast<int32_t>((y_start * screenHeight) / normalizationHeight),
                                  idString, renderer);
        }
    }
    dwRenderBuffer_unmap(n_verts, renderBuffer);
    dwRenderer_renderBuffer(renderBuffer, renderer);
}

void drawBoxesWithLabels(const std::vector<std::pair<dwBox2D, std::string>>& boxesWithLabels,
                         float32_t normalizationWidth, float32_t normalizationHeight,
                         dwRenderBufferHandle_t renderBuffer, dwRendererHandle_t renderer)
{
    if (boxesWithLabels.size() == 0)
        return;

    dwRenderer_setFont(DW_RENDER_FONT_VERDANA_20, renderer);

    float32_t* coords     = nullptr;
    uint32_t maxVertices  = 0;
    uint32_t vertexStride = 0;
    dwRenderBuffer_map(&coords, &maxVertices, &vertexStride, renderBuffer);

    uint32_t n_boxes = static_cast<uint32_t>(boxesWithLabels.size());
    uint32_t n_verts = 8 * n_boxes;
    if (n_verts > maxVertices)
    {
        n_boxes = maxVertices / 8;
        n_verts = 8 * n_boxes;
    }

    dwRect screenRectangle;
    dwRenderer_getRect(&screenRectangle, renderer);
    float32_t screenWidth  = static_cast<float32_t>(screenRectangle.width);
    float32_t screenHeight = static_cast<float32_t>(screenRectangle.height);

    for (uint32_t i = 0U; i < n_boxes; ++i)
    {
        std::pair<dwBox2D, std::string> boxClassIdPair = boxesWithLabels[i];
        dwBox2D& box           = boxClassIdPair.first;
        std::string classLabel = boxClassIdPair.second;
        // transform pixel coords into rendered rectangle
        float32_t x_start = static_cast<float32_t>(box.x) - 0.5f;
        float32_t y_start = static_cast<float32_t>(box.y) - 0.5f;
        float32_t x_end   = static_cast<float32_t>(box.width) + x_start;
        float32_t y_end   = static_cast<float32_t>(box.height) + y_start;

        coords[0] = x_start;
        coords[1] = y_start;
        coords += vertexStride;

        coords[0] = x_start;
        coords[1] = y_end;
        coords += vertexStride;

        coords[0] = x_start;
        coords[1] = y_end;
        coords += vertexStride;

        coords[0] = x_end;
        coords[1] = y_end;
        coords += vertexStride;

        coords[0] = x_end;
        coords[1] = y_end;
        coords += vertexStride;

        coords[0] = x_end;
        coords[1] = y_start;
        coords += vertexStride;

        coords[0] = x_end;
        coords[1] = y_start;
        coords += vertexStride;

        coords[0] = x_start;
        coords[1] = y_start;
        coords += vertexStride;

        dwRenderer_renderText(static_cast<int32_t>((x_start * screenWidth) / normalizationWidth),
                              screenRectangle.height -
                                  static_cast<int32_t>((y_start * screenHeight) / normalizationHeight),
                              classLabel.c_str(), renderer);
    }
    dwRenderBuffer_unmap(n_verts, renderBuffer);
    dwRenderer_renderBuffer(renderBuffer, renderer);
}

bool initSampleApp(int argc, const char** argv,
                   const ProgramArguments* arguments,
                   void (*userKeyPressCallback)(int),
                   uint32_t width, uint32_t height,
                   bool offscreen, int samples)
{
    gUserKeyPressCallback = userKeyPressCallback;

#if (!WINDOWS)
    struct sigaction action = {};
    action.sa_handler       = sig_int_handler;

    sigaction(SIGHUP, &action, NULL);  // controlling terminal closed, Ctrl-D
    sigaction(SIGINT, &action, NULL);  // Ctrl-C
    sigaction(SIGQUIT, &action, NULL); // Ctrl-\, clean quit with core dump
    sigaction(SIGABRT, &action, NULL); // abort() called.
    sigaction(SIGTERM, &action, NULL); // kill command
    sigaction(SIGSTOP, &action, NULL); // kill command
#endif

    if (arguments)
    {
        gArguments = *arguments;

        if (!gArguments.parse(argc, argv))
        {
            exit(1); // Exit if not all require arguments are provided
        }

        std::string argumentString = gArguments.printList();
        if (argumentString.size() > 0)
        {
            std::cout << "Program Arguments:\n"
                      << argumentString << std::endl;
        }
    }
    gRun = true;

    // Setup Window for Output, initialize the GL and GLFW
    try
    {
        gWindow = gWindow ? gWindow : WindowBase::create(width, height, offscreen, samples);
    }
    catch (const std::exception& /*ex*/)
    {
        gWindow = nullptr;
    }

    if (gWindow == nullptr)
        return false;

    gWindow->makeCurrent();
    gWindow->setOnKeypressCallback(keyPressCallback);

    return true;
}

void releaseSampleApp()
{
    // Shutdown
    delete gWindow;
    gWindow = nullptr;
}
