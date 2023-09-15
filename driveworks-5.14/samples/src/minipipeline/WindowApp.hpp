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

#ifndef SMP_WINDOW_APP_HPP_
#define SMP_WINDOW_APP_HPP_

#include <framework/WindowGLFW.hpp>
#include <dwvisualization/core/Renderer.h>
#include <signal.h>

//------------------------------------------------------------------------------
// Variables
//------------------------------------------------------------------------------

extern void* gUserCallbackData;
extern void (*gUserKeyPressCallback)(void*, int, int, int, int);
extern void (*gUserMouseDownCallback)(void*, int, float, float, int);
extern void (*gUserMouseUpCallback)(void*, int, float, float, int);
extern void (*gUserMouseMoveCallback)(void*, float, float);
extern void (*gUserMouseWheelCallback)(void*, float, float);
extern void (*gUserToggleCameraViewMode)(void*);
extern WindowBase* gWindow;
extern bool gRun;

//------------------------------------------------------------------------------
// Functions
//------------------------------------------------------------------------------

// key press event
void keyPressCallback(int key);

// init window application
bool initWindowApp(bool fullScreen, bool offscreen, int32_t width = 1280, int32_t height = 800,
                   void* userCallbackData = nullptr,
                   void (*userKeyPressCallback)(void*, int, int, int, int)      = nullptr,
                   void (*userMouseDownCallback)(void*, int, float, float, int) = nullptr,
                   void (*userMouseUpCallback)(void*, int, float, float, int)   = nullptr,
                   void (*userMouseMoveCallback)(void*, float, float)  = nullptr,
                   void (*userMouseWheelCallback)(void*, float, float) = nullptr);

// release window application
void releaseWindowApp();

#endif // SMP_WINDOW_APP_HPP_
