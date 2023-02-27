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
// SPDX-FileCopyrightText: Copyright (c) 2016-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * <b>NVIDIA DriveWorks API: GL Methods</b>
 *
 * @b Description: This file defines the OpenGL methods of the SDK.
 */

/**
 * @defgroup gl_group OpenGL Interface
 *
 * Defines the OpenGL methods of the SDK.
 * @{
 */

#ifndef DWGL_GL_H_
#define DWGL_GL_H_

#include <dw/core/base/Config.h>

// clang-format off
#ifdef DW_USE_EGL
    #include <GLES3/gl3.h>
    #include <GLES3/gl31.h>
    #include <GLES3/gl32.h> // needed for glFramebufferTexture()
    #include <GLES2/gl2ext.h>
    #define _GLESMODE
#else
    // On non GLES platforms we will use GLEW
    #ifndef USE_GLEW
    #define USE_GLEW
    #endif

    #ifndef GLEW_STATIC
    #define GLEW_STATIC // We will always include GLEW as static
    #endif
    #include <GL/glew.h>
    #include <GL/glu.h>
#endif
// clang-format on

#endif // DWGL_GL_H_

/** @} */
