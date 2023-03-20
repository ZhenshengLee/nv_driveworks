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
// SPDX-FileCopyrightText: Copyright (c) 2015-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "MouseView3D.hpp"
#include <math.h>
#include <algorithm>
#include "MathUtils.hpp"
#include "Mat4.hpp"

MouseView3D::MouseView3D()
    : m_windowAspect(1.0f)
    , m_fovRads(DEG2RAD(60.0f))
    , m_zNear(0.1f)
    , m_zFar(10000.f)
    , m_radius(8)
    , m_mouseLeft(false)
    , m_mouseRight(false)
{
    m_center[0] = 0;
    m_center[1] = 0;
    m_center[2] = 0;

    m_up[0] = 0;
    m_up[1] = 0;
    m_up[2] = 1;

    m_angles[0] = DEG2RAD(180.0f);
    m_angles[1] = DEG2RAD(30.0f);

    m_currentPos[0] = -1.0f;
    m_currentPos[1] = -1.0f;

    updateEye();
    updateMatrices();
}

void MouseView3D::updateEye()
{
    m_eye[0] = m_radius * cos(m_angles[1]) * cos(m_angles[0]) + m_center[0];
    m_eye[1] = m_radius * cos(m_angles[1]) * sin(m_angles[0]) + m_center[1];
    m_eye[2] = m_radius * sin(m_angles[1]) + m_center[2];
}

void MouseView3D::mouseDown(int button, float32_t x, float32_t y)
{
    m_currentPos[0] = x;
    m_currentPos[1] = y;

    m_startAngles[0] = m_angles[0];
    m_startAngles[1] = m_angles[1];

    m_startCenter[0] = m_center[0];
    m_startCenter[1] = m_center[1];
    m_startCenter[2] = m_center[2];

    m_mouseLeft  = (button == 0);
    m_mouseRight = (button == 1);
}

void MouseView3D::mouseUp(int button, float32_t x, float32_t y)
{
    (void)button;
    (void)x;
    (void)y;
    m_mouseLeft  = false;
    m_mouseRight = false;
}

void MouseView3D::mouseMove(float32_t x, float32_t y)
{
    float32_t pos[] = {x, y};

    if (m_mouseLeft)
    {
        // update deltaAngle
        m_angles[0] = m_startAngles[0] - 0.01f * (pos[0] - m_currentPos[0]);
        m_angles[1] = m_startAngles[1] + 0.01f * (pos[1] - m_currentPos[1]);

        // Limit the vertical angle (-30 to 85 degrees)
        m_angles[1] = std::max(std::min(m_angles[1], DEG2RAD(85)), DEG2RAD(-30));

        updateEye();
        updateMatrices();
    }
    else if (m_mouseRight)
    {
        //Translation
        float32_t t[3];
        t[0] = 0.1f * (pos[0] - m_currentPos[0]);
        t[1] = 0;
        t[2] = 0.1f * (pos[1] - m_currentPos[1]);

        float32_t mt[3];
        Mat4_Rtxp(mt, m_modelView.array, t);

        m_center[0] = m_startCenter[0] + mt[0];
        m_center[1] = m_startCenter[1] + mt[1];
        m_center[2] = m_startCenter[2] + mt[2];

        updateEye();
        updateMatrices();
    }
}

void MouseView3D::mouseWheel(float32_t dx, float32_t dy)
{
    (void)dx;

    float32_t tmpRadius = m_radius - dy * 1.5f;

    setRadiusFromCenter(tmpRadius);
}

void MouseView3D::setWindowAspect(float32_t aspect)
{
    m_windowAspect = aspect;
    updateMatrices();
}

void MouseView3D::setCenter(float32_t x, float32_t y, float32_t z)
{
    m_center[0] = x;
    m_center[1] = y;
    m_center[2] = z;
    updateEye();
    updateMatrices();
}

void MouseView3D::setRadiusFromCenter(float32_t zoom)
{
    if (zoom > 0.0f)
    {
        m_radius = zoom;
        updateEye();
        updateMatrices();
    }
}

void MouseView3D::setAngleFromCenter(float32_t yaw, float32_t pitch)
{
    // Limit the pitch angle (-30 to 85 degrees)
    m_angles[0] = yaw;
    m_angles[1] = std::max(std::min(pitch, DEG2RAD(85)), DEG2RAD(-30));

    updateEye();
    updateMatrices();
}

void MouseView3D::updateMatrices()
{
    lookAt(m_modelView.array, m_eye, m_center, m_up);
    perspective(m_projection.array, m_fovRads, 1.0f * m_windowAspect, m_zNear, m_zFar);
}
