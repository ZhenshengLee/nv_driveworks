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
// SPDX-FileCopyrightText: Copyright (c) 2014-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// The ifdef and glfw3.h include are the same
// as in WindowGLFW.hpp. They are duplicated here
// because WindowGLFW undefines "Status", which is
// needed by the X11 includes below
#include <dw/core/display/EGL.h>
#include <dwvisualization/gl/GL.h>
#ifdef DW_USE_EGL
#define GLFW_INCLUDE_ES3
#ifndef VIBRANTE_V5Q
#ifndef GLFW_USE_WAYLAND
#define GLFW_EXPOSE_NATIVE_X11
#endif
#endif
#define GLFW_EXPOSE_NATIVE_EGL
#include <EGL/egl.h>
#endif

#include <GLFW/glfw3.h>

// These includes require the "Status" macro
// but there is no need to make them part of
// the header
#ifdef DW_USE_EGL
#include <GLFW/glfw3native.h>
#include <EGL/eglext.h>
#endif

#if defined(GLFW_EXPOSE_NATIVE_X11)
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xresource.h>
#include <X11/Xatom.h>
#endif

// Include this last because it undefines Status
#include "WindowGLFW.hpp"

#include <iostream>
#include <cstring>
#include <string>
#include <limits.h>

// ----------------------------------------------------------------------------------------------------
// check if the full screen override must be enabled
static int checkFullScreenOverride()
{
    bool overrideFullScreen = false;

    /*
       * Wayland backend causes buffer teardown when the requested resolution does not match
       * the primary monitor connected. Forcing full screen to match primary monitor resolution mode
       * only when wayland is enabled
       */
    FILE* fp;
    char buffer[8];
    std::string cmd = "loginctl show-session $(loginctl | awk '/tty/ {print $1}') -p Type | awk -F= '{print $2}'";
    std::string session;
    std::string op = "wayland";
    fp             = popen(cmd.c_str(), "r");
    if (fp)
    {
        while (!feof(fp))
            if (fgets(buffer, sizeof(buffer), fp) != NULL)
                session.append(buffer);
        pclose(fp);
    }

    /* force full screen if it is wayland session running */
    overrideFullScreen = (strncmp(session.c_str(), op.c_str(), 7) == 0) ? 1 : 0;

    return overrideFullScreen;
}

static void errorCallback(int error, const char* description)
{
    std::cout << "GLFW error: " << error << ", " << description << std::endl;
}

// -----------------------------------------------------------------------------
WindowGLFW::WindowGLFW(const char* title, int width, int height, bool offscreen, int samples,
                       bool initInvisible, bool fullScreen)
    : WindowBase(width, height)
    , m_offscreen(offscreen)
#ifdef DW_USE_EGL
    , m_display(EGL_NO_DISPLAY)
    , m_context(EGL_NO_CONTEXT)
#endif
{
    {
        glfwSetErrorCallback(&errorCallback);
        auto init_code = glfwInit();
        if (init_code == GLFW_FALSE)
        {
            if (!offscreen)
            {
                std::cout << "WindowGLFW: Failed to initialize GLFW" << std::endl;
            }
            throw std::exception();
        }
    }

    // Create a windowed mode window and its OpenGL context
    glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
    glfwWindowHint(GLFW_SAMPLES, samples); // 0 disables MSAA
    glfwWindowHint(GLFW_DEPTH_BITS, 24);   // Enable

    // GLFW_VISIBLE can be used to avoid showing blank window when app is doing initialization.
    if (offscreen || initInvisible)
    {
        // GLFW_VISIBLE specifies whether the windowed mode window will be initially visible.
        // Possible values are GLFW_TRUE and GLFW_FALSE. This hint is ignored for full screen windows.
        // reference: https://www.glfw.org/docs/latest/window_guide.html#window_hints
        glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
    }

#ifdef DW_USE_EGL
    glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_EGL_CONTEXT_API); // Enable EGL as context on Vibrante
#endif

#ifdef _GLESMODE
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
#else
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
#endif

    if (title == nullptr)
    {
        glfwWindowHint(GLFW_DECORATED, GL_FALSE);
        title = "";
    }

    if (!fullScreen)
        fullScreen = checkFullScreenOverride();

    if (fullScreen)
    {
        const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
        if (nullptr != mode)
        {
            m_width  = mode->width;
            m_height = mode->height;

            glfwWindowHint(GLFW_RED_BITS, mode->redBits);
            glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
            glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
            glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
        }
        else
        {
            m_width  = 1920;
            m_height = 1200;
        }

        // glfwCreateWindow function, the fourth param monitor which used for full screen mode, or NULL for windowed mode
        // at fullscreen mode, the visible hint is been ignored, and if offscreen to be true,
        // it will fail to set the window invisible, we will got a black window pop up, change to windowed mode to make offscreen mode clear.
        // reference: https://www.glfw.org/docs/latest/group__window.html#ga5c336fddf2cbb5b92f65f10fb6043344
        if (offscreen)
        {
            m_hWindow = glfwCreateWindow(m_width, m_height, title, NULL, NULL);
        }
        else
        {
            m_hWindow = glfwCreateWindow(m_width, m_height, title, glfwGetPrimaryMonitor(), NULL);
        }
    }
    else
    {
        m_hWindow = glfwCreateWindow(width, height, title, NULL, NULL);
    }

    if (!m_hWindow)
    {
        glfwTerminate();
        std::cout << "WindowGLFW: Failed create window" << std::endl;
        throw std::exception();
    }

    glfwMakeContextCurrent(m_hWindow);

#ifdef USE_GLEW
    // dwRenderer requires glewExperimental
    // because it calls glGenVertexArrays()
    glewExperimental = GL_TRUE;
    GLenum err       = glewInit();
    if (err != GLEW_OK)
    {
        glfwDestroyWindow(m_hWindow);
        glfwTerminate();
        std::cout << "WindowGLFW: Failed to init GLEW: " << glewGetErrorString(err) << std::endl;
        throw std::exception();
    }
    glGetError(); // clears error on init
#endif

    // No vsync
    glfwSwapInterval(0);

    glfwSetInputMode(m_hWindow, GLFW_STICKY_KEYS, GL_FALSE);

    //Callbacks
    glfwSetWindowUserPointer(m_hWindow, this);
    glfwSetKeyCallback(m_hWindow, [](GLFWwindow* win, int key, int scancode, int action, int mods) {
        WindowGLFW* window = reinterpret_cast<WindowGLFW*>(glfwGetWindowUserPointer(win));
        window->onKeyCallback(key, scancode, action, mods);
    });
    glfwSetMouseButtonCallback(m_hWindow, [](GLFWwindow* win, int button, int action, int mods) {
        WindowGLFW* window = reinterpret_cast<WindowGLFW*>(glfwGetWindowUserPointer(win));
        window->onMouseButtonCallback(button, action, mods);
    });
    glfwSetCharModsCallback(m_hWindow, [](GLFWwindow* win, uint32_t codepoint, int32_t mods) {
        WindowGLFW* window = reinterpret_cast<WindowGLFW*>(glfwGetWindowUserPointer(win));
        window->onCharModsCallback(codepoint, mods);
    });
    glfwSetCursorPosCallback(m_hWindow, [](GLFWwindow* win, double x, double y) {
        WindowGLFW* window = reinterpret_cast<WindowGLFW*>(glfwGetWindowUserPointer(win));
        window->onMouseMoveCallback(x, y);
    });
    glfwSetScrollCallback(m_hWindow, [](GLFWwindow* win, double dx, double dy) {
        WindowGLFW* window = reinterpret_cast<WindowGLFW*>(glfwGetWindowUserPointer(win));
        window->onMouseWheelCallback(dx, dy);
    });
    glfwSetFramebufferSizeCallback(m_hWindow, [](GLFWwindow* win, int width, int height) {
        WindowGLFW* window = reinterpret_cast<WindowGLFW*>(glfwGetWindowUserPointer(win));
        window->onResizeWindowCallback(width, height);
    });
    glfwSetDropCallback(m_hWindow, [](GLFWwindow* win, int count, const char** paths) {
        WindowGLFW* window = reinterpret_cast<WindowGLFW*>(glfwGetWindowUserPointer(win));
        window->onDropCallback(count, paths);
    });

#ifdef DW_USE_EGL
    m_display = glfwGetEGLDisplay();
    m_context = glfwGetEGLContext(m_hWindow);

    // Get configuration
    EGLint num_config;
    eglGetConfigs(m_display, nullptr, 0, &num_config);
    m_config.reset(new EGLConfig[num_config]);
    if (eglGetConfigs(m_display, m_config.get(), num_config, &num_config) == EGL_FALSE)
    {
        glfwTerminate();
        std::cout << "WindowGLFW: Failed to get configs" << std::endl;
        throw std::exception();
    }
#endif
}

// -----------------------------------------------------------------------------
WindowGLFW::~WindowGLFW(void)
{
    glfwDestroyWindow(m_hWindow);
    glfwTerminate();
}

// -----------------------------------------------------------------------------
EGLDisplay WindowGLFW::getEGLDisplay(void)
{
#ifdef DW_USE_EGL
    return m_display;
#else
    return 0;
#endif
}

// -----------------------------------------------------------------------------
EGLContext WindowGLFW::getEGLContext(void)
{
#ifdef DW_USE_EGL
    return m_context;
#else
    return 0;
#endif
}

// -----------------------------------------------------------------------------
/*virtual*/
void WindowGLFW::onKeyCallback(int key, int scancode, int action, int mods)
{
    if (action == GLFW_RELEASE && m_keyUpCallback)
        m_keyUpCallback(key, scancode, mods);
    else if (action == GLFW_PRESS && m_keyDownCallback)
        m_keyDownCallback(key, scancode, mods);
    else if (action == GLFW_REPEAT && m_keyRepeatCallback)
        m_keyRepeatCallback(key, scancode, mods);

    if (!m_keyCallback)
        return;

    m_keyCallback(key, scancode, action, mods);
}

// -----------------------------------------------------------------------------
/*virtual*/
void WindowGLFW::onMouseButtonCallback(int button, int action, int mods)
{
    double x, y;
    glfwGetCursorPos(m_hWindow, &x, &y);
    if (action == GLFW_PRESS)
    {
        if (!m_mouseDownCallback)
            return;
        m_mouseDownCallback(button, (float)x, (float)y, mods);
    }
    else if (action == GLFW_RELEASE)
    {
        if (!m_mouseUpCallback)
            return;
        m_mouseUpCallback(button, (float)x, (float)y, mods);
    }
}

// -----------------------------------------------------------------------------
/*virtual*/
void WindowGLFW::onMouseMoveCallback(double x, double y)
{
    if (!m_mouseMoveCallback)
        return;
    m_mouseMoveCallback((float)x, (float)y);
}

// -----------------------------------------------------------------------------
/*virtual*/
void WindowGLFW::onMouseWheelCallback(double dx, double dy)
{
    if (!m_mouseWheelCallback)
        return;
    m_mouseWheelCallback((float)dx, (float)dy);
}

// -----------------------------------------------------------------------------
/*virtual*/
void WindowGLFW::onCharModsCallback(uint32_t codepoint, int32_t mods)
{
    if (!m_charModsCallback)
        return;
    m_charModsCallback(codepoint, mods);
}

// -----------------------------------------------------------------------------
/*virtual*/
void WindowGLFW::onResizeWindowCallback(int width, int height)
{
    m_width  = width;
    m_height = height;

    if (!m_resizeWindowCallback)
        return;
    m_resizeWindowCallback(width, height);
}

// -----------------------------------------------------------------------------
/*virtual*/
void WindowGLFW::onDropCallback(int count, const char** paths)
{
    if (!m_dropCallback)
        return;
    m_dropCallback(count, paths);
}

// -----------------------------------------------------------------------------
bool WindowGLFW::swapBuffers(void)
{
    glfwPollEvents();
    swapBuffersOnly();
    return true;
}

// -----------------------------------------------------------------------------
bool WindowGLFW::swapBuffersOnly(void)
{
    glfwSwapBuffers(m_hWindow);
    return true;
}

// -----------------------------------------------------------------------------
bool WindowGLFW::releaseContext()
{
    glfwMakeContextCurrent(nullptr);
    return true;
}

// -----------------------------------------------------------------------------
void WindowGLFW::resetContext()
{
}

// -----------------------------------------------------------------------------
#ifdef DW_USE_EGL
EGLContext WindowGLFW::createSharedContext() const
{
    // -----------------------
    std::cout << "WindowGLFW: create shared EGL context" << std::endl;

    EGLint ctxAttribs[] = {
        EGL_CONTEXT_CLIENT_VERSION, 3,
        EGL_CONTEXT_OPENGL_ROBUST_ACCESS_EXT, EGL_FALSE,
        EGL_CONTEXT_OPENGL_RESET_NOTIFICATION_STRATEGY_EXT, EGL_NO_RESET_NOTIFICATION_EXT,
        EGL_NONE, EGL_NONE};

    EGLContext shared = eglCreateContext(m_display, *m_config.get(), m_context, ctxAttribs);

    if (shared == EGL_NO_CONTEXT)
    {
        std::cout << "WindowGLFW: Failed to create shared EGL context " << eglGetError() << std::endl;
        throw std::exception();
    }

    EGLBoolean status = eglMakeCurrent(m_display, EGL_NO_SURFACE, EGL_NO_SURFACE, shared);
    if (status != EGL_TRUE)
    {
        std::cout << "WindowGLFW: Failed to make shared EGL context current: " << eglGetError() << std::endl;
        throw std::exception();
    }
    return shared;
}
#else
GLFWwindow* WindowGLFW::createSharedContext() const
{
    GLFWwindow* shared = glfwCreateWindow(m_width, m_height, "", NULL, m_hWindow);
    glfwMakeContextCurrent(shared);
    return shared;
}
#endif

// -----------------------------------------------------------------------------
bool WindowGLFW::makeCurrent()
{
    // Make the window's context current
    glfwMakeContextCurrent(m_hWindow);

    return true;
}

// -----------------------------------------------------------------------------
bool WindowGLFW::resetCurrent()
{
    glfwMakeContextCurrent(nullptr);

    return true;
}

// -----------------------------------------------------------------------------
bool WindowGLFW::setWindowSize(int width, int height)
{
    // Set the window size
    glfwSetWindowSize(m_hWindow, width, height);

    // It will take some time to enter SetFramebufferSize Callback.
    // Preset the window size here so that width()&height() can get the expected value.
    // They will be re-writen in onResizeWindowCallback. This would be a safe change.
    m_width  = width;
    m_height = height;
    return true;
}

// -----------------------------------------------------------------------------
bool WindowGLFW::getDesktopResolution(int& width, int& height)
{
    const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());

    if (nullptr != mode)
    {
        width  = mode->width;
        height = mode->height;
        return true;
    }
    else
    {
        width  = 1280;
        height = 800;
        return false;
    }

    return true;
}

// -----------------------------------------------------------------------------
bool WindowGLFW::setFullScreen()
{
    const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());

#if defined(GLFW_EXPOSE_NATIVE_X11)
    Display* display;
    Atom actualType;
    int actualFormat;
    unsigned long itemCount, bytesAfter;
    Window* windowFromRoot = NULL;

    /* First connect to the display server */
    display = XOpenDisplay(NULL);
    if (!display)
    {
        setWindowSize(mode->width, mode->height);
        std::cout << "WindowGLFW: Failed to connect to the display server " << std::endl;
        return false;
    }

    // Check whether the window manager is running
    const Atom supportingWmCheck =
        XInternAtom(display, "_NET_SUPPORTING_WM_CHECK", False);
    XGetWindowProperty(display,
                       DefaultRootWindow(display),
                       supportingWmCheck,
                       0,
                       LONG_MAX,
                       False,
                       XA_WINDOW,
                       &actualType,
                       &actualFormat,
                       &itemCount,
                       &bytesAfter,
                       (unsigned char**)&windowFromRoot);

    if (actualType != XA_WINDOW)
    {
        setWindowSize(mode->width, mode->height);
        std::cout << "WindowGLFW: The window manager is NOT running" << std::endl;
        if (windowFromRoot)
            XFree(windowFromRoot);
        XCloseDisplay(display);
        return false;
    }

    if (windowFromRoot)
        XFree(windowFromRoot);
    XCloseDisplay(display);
    glfwSetWindowMonitor(m_hWindow, glfwGetPrimaryMonitor(), 0, 0, mode->width, mode->height, mode->refreshRate);
    m_width  = mode->width;
    m_height = mode->height;
    return true;
#else
    setWindowSize(mode->width, mode->height);
    return true;
#endif
}

// -----------------------------------------------------------------------------
bool WindowGLFW::setWindowPosCentered()
{
    int width, height;
    const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
    glfwGetWindowSize(m_hWindow, &width, &height);

    if (width > mode->width || height > mode->height)
    {
        return true;
    }

    glfwSetWindowPos(m_hWindow, (mode->width - width) / 2, (mode->height - height) / 2);
    return true;
}

// -----------------------------------------------------------------------------
bool WindowGLFW::setWindowVisibility(bool visible)
{
    if (visible)
    {
        glfwShowWindow(m_hWindow);
    }
    else
    {
        glfwHideWindow(m_hWindow);
    }

    return true;
}

// -----------------------------------------------------------------------------
bool WindowGLFW::setWindowTitle(const char* title)
{
    if (title)
    {
        glfwSetWindowTitle(m_hWindow, title);
        return true;
    }

    return false;
}
