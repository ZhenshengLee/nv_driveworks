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
// SPDX-FileCopyrightText: Copyright (c) 2017-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <dw/core/base/Version.h>
#include <framework/DriveWorksSample.hpp>

// Include all relevant DriveWorks modules

using namespace dw_samples::common;

//------------------------------------------------------------------------------
// Template of a sample. Put some description what the sample does here
//------------------------------------------------------------------------------
class MySample : public DriveWorksSample
{
private:
    // ------------------------------------------------
    // Driveworks Context and SAL
    // ------------------------------------------------
    dwContextHandle_t m_context           = DW_NULL_HANDLE;
    dwVisualizationContextHandle_t m_viz  = DW_NULL_HANDLE;
    dwRenderEngineHandle_t m_renderEngine = DW_NULL_HANDLE;
    dwSALHandle_t m_sal                   = DW_NULL_HANDLE;

    // ------------------------------------------------
    // Sample specific variables
    // ------------------------------------------------
    dwVector2f m_mousePoint              = {0.0f, 0.0f};
    dwRenderEngineColorRGBA m_mouseColor = {1.0f, 1.0f, 1.0f, 1.0f};

public:
    MySample(const ProgramArguments& args)
        : DriveWorksSample(args) {}

    /// -----------------------------
    /// Initialize Logger and DriveWorks context
    /// -----------------------------
    void initializeDriveWorks(dwContextHandle_t& context) const
    {
        // initialize logger to print verbose message on console in color
        CHECK_DW_ERROR(dwLogger_initialize(getConsoleLoggerCallback(true)));
        CHECK_DW_ERROR(dwLogger_setLogLevel(DW_LOG_VERBOSE));

        // initialize SDK context, using data folder
        dwContextParameters sdkParams = {};

#ifdef VIBRANTE
        sdkParams.eglDisplay = getEGLDisplay();
#endif

        CHECK_DW_ERROR(dwInitialize(&context, DW_VERSION, &sdkParams));
    }

    /// -----------------------------
    /// Initialize everything of a sample here incl. SDK components
    /// -----------------------------
    bool onInitialize() override
    {
        log("Starting my sample application...\n");

        // -----------------------------------------
        // Initialize DriveWorks context and SAL
        // -----------------------------------------
        initializeDriveWorks(m_context);
        CHECK_DW_ERROR(dwSAL_initialize(&m_sal, m_context));

        // -----------------------------
        // Initialize RenderEngine
        // -----------------------------
        CHECK_DW_ERROR(dwVisualizationInitialize(&m_viz, m_context));

        dwRenderEngineParams renderEngineParams{};
        CHECK_DW_ERROR(dwRenderEngine_initDefaultParams(&renderEngineParams,
                                                        getWindowWidth(),
                                                        getWindowHeight()));
        renderEngineParams.defaultTile.backgroundColor = {0.0f, 0.0f, 1.0f, 1.0f};
        CHECK_DW_ERROR_MSG(dwRenderEngine_initialize(&m_renderEngine, &renderEngineParams, m_viz),
                           "Cannot initialize Render Engine, maybe no GL context available?");
        return true;
    }

    ///------------------------------------------------------------------------------
    /// This method is executed when user presses `R`, it indicates that sample has to reset
    ///------------------------------------------------------------------------------
    void onReset() override
    {
        logWarn("My sample has been reset...\n");
        dwRenderEngine_reset(m_renderEngine);
    }

    ///------------------------------------------------------------------------------
    /// This method is executed on release, free up used memory here
    ///------------------------------------------------------------------------------
    void onRelease() override
    {
        if (m_renderEngine != DW_NULL_HANDLE)
        {
            CHECK_DW_ERROR(dwRenderEngine_release(m_renderEngine));
        }

        // -----------------------------------------
        // Release DriveWorks context and SAL
        // -----------------------------------------
        dwSAL_release(m_sal);
        dwVisualizationRelease(m_viz);
        dwRelease(m_context);
        dwLogger_release();
    }

    ///------------------------------------------------------------------------------
    /// Change renderer properties when main rendering window is resized
    ///------------------------------------------------------------------------------
    void onResizeWindow(int width, int height) override
    {
        dwRectf bounds{.x = 0.0f, .y = 0.0f};
        bounds.width  = width;
        bounds.height = height;
        dwRenderEngine_setBounds(bounds, m_renderEngine);

        log("window resized to %dx%d\n", width, height);
    }

    ///------------------------------------------------------------------------------
    /// Main processing of the sample
    ///     - this method is executed for window and console based applications
    ///------------------------------------------------------------------------------
    void onProcess() override
    {
        // this is called from mainloop
        // do some stuff here
    }

    ///------------------------------------------------------------------------------
    /// Render call of the sample, executed for window based applications only
    ///     - render text on screen
    ///------------------------------------------------------------------------------
    void onRender() override
    {
        // render text in the middle of the window
        dwRenderEngine_reset(m_renderEngine);
        dwRenderEngine_setTile(0, m_renderEngine);
        dwRenderEngine_setColor({1.0f, 1.0f, 1.0f, 1.0f}, m_renderEngine);
        dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_32, m_renderEngine);
        dwRectf viewport{};
        dwRenderEngine_getViewport(&viewport, m_renderEngine);
        dwRenderEngine_setCoordinateRange2D({viewport.width, viewport.height}, m_renderEngine);
        dwRenderEngine_renderText2D("Hello World",
                                    {viewport.width * 0.5f - 100.0f, viewport.height * 0.5f}, m_renderEngine);

        dwRenderEngine_setColor(m_mouseColor, m_renderEngine);
        dwRenderEngine_setPointSize(5.0f, m_renderEngine);
        dwRenderEngine_setFont(DW_RENDER_ENGINE_FONT_VERDANA_16, m_renderEngine);
        char mousePointLabel[128];
        snprintf(mousePointLabel, 128, "%.01f, %.01f", m_mousePoint.x, m_mousePoint.y);
        dwRenderEngine_renderWithLabel(DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_2D,
                                       &m_mousePoint,
                                       sizeof(dwVector2f),
                                       0,
                                       mousePointLabel,
                                       1,
                                       m_renderEngine);

        renderutils::renderFPS(m_renderEngine, getCurrentFPS());
    }

    ///------------------------------------------------------------------------------
    /// React to user inputs
    ///------------------------------------------------------------------------------
    void onKeyDown(int key, int /*scancode*/, int /*mods*/) override
    {
        log("key down: %d\n", key);
    }
    void onKeyUp(int key, int /*scancode*/, int /*mods*/) override
    {
        log("key up: %d\n", key);
    }
    void onKeyRepeat(int key, int /*scancode*/, int /*mods*/) override
    {
        log("key repeat: %d\n", key);
    }
    void onMouseDown(int button, float x, float y, int /*mods*/) override
    {
        m_mouseColor = {1.0f, 0.0f, 0.0f, 1.0f};
        log("mouse down %d at %fx%f\n", button, x, y);
    }
    void onMouseUp(int button, float x, float y, int /*mods*/) override
    {
        m_mouseColor = {1.0f, 1.0f, 1.0f, 1.0f};
        log("mouse up %d at %fx%f\n", button, x, y);
    }
    void onMouseMove(float x, float y) override
    {
        m_mousePoint = {x, y};
    }
    void onMouseWheel(float x, float y) override
    {
        m_mouseColor = {0.0f, 1.0f, 0.0f, 1.0f};
        log("mouse wheel press=%f, scroll=%f\n", x, y);
    }
};

//------------------------------------------------------------------------------
int main(int argc, const char** argv)
{
    // -------------------
    // define all arguments used by the application
    // parse user given arguments and bail out if there is --help request or proceed
    ProgramArguments args(argc, argv,
                          {ProgramArguments::Option_t("optionA",
                                                      (dw_samples::SamplesDataPath::get() + "/samples/path/to/data").c_str()),
                           ProgramArguments::Option_t("optionB", "default")},
                          "This is a message shown on console when sample prints help.");

    // -------------------
    // initialize and start a window application (with offscreen support if required)
    MySample app(args);

    app.initializeWindow("My Best Sample", 1280, 800, args.enabled("offscreen"));

    return app.run();
}
