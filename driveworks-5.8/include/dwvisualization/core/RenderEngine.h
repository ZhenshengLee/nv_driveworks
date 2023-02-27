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

/**
 * @file
 * <b>NVIDIA DriveWorks API: Render Engine</b>
 *
 * @b Description: This file defines the C-style interface for Render Engine.
 */

/**
 * @defgroup render_engine_group Render Engine Interface
 *
 * @brief Defines RenderEngine related types, structs, and functions.
 *
 * @{
 * @ingroup vizualization_group
 */
#ifndef DWVISUALIZATION_RENDERENGINE_H_
#define DWVISUALIZATION_RENDERENGINE_H_

#include "Renderer.h"

#include <dwvisualization/image/Image.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Handle for the render engine.
typedef struct dwRenderEngineObject* dwRenderEngineHandle_t;
/// Const handle for the render engine.
typedef const struct dwRenderEngineObject* dwConstRenderEngineHandle_t;

/// The font types for the render engine.
typedef enum {
    DW_RENDER_ENGINE_FONT_VERDANA_8  = DW_RENDER_FONT_VERDANA_8,
    DW_RENDER_ENGINE_FONT_VERDANA_12 = DW_RENDER_FONT_VERDANA_12,
    DW_RENDER_ENGINE_FONT_VERDANA_16 = DW_RENDER_FONT_VERDANA_16,
    DW_RENDER_ENGINE_FONT_VERDANA_20 = DW_RENDER_FONT_VERDANA_20,
    DW_RENDER_ENGINE_FONT_VERDANA_24 = DW_RENDER_FONT_VERDANA_24,
    DW_RENDER_ENGINE_FONT_VERDANA_32 = DW_RENDER_FONT_VERDANA_32,
    DW_RENDER_ENGINE_FONT_VERDANA_48 = DW_RENDER_FONT_VERDANA_48,
    DW_RENDER_ENGINE_FONT_VERDANA_64 = DW_RENDER_FONT_VERDANA_64,
    DW_RENDER_ENGINE_FONT_COUNT      = DW_RENDER_FONT_COUNT
} dwRenderEngineFont;

/// Maximum number of tiles that can be added
const uint32_t DW_RENDER_ENGINE_MAX_TILE_COUNT = 55;

/// When rendering 2D data the coordinate system is always from
/// 0 to 1, unless changed by calling dwRenderEngine_setCoordinateRange2D.
/// When rendering 3D data the coordinate system is NDC.
typedef enum {
    /// Interleaved is x,y,x,y
    /// Min vertex size is 2 floats
    DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_2D,
    /// Interleaved is x,y,z,x,y,z
    /// Min vertex size is 3 floats
    DW_RENDER_ENGINE_PRIMITIVE_TYPE_POINTS_3D,
    /// Interleaved is x,y,x,y
    /// Min vertex size is 2 floats
    DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINES_2D,
    /// Interleaved is x,y,z,x,y,z
    /// Min vertex size is 3 floats
    DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINES_3D,
    /// Interleaved is x,y,x,y
    /// Min vertex size is 2 floats
    DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINESTRIP_2D,
    /// Interleaved is x,y,z,x,y,z
    /// Min vertex size is 3 floats
    DW_RENDER_ENGINE_PRIMITIVE_TYPE_LINESTRIP_3D,
    /// Interleaved is x,y,x,y
    /// Min vertex size is 2 floats
    DW_RENDER_ENGINE_PRIMITIVE_TYPE_TRIANGLES_2D,
    /// Interleaved is x,y,z,x,y,z
    /// Min vertex size is 3 floats
    DW_RENDER_ENGINE_PRIMITIVE_TYPE_TRIANGLES_3D,
    /// Interleaved is x,y,width,height,x,y,width,height
    /// Min vertex size is 4 floats
    DW_RENDER_ENGINE_PRIMITIVE_TYPE_BOXES_2D,
    /// Interleaved is x,y,z,width,height,depth,x,y,z,width,height,depth
    /// Min vertex size is 6 floats
    DW_RENDER_ENGINE_PRIMITIVE_TYPE_BOXES_3D,
    /// Interleaved is 8 sets of (x,y,z) representing the 8 vertices of an oriented box
    /// Min vertex size is 24 floats
    DW_RENDER_ENGINE_PRIMITIVE_TYPE_ORIENTED_BOXES_3D,
    /// Interleaved is center x,y,x radius,y radius,x,y,x radius,y radius
    /// Min vertex size is 4 floats
    DW_RENDER_ENGINE_PRIMITIVE_TYPE_ELLIPSES_2D,
    /// Interleaved is center x,y,z,x radius,y radius
    /// Min vertex size is 5 floats
    DW_RENDER_ENGINE_PRIMITIVE_TYPE_ELLIPSES_3D,
    /// Interleaved is start x,start y,end y,end x
    /// Min vertex size is 4 floats
    DW_RENDER_ENGINE_PRIMITIVE_TYPE_ARROWS_2D,
    /// Interleaved is start x,start y,start z,end y,end x,end z
    /// Min vertex size is 6 floats
    DW_RENDER_ENGINE_PRIMITIVE_TYPE_ARROWS_3D,

    DW_RENDER_ENGINE_PRIMITIVE_TYPE_COUNT
} dwRenderEnginePrimitiveType;

/// @brief An enum that controls the type of plot to render
/// when calling dwRenderEngine_renderPlot2D.
typedef enum {
    /// Render plot as point values. Data layout is x,y,x,y.
    DW_RENDER_ENGINE_PLOT_TYPE_POINTS,
    /// Render plot as line strip. Data layout is x,y,x,y.
    DW_RENDER_ENGINE_PLOT_TYPE_LINESTRIP,
    /// Render plot as histogram. Data layout is y,y,y,y.
    DW_RENDER_ENGINE_PLOT_TYPE_HISTOGRAM
} dwRenderEnginePlotType;

/// Maximum number of plots that can be rendered with a single call of dwRenderEngine_renderPlots2D()
#define DW_RENDER_ENGINE_MAX_PLOTS2D 20

/// @brief RGBA render color.
typedef dwVector4f dwRenderEngineColorRGBA;

static const dwRenderEngineColorRGBA DW_RENDER_ENGINE_COLOR_RED         = {230.0f / 255.0f, 72.0f / 255.0f, 32.0f / 255.0f, 1.0f};
static const dwRenderEngineColorRGBA DW_RENDER_ENGINE_COLOR_DARKRED     = {180.0f / 255.0f, 5.0f / 255.0f, 0.0f / 255.0f, 1.0f};
static const dwRenderEngineColorRGBA DW_RENDER_ENGINE_COLOR_PURPLE      = {255.0f / 255.0f, 0.0f / 255.0f, 240.0f / 255.0f, 1.0f};
static const dwRenderEngineColorRGBA DW_RENDER_ENGINE_COLOR_BLUE        = {32.0f / 255.0f, 72.0f / 255.0f, 230.0f / 255.0f, 1.0f};
static const dwRenderEngineColorRGBA DW_RENDER_ENGINE_COLOR_GREEN       = {32.0f / 255.0f, 230.0f / 255.0f, 32.0f / 255.0f, 1.0f};
static const dwRenderEngineColorRGBA DW_RENDER_ENGINE_COLOR_WHITE       = {255.0f / 255.0f, 255.0f / 255.0f, 255.0f / 255.0f, 1.0f};
static const dwRenderEngineColorRGBA DW_RENDER_ENGINE_COLOR_DARKGREEN   = {45.0f / 255.0f, 100.0f / 255.0f, 44.0f / 255.0f, 1.0f};
static const dwRenderEngineColorRGBA DW_RENDER_ENGINE_COLOR_LIGHTGREEN  = {118.0f / 255.0f, 185.0f / 255.0f, 0.0f / 255.0f, 1.0f};
static const dwRenderEngineColorRGBA DW_RENDER_ENGINE_COLOR_LIGHTGREY   = {185.0f / 255.0f, 185.0f / 255.0f, 185.0f / 255.0f, 1.0f};
static const dwRenderEngineColorRGBA DW_RENDER_ENGINE_COLOR_DARKGREY    = {72.0f / 255.0f, 72.0f / 255.0f, 72.0f / 255.0f, 1.0f};
static const dwRenderEngineColorRGBA DW_RENDER_ENGINE_COLOR_DARKBLUE    = {45.0f / 255.0f, 44.0f / 255.0f, 100.0f / 255.0f, 0.6f};
static const dwRenderEngineColorRGBA DW_RENDER_ENGINE_COLOR_YELLOW      = {230.0f / 255.0f, 230.0f / 255.0f, 10.0f / 255.0f, 1.0f};
static const dwRenderEngineColorRGBA DW_RENDER_ENGINE_COLOR_ORANGE      = {230.0f / 255.0f, 100.0f / 255.0f, 10.0f / 255.0f, 1.0f};
static const dwRenderEngineColorRGBA DW_RENDER_ENGINE_COLOR_LIGHTBLUE   = {51.0f / 255.0f, 153.0f / 255.0f, 255.0f / 255.0f, 0.6f};
static const dwRenderEngineColorRGBA DW_RENDER_ENGINE_COLOR_BLACK       = {0.0f / 255.0f, 0.0f / 255.0f, 0.0f / 255.0f, 1.0f};
static const dwRenderEngineColorRGBA DW_RENDER_ENGINE_COLOR_PINK        = {1.0f, 113.0f / 255.0f, 181.0f / 255.0f, 1.0f};
static const dwRenderEngineColorRGBA DW_RENDER_ENGINE_COLOR_CYAN        = {0.0f, 1.0f, 1.0f, 1.0f};
static const dwRenderEngineColorRGBA DW_RENDER_ENGINE_COLOR_DARKBROWN   = {101.0f / 255.0f, 67.0f / 255.0f, 33.0f / 255.0f, 1.0f};
static const dwRenderEngineColorRGBA DW_RENDER_ENGINE_COLOR_DARKPURPLE  = {70.0f / 255.0f, 39.0f / 255.0f, 89.0f / 255.0f, 1.0f};
static const dwRenderEngineColorRGBA DW_RENDER_ENGINE_COLOR_BITTERSWEET = {1.0f, 112.0f / 255.0f, 94.0f / 255.0f, 1.0f};

/// @brief An enum that controls how primitive colors are rendered.
/// @see dwRenderEngine_setColorByValue
typedef enum {
    /// Render according to tile current color
    DW_RENDER_ENGINE_COLOR_BY_VALUE_COLOR,
    /// Render the hue by distance to origin
    DW_RENDER_ENGINE_COLOR_BY_VALUE_MODE_DISTANCE,
    /// Render the hue by distance in x direction
    DW_RENDER_ENGINE_COLOR_BY_VALUE_MODE_X,
    /// Render the hue by distance in y direction
    DW_RENDER_ENGINE_COLOR_BY_VALUE_MODE_Y,
    /// Render the hue by distance in z direction
    DW_RENDER_ENGINE_COLOR_BY_VALUE_MODE_Z,
    /// Render the hue by distance in xy directions
    DW_RENDER_ENGINE_COLOR_BY_VALUE_MODE_XY,
    /// Render the hue by distance in xz directions
    DW_RENDER_ENGINE_COLOR_BY_VALUE_MODE_XZ,
    /// Render the hue by distance in yz directions
    DW_RENDER_ENGINE_COLOR_BY_VALUE_MODE_YZ,
    /// Render the hue by intensity value
    /// (Last value of primitives. For example,
    /// if rendering 2D boxes with color per box
    /// data layout is x,y,width,height,intensity)
    DW_RENDER_ENGINE_COLOR_BY_VALUE_MODE_INTENSITY,
    /// Render the color with the primitive interleaved
    /// This requires the data have RGBA values interleaved
    /// with the render buffer data. For example, 2D points
    /// using this option would be passed in as:
    /// x1,y1,r1,g1,b1,a1,x2,y2,r2,g2,b2,a2, etc.
    DW_RENDER_ENGINE_COLOR_BY_VALUE_MODE_ATTRIBUTE_RGBA
} dwRenderEngineColorByValueMode;

/// @brief The layout indicator for size and position
/// of each tile.
typedef enum {
    /// Layout the position or size in a fixed position
    /// and size.
    /// This means viewport coordinates should be
    /// in terms of absolute window pixels.
    DW_RENDER_ENGINE_TILE_LAYOUT_TYPE_ABSOLUTE,
    /// Layout the position or size relative to the bounds
    /// of the render engine.
    /// This means viewport coordinates should be
    /// values from 0.0 to 1.0.
    DW_RENDER_ENGINE_TILE_LAYOUT_TYPE_RELATIVE
} dwRenderEngineTileLayoutType;

/// @brief The layout indicator for position
/// of each tile.
typedef enum {
    /// Layout the position relative to the top left
    /// of the bounds.
    /// x and y positions will be offsets from the top
    /// left (x going right, y going down).
    /// Anchor point is top left of tile.
    DW_RENDER_ENGINE_TILE_POSITION_TYPE_TOP_LEFT,
    /// Layout the position relative to the top
    /// of the bounds.
    /// x and y positions will be offsets from the top
    /// center (x going right, y going down).
    /// Anchor point is top center of tile.
    DW_RENDER_ENGINE_TILE_POSITION_TYPE_TOP_CENTER,
    /// Layout the position relative to the top right
    /// of the bounds.
    /// x and y positions will be offsets from the top
    /// right (x going left, y going down).
    /// Anchor point is top right of tile.
    DW_RENDER_ENGINE_TILE_POSITION_TYPE_TOP_RIGHT,
    /// Layout the position relative to the bottom left
    /// of the bounds.
    /// x and y positions will be offsets from the bottom
    /// left (x going right, y going up).
    /// Anchor point is bottom left of tile.
    DW_RENDER_ENGINE_TILE_POSITION_TYPE_BOTTOM_LEFT,
    /// Layout the position relative to the bottom
    /// of the bounds.
    /// x and y positions will be offsets from the bottom
    /// center (x going right, y going up).
    /// Anchor point is bottom center of tile.
    DW_RENDER_ENGINE_TILE_POSITION_TYPE_BOTTOM_CENTER,
    /// Layout the position relative to the bottom right
    /// of the bounds.
    /// x and y positions will be offsets from the bottom
    /// right (x going left, y going up).
    /// Anchor point is bottom right of tile.
    DW_RENDER_ENGINE_TILE_POSITION_TYPE_BOTTOM_RIGHT,
    /// Layout the position relative to the left
    /// of the bounds.
    /// x and y positions will be offsets from the left
    /// center (x going right, y going down).
    /// Anchor point is center left of tile.
    DW_RENDER_ENGINE_TILE_POSITION_TYPE_CENTER_LEFT,
    /// Layout the position relative to the right
    /// of the bounds.
    /// x and y positions will be offsets from the right
    /// center (x going left, y going down).
    /// Anchor point is center right of tile.
    DW_RENDER_ENGINE_TILE_POSITION_TYPE_CENTER_RIGHT,
    /// Layout the position relative to the center
    /// of the bounds.
    /// x and y positions will be offsets from the center
    /// (x going right, y going down).
    /// Anchor point is center of tile.
    DW_RENDER_ENGINE_TILE_POSITION_TYPE_CENTER,
    /// Layout the position in a grid. This enum
    /// should never be chosen but is used internally
    /// when calling dwRenderEngine_addTilesByCount.
    DW_RENDER_ENGINE_TILE_POSITION_TYPE_GRID
} dwRenderEngineTilePositionType;

/// @brief The tile layout for the viewport.
typedef struct dwRenderEngineTileLayout
{
    /// The layout indicator for size.
    dwRenderEngineTileLayoutType sizeLayout;
    /// The layout indicator for position.
    dwRenderEngineTileLayoutType positionLayout;
    /// The position type that determines how
    /// the position coordinates are interpreted.
    dwRenderEngineTilePositionType positionType;
    /// The viewport for the tile.
    dwRectf viewport;
    /// The aspect ratio for the tile.
    float32_t aspectRatio;
    /// Use aspect ratio.
    bool useAspectRatio;
} dwRenderEngineTileLayout;

/// @brief The state for a render engine tile.
typedef struct dwRenderEngineTileState
{
    /// The layout which includes the viewport.
    dwRenderEngineTileLayout layout;
    /// The model view matrix.
    dwMatrix4f modelViewMatrix;
    /// The projection matrix for the camera.
    dwMatrix4f projectionMatrix;
    /// The background color
    dwRenderEngineColorRGBA backgroundColor;
    /// Color of the rendering primitives in the tile.
    dwRenderEngineColorRGBA color;
    /// Color by value mode.
    dwRenderEngineColorByValueMode colorByValueMode;
    /// The line width.
    float32_t lineWidth;
    /// The point size.
    float32_t pointSize;
    /// The font.
    dwRenderEngineFont font;
    /// The normalization factor for 2D rendering
    dwVector2f coordinateRange2D;
} dwRenderEngineTileState;

/// @brief The initialization parameters for a render engine.
typedef struct dwRenderEngineParams
{
    /// The default display bounds.
    dwRectf bounds;
    /// The default tile.
    dwRenderEngineTileState defaultTile;
    /// Default buffer size for rendering primitives in bytes.
    /// This buffer size is used to allocate an internal
    /// GPU buffer which is then used when rendering data.
    /// Any data that is passed into dwRenderEngine_render
    /// will be copied directly to the GPU. This means
    /// that if you choose to have an array of structs
    /// but only the first two entries in that struct
    /// represent the data to be rendered, you can
    /// pass it directly to render without massaging
    /// the data but your initial buffer size must
    /// be large enough to handle it.
    uint32_t bufferSize;
    /// Maximum static buffer count.
    /// This is used for dwRenderEngine_addBuffer and
    /// dwRenderEngine_removeBuffer. It allocates
    /// buffers for static rendering. Internally,
    /// the default bufferId (0) always exists and
    /// any extra buffers are allocated by this amount.
    uint32_t maxBufferCount;
} dwRenderEngineParams;

/**
 * @brief Initialize params to default.
 * @param params The parameters.
 * @param width The raster width.
 * @param height The raster height.
 * @return  DW_SUCCESS When successfully initialized parameters.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_initDefaultParams(dwRenderEngineParams* params,
                                          uint32_t width, uint32_t height);

/**
 * @brief Initializes a render engine handle.
 * @param engine A pointer to a render engine handle.
 * @param params The initialization params.
 * @param context The dw context.
 * @return DW_SUCCESS When successfully initialized.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_initialize(dwRenderEngineHandle_t* engine,
                                   const dwRenderEngineParams* params,
                                   dwVisualizationContextHandle_t context);

/**
 * @brief Releases the render engine handle.
 * @param engine The render engine handle to be released.
 * @return DW_SUCCESS When successfully released.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_release(dwRenderEngineHandle_t engine);

/**
 * @brief Resets the state of all render tiles and clears
 * the display. Sets the current render tile to the default render tile 0.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successfully reset.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_reset(dwRenderEngineHandle_t engine);

/**
 * @brief Resets the state of the current tile and clears
 * the display.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successfully reset.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_resetTile(dwRenderEngineHandle_t engine);

/**
 * @brief Adds a render tile to the render engine.
 * By default,the render engine always has 1 tile (tileId == 0) which is
 * defined by the initialization params of the render engine.
 * @param tileId The output tile id that is added.
 * @param params The tile params to define the settings of the tile.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successfully added a render tile.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_addTile(uint32_t* tileId,
                                const dwRenderEngineTileState* params,
                                dwRenderEngineHandle_t engine);

/**
 * @brief Initialize tile params to default state.
 * @param params The tile parameters.
 * @return  DW_SUCCESS When successfully initialized parameters.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_initTileState(dwRenderEngineTileState* params);

/**
 * @brief Removes a render tile from the render engine.
 * @param tileId The id of the tile to remove.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successfully removed render tile.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_removeTile(uint32_t tileId,
                                   dwRenderEngineHandle_t engine);

/**
 * @brief Adds two render tiles to the render
 * engine that each take up half of the render area (defined by bounds).
 * @param firstTileId The id of the first tile added (either left or top depending
 * on whether or not horizontal option is chosen).
 * @param secondTileId The id of the second tile (either right or bottom).
 * @param horizontal Controls the layout of the two tiles. If true,
 * the tiles are laid out in horizontal rows (top to bottom). If false,the
 * tiles are laid out left to right.
 * @param firstParams The tile parameters for the first tile.
 * @param secondParams The tile parameters for the second tile.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successfully added split screen render tiles.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_addSplitScreenTiles(uint32_t* firstTileId,
                                            uint32_t* secondTileId,
                                            bool horizontal,
                                            const dwRenderEngineTileState* firstParams,
                                            const dwRenderEngineTileState* secondParams,
                                            dwRenderEngineHandle_t engine);

/**
 * @brief Evenly add render tiles to the screen from left
 * to right. If tiles,do not fill up the space of the render area evenly,the last row
 * of the render tiles will not fill the entire row (that's where the left over tiles get placed).
 * @param outTileIds An array of size tileCount that is populated by ids of the newly added
 * render tiles.
 * @param tileCount The number of requested tiles to add.
 * @param maxPerRow The max count of tiles per row.
 * @param paramsList An array of params with size tileCount for each of the new tiles.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successfully added screen render tiles.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_addTilesByCount(uint32_t* outTileIds,
                                        uint32_t tileCount,
                                        uint32_t maxPerRow,
                                        const dwRenderEngineTileState* paramsList,
                                        dwRenderEngineHandle_t engine);

/**
 * @brief Gets the current render tile id.
 * @param tileId Pointer location of where the function will populate the current
 * tile id.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successfully returned the render tile id.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_getTile(uint32_t* tileId,
                                dwConstRenderEngineHandle_t engine);

/**
 * @brief Sets the state of the engine to be a particular tile.
 * All render engine calls that follow will be bound to this tile.
 * @param tileId The id of the tile.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successfully set the render tile.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_setTile(uint32_t tileId,
                                dwRenderEngineHandle_t engine);

/**
 * @brief Get the current tile state parameters.
 * @param state Pointer location with which to populate the tile state.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_getState(dwRenderEngineTileState* state,
                                 dwConstRenderEngineHandle_t engine);

/**
 * @brief Set the current tile state parameters.
 * @param state The state of the tile.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_setState(const dwRenderEngineTileState* state,
                                 dwRenderEngineHandle_t engine);

/**
 * @brief Gets the model view matrix for the current tile.
 * If no tile has been set (@see dwRenderEngine_setTile) it will deliver the
 * model view matrix for the default tile (tileId == 0).
 * @param modelViewMatrix A pointer to the model view matrix of the tile to be filled
 * by this function.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_getModelView(dwMatrix4f* modelViewMatrix,
                                     dwConstRenderEngineHandle_t engine);

/**
 * @brief Sets the model view matrix for the current tile.
 * If no tile has been set (@see dwRenderEngine_setTile) it will use the
 * model view matrix for the default tile (tileId == 0).
 * @param modelViewMatrix A pointer to the model view matrix of the tile.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_setModelView(const dwMatrix4f* modelViewMatrix,
                                     dwRenderEngineHandle_t engine);

/**
 * @brief Gets the projection matrix for the current tile.
 * If no tile has been set (@see dwRenderEngine_setTile) it will deliver the
 * projection matrix for the default tile (tileId == 0).
 * @param projectionMatrix A pointer to the projection matrix of the tile to be filled
 * by this function.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_getProjection(dwMatrix4f* projectionMatrix,
                                      dwConstRenderEngineHandle_t engine);

/**
 * @brief Sets the projection matrix for the current tile.
 * If no tile has been set (@see dwRenderEngine_setTile) it will use the
 * projection matrix for the default tile (tileId == 0).
 * @param projectionMatrix A pointer to the projection matrix of the tile.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_setProjection(const dwMatrix4f* projectionMatrix,
                                      dwRenderEngineHandle_t engine);

/**
 * @brief Sets the model view matrix for the current tile.
 * @param eye The eye vector.
 * @param center The center vector.
 * @param up The up vector.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_setLookAt(dwVector3f eye,
                                  dwVector3f center,
                                  dwVector3f up,
                                  dwRenderEngineHandle_t engine);

/**
 * @brief Defines the camera position on a sphere surface around
 * a center point, with the camera looking towards that center point.
 * The position on the sphere surface is defined by 2 angles and a distance.
 * Defines the look at eye vector by angles and distance
 * with the following equation:
 *         dwVector3f eye{};
 *         eye.x     = distance * cos(yAngleRadians) * cos(xAngleRadians);
 *         eye.y     = distance * cos(yAngleRadians) * sin(xAngleRadians);
 *         eye.z     = distance * sin(yAngleRadians);
 * @param xAngleRadians The x angle in radians for the eye vector.
 * @param yAngleRadians The y angle in radians for the eye vector.
 * @param distance The distance from the center for the eye vector.
 * @param center The center vector.
 * @param up The up vector.
 * @param engine The render engine.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_setLookAtByAngles(float32_t xAngleRadians,
                                          float32_t yAngleRadians,
                                          float32_t distance,
                                          dwVector3f center,
                                          dwVector3f up,
                                          dwRenderEngineHandle_t engine);

/**
 * @brief Sets up an orthographic projection for the current tile.
 * @param left The left of the ortho projection.
 * @param right The right of the ortho projection.
 * @param bottom The bottom of the ortho projection.
 * @param top The top of the ortho projection.
 * @param near The near plane.
 * @param far The far plane.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_setOrthoProjection(float32_t left, float32_t right,
                                           float32_t bottom, float32_t top,
                                           float32_t near, float32_t far,
                                           dwRenderEngineHandle_t engine);

/**
 * @brief Sets the projection matrix for the current tile
 * as defined by the parameters.
 * @param fovRadsY The field of view in the y direction in radians.
 * @param aspect The aspect ratio.
 * @param near The near threshold.
 * @param far The far threshold.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_setPerspectiveProjection(float32_t fovRadsY,
                                                 float32_t aspect,
                                                 float32_t near,
                                                 float32_t far,
                                                 dwRenderEngineHandle_t engine);

/**
 * @brief Gets the user-defined viewport for the current tile. This differs
 * from the render viewport because it is user defined and is used in
 * conjunction with the layout parameters to generate the render viewport.
 * If no tile has been set (@see dwRenderEngine_setTile) it will deliver the
 * viewport for the default tile (tileId == 0).
 * @param viewport A pointer to the viewport of the tile to be filled
 * by this function.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_getViewport(dwRectf* viewport,
                                    dwConstRenderEngineHandle_t engine);

/**
 * @brief Gets the render viewport for the current tile. The render viewport
 * is set internally and cannot be set externally. It is the actual pixel
 * viewport that is used to draw into the tile.
 * If no tile has been set (@see dwRenderEngine_setTile) it will deliver the
 * render viewport for the default tile (tileId == 0).
 * @param viewport A pointer to the render viewport of the tile to be filled
 * by this function.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_getRenderViewport(dwRectf* viewport,
                                          dwConstRenderEngineHandle_t engine);

/**
 * @brief Sets the user-defined viewport which is used in conjunction with
 * the other layout parameters to compute the render viewport.
 * @param viewport The viewport bounds.
 * If no tile has been set (@see dwRenderEngine_setTile) it will use the
 * viewport for the default tile (tileId == 0).
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_setViewport(dwRectf viewport,
                                    dwRenderEngineHandle_t engine);

/**
 * @brief Gets the bounds for the entire render area.
 * This is used for the layout of the render tiles.
 * @param bounds The pointer to a rect that will be populated
 * with the bounds for the render engine.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_getBounds(dwRectf* bounds,
                                  dwRenderEngineHandle_t engine);

/**
 * @brief Sets the bounds for the entire render area.
 * This is used for the layout of the render tiles.
 * For example if you wanted to only use a portion of the window for the
 * render engine, the bounds should be set to only a portion of the window.
 * This would then only allow tiles to be rendered within these bounds.
 * @param bounds The rect defining the bounds in the window.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_setBounds(dwRectf bounds,
                                  dwRenderEngineHandle_t engine);

/**
 * @brief Gets the current tile color.
 * @param color Pointer to location with which to populate the current tile color.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_getColor(dwRenderEngineColorRGBA* color,
                                 dwConstRenderEngineHandle_t engine);

/**
 * @brief Sets the background color of the current tile.
 * @param color The color.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_setBackgroundColor(dwRenderEngineColorRGBA color, dwRenderEngineHandle_t engine);

/**
 * @brief Gets the current tile background color.
 * @param color Pointer to location with which to populate the current tile color.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_getBackgroundColor(dwRenderEngineColorRGBA* color,
                                           dwConstRenderEngineHandle_t engine);

/**
 * @brief Sets the foreground drawing color of the current tile.
 * @param color The color.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_setColor(dwRenderEngineColorRGBA color, dwRenderEngineHandle_t engine);

/**
 * @brief Sets the color of the primitives to be rendered based on their values.
 * @note DW_RENDER_ENGINE_COLOR_BY_VALUE_MODE_INTENSITY only works with
 * an extra value passed in for each interleaved piece of data using
 * this mode. Also, in this mode, scale is ignored.
 * For example, a 2D point would go from x,y,x,y to x,y,i, x,y,i where i is
 * the intensity value. Obviously, this would also increase the stride
 * passed into the dwRenderEngine_render function.
 * @param mode The mode by which to interpolate the color values.
 * @param scale The scale represents how far before the color gamut recycles.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_setColorByValue(dwRenderEngineColorByValueMode mode,
                                        float32_t scale,
                                        dwRenderEngineHandle_t engine);

/**
 * @brief Gets the current tile font.
 * @param font Pointer to location with which to populate the current tile font.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.n
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_getFont(dwRenderEngineFont* font,
                                dwConstRenderEngineHandle_t engine);

/**
 * @brief Sets the font for text drawn in the current tile.
 * @param font The font.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_setFont(dwRenderEngineFont font,
                                dwRenderEngineHandle_t engine);

/**
 * @brief Gets the current tile line width.
 * @param lineWidth Pointer to a location with which to populate the current
 * tile line width.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_getLineWidth(float32_t* lineWidth,
                                     dwConstRenderEngineHandle_t engine);
/**
 * @brief Sets the line width of the current tile.
 * @param lineWidth The line width.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 *
 * @note Maximal line width supported depends on the GL implementation, typical range is 0.5 - 10 px.
 *       Use glGetFloatv(GL_ALIASED_LINE_WIDTH_RANGE, ...) to query for maximal line width range.
 *       In case if smoothed lines are active, i.e. glIsEnabled(GL_LINE_SMOOTH), query using GL_SMOOTH_LINE_WIDTH_RANGE.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_setLineWidth(float32_t lineWidth,
                                     dwRenderEngineHandle_t engine);

/**
 * @brief Gets the current tile point size.
 * @param pointSize Pointer to a location with which to populate the current
 * tile point size.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_getPointSize(float32_t* pointSize,
                                     dwConstRenderEngineHandle_t engine);

/**
 * @brief Sets the point size of the current tile.
 * @param pointSize The point size.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_setPointSize(float32_t pointSize,
                                     dwRenderEngineHandle_t engine);

/**
 * @brief Gets the coordinate range for 2D rendering of the current tile.
 * This controls the normalization of the data passed into render.
 * @param range The pointer to range where x is the width and y is the height.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_getCoordinateRange2D(dwVector2f* range,
                                             dwConstRenderEngineHandle_t engine);

/**
 * @brief Sets the coordinate range for 2D rendering of the current tile.
 * If passing in data that is not from 0 to 1 for 2D rendering,
 * this function will modulate the normalization factors to put
 * the data automatically in the 0 to 1 range. If rendering image
 * space coordinates, this should be the width and height of the image.
 * @param range The range where x is the width and y is the height.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_setCoordinateRange2D(dwVector2f range,
                                             dwRenderEngineHandle_t engine);

/**
 * @brief Gets the layout of the current tile.
 * @param layout Pointer to the location with which to populate
 * the layout.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_getLayout(dwRenderEngineTileLayout* layout,
                                  dwConstRenderEngineHandle_t engine);

/**
 * @brief Sets the layout of the current tile.
 * @param layout The layout of the current tile.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_setLayout(dwRenderEngineTileLayout layout,
                                  dwRenderEngineHandle_t engine);

/**
 * @brief Creates a buffer for static drawing. This function allocates
 * memory on the heap and GPU.
 * When rendering 2D data the coordinate system is always from 0 to 1,
 * unless changed by calling dwRenderEngine_setCoordinateRange2D.
 * The origin is at the top left.
 * When rendering 3D data the coordinate system is NDC.
 * @param bufferId Pointer with which to populate handle to buffer.
 * @param type The type of data to render.
 * @param vertexStrideBytes The stride of each vertex.
 * This should be the total size of
 * each element passed into render. For example, if passing in an array of structs,
 * it should be the size of the struct, even if that struct contains more than just the vertex.
 * @param offsetBytes The offset of each piece of data. This should be to offset in bytes
 * to the primitive data. For example, if passing in an array of structs, and each
 * struct contains an x,y position of a 2D point then this needs to be the offset
 * to the x,y fields. If the x,y fields are the 2nd and 3rd elements of the struct
 * and the first field is an int, then the offset should be sizeof(int).
 * @param primitiveCount The number of primitives in the buffer.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_createBuffer(uint32_t* bufferId,
                                     dwRenderEnginePrimitiveType type,
                                     uint32_t vertexStrideBytes,
                                     uint32_t offsetBytes,
                                     uint32_t primitiveCount,
                                     dwRenderEngineHandle_t engine);

/**
 * @brief Destroys a buffer for static drawing.
 * @param bufferId Handle to a buffer.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_destroyBuffer(uint32_t bufferId,
                                      dwRenderEngineHandle_t engine);

/**
 * @brief Get the max renderable primitive count for a buffer. This is the max primitive
 * count that can be rendered after dwRenderEngine_setBuffer or
 * dwRenderEngine_setBufferPlanarGrid2D/3D or
 * dwRenderEngine_setBufferEllipiticalGrid2D/3D has been called.
 * @param[out] maxPrimitiveCount A pointer with which to populate the max primitive count.
 * @param bufferId Handle to a buffer added with dwRenderEngine_addBuffer.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_getBufferMaxPrimitiveCount(uint32_t* maxPrimitiveCount,
                                                   uint32_t bufferId,
                                                   dwRenderEngineHandle_t engine);

/**
 * @brief Set the data for a buffer. This is useful for static data.
 * @param bufferId Handle to a buffer added with dwRenderEngine_addBuffer.
 * @param type The type of data to render.
 * @param buffer The data.
 * @param vertexStrideBytes The stride of each vertex.
 * This should be the total size of
 * each element passed into render. For example, if passing in an array of structs,
 * it should be the size of the struct, even if that struct contains more than just the vertex.
 * @param offsetBytes The offset of each piece of data. This should be to offset in bytes
 * to the primitive data. For example, if passing in an array of structs, and each
 * struct contains an x,y position of a 2D point then this needs to be the offset
 * to the x,y fields. If the x,y fields are the 2nd and 3rd elements of the struct
 * and the first field is an int, then the offset should be sizeof(int).
 * @param primitiveCount The number of primitives to render.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_setBuffer(uint32_t bufferId,
                                  dwRenderEnginePrimitiveType type,
                                  const void* buffer,
                                  uint32_t vertexStrideBytes,
                                  uint32_t offsetBytes,
                                  uint32_t primitiveCount,
                                  dwRenderEngineHandle_t engine);

/**
 * @brief maps internal buffer for data operations. Populate
 * this buffer to render data.
 * @param bufferId Handle to a buffer added with dwRenderEngine_addBuffer.
 * @param buffer The buffer to populate the data with.
 * @param offsetBytes The offset into the internal buffer.
 * @param sizeBytes The size to map.
 * @param type The primitive type to map.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_mapBuffer(uint32_t bufferId,
                                  void** buffer,
                                  uint32_t offsetBytes,
                                  uint32_t sizeBytes,
                                  dwRenderEnginePrimitiveType type,
                                  dwRenderEngineHandle_t engine);

/**
 * @brief Unmaps the internal buffer.
 * @param bufferId Handle to a buffer added with dwRenderEngine_addBuffer.
 * @param type The primitive type to unmap.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_unmapBuffer(uint32_t bufferId,
                                    dwRenderEnginePrimitiveType type,
                                    dwRenderEngineHandle_t engine);

/**
 * @brief Render the buffer.
 * When rendering 2D data the coordinate system is always from 0 to 1,
 * unless changed by calling dwRenderEngine_setCoordinateRange2D.
 * The origin is at the top left.
 * When rendering 3D data the coordinate system is NDC.
 * @param bufferId Handle to a buffer added with dwRenderEngine_addBuffer.
 * @param primitiveCount The number of primitives to render.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_renderBuffer(uint32_t bufferId,
                                     uint32_t primitiveCount,
                                     dwRenderEngineHandle_t engine);

/**
 * @brief Render the internal buffer with labels.
 * @param bufferId Handle to a buffer added with dwRenderEngine_addBuffer.
 * @param type The type of data to render.
 * @param vertexStrideBytes The stride of each vertex.
 * This should be the total size of
 * each element passed into render. For example, if passing in an array of structs,
 * it should be the size of the struct, even if that struct contains more than just the vertex.
 * @param offsetBytes The offset of each piece of data. This should be to offset in bytes
 * to the primitive data. For example, if passing in an array of structs, and each
 * struct contains an x,y position of a 2D point then this needs to be the offset
 * to the x,y fields. If the x,y fields are the 2nd and 3rd elements of the struct
 * and the first field is an int, then the offset should be sizeof(int).
 * @param labels The array of text labels.
 * @param primitiveCount The number of primitives to render.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_renderBufferWithLabels(uint32_t bufferId,
                                               dwRenderEnginePrimitiveType type,
                                               uint32_t vertexStrideBytes,
                                               uint32_t offsetBytes,
                                               const char** labels,
                                               uint32_t primitiveCount,
                                               dwRenderEngineHandle_t engine);

/**
 * @brief Render an external buffer. This method will copy data
 * to the internal buffer.
 * When rendering 2D data the coordinate system is always from 0 to 1,
 * unless changed by calling dwRenderEngine_setCoordinateRange2D.
 * The origin is at the top left.
 * When rendering 3D data the coordinate system is NDC.
 * @param type The type of data to render.
 * @param buffer The data.
 * @param vertexStrideBytes The stride of each vertex.
 * This should be the total size of
 * each element passed into render. For example, if passing in an array of structs,
 * it should be the size of the struct, even if that struct contains more than just the vertex.
 * @param offsetBytes The offset of each piece of data. This should be to offset in bytes
 * to the primitive data. For example, if passing in an array of structs, and each
 * struct contains an x,y position of a 2D point then this needs to be the offset
 * to the x,y fields. If the x,y fields are the 2nd and 3rd elements of the struct
 * and the first field is an int, then the offset should be sizeof(int).
 * @param primitiveCount The number of primitives to render.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_render(dwRenderEnginePrimitiveType type,
                               const void* buffer,
                               uint32_t vertexStrideBytes,
                               uint32_t offsetBytes,
                               uint32_t primitiveCount,
                               dwRenderEngineHandle_t engine);

/**
 * @brief Render an external buffer with labels.
 *  This method will copy data to the internal buffer.
 * When rendering 2D data the coordinate system is always from 0 to 1,
 * unless changed by calling dwRenderEngine_setCoordinateRange2D.
 * The origin is at the top left.
 * When rendering 3D data the coordinate system is NDC.
 * @param type The type of data to render.
 * @param buffer The data.
 * @param vertexStrideBytes The stride of each vertex.
 * This should be the total size of
 * each element passed into render. For example, if passing in an array of structs,
 * it should be the size of the struct, even if that struct contains more than just the vertex.
 * @param offsetBytes The offset of each piece of data. This should be to offset in bytes
 * to the primitive data. For example, if passing in an array of structs, and each
 * struct contains an x,y position of a 2D point then this needs to be the offset
 * to the x,y fields. If the x,y fields are the 2nd and 3rd elements of the struct
 * and the first field is an int, then the offset should be sizeof(int).
 * @param labels The array of text labels.
 * @param primitiveCount The number of primitives to render.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_renderWithLabels(dwRenderEnginePrimitiveType type,
                                         const void* buffer,
                                         uint32_t vertexStrideBytes,
                                         uint32_t offsetBytes,
                                         const char* labels[],
                                         uint32_t primitiveCount,
                                         dwRenderEngineHandle_t engine);

/**
 * @brief Render an external buffer with one label.
 *  This method will copy data to the internal buffer.
 * When rendering 2D data the coordinate system is always from 0 to 1,
 * unless changed by calling dwRenderEngine_setCoordinateRange2D.
 * The origin is at the top left.
 * When rendering 3D data the coordinate system is NDC.
 * @param type The type of data to render.
 * @param buffer The data.
 * @param vertexStrideBytes The stride of each vertex.
 * This should be the total size of
 * each element passed into render. For example, if passing in an array of structs,
 * it should be the size of the struct, even if that struct contains more than just the vertex.
 * @param offsetBytes The offset of each piece of data. This should be to offset in bytes
 * to the primitive data. For example, if passing in an array of structs, and each
 * struct contains an x,y position of a 2D point then this needs to be the offset
 * to the x,y fields. If the x,y fields are the 2nd and 3rd elements of the struct
 * and the first field is an int, then the offset should be sizeof(int).
 * @param label The text label.
 * @param primitiveCount The number of primitives to render.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_renderWithLabel(dwRenderEnginePrimitiveType type,
                                        const void* buffer,
                                        uint32_t vertexStrideBytes,
                                        uint32_t offsetBytes,
                                        const char* label,
                                        uint32_t primitiveCount,
                                        dwRenderEngineHandle_t engine);

/**
 * @brief Reads a file from disk and
 * puts it into an dwImageGL. Currently only supports png format.
 * @param img Pointer to the image.
 * @param filename File name of the image.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_createImageGLFromFile(dwImageGL* img,
                                              const char* filename,
                                              dwRenderEngineHandle_t engine);

/**
 * @brief Releases the memory of a dwImageGL.
 * @param image The image.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_destroyImageGL(const dwImageGL* image,
                                       dwRenderEngineHandle_t engine);

/**
 * @brief Renders a 2D image within the current tile.
 * When rendering 2D data the coordinate system is always from 0 to 1,
 * unless changed by calling dwRenderEngine_setCoordinateRange2D.
 * The origin is at the top left.
 * @param img The image to render.
 * @param plane The bounds of the image.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_renderImage2D(const dwImageGL* img,
                                      dwRectf plane,
                                      dwRenderEngineHandle_t engine);

/**
 * @brief Renders a 2D image in 3D space within the current tile.
 * When rendering 3D data the coordinate system is NDC.
 * @param img The image to render.
 * @param plane The plane on which to render the image.
 * @param modelMatrix The model matrix of the plane within the world.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_renderImage3D(const dwImageGL* img,
                                      dwRectf plane,
                                      const dwMatrix4f* modelMatrix,
                                      dwRenderEngineHandle_t engine);

/**
 * @brief Renders text within the current tile.
 * When rendering 2D data the coordinate system is always from 0 to 1,
 * unless changed by calling dwRenderEngine_setCoordinateRange2D.
 * The origin is at the top left.
 * @param text The pointer to the text structure.
 * @param pos The position in 2D of the text.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_renderText2D(const char* text,
                                     dwVector2f pos,
                                     dwRenderEngineHandle_t engine);

/**
 * @brief Renders text within the current tile.
 * When rendering 3D data the coordinate system is NDC.
 * @param text The pointer to the text structure.
 * @param pos The position of the text to be rendered in 3D. This will
 * ultimately be mapped to the correct 2D position by the function,as
 * text only renders in 2D.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_renderText3D(const char* text,
                                     dwVector3f pos,
                                     dwRenderEngineHandle_t engine);
/**
 * @brief Fills a buffer with a grid in 2D.
 * @param bufferId The buffer id to fill.
 * @param plane This plane defines the center point (x,y) and
 * the radius in the x, y directions (width, height).
 * @param xSpacing The spacing of the lines in the x direction.
 * @param ySpacing The spacing of the lines in the y direction.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_setBufferPlanarGrid2D(uint32_t bufferId,
                                              dwRectf plane,
                                              float32_t xSpacing,
                                              float32_t ySpacing,
                                              dwRenderEngineHandle_t engine);

/**
 * @brief Renders a grid in 2D.
 * @param plane This plane defines the center point (x,y) and
 * the radius in the x, y directions (width, height).
 * @param xSpacing The spacing of the lines in the x direction.
 * @param ySpacing The spacing of the lines in the y direction.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_renderPlanarGrid2D(dwRectf plane,
                                           float32_t xSpacing,
                                           float32_t ySpacing,
                                           dwRenderEngineHandle_t engine);

/**
 * @brief Fills a buffer with a grid in 3D.
 * The planeis rendered on the x,y plane with z coordinates set to zero.
 * The plane can then be transformed in 3D by the model matrix.
 * @param bufferId The buffer id to fill.
 * @param plane This plane defines the center point (x,y) and
 * the radius in the x, y directions (width, height).
 * @param xSpacing The spacing of the lines in the x direction.
 * @param ySpacing The spacing of the lines in the y direction.
 * @param modelMatrix The model matrix that defines where this grid is in the world.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_setBufferPlanarGrid3D(uint32_t bufferId,
                                              dwRectf plane,
                                              float32_t xSpacing,
                                              float32_t ySpacing,
                                              const dwMatrix4f* modelMatrix,
                                              dwRenderEngineHandle_t engine);

/**
 * @brief Renders a grid in 3D.
 * The plane is rendered on the x,y plane with z coordinates set to zero.
 * The plane can then be transformed in 3D by the model matrix.
 * @param plane This plane defines the center point (x,y) and
 * the radius in the x, y directions (width, height).
 * @param xSpacing The spacing of the lines in the x direction.
 * @param ySpacing The spacing of the lines in the y direction.
 * @param modelMatrix The model matrix that defines where this grid is in the world.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_renderPlanarGrid3D(dwRectf plane,
                                           float32_t xSpacing,
                                           float32_t ySpacing,
                                           const dwMatrix4f* modelMatrix,
                                           dwRenderEngineHandle_t engine);

/**
 * @brief Fills a buffer with an elliptical grid in 2D.
 * @param bufferId The buffer id to fill.
 * @param plane This plane defines the center point (x,y) and
 * the radius in the x, y directions (width, height).
 * @param xSpacing The spacing of the lines in the x direction.
 * @param ySpacing The spacing of the lines in the y direction.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_setBufferEllipticalGrid2D(uint32_t bufferId,
                                                  dwRectf plane,
                                                  float32_t xSpacing,
                                                  float32_t ySpacing,
                                                  dwRenderEngineHandle_t engine);

/**
 * @brief Renders an elliptical grid in 2D.
 * @param plane This plane defines the center point (x,y) and
 * the radius in the x, y directions (width, height).
 * @param xSpacing The spacing of the lines in the x direction.
 * @param ySpacing The spacing of the lines in the y direction.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_renderEllipticalGrid2D(dwRectf plane,
                                               float32_t xSpacing,
                                               float32_t ySpacing,
                                               dwRenderEngineHandle_t engine);

/**
 * @brief Fills a buffer with an elliptical grid in 3D.
 * The ellipse is rendered on the x,y plane with z coordinates set to zero.
 * The ellipse can then be transformed in 3D by the model matrix.
 * @param bufferId The buffer id to fill.
 * @param plane This plane defines the center point (x,y) and
 * the radius in the x, y directions (width, height).
 * @param xSpacing The spacing of the lines in the x direction.
 * @param ySpacing The spacing of the lines in the y direction.
 * @param modelMatrix The model matrix that defines where this grid is in the world.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_setBufferEllipticalGrid3D(uint32_t bufferId,
                                                  dwRectf plane,
                                                  float32_t xSpacing,
                                                  float32_t ySpacing,
                                                  const dwMatrix4f* modelMatrix,
                                                  dwRenderEngineHandle_t engine);

/**
 * @brief Renders an elliptical grid in 3D.
 * The ellipse is rendered on the x,y plane with z coordinates set to zero.
 * The ellipse can then be transformed in 3D by the model matrix.
 * @param plane This plane defines the center point (x,y) and
 * the radius in the x, y directions (width, height).
 * @param xSpacing The spacing of the lines in the x direction.
 * @param ySpacing The spacing of the lines in the y direction.
 * @param modelMatrix The model matrix that defines where this grid is in the world.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_renderEllipticalGrid3D(dwRectf plane,
                                               float32_t xSpacing,
                                               float32_t ySpacing,
                                               const dwMatrix4f* modelMatrix,
                                               dwRenderEngineHandle_t engine);

/**
 * @brief Renders a plot in 2D.
 * @param type The type of plot to render.
 * @param buffer The data to render.
 * @param vertexStrideBytes The stride of each vertex.
 * This should be the total size of
 * each element passed into render. For example, if passing in an array of structs,
 * it should be the size of the struct, even if that struct contains more than just the vertex.
 * @param offsetBytes The offset of each piece of data. This should be to offset in bytes
 * to the primitive data. For example, if passing in an array of structs, and each
 * struct contains an x,y position of a 2D point then this needs to be the offset
 * to the x,y fields. If the x,y fields are the 2nd and 3rd elements of the struct
 * and the first field is an int, then the offset should be sizeof(int).
 * @param primitiveCount The number of primitives in the data.
 * @param label The label of the plot data.
 * @param range The range for the data. The range for the the lower limits can be
 * determined automatically if infinity is passed in for any of the values.
 * X and y represent min x and min y, respectively and z and w represent
 * max x and max y respectively.
 * @param plane The plane to render the plot in.
 * @param axesColor The color of the axes.
 * @param axesWidth The line width of the axes.
 * @param title The title of the plot.
 * @param xLabel The label for the x-axis.
 * @param yLabel The label for the y-axis.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_renderPlot2D(dwRenderEnginePlotType type,
                                     const void* buffer,
                                     uint32_t vertexStrideBytes,
                                     uint32_t offsetBytes,
                                     uint32_t primitiveCount,
                                     const char* label,
                                     dwVector4f range,
                                     dwRectf plane,
                                     dwRenderEngineColorRGBA axesColor,
                                     float32_t axesWidth,
                                     const char* title,
                                     const char* xLabel,
                                     const char* yLabel,
                                     dwRenderEngineHandle_t engine);

/**
 * @brief Renders multiple plots in one graph.
 * @param types The types of plots to render.
 * @param buffers The buffers for each plot.
 * @param vertexStridesBytes The strides for each plot
 * of each piece of data. This should be the total size of
 * each element passed into render. For example, if passing in an array of structs,
 * it should be the size of the struct, even if that struct contains more than just the vertex.
 * @param offsetsBytes The offsets for each plot
 * of each piece of data. This should be to offset in bytes
 * to the primitive data. For example, if passing in an array of structs, and each
 * struct contains an x,y position of a 2D point then this needs to be the offset
 * to the x,y fields. If the x,y fields are the 2nd and 3rd elements of the struct
 * and the first field is an int, then the offset should be sizeof(int).
 * @param bufferCounts The sizes of each buffer in buffers.
 * @param colors The colors for each plot. If null then tile color will be used.
 * @param lineWidths The line widths or point sizes for each plot.
 * If null, then the tile state will be used.
 * @param labels The labels for each plot. If null, then no label will
 * be given for each plot.
 * @param plotsCount The number of buffers.
 * @param range The range for the data. The range for the the lower limits can be
 * determined automatically if infinity is passed in for any of the values.
 * X and y represent min x and min y, respectively and z and w represent
 * max x and max y respectively.
 * @param plane The plane to render the plot in.
 * @param axesColor The color of the axes.
 * @param axesWidth The line width of the axes.
 * @param title The title of the plot.
 * @param xLabel The label for the x-axis.
 * @param yLabel The label for the y-axis.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_renderPlots2D(const dwRenderEnginePlotType* types,
                                      const void** buffers,
                                      const uint32_t* vertexStridesBytes,
                                      const uint32_t* offsetsBytes,
                                      const uint32_t* bufferCounts,
                                      const dwRenderEngineColorRGBA* colors,
                                      const float32_t* lineWidths,
                                      const char** labels,
                                      uint32_t plotsCount,
                                      dwVector4f range,
                                      dwRectf plane,
                                      dwRenderEngineColorRGBA axesColor,
                                      float32_t axesWidth,
                                      const char* title,
                                      const char* xLabel,
                                      const char* yLabel,
                                      dwRenderEngineHandle_t engine);

/**
 * @brief Get the world coordinate in 3D based on screen
 * pixel input.
 * @param worldPosition The output world position in 3D.
 * @param pixelPosition The pixel position.
 * @param screenSize The size of the window screen.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_screenToWorld3D(dwVector3f* worldPosition,
                                        dwVector2f pixelPosition,
                                        dwVector2f screenSize,
                                        dwRenderEngineHandle_t engine);

/**
 * @brief Get the screen coordinate in based on world 3D input.
 * @param pixelPosition The output pixel position.
 * @param worldPosition The input world position in 3D.
 * @param screenSize The size of the window screen.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_world3DToScreen(dwVector2f* pixelPosition,
                                        dwVector3f worldPosition,
                                        dwVector2f screenSize,
                                        dwRenderEngineHandle_t engine);

/**
 * @brief Gets the tile id that surrounds the
 * input pixel position. If multiple tiles surround the same point, the most recently
 * added tile id will be returned.
 * @param tileId The output of the tile that surrounds the input point.
 * @param pixelPosition The input point.
 * @param screenSize The size of the window screen.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_getTileByScreenCoordinates(uint32_t* tileId,
                                                   dwVector2f pixelPosition,
                                                   dwVector2f screenSize,
                                                   dwConstRenderEngineHandle_t engine);

/**
 * @brief Signal the start of a series of text-drawing calls. Issuing
 * non-text-rendering calls between beginTextBatch and endTextBatch will
 * cause undefined rendering results.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_beginTextBatch(dwRenderEngineHandle_t engine);

/**
 * @brief Signal the end of a series of text-drawing calls. Issuing
 * non-text-rendering calls between beginTextBatch and endTextBatch will
 * cause undefined rendering results.
 * @param engine The render engine handle.
 * @return DW_SUCCESS When successful.
 *          DW_FAILURE When unsuccessful.
 *          DW_INVALID_ARGUMENT When an invalid argument is passed in.
 *          DW_GL_ERROR When there is an OpenGL error.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderEngine_endTextBatch(dwRenderEngineHandle_t engine);
#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_VISUALIZATION_RENDERENGINE_H_
