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
 * <b>NVIDIA DriveWorks API: Renderer</b>
 *
 * @b Description: This file defines the C-style interface for Renderer.
 */

/**
 * @defgroup renderer_group Renderer Interface
 *
 * @brief Defines Renderer related types, structs, and functions.
 * @{
 * @ingroup vizualization_group
 */

#ifndef DWVISUALIZATION_RENDERER_H_
#define DWVISUALIZATION_RENDERER_H_

#include <dwvisualization/core/Visualization.h>

#include <dwvisualization/gl/GL.h>

#ifdef __cplusplus
extern "C" {
#endif

//#######################################################################################
const dwVector4f DW_RENDERER_COLOR_BLACK     = {0.0f / 255.0f, 0.0f / 255.0f, 0.0f / 255.0f, 1.0f};
const dwVector4f DW_RENDERER_COLOR_WHITE     = {255.0f / 255.0f, 255.0f / 255.0f, 255.0f / 255.0f, 1.0f};
const dwVector4f DW_RENDERER_COLOR_DARKGREY  = {72.0f / 255.0f, 72.0f / 255.0f, 72.0f / 255.0f, 1.0f};
const dwVector4f DW_RENDERER_COLOR_GREY      = {128.0f / 255.0f, 128.0f / 255.0f, 128.0f / 255.0f, 1.0f};
const dwVector4f DW_RENDERER_COLOR_LIGHTGREY = {185.0f / 255.0f, 185.0f / 255.0f, 185.0f / 255.0f, 1.0f};

const dwVector4f DW_RENDERER_COLOR_DARKBLUE  = {45.0f / 255.0f, 44.0f / 255.0f, 100.0f / 255.0f, 0.6f};
const dwVector4f DW_RENDERER_COLOR_BLUE      = {32.0f / 255.0f, 72.0f / 255.0f, 230.0f / 255.0f, 1.0f};
const dwVector4f DW_RENDERER_COLOR_LIGHTBLUE = {51.0f / 255.0f, 153.0f / 255.0f, 255.0f / 255.0f, 0.6f};

const dwVector4f DW_RENDERER_COLOR_DARKCYAN  = {5.0f / 255.0f, 150.0f / 255.0f, 150.0f / 255.0f, 1.0f};
const dwVector4f DW_RENDERER_COLOR_CYAN      = {40.0f / 255.0f, 130.0f / 255.0f, 255.0f / 255.0f, 1.0f};
const dwVector4f DW_RENDERER_COLOR_LIGHTCYAN = {150.0f / 255.0f, 230.0f / 255.0f, 230.0f / 255.0f, 1.0f};

const dwVector4f DW_RENDERER_COLOR_DARKGREEN    = {45.0f / 255.0f, 100.0f / 255.0f, 44.0f / 255.0f, 1.0f};
const dwVector4f DW_RENDERER_COLOR_GREEN        = {32.0f / 255.0f, 230.0f / 255.0f, 32.0f / 255.0f, 1.0f};
const dwVector4f DW_RENDERER_COLOR_LIGHTGREEN   = {118.0f / 255.0f, 185.0f / 255.0f, 0.0f / 255.0f, 1.0f};
const dwVector4f DW_RENDERER_COLOR_NVIDIA_GREEN = {0.4609375f, 0.72265625f, 0.0f, 1.0f};
const dwVector4f DW_RENDERER_COLOR_LIME         = {227.0f / 255.0f, 255.0f / 255.0f, 0.0f / 255.0f, 1.0f};

const dwVector4f DW_RENDERER_COLOR_DARKYELLOW  = {242.0f / 255.0f, 186.0f / 255.0f, 73.0f / 255.0f, 1.0f};
const dwVector4f DW_RENDERER_COLOR_YELLOW      = {230.0f / 255.0f, 230.0f / 255.0f, 10.0f / 255.0f, 1.0f};
const dwVector4f DW_RENDERER_COLOR_LIGHTYELLOW = {253.0f / 255.0f, 255.0f / 255.0f, 0.0f / 255.0f, 1.0f};

const dwVector4f DW_RENDERER_COLOR_ORANGE      = {230.0f / 255.0f, 100.0f / 255.0f, 10.0f / 255.0f, 1.0f};
const dwVector4f DW_RENDERER_COLOR_DARKERBROWN = {101.0f / 255.0f, 67.0f / 255.0f, 33.0f / 255.0f, 1.0f};
const dwVector4f DW_RENDERER_COLOR_DARKBROWN   = {155.0f / 255.0f, 103.0f / 255.0f, 60.0f / 255.0f, 1.0f};
const dwVector4f DW_RENDERER_COLOR_LIGHTBROWN  = {188.0f / 255.0f, 158.0f / 255.0f, 130.0f / 255.0f, 1.0f};

const dwVector4f DW_RENDERER_COLOR_DARKRED  = {180.0f / 255.0f, 5.0f / 255.0f, 0.0f / 255.0f, 1.0f};
const dwVector4f DW_RENDERER_COLOR_RED      = {230.0f / 255.0f, 72.0f / 255.0f, 32.0f / 255.0f, 1.0f};
const dwVector4f DW_RENDERER_COLOR_LIGHTRED = {250.0f / 255.0f, 100.0f / 255.0f, 100.0f / 255.0f, 1.0f};

const dwVector4f DW_RENDERER_COLOR_DARKMAGENTA  = {120.0f / 255.0f, 50.0f / 255.0f, 120.0f / 255.0f, 1.0f};
const dwVector4f DW_RENDERER_COLOR_MAGENTA      = {230.f / 255.f, 25.f / 255.f, 230.f / 255.f, 1.f};
const dwVector4f DW_RENDERER_COLOR_LIGHTMAGENTA = {170.0f / 255.0f, 100.0f / 255.0f, 170.0f / 255.0f, 1.0f};

const dwVector4f DW_RENDERER_COLOR_DARKPURPLE  = {81.0f / 255.0f, 4.0f / 255.0f, 189.0f / 255.0f, 1.0f};
const dwVector4f DW_RENDERER_COLOR_PURPLE      = {255.0f / 255.0f, 0.0f / 255.0f, 240.0f / 255.0f, 1.0f};
const dwVector4f DW_RENDERER_COLOR_LIGHTPURPLE = {181.0f / 255.0f, 138.0f / 255.0f, 242.0f / 255.0f, 1.0f};

const dwVector4f DW_RENDERER_COLOR_DARKSALMON  = {233.0f / 255.0f, 150.0f / 255.0f, 122.0f / 255.0f, 1.0f};
const dwVector4f DW_RENDERER_COLOR_SALMON      = {250.0f / 255.0f, 128.0f / 255.0f, 114.0f / 255.0f, 1.0f};
const dwVector4f DW_RENDERER_COLOR_LIGHTSALMON = {255.0f / 255.0f, 160.0f / 255.0f, 122.0f / 255.0f, 1.0f};

const dwVector3f DW_RENDERER_HSV_RED    = {0.0f, 1.0f, 0.75f};
const dwVector3f DW_RENDERER_HSV_BLUE   = {240.0f, 1.0f, 0.75f};
const dwVector3f DW_RENDERER_HSV_PURPLE = {270.0f, 1.0f, 0.75f};

/// The maximal number of views that can be created in with @see dwRenderBuffer_createView().
static const uint32_t DW_RENDERBUFFER_MAX_VIEWS = 10;

//#######################################################################################
/// @brief Color map scales - determine the bandwidth of the color spectrum
const dwVector3f DW_RENDER_COLORMAPSCALE_75M  = {75.0f, 75.0f, 75.0f};
const dwVector3f DW_RENDER_COLORMAPSCALE_100M = {100.0f, 100.0f, 100.0f};
const dwVector3f DW_RENDER_COLORMAPSCALE_130M = {130.0f, 130.0f, 130.0f};
const dwVector3f DW_RENDER_COLORMAPSCALE_150M = {150.0f, 150.0f, 150.0f};

//#######################################################################################
/// @brief Font types and sizes supported by the renderer.
typedef enum dwRendererFonts {
    DW_RENDER_FONT_VERDANA_8  = 0,
    DW_RENDER_FONT_VERDANA_12 = 1,
    DW_RENDER_FONT_VERDANA_16 = 2,
    DW_RENDER_FONT_VERDANA_20 = 3,
    DW_RENDER_FONT_VERDANA_24 = 4,
    DW_RENDER_FONT_VERDANA_32 = 5,
    DW_RENDER_FONT_VERDANA_48 = 6,
    DW_RENDER_FONT_VERDANA_64 = 7,
    DW_RENDER_FONT_COUNT
} dwRendererFonts;

//#######################################################################################
/// @brief Vertex component channel count and format.
typedef enum dwRenderBufferFormat {
    DW_RENDER_FORMAT_NULL               = 0,
    DW_RENDER_FORMAT_R32G32B32A32_FLOAT = 1,
    DW_RENDER_FORMAT_R32G32B32_FLOAT    = 2,
    DW_RENDER_FORMAT_R32G32_FLOAT       = 3,
    DW_RENDER_FORMAT_R32_FLOAT          = 4
} dwRenderBufferFormat;

/**
 * @brief Vertex component semantics.
 * Origin of POS_XY coordinates is top-left.
 */
typedef enum dwRenderBufferPositionSemantic {
    DW_RENDER_SEMANTIC_POS_NULL = 0,
    DW_RENDER_SEMANTIC_POS_XY   = 1,
    DW_RENDER_SEMANTIC_POS_XYZ  = 2,
} dwRenderBufferPositionSemantic;

/// @brief Vertex component semantics.
typedef enum dwRenderBufferColorSemantic {
    DW_RENDER_SEMANTIC_COL_NULL = 0,
    DW_RENDER_SEMANTIC_COL_R    = 1,
    DW_RENDER_SEMANTIC_COL_A    = 2,
    DW_RENDER_SEMANTIC_COL_RGB  = 3,
    DW_RENDER_SEMANTIC_COL_RGBA = 4,
    DW_RENDER_SEMANTIC_COL_LUT  = 5,
    DW_RENDER_SEMANTIC_COL_HUE  = 6
} dwRenderBufferColorSemantic;

/// @brief Vertex component semantics.
typedef enum dwRenderBufferTexSemantic {
    DW_RENDER_SEMANTIC_TEX_NULL = 0,
    DW_RENDER_SEMANTIC_TEX_S    = 1,
    DW_RENDER_SEMANTIC_TEX_ST   = 2
} dwRenderBufferTexSemantic;

/// @brief Vertex layout describing format and semantics for position, color, and texture.
typedef struct dwRenderBufferVertexLayout
{
    dwRenderBufferFormat posFormat;             /*!< format of the position data */
    dwRenderBufferPositionSemantic posSemantic; /*!< description of the color data */
    dwRenderBufferFormat colFormat;             /*!< format of the color data */
    dwRenderBufferColorSemantic colSemantic;    /*!< description of the color data */
    dwRenderBufferFormat texFormat;             /*!< format of the texture coordinates data */
    dwRenderBufferTexSemantic texSemantic;      /*!< description of the color data */
} dwRenderBufferVertexLayout;

/// @brief Render primitives supported by the renderer.
typedef enum dwRenderBufferPrimitiveType {
    DW_RENDER_PRIM_POINTLIST    = 0,
    DW_RENDER_PRIM_LINELIST     = 1,
    DW_RENDER_PRIM_TRIANGLELIST = 2,
    DW_RENDER_PRIM_LINESTRIP    = 3,
    DW_RENDER_PRIM_LINELOOP     = 4,
    DW_RENDER_PRIM_COUNT        = 5
} dwRenderBufferPrimitiveType;

/// The maximum number of vertices that can be passed to a call dwRenderer_renderData3D() or dwRenderer_renderData2D()
static const uint32_t DW_RENDERER_DEFAULT_BUFFER_SIZE = 10000;

/// Handle representing vertex data for rendering.
typedef struct dwRenderBufferObject* dwRenderBufferHandle_t;

/// Const handle representing vertex data for rendering.
typedef const struct dwRenderBufferObject* dwConstRenderBufferHandle_t;

/// Handle for the renderer.
typedef struct dwRendererObject* dwRendererHandle_t;

/// Const handle for the renderer.
typedef const struct dwRendererObject* dwConstRendererHandle_t;

//#######################################################################################
// RenderBuffer
//#######################################################################################

/**
 * @brief Initializes a `RenderBuffer` structure for rendering. It must be initialized on
 * a thread with a valid current OpenGL context. This call creates data both on the heap
 * and on the GPU.
 *
 * @param[out] renderbuffer A pointer to an initialized render buffer object.
 * @param[in] layout Specifies the stride of a vertex.
 * @param[in] primType Specifies how many vertices are in a primitive.
 * @param[in] primCount Specifies how many primitives the buffer should contain.
 * @param[in] context Specifies a handle to the context under which it is created.
 *
 * @return DW_INVALID_ARGUMENT, DW_SUCCESS
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderBuffer_initialize(dwRenderBufferHandle_t* renderbuffer,
                                   const dwRenderBufferVertexLayout layout,
                                   const dwRenderBufferPrimitiveType primType,
                                   const uint32_t primCount,
                                   dwVisualizationContextHandle_t context);
/**
 * @brief Creates an additional view for a render buffer, allowing you to change
 * layout and primitive type, but not vertex stride.
 *
 * @note This method performs allocations on the heap.
 *
 * @param[out] slot A pointer to the slot this view is allocated in. Slot 0 is the default view.
 * @param[in] renderbuffer Specifies the `RenderBuffer` with the data.
 * @param[in] newLayout Specifies the new size of a vertex.
 * @param[in] newPrimType Specifies how many vertices are in a primitive.
 * @param[in] context Specifies a handle to the context under which it is created.
 *
 * @return DW_INVALID_ARGUMENT, DW_SUCCESS, DW_FAILURE
 *
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderBuffer_createView(uint32_t* slot,
                                   dwRenderBufferHandle_t renderbuffer,
                                   const dwRenderBufferVertexLayout newLayout,
                                   const dwRenderBufferPrimitiveType newPrimType,
                                   dwVisualizationContextHandle_t context);
/**
 * @brief Maps the CPU copy of a `RenderBuffer` to the application for updating. The  pointer
 * provided is only valid until the corresponding unmap call. The application can overwrite any portion
 * of the buffer. Note that the number of valid vertices in the buffer is
 * updated to `nVerts` in `unmap`. This is a lightweight call. Mapping a currently mapped `RenderBuffer`
 * returns an error.
 *
 * @param[out] map A pointer to the CPU side of the `RenderBuffer` data.
 * @param[out] maxVertices A pointer to the maximum number of vertices in the allocated data.
 * @param[out] vertexStride A pointer to the size (in `float32_t`) of a vertex.
 * @param[in] renderbuffer Specifies the `RenderBuffer` being mapped to the CPU.
 *
 * @return DW_INVALID_ARGUMENT, DW_SUCCESS, DW_FAILURE
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderBuffer_map(float32_t** map,
                            uint32_t* maxVertices,
                            uint32_t* vertexStride,
                            dwRenderBufferHandle_t renderbuffer);

/**
 * @brief Maps a contiguous range of the CPU copy of a `RenderBuffer` to the application for updating. The
 * pointer provided is only valid until the corresponding unmap call. The application can overwrite parts or
 * the entirety of the specified range. This is a lightweight call. Mapping a currently mapped `RenderBuffer`
 * returns an error. The vertices before `startVertex` must be valid.
 *
 * @param[out] map A pointer to the CPU side of the `RenderBuffer` data.
 * @param[out] maxVertices A pointer to the maximum number of vertices that can be mapped.
 * @param[out] vertexStride A pointer to the size (in floats) of a vertex.
 * @param[in] startVertex Specifies the first vertex to map. This corresponds to map[0].
 * @param[in] renderbuffer Specifies the `RenderBuffer` being mapped to the CPU.
 *
 * @return DW_CALL_NOT_ALLOWED, DW_SUCCESS
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderBuffer_mapRange(float32_t** map,
                                 uint32_t* maxVertices,
                                 uint32_t* vertexStride,
                                 const uint32_t startVertex,
                                 dwRenderBufferHandle_t renderbuffer);
/**
 * @brief After a `map`/`mapRange`, returns the updated buffer to the `RenderBuffer` structure and updates the GPU
 * portion. Since a host to device copy is required, this method is not lightweight. If the GPU buffer is in
 * use, this can incur an extra cost. If the provided `RenderBuffer` has not been mapped, it returns an error.
 *
 * @param[in] nVerts Specifies how many vertices have been updated. Assumes a contiguous update from the
 *                   origin to `nVerts`.
 *                   map:      Current number of valid vertices in buffer is updated to `nVerts`.
 *                   mapRange: Current number of valid vertices extended if necessary, but not reduced.
 * @param[in] renderbuffer Specifies the `RenderBuffer` being mapped to the CPU.
 *
 * @return DW_INVALID_HANDLE, DW_SUCCESS, DW_CALL_NOT_ALLOWED
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderBuffer_unmap(uint32_t nVerts, dwRenderBufferHandle_t renderbuffer);

/**
 * @brief Releases the `RenderBuffer` data structure. It frees both the CPU and GPU memory allocated during
 * the initialization call. The `RenderBuffer` is assigned a NULL value.
 *
 * @param[out] renderbuffer The RenderBuffer to be released.
 *
 * @return DW_CALL_NOT_ALLOWED, DW_SUCCESS
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderBuffer_release(dwRenderBufferHandle_t renderbuffer);

/**
 * @brief Sets 2D normalization values when 2D coordinates are specified in pixel values and
 * not in 0..1 range. To reset to normalized values, set values to 1.0.
 *
 * @param[in] width Specifies the pixels to scale the X coordinate with.
 * @param[in] height Specifies the pixels to scale the Y coordinate with.
 * @param[in] renderbuffer Specifies the `RenderBuffer` to be released.
 *
 * @return DW_INVALID_HANDLE, DW_SUCCESS
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderBuffer_set2DCoordNormalizationFactors(const float32_t width, const float32_t height,
                                                       dwRenderBufferHandle_t renderbuffer);

//#######################################################################################
// Renderer
//#######################################################################################
/**
 * @brief Initializes a Renderer. This initialization implies creation of various graphics resources
 * including shaders, vertex data, and textures. It must be initialized on
 * a thread with a valid current OpenGL context. While multiple renderers may coexist, resources
 * among them are not shared.
 *
 * @param[out] renderer A pointer to an initialized renderer object.
 * @param[in] context Specifies a handle to the context under which it is created.
 *
 * @return DW_INVALID_ARGUMENT, DW_FAILURE, DW_SUCCESS
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderer_initialize(dwRendererHandle_t* renderer, dwVisualizationContextHandle_t context);

/**
 * @brief Resets renderer state. THis is a Lightweight call that does not destroy or create resources.
 *
 * @param[in] renderer A pointer to Renderer object to be reset.
 *
 * @return DW_INVALID_ARGUMENT, DW_SUCCESS
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderer_reset(dwRendererHandle_t renderer);

/**
 * @brief Released the renderer. This call frees all GPU resources. The handle points to NULL
 * on success.
 *
 * @param[in] renderer The Renderer object to be released.
 *
 * @return DW_INVALID_ARGUMENT, DW_SUCCESS
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderer_release(dwRendererHandle_t renderer);

//#######################################################################################
// Get state
//#######################################################################################
/**
 * @brief Gets current rendering screen area expressed in pixel values.
 *
 * @param[out] rect A pointer to a structure containing the current react area.
 * @param[in] obj Specifies a handle to the renderer object.
 *
 * @return DW_INVALID_ARGUMENT, DW_SUCCESS
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderer_getRect(dwRect* rect, dwConstRendererHandle_t obj);

/**
 * @brief Gets current rendering color.
 *
 * @param[out] color Specifies a float[4] vector in 0..1 range.
 * @param[in] obj Specifies a handle to the renderer object.
 *
 * @return DW_INVALID_ARGUMENT, DW_SUCCESS
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderer_getColor(dwVector4f* color, dwConstRendererHandle_t obj);

/**
 * @brief Gets current rendering color map scale.
 *
 * @param[out] colorMapScale Specifies a scale factor via float32_t[3] in meters for the saturation range. colorScale[0],
 * colorScale[1] and colorScale[2] to x, y, z respectively.
 * E.g. {150, 150, std::numeric_limits<double>::infinity()} indicates the color will wrap after 150 meters in
 * the x,y plane (z component is ignored).
 * @param[in] obj Specifies a handle to the renderer object.
 *
 * @return DW_INVALID_ARGUMENT, DW_SUCCESS
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderer_getColorMapScale(dwVector3f* colorMapScale, dwConstRendererHandle_t obj);

/**
 * @brief Gets current ModelView matrix.
 *
 * @param[out] matrix Specifies a 4D homogeneous matrix as float[16] col major.
 * @param[in]  obj Specifies a handle to renderer object.
 *
 * @return DW_INVALID_ARGUMENT, DW_SUCCESS
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderer_getModelView(dwMatrix4f* matrix, dwConstRendererHandle_t obj);

/**
 * @brief Gets current projection matrix.
 *
 * @param[out] matrix Specifies a 4D homogeneous matrix as float[16] col major
 * @param[in]  obj Specifies a handle to renderer object.
 *
 * @return DW_INVALID_ARGUMENT, DW_SUCCESS
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderer_getProjection(dwMatrix4f* matrix, dwConstRendererHandle_t obj);

/**
 * @brief Gets current point size.
 *
 * @param[out] value Specifies the size of a point, in pixels.
 * @param[in]  obj Specifies a handle to renderer object.
 *
 * @return DW_INVALID_ARGUMENT, DW_SUCCESS
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderer_getPointSize(float32_t* value, dwConstRendererHandle_t obj);

/**
 * @brief Gets current line width, in pixels.
 *
 * @param[out] value Specifies the width of a line, in pixels.
 * @param[in]  obj Specifies a handle to the renderer object.
 *
 * @return DW_INVALID_ARGUMENT, DW_SUCCESS
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderer_getLineWidth(float32_t* value, dwConstRendererHandle_t obj);

/**
 * @brief Gets current font.
 *
 * @param[out] value Specifies the active valid font.
 * @param[in]  obj Specifies a handle to the renderer object.
 *
 * @return DW_INVALID_ARGUMENT, DW_SUCCESS
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderer_getFont(dwRendererFonts* value, dwConstRendererHandle_t obj);

/**
 * @brief Gets current 2D transformation applied when rendering to the screen.
 *
 * @param[out] matrix Specifies a 3x3 matrix containing a 2D rasterization transform applied during
 * rendering.
 * @param[in] obj Specifies a handle to the renderer object.
 *
 * @return DW_INVALID_ARGUMENT, DW_SUCCESS
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderer_get2DTransform(dwMatrix3f* matrix, dwConstRendererHandle_t obj);

//#######################################################################################
// Set state
//#######################################################################################

/**
 * @brief Sets current rendering screen area expressed in pixel values.
 * Origin of rectangle coordinates is bottom-left.
 *
 * @param[out] rect Specifies the structure containing the current react area.
 * @param[in] obj Specifies a handle to renderer object.
 *
 * @return DW_SUCCESS
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderer_setRect(const dwRect rect, dwRendererHandle_t obj);

/**
 * @brief Sets viewport and scissor rectangles in pixel values.
 * Origin of rectangle coordinates is bottom-left.
 *
 * @param[in] viewport Specifies the desired viewport rectangle.
 * @param[in] scissor Specifies the desired scissor rectangle.
 * @param[in] obj Specifies a handle to renderer object.
 *
 * @return DW_SUCCESS
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderer_setViewportAndScissorRects(const dwRect viewport, const dwRect scissor, dwRendererHandle_t obj);

/**
 * @brief Sets current rendering color.
 *
 * @param[in] color Specifies the color as float[4] values in 0..1 range.
 * @param[in] obj Specifies a handle to renderer object.
 *
 * @return DW_INVALID_ARGUMENT, DW_SUCCESS
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderer_setColor(const dwVector4f color, dwRendererHandle_t obj);

/**
 * @brief Sets current rendering color mapping scale
 *
 * @param[in] colorMapScale Specifies a scale factor via float32_t[3] in meters for the saturation range.
 * @param[in] obj Specifies a handle to renderer object.
 *
 * @return DW_INVALID_ARGUMENT, DW_SUCCESS
 * @note colorScale[0], colorScale[1] and colorScale[2] to x, y, z respectively.
 * E.g. colorMapScale = {150, 150, std::numeric_limits<double>::infinity()} indicates the color will wrap after
 * 150 meters in the x-y plane.
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderer_setColorMapScale(const dwVector3f colorMapScale, dwRendererHandle_t obj);

/**
 * @brief Sets current ModelView matrix.
 * @param[in] matrix Specifies the 4D homogeneous matrix as float[16] col major. OpenGL ModelView matrix.
 * @param[in] obj Specifies a handle to the renderer object.
 * @return DW_INVALID_ARGUMENT, DW_SUCCESS
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderer_setModelView(const dwMatrix4f* matrix, dwRendererHandle_t obj);

/**
 * @brief Sets current projection matrix.
 *
 * @param[in] matrix Specifies the 4D homogeneous matrix as float[16] col major. OpenGL projection matrix.
 * @param[in] obj Specifies a handle to renderer object.
 *
 * @return DW_INVALID_ARGUMENT, DW_SUCCESS
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderer_setProjection(const dwMatrix4f* matrix, dwRendererHandle_t obj);

/**
 * @brief Sets current point size.
 *
 * @param[in] value Specifies the float value representing the size of a point, in pixels.
 * @param[in] obj Specifies a handle to renderer object.
 *
 * @return DW_INVALID_ARGUMENT, DW_SUCCESS
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderer_setPointSize(const float32_t value, dwRendererHandle_t obj);

/**
 * @brief Sets current line width, in pixels.
 *
 * @param[in] value Specifies the float value representing the width of a line, in pixels.
 * @param[in] obj Specifies a handle to the renderer object.
 *
 * @return DW_INVALID_ARGUMENT, DW_SUCCESS
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderer_setLineWidth(const float32_t value, dwRendererHandle_t obj);

/**
 * @brief Sets current font.
 *
 * @param[in] value Specifies the enum value corresponding to the desired valid font.
 * @param[in] obj Specifies a handle to the renderer object.
 *
 * @return DW_INVALID_ARGUMENT, DW_SUCCESS
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderer_setFont(const dwRendererFonts value, dwRendererHandle_t obj);

/**
 * @brief Sets current 2D transformation applied when rendering to the screen.
 *
 * @param[in] matrix Specifies the 3x3 matrix containing the 2D rasterization applied to the data being rendered.
 * @param[in] obj Specifies a handle to the renderer object.
 *
 * @return DW_INVALID_ARGUMENT, DW_SUCCESS
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderer_set2DTransform(const dwMatrix3f* matrix, dwRendererHandle_t obj);

//#######################################################################################
// Render calls
//#######################################################################################

/**
 * @brief Renders a `RenderBuffer`.
 *
 * @param[in] buffer Specifies the buffer to render.
 * @param[in] obj Specifies a handle to the renderer object.
 *
 * @return DW_SUCCESS
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderer_renderBuffer(dwConstRenderBufferHandle_t buffer, dwRendererHandle_t obj);

/**
 * @brief Renders a set of 2D vertices. This is equivalent to rendering a RenderBuffer with only xy positions. No 2d normalization factors are used.
 *
 * @param[in] buffer Specifies the vertices to render. Vertices are in opengl [-1..+1] range.
 * @param[in] count Number of vertices.
 * @param[in] primitiveType GL primitive type.
 * @param[in] obj Specifies a handle to the renderer object.
 *
 * @return DW_SUCCESS
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderer_renderData2D(const dwVector2f* buffer, size_t count, dwRenderBufferPrimitiveType primitiveType, dwRendererHandle_t obj);

/**
 * @brief Renders a set of 3D vertices. This is equivalent to rendering a RenderBuffer with only xyz positions.
 *
 * @param[in] buffer Specifies the buffer to render. The active projection matrix determines the expected range of the vertices.
 * @param[in] count Number of vertices.
 * @param[in] primitiveType GL primitive type.
 * @param[in] obj Specifies a handle to the renderer object.
 *
 * @return DW_SUCCESS
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderer_renderData3D(const dwVector3f* buffer, size_t count, dwRenderBufferPrimitiveType primitiveType, dwRendererHandle_t obj);

/**
 * @brief Renders a 2D circle.
 *
 * @param[in] center Center of circle.
 * @param[in] radius Radius of circle.
 * @param[in] obj Specifies a handle to the renderer object.
 *
 * @return DW_SUCCESS
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderer_renderCircle(dwVector2f center, float32_t radius, dwRendererHandle_t obj);

/**
 * @brief Renders a textured quad on the screen. If this texture comes from an external device such as a
 * camera, then use `GL_TEXTURE_EXTERNAL_OES` as format instead of a regular `GL_TEXTURE_2D` for the rest.
 *
 * @param[in] inputTexture Specifies the GL texture ID to render.
 * @param[in] textureTarget Specifies the GL texture type. Only `GL_TEXTURE_2D` and `GL_TEXTURE_EXTERNAL_OES` are supported.
 * @param[in] obj Specifies a handle to the renderer object.
 *
 * @return DW_SUCCESS
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderer_renderTexture(const GLuint inputTexture, const GLenum textureTarget,
                                  dwRendererHandle_t obj);

/**
 * @brief Renders a subrect of a textured quad on the screen. If this texture comes from an external device
 * such as a camera, then use `GL_TEXTURE_EXTERNAL_OES` as the format instead of a regular `GL_TEXTURE_2D` for
 * the rest.
 *
 * @param[in] inputTexture Specifies the GL texture ID to render.
*  @param[in] textureTarget Specifies the GL texture type. Only `GL_TEXTURE_2D` and `GL_TEXTURE_EXTERNAL_OES` are supported.
 * @param[in] minx Specifies the smallest value of the x coordinate in -1..1 range.
 * @param[in] miny Specifies the smallest value of the y coordinate in -1..1 range.
 * @param[in] maxx Specifies the largest value of the x coordinate in -1..1 range.
 * @param[in] maxy Specifies the largest value of the y coordinate in -1..1 range.
 * @param[in] obj Specifies a handle to renderer object.
 *
 * @return DW_SUCCESS
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderer_renderSubTexture(const GLuint inputTexture, const GLenum textureTarget,
                                     const float32_t minx, const float32_t miny,
                                     const float32_t maxx, const float32_t maxy,
                                     dwRendererHandle_t obj);
/**
 * @brief Renders a text line on the screen.
 *
 * @param[in] x Specifies the coordinate of the start of the string, in pixel values.
 * @param[in] y Specifies the coordinate of the start of the string, in pixel values.
 * @param[in] text Specifies the text to render.
 * @param[in] obj Specifies a handle to renderer object.
 *
 * @return DW_INVALID_ARGUMENT, DW_SUCCESS
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderer_renderText(const int32_t x, const int32_t y, const char* text,
                               dwRendererHandle_t obj);

/**
 * @brief Renders a text line on the screen.
 *
 * @param[in] normX Specifies the x coordinate of the start of the string, in 0..1 range.
 * @param[in] normY Specifies the y coordinate of the start of the string, in 0..1 range.
 * @param[in] text Specifies the text to render.
 * @param[in] obj Specifies a handle to renderer object.
 *
 * @return DW_INVALID_ARGUMENT, DW_SUCCESS
 */
DW_VIZ_API_PUBLIC
dwStatus dwRenderer_renderTextNorm(const float32_t normX, const float32_t normY,
                                   const char* text,
                                   dwRendererHandle_t obj);

#ifdef __cplusplus
}
#endif
/** @} */
#endif // DW_VISUALIZATION_RENDERER_H_
