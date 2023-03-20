# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_render_engine_sample Rendering Engine Sample
@tableofcontents

@section dwx_render_engine_description Description

The Rendering Engine sample demonstrates how to draw various primitives such as points, lines, triangles, ellipses, boxes, arrows, plots, grids, and images both in 2D and 3D space. The purpose is to add an easy way to draw these primitives.

All rendering states - font, color, line size, point size, projection and ModelView matrices - are tile based. This means that anytime the render engine is used for rendering it will render into a tile which is either part or all of the viewport.

@section dwx_render_engine_running Running the Sample

The command line for the Rendering Engine sample is:

    ./sample_render_engine

@section dwx_render_engine_output Output

The sample creates a window and displays several tiled sections of the window with different rendered primitives.

![Render Engine Sample](sample_render_engine.png)

@section dwx_render_engine_more Additional information

For more details see @ref renderer_usecase2.
