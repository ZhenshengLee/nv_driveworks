# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_renderer_sample Rendering Sample
@tableofcontents

@section dwx_renderer_description Description

The Renderer sample demonstrates some basic features of the rendering helpers
provided with NVIDIA<sup>&reg;</sup> DriveWorks. These helpers provide basic rendering of points,
lines, triangles, and textures in 2D and 3D, as well as text rendering. The
purpose of this module is not to replace a renderer but to help simple rendering
for debug visualization. The renderer uses the OpenGL ES 3.1 API.

@section dwx_renderer_running Running the Sample

The command line for the sample is:

    ./sample_renderer

@section dwx_renderer_Output Output

The sample creates a window, displays text in various sizes, and shows an
animated 2D point list.

![Rendering Sample](sample_renderer.png)

@section dwx_renderer_more Additional Information

For more details see @ref renderer_usecase1 .
