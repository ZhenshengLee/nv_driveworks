# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_image_streamer_simple_sample Simple Image Streamer Sample
@tableofcontents

@section dwx_image_streamer_simple_description Description

The Simple Image Streamer sample demonstrates how to use an image streamer.

The sample has no inputs and is intended as a guide on how to properly create, setup, use and release
an image streamer. The sample does the following:
1. Manually creates a dwImageCPU object.
2. Streams it to a dwImageCUDA object.
3. Applies an NVIDIA<sup>&reg;</sup> CUDA<sup>&reg;</sup> kernel on it.
4. Streams the resulting image to dwImageGL object.
5. Renders the image on screen.

@section dwx_image_streamer_simple_running Running the Sample

The command line for the sample is:

    ./sample_image_streamer_simple

@section dwx_image_streamer_simple_output Output

The sample creates a window and renders a colored pattern.

![simple image streamer](image_streamer_simple.png)

@section dwx_image_streamer_simple_more Additional information

For more details see @ref image_mainsection.
