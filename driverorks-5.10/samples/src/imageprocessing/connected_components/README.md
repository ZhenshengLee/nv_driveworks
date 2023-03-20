# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_connected_components_sample Connected Components Sample
@tableofcontents

@section dwx_connected_components_description Description

The Connected Components sample demonstrates how to utilize the connected components
in dw_imageprocessing module.

@section dwx_connected_components_running Running the Sample

The command line for the sample is:

    ./sample_connected_components  --video=<path/to/video/file>

where

    --video=<path/to/video/file>
        Path to input video file.
        Default: path/to/data/samples/recordings/suburb0/video_0_roof_front_120.mp4

@section dwx_connected_components_output Output

The sample creates a window and plays a video processed with a connected components labeling algorithm. Every frame is converted to grey scale image and threshold is applied to binarize it. The algorithm then assigns unique label to every connected region. Note there is no any tracking included to the sample. Thus as assigned labels may differ from frame to frame the video may flicker a bit.

![Labeled image](sample_ccl.png)

@section dwx_connected_components_more Additional information

For more details on camera calibration see @ref connectedcomponents_mainsection.
