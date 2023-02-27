# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_imagetransformation Image Transformation Sample
@tableofcontents

@section dwx_imagetransformation_description Description

The Image Transformation sample demonstrates the basic image processing functions within DriveWorks.

It takes an H.264 video as input, reads the frames sequentially, processes the image, and displays the output next to the original video.

@section dwx_imagetransformation_usage Running the Sample

The command line for the sample is:

    ./sample_imagetransformation --video=[path/to/video]

where

    --video=[path/to/video]
        Points to a recorded video.
        Default value: path/to/data/samples/sfm/triangulation/video_0.h264

@section dwx_imagetransformation_output Output

The sample creates a window, processes the image, and displays the output to the right of the original video. The simplest output is a cropeed, resized and converted to monochrome image
The sample allows the user to select the are to transform.

Press T to toggle Thresholding and switch between different modes

![Image Transformation Output](image_transformation_sample.png)

@section dwx_imagetransformation_more Additional information

For more details, see @ref imagetransformation_mainsection.
