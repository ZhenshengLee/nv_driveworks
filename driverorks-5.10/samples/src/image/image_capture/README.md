# SPDX-FileCopyrightText: Copyright (c) 2018-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_image_capture_sample Image Capture Sample
@tableofcontents

@section dwx_image_capture_description Description

The Image Capture sample shows how to record a video from an NVIDIA<sup>&reg;</sup> CUDA<sup>&reg;</sup> image or
directly from the rendering window. To differentiate the two sources, the sample renders different text on the screen.

@section dwx_image_capture_running Running the Sample

The command line for the sample is:

    ./sample_image_capture --capture-bitrate=[integer]
                           --capture-file=[path/to/output/h264/file]
                           --capture-framerate=[integer]
                           --capture-screen=[0|1]

where

    --capture-bitrate=[integer]
        Defines the suggested bitrate for the output file. The actual bitrate might be different, depending on the input.
        For example, if the input is a black windows the output frame rate will be very close to 0.
        Default value: 10000000

    --capture-file=[path/to/output/h264/file]
        Specifies the path to the captured video.
        Default value: capture.h264

    --capture-framerate=[integer]
        Specifies the rate at which frames are captured.
        Default value: 30

    --capture-screen=[0|1]
        Specifies whether to capture the current window on a synthetic CUDA image.
        Default value: 1

@subsection dwx_image_capture_examples Example

#### To record with default parameters in a specific output file

    ./sample_image_capture --capture-file=path/to/captureOutput.h264

@section dwx_image_capture_output Output

The sample creates a window and renders a coloured pattern and a floating "DriveWorks" text.

- When recording directly from screen, the text is also visible in the recording.
- When recording from a synthetic CUDA image, there is no rendered text.

![image screen capture](image_capture.png)

@section dwx_image_capture_more Additional information

For more information, see @ref image_mainsection .
