# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_camera_usb_sample USB Camera Capture Sample
@tableofcontents

@section dwx_camera_usb_sample_description Description

The USB Camera Capture sample captures video input from USB cameras, and displays the output in an opened window. This sample is compatible with Linux and NVIDIA DRIVE<sup>&trade;</sup> platforms.

@section dwx_camera_usb_sample_running Running the Sample

A camera must be connected to a USB port.
The USB Camera Capture sample, sample_camera_usb, accepts the following parameters:

    ./sample_camera_usb --device=[integer]
                        --mode=[a|b|integer]
                        --record-file=[path/to/output/file]

Where:

    --device=[integer]
            Is the device ID of the camera.
            Default value: 0

    --mode=[a|b|integer]
            Applicable for generic camera only. Specifies
            a method for selecting capture settings:
            `a`: choose mode with maximum resolution
            `b`: choose mode with maximum fps
            integer number: choose mode by index
            Default value: 0

    --record-file=[path/to/output/file]
            Specifies the path to the captured video.
            This option is only available on x86.
            Default value: none

@subsection dwx_camera_usb_sample_examples Examples

#### To test the first camera present on the system

    ./sample_camera_usb

#### To test the third camera present on the system

    ./sample_camera_usb --device=2

@section dwx_camera_usb_sample_output Output

The sample opens a window displaying the input from the camera.

![Single consumer-grade USB camera capturing input](sample_camera_usb.png)

@section dwx_camera_usb_sample_more Additional information

For more details see @ref camera_mainsection.
