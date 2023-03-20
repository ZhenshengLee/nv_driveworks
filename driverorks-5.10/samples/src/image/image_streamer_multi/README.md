# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_image_streamer_multi_sample Multi-Thread Image Streamer Sample
@tableofcontents

@section dwx_image_streamer_multi_description Description

The Multi-Thread Image Streamer sample demonstrates how to use an image streamer in a multi-thread environment.
It consumes a CPU image.

The sample shows how to create, setup, use and release
an image streamer in multi-thread. It does the following:
1. Manually creates a dwImageCPU object.
2. Streams the dwImageCPU to a dwImageCUDA object.
3. Applies a NVIDIA<sup>&reg;</sup> CUDA<sup>&reg;</sup> kernel on it.
4. Streams the resulting image to a dwImageGL object.
5. Renders it on screen.

@section dwx_image_streamer_multi_running Running the Sample

The command line for the sample is:

    ./sample_image_streamer_multi

@section dwx_image_streamer_multi_output Output

The sample creates a window and renders a colored pattern.

![multi thread image streamer](image_streamer_multi.png)

At the same time the state of the two threads is printed on console:

    nvidia@tegra-ubuntu:/usr/local/driveworks/bin$ ./sample_image_streamer_multi
    This sample illustrates how to use an image streamer given a CPU image. This will create an empty dwImageCPU, stream it to a dwImageCUDA, apply some simple operations in a kernel and then stream it to a dwImageGL for rendering. The purpose is to show how to properly create, use and destroy an image streamer.
    [10-8-2018 9:5:10] Initialize DriveWorks SDK v1.2.227
    [10-8-2018 9:5:10] Release build with GNU 4.9.4 from v1.2.0-rc6-0-g79beb2a against Vibrante PDK v5.0.10.3
    [10-8-2018 9:5:10] Platform: Detected Drive PX2 - Tegra A
    [10-8-2018 9:5:10] TimeSource: monotonic epoch time offset is 1533299678306576
    [10-8-2018 9:5:10] TimeSource: PTP ioctl returned error. Synchronized time will not be available.
    [10-8-2018 9:5:10] TimeSource: Could not detect valid PTP time source at 'eth0'. Fallback to CLOCK_MONOTONIC.
    [10-8-2018 9:5:10] Platform: number of GPU devices detected 2
    [10-8-2018 9:5:10] Platform: currently selected GPU device discrete ID 0
    [10-8-2018 9:5:10] SDK: Resources mounted from .././data/resources
    [10-8-2018 9:5:10] SDK: Create NvMediaDevice
    [10-8-2018 9:5:10] SDK: Create NvMediaIPPManager
    [10-8-2018 9:5:10] SDK: use EGL display as provided
    Starting producer...
    WindowGLFW: create shared EGL context
    Consumer, acquiring...
    Producer, posting...
    Producer, posted, now waiting...
    Consumer, completed
    Consumer, acquiring...
    Producer, completed.
    Producer, posting...
    Producer, posted, now waiting...
    Consumer, completed
    Consumer, acquiring...
    Producer, completed.
    Producer, posting...
    Producer, posted, now waiting...
    Consumer, completed
    Timing results:
    Thread main:
    -onProcess                CPU:    21us, std= 213       | GPU:    10us, std= 255       | samples=35
    -onRender                 CPU: 10394us, std=165141       | GPU: 10399us, std=165143       | samples=35

    [10-8-2018 9:5:13] SDK: Release NvMediaDevice
    [10-8-2018 9:5:13] Driveworks SDK released
    [10-8-2018 9:5:13] SDK: Release NvMedia2D

@section dwx_image_streamer_multi_more Additional information

For more details see @ref image_mainsection.
