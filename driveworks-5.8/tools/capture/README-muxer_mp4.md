# SPDX-FileCopyrightText: Copyright (c) 2020-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_muxer_mp4_tool Muxer mp4

This tool is available on the x86 Host System.

The NVIDIA<sup>&reg;</sup> DriveWorks Muxer mp4 tool exports a h264 stream to mp4 format.

# Usage

Run this tool by executing:

    ./muxer_mp4 --video=<input h264 video>

Running `./muxer_mp4 --help` displays the following:

    ./muxer_mp4 --help
    --gopSize: default=1
        GOP spacing
    --offscreen: default=0
        Used for running windowed apps in headless mode. 0 = show window, 1 = offscreen window, 2 = no window created
    --output: default=
        output file
    --profiling: default=1
        Enables/disables sample profiling
    --quality: default=0
        quality factor
    --timestamp: default=
        path to timestamp file
    --video: default=
        path to input h264 stream

### Optional Arguments

The following arguments are optional:
- `--timestamp`: Input timestamp file for h264 video.
- `--output`: Full path to output mp4 file.
