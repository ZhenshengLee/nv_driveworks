# SPDX-FileCopyrightText: Copyright (c) 2017-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_video_exporter_tool Video Exporter

This tool is not available on QNX.

The NVIDIA<sup>&reg;</sup> DriveWorks Video Exporter tool exports demosaiced and tonemapped
 H.264 video in a MP4 container from the various video formats. For more
information, see the <em>NVIDIA DriveWorks Release Notes</em>.

# Usage

Run this tool by executing:

    ./video_exporter --input-file=<input raw or encoded video> --output-file=<output h264, h265 or mp4>

### Optional Arguments

The following arguments are optional:
- `--timestamp-file`: Output file containing timestamps in case raw or lraw video, or input timestamp file for h264 input.
- `--useSoftISP`: if 1 uses deprecated SoftISP for raw to h264/mp4 conversion, otherwise it uses TegraISP.
- `--denoise`: Denoising method to use with SoftISP. Options are 'none' or 'bilateral'. The second is by default.
- `--demosaic`: Demosaicing method to use with SoftISP. Options are 'interpolate' or 'downsample'. Default is 'downsample'.
- `--start` : Start frame of output video. 0 by default.
- `--duration` : Number of frames in output video.
- `--interval` : Number of frames to skip per output frame [0-300].
- `--quality` : quality factor for rate control [0-50], 0 - best quality, 50 - worse quality.
- `--camera-name` : camera name override to be used when searching in the SIPL database.
- `--camera-timeout` : Timeout (in microseconds) to read frames from the camera sensor.
- `--verbose` : if 1 enables verbose logging from DW

@note When using h264 to mp4 conversion input video is re-encoded with specified input parameters. If the goal is to wrap the stream with mp4 container as is without changes use muxer_mp4.
