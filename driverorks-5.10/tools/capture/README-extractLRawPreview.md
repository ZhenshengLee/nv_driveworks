# SPDX-FileCopyrightText: Copyright (c) 2018-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_tools_extractlraw LRAW Preview Extraction Tool
@tableofcontents

This tool is available on the x86 Host System and NVIDIA DRIVE<sup>&trade;</sup> OS Linux.

The NVIDIA<sup>&reg;</sup> DriveWorks LRAW Preview Extract tool extracts the
encoded H.264 lossy preview data in lraw2.0 file and outputs to a file.

Every thirtieth frame (1 FPS) in an LRAW2.0 file contains
lossy encoded Preview data.
This data can be used for quick preview of contents.
LRAW2.0 file format defines these encoding types:
- Lossless compression of Raw RCCB camera input using h264 Lossless Hi444 Profile(244).
  This is decompressed and played out by all sample applications like `sample_camera_replay`.
- Lossy compression of YUV420 preview frame provided by camera, downscaled to 720p and compressed in h264 lossy mode.
  We encode 1 in 30 preview frames and embed it in .lraw file for quick preview.
  This utility extractLRawPreview, parses the lraw file and extracts the embedded preview data in that and dumps it to a file,
  so it can be decoded offline and seen through any h264 player.

@section dwx_tools_extractLRawPreview_arguments Input Arguments

This tool supports following arguments:

    --inputfile=<input.lraw file>       The fullpath of the input lraw recording in LRAW V2 format.
    --outputfile=<outputfile.h264>      The fullpath of the output preview data in h264 format.

@section dwx_tools_extractLRawPreview_running Running the Tool

The usage is `./extractLRawPreview`, for which the preview data is extracted from the provided LRAW input.

    ./extractLRawPreview --inputfile=<input.lraw file> --outputfile=<outputfile.h264>
