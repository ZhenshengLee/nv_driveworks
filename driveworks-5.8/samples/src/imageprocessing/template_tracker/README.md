# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_template_tracker_sample Template Tracker Sample
@tableofcontents

@section dwx_template_tracker_description Description

The Template Tracker sample demonstrates the template tracking capabilities of
the @ref imageprocessing_tracking_mainsection module. It loads a video stream and
reads the images sequentially. For each frame, it tracks templates from the
previous frame.

@section dwx_template_tracker_running Running the Sample

The template tracker sample, sample_template_tracker, accepts the following optional parameters.
If none are specified, it will perform detections on a supplied pre-recorded video.

    ./sample_template_tracker --video=[path/to/file.h264]
                              --trackMode=[0|1]
                              --pyramidLevel=[positive number]
                              --pva=[0|1]
                              --pvaEngineNo=[0|1]

where

    --video=[path/to/video.h264]
        Is the absolute or relative path of a h264 video.
        Containers such as AVI, MKV, etc. are not supported.
        Default value: path/to/data/samples/sfm/triangulation/video_0.h264.

    --trackMode=[0|1]
        Allows to select the algorithm used for tracking.
        0: Inverse Additive KLT.
        1: Inverse Compositional KLT.
        Default value: 0

    --pyramidLevel=[positive number]
        Defines the number of pyramid levels to build image while tracking.
        Higher level provides better results for large motion, but takes more time.
        This does not have any effect if pva is enabled.
        Default value: 1

    --pva=[0|1]
        When set to 1, the template tracker algorithm is performed on PVA hardware.
        Otherwise, it is performed on GPU.
        Note that PVA can only be used in combination with --trackMode=1.
        Default value: 0

    --pvaEngineNo=[0|1]
        If pva is enabled, determines on which PVA the tracker should run on.
        Default value: 0

@section dwx_template_tracker_output Output

The sample creates a window, displays the video, and overlays the list of template boxes. It does not detect new templates.

1. Press space to pause the video.
2. Use your mouse to draw bounding boxes.
3. Press space to start the video again.

The video pauses automatically if there are no templates in the video.

![Tracked template bounding boxes on a single H.264 stream](sample_template_tracker.png)

@section dwx_template_tracker_more Additional Information
The sample will restart from beginning with the initial bounding boxes when hitting 'R'.

For more details see @ref imageprocessing_tracking_usecase1.
