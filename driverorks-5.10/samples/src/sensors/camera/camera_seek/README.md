# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_camera_seek_sample Camera Seek Sample
@tableofcontents

@section dwx_camera_seek_sample_description Description

The Camera Seek sample demonstrates how to replay a video and use the 'seek to timestamp/event' feature to seek to any point in the video.

@section dwx_camera_seek_sample_running Running the sample

The Camera Seek sample, sample_camera_seek, accepts the following parameters:

    ./sample_camera_seek --video=[path/to/video]

where:

    --video=[path/to/video]
        Path to the video file.
        Default value: path/to/data/samples/sfm/triangulation/video_0.h264

The sample works with the following keyboard inputs:

    T  : changes the seek mode to timestamp.
    F  : changes the seek mode to frame event.
    Left Arrow  : steps backward (10 for frames, 1000000 for timestamp).
    Right Arrow  : steps forward (10 for frames, 1000000 for timestamp).
    Space : pauses the video.

Seeking is available in both normal and paused states.

@subsection dwx_camera_seek_sample_examples Examples

#### Replay .h264 file

    ./sample_camera_seek --video=/path/to/file.h264

@section dwx_camera_seek_sample_output Output

The sample creates a window and displays a video.

![Single H.264 stream](sample_camera_seek.png)

@section dwx_camera_seek_sample_more Additional Information

For more details see @ref camera_mainsection .
