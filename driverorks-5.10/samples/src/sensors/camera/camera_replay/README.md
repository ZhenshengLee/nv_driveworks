# SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_camera_replay_sample Camera Replay Sample
@tableofcontents

@section dwx_camera_replay_sample_description Description

The Camera Replay sample demonstrates H.264/265/MP4 playback by using a hardware decoder, and replaying RAW/LRAW recorded files on the target hardware or desktop emulation. It opens a window to play back the provided video file.

@section dwx_video_replay_sample_running Running the Sample

The Camera Replay sample, sample_camera_replay, accepts the following parameters

    ./sample_camera_replay --video=[path/to/video]

Where:

    --video=[path/to/video]
        Path to the video file.
        Default value: path/to/data/samples/sfm/triangulation/video_0.h264

@subsection dwx_video_replay_sample_examples Examples

#### Replay .h264 file

    ./sample_camera_replay --video=/path/to/file.h264

#### Replay RAW file

    ./sample_camera_replay --video=/path/to/file.raw

#### Replay LRAW file:

  For playback of LRAW recorded files the option for using PinnedMemory
  to save cycles during CudaMemCopy has been added to this utility.

    ./sample_camera_replay --video=/path/to/file.lraw

@section dwx_video_replay_sample_output Output

![Single H.264 stream](sample_camera_replay.png)

The sample creates a window and displays a video.

@section dwx_video_replay_sample_more Additional Information

For more details see @ref camera_mainsection.

Press Enter to toggle between Software ISP and Xavier ISP output when replaying RAW/LRAW.
Press 'F' to take a frame grab. Press 'S' to take a screenshot.