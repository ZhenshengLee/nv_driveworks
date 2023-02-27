# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_video_rectifier_sample Video Rectification Sample
@tableofcontents

@section dwx_video_rectifier_description Description

The Video Rectification sample demonstrates how to remove fisheye distortion
from a video captured on a camera with a fisheeye lens.

The sample reads frames from a video input recorded from an 180 fov camera and takes the
calibration from the `rig.json` file. It then performs rectification and displays
both the original and rectified video side-by-side.

@section dwx_video_rectifier_running Running the Sample

    ./sample_video_rectifier --video=[path/to/video]
                             --rig=[path/to/rig/file]
                             --camera-name=[string]
                             --fovX=[integer]
                             --fovY=[integer]
                             --record-video=[h264|h265|mp4]

where

    --video=[path/to/video]
        Points to a recorded video.
        Default value: .././data/samples/sfm/triangulation/video_0.h264

    --rig=[path/to/rig/file]
        Points to the rig file that contains calibration information.
        Default value: .././data/samples/sfm/triangulation/rig.json

    --camera-name=[string]
        Is the camera name in the rig configuration file.
        Default value: SVIEW_FR

    --fovX=[integer]
        Defines the field of view of the camera along the X axis.
        This property is only applicable to pinhole cameras.
        Default value: 120

    --fovY=[integer]
        Defines the field of view of the camera along the Y axis.
        This property is only applicable to pinhole cameras.
        Default value: 90

    --record-video=[h264|h265|mp4]
        Records the rectified output to a specified video type.
        Default value: N/A

It is possible to apply a basic homography transformation in the form of translations to the input camera.
By using LEFT/RIGHT the camera moves left and right, UP/DOWN moves forwards and backwards and
U/J moves up and down. The resulting homopgraphy is applied along with the undistortion and is
visible in the output image.

@subsection dwx_video_rectifier_examples Examples

#### To play a custom video and with a corresponding rig calibration file

    ./sample_video_rectifier --video=<video file.h264> --rig=<rig.json>

#### To use a different field of view

    ./sample_video_rectifier --fovX=90 --fovY=30

@section dwx_video_rectifier_output Output

The sample creates a window and displays the original video and the rectified version.

![Fisheye video and undistorted version](sample_rectifier.png)

@section dwx_video_rectifier_more Additional information

For more details see @ref rectifier_mainsection .
