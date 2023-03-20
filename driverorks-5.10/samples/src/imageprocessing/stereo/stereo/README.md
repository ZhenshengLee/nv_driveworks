# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_stereo_disparity_sample Stereo Disparity Sample
@tableofcontents

@section dwx_stereo_disparity_description Description

The Stereo Disparity sample demonstrates the stereo pipeline.

The sample reads frames from two stereo videos and rectifies them on the basis
of the camera calibration contained in the rig configuration file. Then, it runs the
frames through the stereo pipeline and displays a confidence map and final stereo
output.

@section dwx_stereo_disparity_running Running the Sample

The command line for the sample is:

    ./sample_stereo_disparity --rigconfig=[path/to/rig/configuration/file]
                              --video-left=[path/to/the/left/video]
                              --video-right=[path/to/the/right/video]
                              --sensor-left=[name/of/left/camera/in/rig/file]
                              --sensor-right=[name/of/right/camera/in/rig/file]
                              --level=[0|1|2|3]
                              --single_side=[0|1]
                              --maxDisparity=[maximum/disparity/for/stereo/match]
                              --maxDistance=[maximum/distance/in/meters/for/depth/rendering]

where

    --rigconfig=[path/to/rig/configuration/file]
        Points to the rig file that contains the intrinsics and extrinsics for the stereo camera.
        Default value: path/to/data/samples/stereo/full.json

    --video-left=[path/to/the/left/video]
        Is the path to the video recorded from the left camera.
        Default value: path/to/data/samples/stereo/left_1.h264

    --sensor-left=[name/of/left/camera/in/rig/file]
        Is the name of the left input sensor in the rig file.
        Defaul value: left_60FOV

    --sensor-right=[name/of/right/camera/in/rig/file]
        Is the name of the right input sensor in the rig file.
        Defaul value: right_60FOV

    --video-right=[path/to/the/right/video]
        Is the path to the video recorded from the right camera.
        Default value: path/to/data/samples/stereo/right_1.h264

    --level=[0|1|2|3]
        Defines the pyramid level to display the disparity, depends on the number of levels.
        Default value: 1

    --single_side=[0|1]
        If `--single_side` is 0, the sample computes left and right stereo images
        and performs complete stereo pipeline. Otherwise, it computes only the left
        image and approximates occlusions by thresholding the confidence map.
        Default value: 0

    --maxDisparity=[maximum/disparity/for/stereo/match]
        Maximum disparity value for stereo matching between left and right camera.
        Default value: 128

    --maxDistance=[maximum/distance/in/meters/for/depth/rendering]
        Maximum distance in meters for depth rendering.
        Every pixel with a depth greater than maxDistance is rendered as blue
        Default value: 40.0

It is possible to use keyboard input to change parameters at runtime:

    0-6: changes the level of refinement (0 no refinement)
    O  : toggles occlusions
    K  : infills occlusions (only if on)
    +,-: changes invalidy threshold (appears as white pixels)
    I  : toggle horizontal infill of invalidity pixels
    L  : switch between disparity view and rectification view

It is possible to left click on the disparity images to obtain the correspoding point depth value.

@section dwx_stereo_disparity_output Output

The sample creates a window and displays the disparity view composed by:

- Top: Anaglyph of left and right image
- Bottom: Stereo images

The stereo output is color coded for clarity and some pixels are masked if they
are occluded or invalid.

![stereo disparity](sample_stereo_disparity.png)

By pressing `L` the view switches the rectification view composed by:
- Top: Input left and right image with parallel red line showing that epipolar lines are not parallel.
- Bottom: Rectified images.

![Unrectified and rectified images](sample_stereo_rectifier.png)

@section dwx_stereo_disparity_more Additional information

For more details see @ref stereo_mainsection .
