# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_struct_from_motion_sample Structure from Motion (SFM) Sample
@tableofcontents

@section dwx_struct_from_motion_description Description

The Structure from Motion (SFM) sample demonstrates the triangulation functionality of
the SFM module; a car pose is estimated entirely from CAN data using the NVIDIA<sup>&reg;</sup>
DriveWorks egomotion module. The car has a 4-fisheye camera rig that is pre-calibrated.
Features are detected and tracked using the features module. Points are
triangulated for each frame by using the estimated pose and tracked features. Vehicle motion is parsed from CAN data
using a custom VehicleIO plugin.

@section dwx_struct_from_motion_running Running the Sample

The structure from motion sample, sample_sfm, accepts the following optional parameters. If none are specified, it will process four supplied pre-recorded video.

    ./sample_sfm --baseDir=[path/to/rig/dir]
                 --rig=[rig.json]
                 --maxFeatureCount=[integer]
                 --trackMode=[0|1]
                 --useHalf=[0|1]
                 --displacementThreshold=[fp_number]
                 --enableAdaptiveWindowSize=[0|1]

where

    --baseDir=[path/to/rig/dir]
        Path to the folder containint the rig.json file.
        Default value: path/to/data/samples/sfm/triangulation

    --rig=[rig.json]
        A `rig.json` file as serialized by the DriveWorks rig module, or as produced by the DriveWorks calibration tool.
        The rig must include the 4 camera sensors and one CAN sensor.
        The rig sensors must contain valid protocol and parameter properties to open the virtual sensors.
        The video files must be a H.264 stream.
        Video containers as MP4, AVI, MKV, etc. are not supported.
        The rig file also points to a video timestamps text file where each row contains the frame index
        (starting at 1) and the timestamp for all cameras. It is read by the camera virtual sensor.
        The rig file must contain a valid VehicleIO driver.
        Default value: rig.json

    --maxFeatureCount=[integer]
        The manixum amount of features stored for tracking.
        Default value: 2000

    --trackMode=[0|1]
        Defines feature tracking mode.
        0: translation-only KLT tracker or Sparse LK PVA tracker
        1: translation-scale fast KLT tracker
        Default value: 0

    --useHalf=[0|1]
        Defines whether to use fp16 for tracking.
        --useHalf=0 uses fp32 for tracking.
        --useHalf=1 uses fp16 for tracking.
        This parameter only takes effect when --trackMode=1.
        Default value: 0

    --displacementThreshold=[fp_number]
        Defines the early stop threshold during translation-only tracking.
        This parameter only takes effect when --trackMode=1.
        Default value: 0.1

    --enableAdaptiveWindowSize=[0|1]
        Defines whether to use full window size at the lowest and the highest levels,
        and smaller window size at the rest of levels during tracking.
        This parameter only takes effect when --trackMode=1.
        Default value: 1

If a mouse is available, the left button rotates the 3D view, the right button
translate, and the mouse wheel zooms.

While the sample is running the following commands are available:
- Press V to enable / disable pose estimation.
- Press F to enable / disable feature position prediction.
- Press Space to pause / resume execution.
- Press Q to switch between different camera views.
- Press R to restart playback.

@section dwx_struct_from_motion_output Output

The left side of the screen shows the 4 input images; tracked features are shown
in green. Triangulated points are reprojected back onto the camera and shown in
red. The right side shows a 3D view of the triangulated point cloud.

In 3D, the colors are:

- Red points = points from frontal camera
- Green points = points from rear camera
- Blue points = points from left camera
- Yellow points = points from right camera
- Green/red line = path of the car

![Structure from Motion sample](sample_triangulation.png)

@section dwx_struct_from_motion_more Additional Information

For more details see @ref sfm_mainsection .
