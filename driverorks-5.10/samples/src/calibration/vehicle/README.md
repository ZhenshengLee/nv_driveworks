# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_vehicle_steering_calibration_sample Steering Calibration Sample
@tableofcontents

@section dwx_vehicle_steering_calibration_description Description

The Steering Calibration sample demonstrates estimating vehicle steering offset parameter with the
NVIDIA<sup>&reg;</sup> DriveWorks Calibration Engine.

@section dwx_vehicle_calibration_running Running the Sample

The Steering Calibration sample, `sample_calibration_steering`, accepts the following optional parameters. If none are specified, the vehicle parameter are estimated on
a default dataset.

    ./sample_calibration_steering --rig=[path/to/rig/configuration/file]
                                  --cameraIndex=[integer]
                                  --canIndex=[integer]
                                  --imuIndex=[integer]

where

    --rig=[path/to/rig/configuration/file]
        Path to the rig configuration file.
        Default value: path/to/data/samples/recordings/highway0/rig.json

    --cameraIndex=[integer]
        The index of the camera in the rig configuration file
        Default value: 0

    --canIndex=[integer]
        The index of the vehicle CAN in the rig configuration file
        Default value: 0

    --imuIndex=[integer]
        The index of the IMU in the rig configuration file
        Default value: 0

@section dwx_vehicle_calibration_output Output

The sample does the following:
- Creates a window.
- Displays a video. The calibration state and final calibration result are shown in the bottom left corner.<br>
  The speed at which the video is displayed differs, depending on convergence.<br>
  After convergence, the sample slows the video playback to the usual 30-frames per second.

After convergence, the sample runs the sample data in a loop, during which the
calibration is further refined.

![Vehicle Steering Calibration](sample_calibration_steering.png)

@section dwx_vehicle_calibration_more Additional information

For more information on vehicle calibration, see @ref calibration_usecase_vehicle .
