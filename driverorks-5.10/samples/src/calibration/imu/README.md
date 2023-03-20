# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_imu_calibration_sample IMU Calibration Sample
@tableofcontents

@section dwx_imu_calibration_description Description

This sample demonstrates estimating IMU extrinsics with the
NVIDIA<sup>&reg;</sup> DriveWorks Calibration Engine.

@section dwx_imu_calibration_running Running the Sample

The IMU calibration sample, `sample_calibration_imu`, accepts the following
optional parameters. If none are specified, the IMU extrinsics are estimated on
a default dataset.

    ./sample_calibration_imu --rig=[path/to/rig/configuration/file]
                             --imu-sensor=[integer/sensor-name]
                             --can-sensor=[integer/sensor-name]
                             --camera-sensor=[integer/sensor-name]

where

    --rig=[path/to/rig/configuration/file]
        Path to the rig configuration file.
        Default value: path/to/data/samples/recordings/suburb0/imu_offset_rig.json

    --imu-sensor=[integer]
        The index or name of the IMU sensor in the rig configuration file to calibrate
        Default value: 0

    --can-sensor=[integer]
        The index or name of the CAN sensor in the rig configuration file
        Default value: 0

    --camera-sensor=[integer]
        The index or name of the camera sensor in the rig configuration file (used for visualization only)
        Default value: 0

@section dwx_imu_calibration_output Output

The sample does the following:
- Creates a window.
- Displays a video. The speed at which the video is displayed differs, depending
  on convergence:
  - Before convergence, the sample does not limit the video playback. As a result,
    the visualization appears to be sped up.
  - After convergence, the sample slows the video playback to the usual 30-frames
    per second.
- Displays nominal calibration indicators (blue) and, after convergence, a
  corrected calibration indicator (green). The indicator shows the estimated
  rig horizon as seen in the camera's frame. The rig horizon shows the estimated
  IMU roll and pitch.

After convergence, the sample runs the sample data in a loop, during which the
calibration is further refined.

![IMU Calibration](sample_calibration_imu.png)

@section dwx_imu_calibration_more Additional information

For more information on IMU calibration, see @ref calibration_usecase_imu .
