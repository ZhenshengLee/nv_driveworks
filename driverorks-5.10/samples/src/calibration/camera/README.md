# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_camera_calibration_sample Camera Calibration Sample
@tableofcontents

@section dwx_camera_calibration_description Description

This sample demonstrates the estimation of camera extrinsic calibration parameters using the
NVIDIA<sup>&reg;</sup> DriveWorks Calibration Engine.

@section dwx_camera_calibration_running Running the Sample

The camera calibration sample, `sample_calibration_camera`, accepts the following optional parameters.
If none are specified, the camera extrinsics are estimated on a default dataset.

    ./sample_calibration_camera --rig=[path/to/rig/configuration/file]
                                --camera-sensor=[integer/sensor-name]
                                --imu-sensor=[integer/sensor-name]
                                --can-sensor=[integer/sensor-name]
                                --signals='default', or any combination of ['pitchyaw','roll','height'] substrings
                                --feature-max-count=[integer]

where

    --rig=[path/to/rig/configuration/file]
        Path to the rig configuration file.
        Default value: path/to/data/samples/recordings/suburb0/rig.json

    --camera-sensor=[integer]
        The index or name of the camera sensor in the rig configuration file to calibrate
        Default value: 0

    --imu-sensor=[integer]
        The index or name of the IMU sensor in the rig configuration file
        Default value: 0

    --can-sensor=[integer]
        The index or name of the CAN sensor in the rig configuration file
        Default value: 0

    --signals='default', or any combination of ['pitchyaw','roll','height'] substrings
        The camera extrinsic parameters to estimate, given as any combination of
        ['pitchyaw','roll','height'] substrings, or 'default'.
        For instance, the combination `pitchyawroll` will enable estimation of all
        orientation components (roll+pitch+yaw), if supported by the chosen method.
        'default' enables signals that are well-supported by the calibrated sensor.
        For instance, the calibration of less signals might be activated by default
        for side-facing cameras compared to front-facing cameras.
        Default value: default

    --fast-acceptance: ['default', 'enabled', 'disabled']
        If previously accepted estimates are available, fast-acceptance is a method
        to reduce re-calibration times in case the previous estimates can be
        validated with latest measurements. This option allows to configure the
        fast-acceptance behaviour of the camera calibration routine.
        Default value: 'disabled' (to illustrate calibration from scratch)

    --feature-max-count=[integer]
        The maximum number of features for the tracker.
        Default value: 800

@section dwx_camera_calibration_output Output

The sample does the following:
- Creates a window.
- Displays a video.
- Displays nominal calibration indicators (blue) and, after convergence,
  corrected calibration indicators (green).

Indicators show the estimated rig horizon and forward directions,
as seen from the camera's pose relative to the rig. The resulting display
shows a visualization of the nominal and estimated extrinsic calibration parameters.

![Camera Calibration](sample_calibration_camera.png)

@section dwx_camera_calibration_more Additional information

For more details on camera calibration see @ref calibration_usecase_features .
