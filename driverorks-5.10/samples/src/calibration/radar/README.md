# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_radar_calibration_sample Radar Calibration Sample
@tableofcontents

@section dwx_radar_calibration_descrption Description

The NVIDIA<sup>&reg;</sup> DriveWorks Radar self-calibration sample shows how to use the
DriveWorks Radar self-calibration module. The sample uses RadarDopplerMotion to determine the radar motion for each radar scan.
Then, use the corresponding radar motion to estimate the radar yaw angle. Using radar we can calibrate
also other odometry properties. That is `velocity_factor`, which maps speed measurement as reported by the odometry to speed as measured using radars,
and `wheel_radius[]` which is the radius of each wheel.

@section dwx_radar_calibration_running Running the Sample

The Radar calibration sample, `sample_calibration_radar`, accepts the following optional parameters. If none are specified, the Radar extrinsics are estimated on
a default dataset.

    ./sample_calibration_radar --rig=[path/to/rig/configuration/file]
                               --output-rig=[output/rig/file]
                               --radar-sensor=[integer/sensor-name]
                               --camera-sensor=[integer/sensor-name]
                               --can-sensor=[integer/sensor-name]
                               --calibrate-odometry-properties=[1]

where

    --rig=[path/to/rig/configuration/file]
        Path to the rig configuration file.
        Default value: data/samples/recordings/highway0/rig8Radars.json

    --output-rig=[output/rig/file]
        Output rig configuration file, which contains updated Radar Extrinsics, velocity_factor, and updated wheel radii.
        Default value: rig_updated.json

    --radar-sensor=[integer]
        The index or name of the radar sensor in the rig configuration file to calibrate
        Default value: 0

    --camera-sensor=[integer]
        The index or name of the camera sensor in the rig configuration file (used for visualization only)
        Default value: 0

    --can-sensor=[integer]
        The index or name of the CAN sensor in the rig configuration file
        Default value: 0

    --calibrate-odometry-properties=[1]
        Bitwise combination of: [1] wheel radius calibration using selected radar [2] enable odometry speed factor. Specifying [3] will enable both.

To pause the sample, press `SPACE`.
\n To exit the sample, press `ESC`.

@note Depending on `--rigOutFile`, you may need to start the sample with **sudo**

@section dwx_radar_calibration_output Output

![Radar Calibration Sample](sample_calibration_radar.png)

@section dwx_radar_calibration_more Additional information

For more information on Radar calibration, see @ref calibration_usecase_radar.
