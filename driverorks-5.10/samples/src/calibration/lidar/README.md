# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_lidar_calibration_sample Lidar Calibration Sample
@tableofcontents

@section dwx_lidar_calibration_description Description

The NVIDIA<sup>&reg;</sup> DriveWorks Lidar Calibration sample demonstrates how to use DRIVE Calibration's @ref calibration_mainsection module. The sample uses ICP (Iterative
Closest Point) to determine the relative transform between consecutive
sweeps of Lidar. Additionally, the sample performs computations on
the full point cloud of the latest sweep. Those additional computations are
independent of the relative transformation.

@section dwx_lidar_calibration_running Running the Sample

The Lidar Calibration sample, `sample_calibration_lidar`, accepts the following optional parameters. If none are specified, the Lidar extrinsics are estimated on
a default dataset.

    ./sample_calibration_lidar --rig=[path/to/rig/configuration/file]
                               --lidar-sensor=[integer/sensor-name]
                               --imu-sensor=[integer/sensor-name]
                               --can-sensor=[integer/sensor-name]
                               --output-rig=[output/rig/file]
                               --run-once=[0|1]
                               --use-ego-pose=[0|1]
                               --verbose=[0|1]

where

    --rig=[path/to/rig/configuration/file]
        Path to the rig configuration file.
        Default value: path/to/data/samples/lidar/rig_perturbed.json

    --lidar-sensor=[integer]
        The index or name of the lidar sensor in the rig configuration file to calibrate
        Default value: 0

    --imu-sensor=[integer]
        The index or name of the IMU sensor in the rig configuration file
        Default value: 0

    --can-sensor=[integer]
        The index or name of the CAN sensor in the rig configuration file
        Default value: 0

    --output-rig=[output/rig/file]
        Output rig configuration file, which contains updated Lidar Extrinsics.
        Default value: rig_updated.json

    --run-once=[0|1]
        Controls whether the runs through a dataset once, rather than a loop.
        Default value: 0

    --use-ego-pose=[0|1]
        Whether or not ego-motion pose is fed to Lidar calibration (in addition to ICP pose).
        Default value: 0

    --verbose=[0|1]
        Whether or not the sample prints detailed estimation together with Lidar sweep number.
        Default value: 0

To pause the sample, press `SPACE`.
\n To exit the sample, press `ESC`.
\n To rotate and move the camera while the sample is paused, select and drag the image.

@note Depending on `--rigOutFile`, you may need to start the sample with **sudo**.

@section dwx_lidar_calibration_output Output

The sample application shows the previous sweep (Red) and the aligned current sweep (Green).
After the calibration converges, the contour rectangle on the ground plane should
be aligned co-planar with the ground Lidar points and parallel to the driving direction.

The sample application also shows the state of the calibration and the percentage completed,
along with the computed update to the Lidar extrinsics calibration.

![Lidar Calibration Sample](sample_lidar_self_calib.png)

@section dwx_lidar_calibration_more Additional information

For more information on Lidar calibration, see @ref calibration_usecase_lidar.
