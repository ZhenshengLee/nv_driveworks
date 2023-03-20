# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_egomotion_sample Egomotion Sample
@tableofcontents

@section dwx_egomotion_description Description

The Egomotion sample application shows how to use steering angle and velocity CAN measurements.
It also explains how to use IMU and GPS measurements to compute vehicle position and orientation
within the world coordinate system.

@section dwx_egomotion_running Running the Sample

The command line for the sample is:

    ./sample_egomotion --camera-sensor-name=[name]
                       --vehicle-sensor-name=[name]
                       --imu-sensor-name=[name]
                       --gps-sensor-name=[name]
                       --mode=[0|1]
                       --output=[path/to/output/file]
                       --outputkml=[path/to/output/file]
                       --rig=[path/to/rig/file]
                       --speed-measurement-type=[0|1|2]
                       --enable-suspension=[0|1]

where

    --camera-sensor-name=[name]
        Name of the camera sensor in the given rig file.
        Default value: First camera sensor found in the rig file.

    --vehicle-sensor-name=[name]
        Name of the sensor providing vehicle data in the given rig file.
        Default value: First CAN sensor found in the rig file.

    --gps-sensor-name=[name]
        Name of the GPS sensor in the given rig file.
        Default value: First GPS sensor found in the rig file.

    --imu-sensor-name=[name]
        Name of the IMU sensor in the given rig file.
        Default value: First IMU sensor found in the rig file.

    --mode=[0|1]
        The sample application supports different egomotion estimation modes.
        To switch the mode, pass `--mode=0/1` as the argument.
        Mode 0 represents odometry-based egomotion estimation.
        The vehicle motion is estimated using Ackerman principle.
        Mode 1 uses IMU measurements to estimate vehicle motion.
        Gyroscope and linear accelerometers are filtered and fused to estimate vehicle orientation.
        Using speed vehicle odometry reading, the vehicle's traveling path can be estimated.
        This mode also filters GPS locations if they are passed to the module.
        Default value: 1

    --output=[path/to/output/file]
        If specified, the sample application outputs the odometry data to this file. The vehicle's position
        in world coordinates (x,y) together with a timestamp in microseconds will be written out as:
        ....
        5680571648,-7.67,118.30
        5680604981,-7.75,118.33
        5680638314,-7.83,118.36
        5680671647,-7.91,118.38
        5680704980,-7.98,118.41
        Default value: none

    --outputkml=[path/to/output/file]
        If specified, the sample application outputs the GPS and estimated location.
        Default value: none

    --rig=[path/to/rig/file]
        Rig file containing all information about vehicle sensors and calibration.
        Default value: path/to/data/samples/recordings/suburb0/rig.json

    --speed-measurement-type=[0|1|2]
        Speed measurement to be used, refer to dwEgomotionSpeedMeasurementType.
        Default value: 1

    --enable-suspension=[0|1]
        Enables egomotion suspension modeling. It requires Odometry+IMU [--mode=1].
        Default value: 0

You must provide the following file:
- Rig file: contains the rig configuration and sensor parameters.

@note for more details on sensors parameters and usage refer to @ref canbus_mainsection, @ref gps_mainsection, @ref imu_mainsection.

@subsection dwx_egomotion_examples Examples

#### Running the sample with default arguments

    ./sample_egomotion

#### Running the sample with output file

    sudo ./sample_egomotion --output=/home/nvidia/out.txt

@section dwx_egomotion_output Output

The sample application creates a window, displays a video, and plots the vehicle's position at a 30 Hertz sampling rate.
Current speed, roll, pitch, and yaw are also printed.

![Egomotion Sample](sample_egomotion.png)

@section dwx_egomotion_more Additional Information

For more details see @ref egomotion_mainsection.
