# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_imu_calibration_tool IMU Calibration Tool
@tableofcontents

@section dwx_imu_calibration_tool_description Description

The NVIDIA<sup>&reg;</sup> DriveWorks IMU Calibration tool calibrates the orientation of IMU with respect to the
car ("Rig") coordinate system, and displays it on screen.

If the orientation is successfully calibrated, a new rig file containing this orientation is created in the same folder as the input rig file. \n This new rig file is labelled `<input_rig_filename>_calibrated.json`.

@section dwx_imu_calibration_prereqs Prerequisites

This tool is available on the x86 Host System and NVIDIA DRIVE<sup>&trade;</sup> OS Linux.

This tool creates output files that are placed into the current working directory by default. Please ensure the following for your convenience:
- Write permissions are enabled for the current working directory.
- Include the tools folder in the binary search path of the system.
- Execute from your home directory.

To use this tool, you first must capture IMU and CAN data, with the vehicle performing
a special calibration sequence. For more information, see how @ref dwx_imu_calibration_IMUsequence.

For an example of a suitable input rig file, please navigate to `path/to/data/tools/calibration_imu/rig.json`.

@section dwx_imu_calibration_usage Running the Tool

When you have obtained the IMU and CAN data, run the tool by executing:

    ./calibration_imu --rig=rig.json
                      --imu-sensor=[IMU sensor or name]
                      --can-sensor=[CAN sensor or name]
                      [--trigger=[float number]]
                      [--angleStdDev=[float number]]

@subsection dwx_imu_calibration_parameters Parameters

    --rig=[path to vehicle rig file]
            Description: The rig file must specify file paths to the IMU, CAN, and DBC files.
            Example: --rig=rig.json

    --imu-sensor=[IMU name or number]
            Description: The IMU sensor name or number specified in the input rig file.
            Example: --imu-sensor=imu:xsens

    --can-sensor=[IMU name or number]
            Description: The CAN sensor name or number specified in the input rig file.
            Example: --can-sensor=can:vehicle

    --trigger=[float number]
            Description: Specifies a forward acceleration trigger to record in (m/s^2).
                         Heavy vehicles such as trucks may require a smaller acceleration trigger.
                         This parameter is optional.
            Default value: 2.0
            Example: --trigger=6.1

	--angleStdDev=[float number]
            Description: The angular standard deviation threshold in degrees.
                         This parameter is optional.
            Default value: 10.0
            Example: --angleStdDev=7.6

@section dwx_imu_calibration_example Example

@subsection dwx_imu_calibration_displaying Displaying the IMU Sensor's Orientation on Screen

    ./calibration_imu --rig=rig.json
                      --imu-sensor=imu:xsens
                      --can-sensor=can:vehicle

@section dwx_imu_calibration_additional Additional Information

@subsection dwx_imu_calibration_IMUsequence To Capture IMU and CAN Data

1. Find a level surface. It is important for the surface to have as little incline as possible.

2. Start recording the IMU data with the car stopped. (To obtain the IMU data,
   use the @ref dwx_tools_recording that NVIDIA<sup>&reg;</sup> DriveWorks provides.)

3. With a Recording Tool turned on, **wait at least 5 seconds before driving**. This is for the tool to find the gravity vector.

4. Rapidly accelerate the vehicle, and rapidly stop multiple times. Try to avoid skidding.
   This is needed for the calibration utility to find forward acceleration.

5. Stop the Recording Tool.
