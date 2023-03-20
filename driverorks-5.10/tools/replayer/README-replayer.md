# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_replayer_tool Replayer Tool
@tableofcontents

@section dwx_replayer_tool_description Description

The Replayer tool replays sensor data captured with the @ref dwx_recording_tools, and visualizes all recorded data in the rig configuration file. The visualization settings for each individual sensor is specified in the command line.

@section dwx_replayer_tool_prerequisites Prerequisites

This tool is available on the x86 Host System and NVIDIA DRIVE<sup>&trade;</sup> OS Linux.

On DRIVE platforms it is only supported on iGPU.
To specify the iGPU, you must prepend `CUDA_VISIBLE_DEVICES=1` before the tool command.

@section dwx_replayer_tool_usage Running the Tool

@subsection dwx_replayer_tool_rigfile Playing Sensors Listed in Rig File

Run the tool by executing:

    export CUDA_VISIBLE_DEVICES=1
    ./replayer --rig=<pathToRigConfigurationFile> 
	           [--rig-sensor-filter=<sensor name filter REGEX, e.g., 'camera|lidar']
			   [--motion-compensation=1]

@subsection dwx_replayer_tool_individual Playing Individual Sensors

Run the tool by executing:

    ./replayer --camera=[path to video]
			   --timestamp=[path to timestamp]
	           --can=[path to CAN data]
	           --candbc=[path to DBC file]
               --imu=[path to IMU data]
	           --gps=[path to GPS data]
	           --lidar=[path to Lidar data]
               --radar=[path to Radar data]

where

	--camera=[path to video]
			Description:

@section dwx_recording_tools_output Output

![Replayer Tool](tool_replayer.png)
