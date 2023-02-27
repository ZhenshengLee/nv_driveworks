# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_rig_viewer_tool Rig Viewer Tool
@tableofcontents

@section dwx_rig_viewer_tool_description Description

The NVIDIA<sup>&reg;</sup> DriveWorks Rig Viewer Tool displays the rig configuration file passed as an input parameter in 3D.<br>
It visualizes the ego-car and its sensors as oriented boxes based on information specified in the rig configuration file.<br>
As an option, it can also visualize the coverage of Lidar-sensors on the ground plane around the vehicle.

@section dwx_rig_viewer_tool_prereqs Prerequisites

This tool is available on the x86 Host System and NVIDIA DRIVE<sup>&trade;</sup> OS Linux.

This tool creates output files that are placed into the current working directory by default. Please ensure the following for your convenience:
- Write permissions are enabled for the current working directory.
- Include the tools folder in the binary search path of the system.
- Execute from your home directory.

@section dwx_rig_viewer_tool_usage Running the Tool

Run this tool by executing:

    ./rig_viewer --rig=[input file]
    			 [--offscreen=[0|1|2]]
    			 [--profiling=[0|1]]

@subsection dwx_rig_viewer_tool_params Parameters

    --rig=[path to input file]
	        Description: The rig file containing all the sensor configurations required for the tool.
	        			 If there is no rig file present, the tool will not launch.
	        Example: --rig=wwdc_rig.json
	        Default value: ../../data/tools/rig_viewer/wwdc_rig.json

    --offscreen=[0|1|2]
	        Description: Used to run windowed apps in headless mode:
	        			 '0' = Display window.
	        			 '1' = Offscreen window.
	        			 '2' = No window created.
	        Default value: 0

	--profiling=[0|1]
			Description: Enables or disables sample profiling.
						 '0' = Disables sample profiling.
						 '1' = Enables sample profiling.
			Default Value: 1

@section dwx_rig_viewer_example Example

@subsection dwx_rig_viewer_example_launch Launching in Offscreen Window while Sample Profiling is Enabled

	./rig_viewer --rig=../../data/tools/rig_viewer/wwdc_rig.json
				 --offscreen=0
				 --profiling=1

@section dwx_rig_viewer_tool_output Output

The Rig Viewer Tool displays a window where the ego-vehicle is represented by a green bounding box, and each sensor is represented by yellow bounding boxes.<br>
Clicking on each sensor allows you to view additional information on the bottom left of the screen. You can also click and drag the cursor around the screen to manipulate the view.

![Basic View](tool_rig_viewer.png)
