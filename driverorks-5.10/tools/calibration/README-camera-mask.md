# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_calibration_camera_mask Camera Mask Calibration Tool
@tableofcontents

@section dwx_calibration_camera_mask_description Description

The NVIDIA<sup>&reg;</sup> DriveWorks Camera Mask tool serializes camera masks in an output rig file, given an input rig file and a corresponding car model. \n The masks are used to discriminate the pixels that belong to the vehicle.

@section dwx_calibration_camera_mask_prerequisites Prerequisites

This tool is available on the x86 Host System.

This tool creates output files that are placed into the current working directory by default. Please ensure the following for your convenience:
- Write permissions are enabled for the current working directory.
- Include the tools folder in the binary search path of the system.
- Execute from your home directory.

@section dwx_calibration_camera_mask_usage Running the Tool

Run this tool by executing:

    ./calibration-camera-mask --rig=[rig file]
    						  --model=[OBJ car model]
    						  --output_rig=[output rig]
    						  [--extra_leeway=[value from 0-100]]
    						  [--wireframe]

@subsection dwx_calibration_camera_mask_parameters Parameters

	--rig=[path to input rig file]
			Description: The path for the input rig file containing the ftheta camera definitions to be rendered.
						 Only ftheta cameras are supported at this time.
			Example: --rig=.././data/tools/camera_mask/ford-fusion-rig.json

	--model=[path to OBJ file]
			Description: The path for the OBJ file of the vehicle's 3D model.
			Example: --model=.././data/tools/camera_mask/ford-fusion.obj

	--output_rig=[path to output rig file]
			Description: The path for the modified output rig file with an encoded camera mask.
			Example: --output_rig=output_rig.json

	--extra_leeway=[value from 0-100]
			Description: Extra margin for the camera mask in percent value.
						 This parameter is optional.
			Default value: 5
			Example: --extra_leeway=15

	--wireframe
			Description: If this argument is specified, it enables a point-like reprojection of the CAD model,
						 instead of creating a silhouette mask. It can be used for debugging purposes.
						 This parameter is optional.

@section dwx_calibration_camera_mask_example Example

    ./calibration-camera-mask --rig=.././data/tools/camera_mask/ford-fusion-rig.json
    						  --model=.././data/tools/camera_mask/ford-fusion.obj
    						  --output_rig=output_rig.json

@section dwx_calibration_camera_mask_output Output

A visualization of the mask with a corresponding image is demonstrated below.

![Camera Mask Calibration Tool Output] (tool_camera_mask.png)