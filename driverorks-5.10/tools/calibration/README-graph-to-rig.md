# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_calibration_graph_to_rig Calibrated Graph to Rig File Tool
@tableofcontents

@section dwx_calibration_graph_to_rig_description Description

The NVIDIA<sup>&reg;</sup> DriveWorks Calibrated Graph to Rig File tool extracts the relevant parts from a JSON calibrated graph file as input, into a JSON rig file as output.<br>
This output file can be parsed by the @ref rig_mainsection module within DriveWorks.

@section dwx_calibration_graph_to_rig_prereqs Prerequisites

This tool is available on the x86 Host System and NVIDIA DRIVE<sup>&trade;</sup> OS Linux.

This tool creates output files that are placed into the current working directory by default. Please ensure the following for your convenience:
- Write permissions are enabled for the current working directory.
- Include the tools folder in the binary search path of the system.
- Execute from your home directory.

@section dwx_calibration_graph_to_rig_usage Running the Tool

Run the tool by executing:

    /calibration-graph-to-rig --graph=[path to input calibrated-graph.json file]
                              --rig=[path to input rig.json file]
                              --output=[path to output rig file]

@subsection dwx_calibration_graph_to_rig_parameters Parameters

    --graph=[path to input calibrated-graph.json file]
            Description: The input calibrated graph file.
                         If only this parameter is provided, a `rig.json` file is created from scratch.

    --rig=[path to input rig.json file]
            Description: The input rig file.
                         If this parameter and `--graph` are both provided, the calibration data from the `calibrated-graph.json` file
                         is merged with the input `rig.json` file.

    --output=[path to output rig file]
            Description: The output rig file. If this parameter is specified, the `calibrated-graph.json` and `rig.json` files
                         are saved to a separate file. The default setting overwrites the input `rig.json` file.

@section dwx_calibration_graph_to_rig_output Output

The tool generates a `rig.json` file in the current folder as output. This file lists all cameras used during calibration with their intrinsic \n and extrinsic calibration data. If an existing `rig.json` file is passed to the application as input, all camera entries are modified with the new calibration results.<br>
For additional information regarding the format for the produced JSON file, please refer to @ref rig_mainsection and @ref rigconfiguration_usecase0.

@section dwx_calibration_graph_to_rig_additional Additional Information

This tool uses the center of the rear axle projected to the ground as a coordinate system origin.<br>
For additional information regarding this coordinate system, please refer to @ref dwx_coordinate_systems.
