# SPDX-FileCopyrightText: Copyright (c) 2020-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_rig_json2json_tool Rig Reserializer Tool
@tableofcontents

@section dwx_rig_json2json_tool_description Description

The NVIDIA<sup>&reg;</sup> DriveWorks Rig Reserializer Tool deserializes and serializes an input rig file, upgrading it to the latest version in the process.<br>

@section dwx_rig_json2json_tool_prereqs Prerequisites

This tool is available on the x86 Host System, NVIDIA DRIVE<sup>&trade;</sup> OS Linux and NVIDIA DRIVE<sup>&trade;</sup> OS QNX.

This tool creates an output file that is:
- Named [input_rig_name]-new.json if no output rig file is specified
- Stored at the same path as the input rig file if no absolute path for the output rig file is specified
- Stored at the absolute path provided via the output rig file argument

@section dwx_rig_json2json_tool_usage Running the Tool

Run this tool by executing:

    ./rig_json2json input.json [output.json]

@subsection dwx_rig_json2json_tool_params Positional parameters

    [path to input file]
	    Description: The rig file to be re-serialized.
	        			 If there is no rig file present, the tool will not launch.
	        Default value: N/A

    [path to output file]
	    Description: The rig file containing the resulting serialized values with the latest version.
	        Default value: <input-rig>-new.json

@section dwx_rig_json2json_examples Examples

	./rig_json2json /home/user/input-rig.json
            Results in /home/user/input-rig-new.json

	./rig_json2json /home/user/input-rig.json /home/user/result/output-rig.json
            Results in /home/user/result/output-rig.json

	./rig_json2json /home/user/input-rig.json output-rig.json
            Results in /home/user/output-rig.json
