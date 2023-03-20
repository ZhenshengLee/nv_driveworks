# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_sensor_indexer_tool Sensor Indexer Tool
@tableofcontents

@section dwx_sensor_indexer_tool_description Description

The NVIDIA<sup>&reg;</sup> DriveWorks Sensor Indexer Tool creates seeking table files for all supported sensors.<br>
Any virtual sensor that supports index seeks can use those index tables for random access to the data streams.<br>

@section dwx_sensor_indexer_tool_prereqs Prerequisites

This tool is available on the x86 Host System and NVIDIA DRIVE<sup>&trade;</sup> OS Linux.

This tool creates output files that are placed into the current working directory by default. Please ensure the following for your convenience:
- Write permissions are enabled for the current working directory.
- Include the tools folder in the binary search path of the system.
- Execute from your home directory.

@section dwx_sensor_indexer_tool_usage Running the Tool

Run this tool by executing:

    ./sensor_indexer --sensor=[virtual sensor name]
                     --input=[path to sensor file]
                     --timestamp=[path to timestamp file]
                     [--output=[path to output file]]

@subsection dwx_sensor_indexer_tool_params Parameters

    --sensor=[virtual sensor name]
            Description: The name of the virtual sensor used to parse the file.

    --input=[path to sensor file]
            Description: The input file containing recorded data for this sensor.

    --timestamp=[path to timestamp file]
            Optional: The input file containing timestamp for h264 stream

    --output=[path to output file]
            Optional: The path where the index file needs to be stored. If not provided [input].seek will be used.

@section dwx_sensor_indexer_tool_output Output

The Sensor Indexer Tool creates a `.seek` file with the same name as the input file. It is placed into the current working directory by default.<br>
Ensure write permissions are enabled for the current working directory.
