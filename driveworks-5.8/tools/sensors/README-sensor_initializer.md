# SPDX-FileCopyrightText: Copyright (c) 2017-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_sensor_initializer_tool Sensors Initialization Tool

The NVIDIA<sup>&reg;</sup> DriveWorks Sensor Initializer tool initializes and starts
sensors described in a rig configuration file.

Run the tool by executing:

    ./sensor_initializer --rig=<path/to/rig/file>

This tool does not actually perform any specific operation. It simply starts
the sensor and consumes the incoming data.

### Command Line Options ###

The following lists the required and optional command line arguments.

#### Required Arguments ####
- `--rig`: Path to the rig file describing the sensor to be initialized (e.g., `--input=/home/nvidia/rig.json`)
