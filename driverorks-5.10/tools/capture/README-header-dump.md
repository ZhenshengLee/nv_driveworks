# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_headerdump_tool Recording Header Dump

This tool is available on the x86 Host System and NVIDIA DRIVE<sup>&trade;</sup> OS Linux.

This tool prints common header information of specified recording file.

# Usage

    ./header-dump --file=path/to/file

### Optional Arguments

The following arguments are optional:

- `--sensor-type`: argument is optional but is recommended to use. Otherwise some information like sensor parameters is not printed.
- `--scan-events`: forces index table to be created (if not exists) to output timestamp data and number of events in a file. The option may slow down the tool.
