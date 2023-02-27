# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_recorder_tool Basic Recording Tool

@note This tool is available in both **NVIDIA DriveWorks** and **NVIDIA DRIVE Software** releases.

This tool is available on the x86 Host System, NVIDIA DRIVE<sup>&trade;</sup> OS Linux, and NVIDIA DRIVE<sup>&trade;</sup> OS QNX.

This is a simple recording tool intended to be used as a back-end process
for more advanced recording applications. While it can be used as is, it is encouraged
to use either the @ref dwx_recorder_textui_tool or @ref dwx_gui_recording2_tool.

@section recorder_starting Starting the Recording Application

Run this tool by executing:

    sudo -s
    cd /usr/local/driveworks/tools/capture
    ./recorder <rig-file>

The tool supports the following commands:

| Command                  | Action          |
| ------------------------ | --------------- |
| `s [record_path]<Enter>` | start recording |
| `s /dev/null<Enter>`     | stop recording  |
| `q <Enter>`              | quit            |

If no `record_path` is specified using the start command above, the tool uses the
current working directory as its recording path.

For related information, see:
- @ref dwx_recording_devguide_group
- @ref dwx_config_ref

@section recorder_configuring Configuring the Recorder

See @ref dwx_config_ref
