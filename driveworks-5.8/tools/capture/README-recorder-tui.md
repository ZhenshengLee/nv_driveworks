# SPDX-FileCopyrightText: Copyright (c) 2020-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_recorder_textui_tool TextUI Recording Tool

@note This tool is available in both **NVIDIA DriveWorks** and **NVIDIA DRIVE Software** releases.

This tool is available on the x86 Host System and NVIDIA DRIVE<sup>&trade;</sup> OS Linux.

This is a front-end for the @ref dwx_recorder_tool, and provides the following additional
features:

1. @ref tui_auto_storage_selection
2. @ref tui_manual_storage_selection
3. @ref tui_sensor_statistics
4. @ref tui_multissd_recording
5. @ref tui_distributed_recording
6. @ref black_box_recorder

@section tui_starting Starting the Recording Application

Run this tool by executing:

    cd /usr/local/driveworks/tools/capture
    ./recorder-tui <rig file> OR <rig_directory> [--bbr]

    `bbr` starts the recorder in black box recording (bbr) mode.

After initialization, the tool supports the following commands:

| Command              | Action                    |
| -------------------- | --------------------------|
| `s <Enter>`          | toggle recording          |
| `q <Enter>`          | quit                      |
| `start <Enter>`      | start recording           |
| `stop <Enter>`       | stop recording            |
| `bbr <Enter>`        | bbr recording mode        |
| `continuous <Enter>` | continuous recording mode |
| `h264 <Enter>`       | record camera in h264     |
| `lraw <Enter>`       | record camera in lraw     |

`bbr`, `continuous`, `h264` and `lraw` switch is supported only when the recorder is in stopped state.

@section tui_user_interface User Interface

The textual interface of the tool is illustrated in the screenshot below:

1. @ref tui_sensor_statistics
2. Last error/output message

![Recording Tool - TextUI Interface](tool_tui_recorder.png)

@section tui_auto_storage_selection Automatic Storage Selection

This tool can discover SSD/eSATA disks that are mounted on the target and
meet the following criteria:

1. `ext4` filesystem format
2. `> 5GB` free space available
3. visible in `/proc/mounts`
4. Mountpoint does not contain any spaces or non-standard characters

This tool automatically selects the first available disk for recording.
Once a disk becomes full (free space < 5GB), output files will be put into the
next available disk automatically.

@section tui_manual_storage_selection Manual Storage Selection

If automatic disk selection is not convenient, one can manually specify
the recording directory in the rig configuration file using the "recorder_storage_paths":

    {
        "rig": {
            "recorder_storage_map": {
                "<sensor_name_1>": "1",
                "<sensor_name_2>": "0",
                "<sensor_name_3>": "0",
                <...>
            },

            "recorder_storage_paths": [
                "<your_storage_path_1>",
                "<your_storage_path_2>",
                <...>
            ],
            <...>
        },
        <...>
    }

"recorder_storage_map" is used to map the sensor to the index of "recorder_storage_paths" list. Example: "sensor_name_1"
is mapped to index "1" which is "your_storage_path_2". If the sensor map is not provided, then the default value of
the storage index "0" will be selected.

@note The manually specified mountpoint must still be on an `ext4` filesystem
and have `> 5GB` in free space.

@section tui_sensor_statistics Sensor Throughput Statistics

This tool measures sensor statistics (at disk sink).
1. Data throughput per sensor
2. Data written per sensor
3. Data written for current recording on current sink

@section tui_multissd_recording Multi-SSD Recording

The @ref tui_manual_storage_selection feature can be used to perform multi-SSD
recording on a single Xavier device.

To achieve this, a user must create multiple rig files, each with
their specified storage path and subset of sensors, and place them
inside a `<rig_directory>`.

The tool can then be launched as below:

    ./recorder-tui <rig_directory>

@section black_box_recorder Black Box Recorder (BBR)

Black Box Recorder (BBR) records sensor data clips around a vehicle disengagement trigger event.
BBR automatically deletes the recording folders of current recording session expect the last three.
If a lateral or longitudinal disengagement occurs, BBR preserves the 30 seconds before and after the
point of disengagement. The disengagement folder will have two or three recording folders depending
on the exact time of the disengagement. BBR internally uses recorder-tui, so the recording data format
and directory structure will be identical to the normal recording case.

    ./recorder-tui <rig_directory> --bbr

@note BBR relies on Dataspeed CAN messages to find the disengagement triggers.

@section tui_distributed_recording Distributed Recording

See @ref dwx_devguide_rec_distrec

@section tui_configuring Configuring the Recorder

See @ref dwx_config_ref

@section Session Recording Tags

When using recorder-tui to record a session, it can be useful to be able to tag these sessions.
With recorder-tui you can pass in `--tag <tag-name>` to set a tag for the recorded session which
will appear in the aux_info file. The valid values for this tag will need to be specified under the
`tags.txt` file which should appear under your configuration directory that you pass to recorder-tui.

@section Encryption Recording

When using recorder-tui to record a session, we can enable or disable encryption. It can be passed as a
parameter `--disable-encryption` or `--enable-encryption`, by default, encryption is disabled. When encryption
is enabled, encryption key is required to encrypt the recording. Encryption key can be provided as a
parameter `--rsa-key <path-to-encryption>`, by default "/home/nvidia/.ssh/recorder-aiinfra.pem" path is used.

@section Input Fields

The recorder-tui requires input arguments to collect metadata for a given drive. These fields values are
collected as command-line inputs before the recorder launches a recording session. These fields are stored
in the yaml file 'auxFile', which is written to the recording session directory.

Fields required:
**Route**: Four digit code used to identify the route recorded in the given session.

**Lane**: (optional) Certain data collection use-cases require a single lane to be selected for drives along a given route.
This field can be left blank to indicate that no particular lane is being driven in the current drive session.
Lanes are designated 1-8 with 1 being the rightmost lane and indexes increasing right-to-left like so:

 | |   |   |   |   |
 |M|   |   |   |   |
 |e|   |   |   |   |
 |d|   |   |   |   |
 |i|   |   |   |   |
 |a| 4 | 3 | 2 | 1 | 0 = No lane selected
 |n|   |   |   |   |
 | |   |   |   |   |

If you are not performing a data collection drive in a specific lane leave this section blank.

**Login-ID**: Name of the car operator.

**Description**: (optional) Text entry field for the car operator to include manual notes.
