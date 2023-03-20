# SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_gui_recording2_tool GUI Recording Tool

@note This tool is available in both **NVIDIA DriveWorks** and **NVIDIA DRIVE Software** releases.

This tool is available on the x86 Host System and NVIDIA DRIVE<sup>&trade;</sup> OS Linux.

The GUI recording tool builds on top of the @ref dwx_recorder_textui_tool,
but provides a graphical user interface for operation and visualizing
sensor status.

@section gtgui2_starting Starting the Recording Application

Run this tool by executing:

    cd /usr/local/driveworks/tools/capture
    ./recorder-qtgui <rig file> OR <rig_directory>

@section qtgui2_using_interface Using the Interface

The home screen of the recorder GUI is composed of several parts as
illustrated in the following screenshot.
1. [Recorder](@ref Recorder) button.
2. [Settings](@ref Settings) button.
3. Exit button.

![Recording Tool GUI - Home](tool_gui_recorder_home.png)

<a name="recorder">
## Recorder {#Recorder}

The main screen of the recorder GUI is composed of several parts as
illustrated in the following screenshot:
1. Main button to return to Home screen.
2. [Route](@ref Route) setting drop boxes.
3. [Start Recording](@ref Start_Recording) control button.
4. [Storage Status](@ref storage_status) info.
5. [Sensor Status](@ref sensor_status).
6. VIN info.

![Recording Tool GUI - Recorder Screen](tool_gui_recorder_recorder.png)

<a name="route">

### Route {#Route}

The route can be set by selecting numbers from the 4 drop boxes.
This semantics of route are user-defined, but a file containing this
route number will be included in the recording.

![Recording Tool GUI - Route](tool_gui_recorder_route.png)

<a name="start_recording">

### Start Recording {#Start_Recording}

When you press the Record button, the application starts recording and
the Start button is replaced by a Stop Recording button.

![Recording Tool GUI - Recording](tool_gui_recorder_recorder_recording.png)

The Elapsed time item marked in the picture indicates the time elapsed
since the start of the recording. Pressing the RECORD button again stops
the current recording procedure.

@note The RECORD button will be disabled with the message `Disk Full/No Disk`
in case no valid storage medium is found or the available disks are full.

@note The RECORD button will be disabled with the message `Failure` in case
one of the sensor initialization fails during startup. The log to diagnose
the failure can be found on the console.

<a name="sensor_status">

### Sensor Status {#sensor_status}

If sensor initialization succeeds, a block for the sensor containing its name
is added to the grid in the recoder window. These blocks also indicate the
data rate (measured at disk sink) for that sensor. The color of this block
is used to indicate the status - green and red, respectively.

<a name="storage_status">

### Storage Status {#storage_status}

The Storage field shows the storage usage, including:

1. Total free percentage.
2. Estimated remaining recording time.
3. Usage of each available storage device.

The selection of storage device for the current recording can be made by
opening the storage drop box, as shown below.

![Recording Tool GUI - Storage](tool_gui_recorder_storage.png)

<a name="settings">

## Settings {#Settings}

<a name="storage_management">
### Storage Management

This screen allows one to erase disk volumes that are connected to the device.
In a distributed recording environment, this screen has capability to erase
disks on all slave devices, as well.

![Recording Tool GUI - Storage Management](tool_gui_recorder_settings_storage.png)

<a name="vin_selection">
### VIN selection
On the general setting page, the last 6 digits of the VIN number can be selected,
and will be in a file in the recording folder.

![Recording Tool GUI - VIN Selection](tool_gui_recorder_settings_general.png)

<a name="driver_selection">
### Driver selection
On the general settings page, the name of the driver performing the recording can be
selected from a list. This will be included as part of the metadata file (named "aux_info")
that is stored in the recording folder.

The list of available driver names is specified in a text file, named "login.txt".
This file should contain a single driver name per line, as shown in the example below:

    Test Driver #1
    Test Driver #2

The "login.txt" file shall be placed in the same directory as the rig configuration file.
In case a rig_directory is used, place this file in the rig_directory.

@section qtgui2_distributed_recording Distributed Recording

See @ref dwx_devguide_rec_distrec

@section qtgui2_configuring Configuring the Recorder

See @ref dwx_config_ref
