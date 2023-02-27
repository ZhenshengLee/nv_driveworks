# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_radar_plugin_sample Radar Plugin Sample
@tableofcontents

@section dwx_radar_plugin_sample_description Description

The RADAR Plugin sample implements a sensor driver for a UDP/TCP-based RADAR using the comprehensive sensor plugin framework. It can be used to replay sample raw data provided with the SDK (see @ref dwx_radar_replay_sample), visualize and record live data created by the sensor simulation tool. (see @ref dwx_sensor_simulator_tool)

It also provides sources for a refcounted-`BufferPool` data structure that may be used as reference for other implementations.

@section dwx_radar_plugin_sample_run Using The NV Radar Plugin

This radar plugin compiles into a shared library (.so) that serves as a reference on how to implement a radar plugin that is compatible with the DriveWorks Sensor Abstraction Layer (SAL).
<br>

@note The precompiled decoder library is called 'libsample_radar_plugin.so'.

<br>

<b>Live data visualization:</b>

    ./sample_radar_replay --protocol=radar.custom
                          --params=device=CUSTOM_EX,
                                   ip=XXX.XXX.XXX.XXX,
                                   port=XXXXX,
                                   protocol=[udp|tcp],
                                   decoder-path=<path_to_radar_plugin>

<b>Data replay:</b>

    ./sample_radar_replay --protocol=radar.virtual
                          --params=file=<path_to_radar_recording.bin>,
                                   decoder-path=<path_to_radar_plugin>

<b>Data recording(see @ref dwx_recording_tools):</b>

    ./recorder <path_to_rig_file>

@note A rig file that is setup to use the plugin is already provided with the SDK and is called nv_radar_plugin_rig.json.

In the following sections, an overview of the implementation details and suggested project structure when developing a plugin for a custom radar sensor will be provided.
<br>

@note For more details about the plugin interface for radars see @ref sensorplugins_radarsensor.
<br>

@section dwx_radar_plugin_sample_walk_through NV Radar Plugin Walkthrough

The NV RADAR is a generic radar sensor simulation that mimicks a radar with the following specifications:

    Supported protocols: UDP/TCP
    Scan-frequency:      20 Hz
    Maximum detections:  100

See @ref dwx_sensor_simulator_tool for more information on how to start the simulation of this radar.

@subsection dwx_radar_plugin_sample_plugin_interface Radar Plugin Interface Definition

The plugin framework defines a set of function pointer definitions which must be implemented and exported to the SAL. For Radar sensors, the plugin must have implementations for the function pointers defined in:

* @ref sensor_plugins_ext_common_group
* @ref sensor_plugins_ext_radar_group

In addition, the plugin must implement & export the function, `dwSensorRadarPlugin_getFunctionTable()`, which is the only C function that needs to be exposed from the shared object.
This allows the developer flexibility to implement internals of their plugin in C++, if necessary.

@subsection dwx_radar_plugin_sample_project_structure Project structure

The file setup for the plugin implementation is depicted in the following image. It is not an obligatory project setup, but rather a recommendation which is easy to maintain and which this plugin implementation follows.

@note The provided sample implementation can be used as a template for other sensor implementations.

![NV Radar Plugin Project Structure](radar_plugin_file_overview.png)
<br>

The project is split into three components:

* Implementation of sensor class with decoding logic:
    * NVRadar.cpp
    * NVRadar.h
    * NVRadar.hpp
* Plugin interface definition:
    * SensorCommonPlugin.h
    * RadarPlugin.h
* Mapping of sensor class funcitons to plugin function calls:
    * main.cpp
<br>

@subsection dwx_radar_plugin_sample_custom_radar_class Implementing The NV Radar Class

<b>NVRadar.cpp</b> and <b>NVRadar.h</b> contain the sensor specific functionality needed to process the data being provided by the sensor along with its initialization and life cycle management.

<b>NVRadar_Properties.hpp </b>contains characteristics of the data stream which are utilized in the decoding logic as well as information on the layout of the data structures the received raw data is expected to map to.

The specific implementation details are all accessible in the respective project files for the NV Radar plugin.

Specifics regarding the API function calls can be found in the @ref sensorplugins_radarsensor section.

@subsection dwx_radar_plugin_sample_function_mapping Function Mapping

Based on the interface definition the functions in the NVRadar class are mapped accordingly to their respective common sensor function call and sensor type specific funciton calls that the plugin API exposes. (see tables below)
<br>

<b> Common Functions </b>(see @ref sensor_plugins_ext_common_group): <br>

| API function | NVRadar member function |
|:---|:---|
| `dwSensorPlugin_createHandle()` | `NVRadar()` |
| `dwSensorPlugin_release()` | `CloseFileDescriptor()` |
| `dwSensorPlugin_createSensor()` | `CreateSensor()` |
| `dwSensorPlugin_start()` | `StartSensor()` |
| `dwSensorPlugin_stop()` | `StopSensor()` |
| `dwSensorPlugin_reset()` | `ResetSensor()` |
| `dwSensorPlugin_readRawData()` | `ReadRawData()` |
| `dwSensorPlugin_returnRawData()` | `ReturnRawData()` |
| `dwSensorPlugin_pushData()` | `PushRawData()` |

<b> Radar Specific Functions </b>(see RadarPlugin.h): <br>

| API function | NVRadar member function |
|:---|:---|
| `dwSensorRadarPlugin_parseDataBuffer()` | `ParseDataBuffer()` |
| `dwSensorRadarPlugin_getDecoderConstants()` | `GetDecoderConstants()` |
| `dwSensorRadarPlugin_validatePacket()` | `ValidatePacket()` |

Naming of the sensor class functions does not have to follow the above chosen names as it is merely a suggestion.

Once the sensor class is implemented, one can proceed to map the functions according to the table lined out above.

This happens in the project file <b>main.cpp</b> which is the missing link between the API interface calls and the custom sensor class.

Once the respective functions are populated with their counter part in the custom radar sensor class in the main.cpp project file, the last step is to map the those functions in the function table that is used by the SAL to access them (see `dwSensorRadarPlugin_getFunctionTable()` in main.cpp).

At this point the plugin implementation is complete and the project can be compiled to a shared library, ready to be used with DriveWorks. This enables processing of the custom radar sensor data in a format that is usable within DriveWorks.
