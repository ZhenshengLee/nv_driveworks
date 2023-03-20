# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_gps_plugin_sample GPS Plugin Sample
@tableofcontents

@section dwx_gps_plugin_sample_description Description

The GPS Plugin sample implements a sensor driver for a CAN-based GPS
using the comprehensive sensor plugin framework. This uses a DriveWorks
@ref canbus_mainsection sensor to provide raw data.

@section dwx_gps_plugin_sample_run Running the sample

This sample compiles as a shared library (.so) that can be used with the
DriveWorks Sensor Abstraction Layer (SAL).

This plugin can be used in conjunction with the @ref dwx_gps_loc_sample to test
and verify functionality:

    ./sample_gps_logger --driver=gps.custom
                        --params=decoder-path=[path_to_decoder.so],
                                 can-proto=[can.virtual|can.socket],
                                 [file=<path_to_can_recording.bin>|device=<can_device>]
                        --timestamp-trace=[true|false]

For playing back GPS data recorded with a custom plugin, the following
command line can be used:

    ./sample_gps_logger --driver=gps.virtual
                        --params=file=[path_to_recording.bin],
                                 decoder-path=[path_to_decoder.so]

@note The precompiled decoder library is called `libsample_gps_plugin.so`.

@section dwx_gps_plugin_sample_more Additional Information

For more information, please refer to @ref sensorplugins_gpssensor.
