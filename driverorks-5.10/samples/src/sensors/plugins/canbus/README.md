# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_canbus_plugin_sample CAN Plugin Sample
@tableofcontents

@section dwx_can_plugin_sample_description Description

The CAN Plugin sample implements a sensor driver for a CAN
using the comprehensive sensor plugin framework. This uses a DriveWorks
@ref canbus_mainsection sensor to provide raw data.

It also provides sources for a refcounted-`BufferPool` data structure
that may be used as reference for other implementations.

@section dwx_can_plugin_sample_run Running the sample

This sample compiles as a shared library (.so) that can be used with the
DriveWorks Sensor Abstraction Layer (SAL).

This plugin can be used to test and verify functionality:

    ./sample_canbus_logger --driver=can.custom
                           --params=decoder-path=[path_to_decoder.so]

For playing back CAN data recorded with a custom plugin, the following
command line can be used:

    ./sample_canbus_logger --driver=can.virtual
                           --params=file=[path_to_recording.bin],
                                    decoder-path=[path_to_decoder.so]

@note The precompiled decoder library is called `libsample_can_plugin.so`.

@section dwx_can_plugin_sample_more Additional Information

For more information, please refer to @ref sensorplugins_canbussensor.
