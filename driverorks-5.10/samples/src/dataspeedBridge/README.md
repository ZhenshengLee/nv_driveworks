# SPDX-FileCopyrightText: Copyright (c) 2018-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_dataspeedBridge_sample Dataspeed Bridge Sample
@tableofcontents

@section dwx_dataspeedBridge_description Description

Dataspeed Bridge sample shows how to convert CAN signals coming to/from a Dataspeed
drive-by-wire system to a generic NVIDIA-defined CAN bus format, as defined in the NVIDIA-
provided Database CAN (DBC) file.

The sample expects that the Dataspeed CAN bus messages are flowing over the actual CAN bus
or a virtual CAN network (say vcan0).

The sample intercepts the CAN bus messages, converts
them to the NVIDIA `generic` drive-by-wire format, and puts the converted messages back on the CAN bus. If the vehicleio sample
is running with `type=generic`, it picks up and displays vehicle parameters
gleaned from these generic messages, such as the status for brake, throttle, and steering.
For information on vehicleio, see @ref dwx_dataspeedBridge_more (end of this section).

The sample can also send CAN bus messages with
generic commands that influence vehicle controls such as throttle, brake, and steering.
It does so by converting the messages to Dataspeed commands and then executing them.

@section dwx_dataspeedBridge_running Running the Sample

The command line for the sample is:

    ./sample_dataspeedBridge --dbc=[path/to/dbc/file]
                             --driver=[can.socket]
                             --params=[device=canN]
                             --rig=[path/to/rig/file]

where

    --dbc=[path/to/dbc/file]
        DBC file with which to parse CAN data.
        Default value: path/to/data/samples/sensors/can/AutonomousVehicleCANSignals.dbc

    --driver=[can.socket]
        Specifies the can driver.
        Default value: can.socket

    --params=[device=canN]
        Specifies the actual can device.
        Default value: device=can0

    --rig=[path/to/rig/file]
        Path to the rig configuration file.
        Default value: path/to/data/samples/vehicleio/rig-dataspeedBridge.json

If you do not have an access to the actual Dataspeed-equipped vehicle, please see
@ref dwx_vehicleio_sample for information on how to replay pre-recorded
Dataspeed messages on either virtual CAN or physical CAN using Linux.

@if QNX
@section dwx_dataspeedBridge_qnx_notes QNX-Specific Notes

On QNX, outgoing CAN messages cannot be seen locally, thus it is not possible to
run both Dataspeed Bridge Sample and VehicleIO Sample on the same SoC, such as
Xavier 1 or 2.
You can work-around this limitation by using two separate SoCs connected to
the same CAN Bus, as shown below.

```
+--+---(Physical Dataspeed CAN Bus: Vehicle messages or replayed CAN messages from Linux host system)
|  |
|  +
|  (QNX SoC 1: Dataspeed Bridge Sample)
+
(QNX SoC 2: VehicleIO Sample using Generic VehicleIO back-end)
```
@endif

@section dwx_dataspeedBridge_more Additional information

For more information, see:
- @ref canbus_mainsection
- @ref vehicleio_mainsection
