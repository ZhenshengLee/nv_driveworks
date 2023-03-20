# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_vehicleio_sample VehicleIO Sample
@tableofcontents

@section dwx_vehicleio_description Description

The VehicleIO sample demonstrates how to read and write the state of
the vehicle actuators (i.e. throttle, brake, steering, etc.) It can talk to
different vehicle control systems, one of which is Dataspeed. The
@ref vehicleio_mainsection module abstracts the underlying physical protocols and
connections. The internal "driver layer" handles the protocol interface and
the VehicleIO module handles the state.

The VehicleIO sample has been tested with CAN-based Dataspeed vehicle control system and
validated with Ford MKZ and Ford Fusion vehicles. To send the
steering, brake or throttle commands, use a keyboard or a USB joystick
(tested with Logitech Gamepad F310).

The VehicleIO sample can run with the Dataspeed drive-by-wire (DbW) systems.
It can also run with a generic NVIDIA DbW driver. The
`AutonomousVehicleCANSignals.dbc` DBC file describes the CAN commands for that
driver.

If you run the vehicleIO sample with the generic driver, you must also run the
DataspeedBridge utility. That utility converts the CAN messages and commands
to/from DbW systems to a generic type.

@section dwx_vehicleio_running Running the Sample

The command line for the sample is:

    ./sample_vehicleio [--rig=path/to/rig/file] [--allow-no-safety=[no|yes]]

where

    --rig=[path/to/rig/file]
        Points to the rig file.
        Default value: path/to/data/samples/samples/vehicleio/rig.json

    --allow-no-safety=[no|yes]
        Set to 'yes' to allow issuing actuation commands without
        software-imposed limits.
        Default value: no

While the sample is running, the following commands are available:

- E - enable the vehicle control system
- D - disable the vehicle control system
- Arrow key UP - incrementel throttle with every arrow key UP press
- Arrow key DOWN - incremental braking with every arrow key DOWN press
- Arrow key LEFT - incremental steering rotation (counter-clockwise) with every key press
- Arrow key RIGHT - incremental steering rotation (clockwise) with every key press

@note The vehicle control system is only available when using real CAN or linux VCAN.

@subsection dwx_vehicleio_examples Examples

#### Run VehicleIO sample with the Dataspeed driver

To use with the pre-recorded CAN data, which is provided with the sample,
run the sample without any arguments:

    ./sample_vehicleio

To use a custom pre-recorded CAN data or to run live, one should provide a
custom rig file and use `--rig` argument:

    ./sample_vehicleio --rig=rig.json

One can use the provided (default) rig.json file as a template (exact location
of that file can be found using `./sample_vehicleio --help`.)

#### Run VehicleIO sample with the generic DbW Driver

Before you can run the sample with a generic driver, you must run the utility
DataspeedBridge. This sample converts the messages from Dataspeed to a generic type and
vice-versa.

You can use the CAN0 interface in a vehicle equipped with the Dataspeed DbW
system. With that setup, you can use the generic vehicleIO driver directly in a
CAR, where the Dataspeed system is connected to CAN0.

The following steps assume your DbW system is connected to CAN0.

After running the dataspeedBridge, run the vehicleio sample with a Rig
Configuration that has only generic VehicleIO node:

```
{
    "rig": {
        "vehicleio": [
            {
                "type": "generic",
                "parent-sensor": "can:vehicle:generic",
                "dbc-file": "AutonomousVehicleCANSignals.dbc"
            }
        ],
        "sensors": [
            {
                "name": "can:vehicle:generic",
                "parameter": "device=can0",
                "protocol": "can.socket",
                "properties": null,
                "nominalSensor2Rig": { "quaternion": [ 0.0, 0.0, 0.0, 1.0 ], "t": [ 0.0, 0.0, 0.0 ] },
                "sensor2Rig": { "quaternion": [ 0.0, 0.0, 0.0, 1.0 ], "t": [ 0.0, 0.0, 0.0 ] }
            }
        ],
...
}
```

Then run the sample with this new rig.json file:

    ./sample_vehicleio --rig=rig.json

#### Using Generic VehicleIO via VCAN0 interface on Bench

If you want to use the generic vehicleio driver in a bench setup, you can use
the VCAN interface.

1. Enable the VCAN interface.

       sudo modprobe vcan
       sudo ip link add dev vcan0 type vcan
       sudo ip link set up vcan0

2. Start transmitting Dataspeed can messages over the VCAN interface.

       while true; do cat candump_dataspeed.log | canplayer -l i vcan0=can0; done

3. Run `sample_dataspeedBridge` to convert Dataspeed messages to generic
   vehicleio messages (refer to @ref dwx_dataspeedBridge_sample for details).

4. Prepare a rig file that has vcan0 sensor and Generic VehicleIO node (as
   described above), then run the sample:

       ./sample_vehicleio --rig=rig.json

@section dwx_vehicleio_output Output

The sample opens a window to display the various vehicle control parameters,
such as throttle, braking, and steering. Once the window is in the focus, the
throttle, brakes, steering of the vehicle can be controlled via a keyboard or a
USB joystick, as described above.

The sample prints the basic vehicle state information to the terminal.

![VehicleIO sample](sample_vehicleio.png)

@section dwx_vehicleio_more Additional information

For more information, see @ref vehicleio_mainsection .
