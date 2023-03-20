# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_canbus_message_sample CAN Message Interpreter Sample
@tableofcontents

@section dwx_canbus_message_sample_description Description

The CAN Message Interpreter sample is a simple CAN bus interpreter sample.
An interpreter is built based either on the definition in a DBC file or
a set of user-provided callbacks and input CAN messages are then decoded
by the interpreter. In this sample, information about car steering and
speed is transmitted in CAN messages, and the sample decodes and displays
all received CAN messages. By default, the sample demonstrates the usage
with virtual CAN, using an offline message file.

You may also set the input arguments for real CAN message input (e.g., can0 or
vcan0) on Linux desktop or can.aurix on NVIDIA DRIVE<sup>&trade;</sup> platforms.

Use this sample application with:

- Virtual CAN bus defined in NVIDIA<sup>&reg;</sup> DriveWorks with offline binary CAN message files
- Virtual CAN bus defined by SocketCAN on Linux
- Aurix CAN bus on NVIDIA DRIVE<sup>&trade;</sup> platforms (see @ref dwx_canbus_logger_sample for Aurix based parameters and limitations of sending/receiving using AurixCAN)
- Real CAN device, on Linux or QNX

@section dwx_canbus_message_sample_running Running the Sample

The CAN Message Interpreter sample, sample_canbus_interpreter, accepts the following parameters:

    ./sample_canbus_interpreter --driver=[can.virtual|can.socket|can.aurix]
                                --dbc=[path/to/dbc/file]
                                --params=[comma/separated/key/value/pairs]

Where:

    --driver=[can.virtual|can.socket|can.aurix]
        Allows to specify which CAN interface to use.
        Devault value: can.virtual

    --dbc=[path/to/dbc/file]
        Location of DBC file.
        Default value: /path/to/data/samples/sensors/can/sample.dbc

    --params=[comma/separated/key/value/pairs]
        Different parameters are available for each CAN driver.
        Default value: file=path/to/data/samples/sensors/can/canbus_dbc.can

@note For a full list of key/value pairs that can be passed to --params see @ref dwx_sensor_enum_sample .

@subsection dwx_canbus_message_sample_examples Examples

#### Offline CAN Messages

    ./sample_canbus_interpreter

By default, the sample reads `data/samples/sensors/can/sample.dbc` to build the
interpreter, and opens a virtual CAN bus with messages defined in
`data/samples/sensors/can/canbus_dbc.can`.

The output on the console is the car steering and speed. For example:

    5662312708 [0x100] -> 0x9d 0xfe  Car steering -0.0221875 rad at [5662312708]
    5662312708 [0x200] -> 0x1f 0x21  Car speed 8.479 m/s at [5662312708]

#### DBC interpreter

The sample can be used with a DBC file provided by the user. In order to start
interpreter sample using user provided DBC file, pass `--dbc` parameter to the
application:

    ./sample_canbus_interpreter --dbc=user_path_to/file.dbc

#### Callback-based "plugin" interpreter

The sample can be used with an interpreter using user-provided callbacks. In order
to start the callback-based interpreter, pass `--dbc=plugin` as parameter to the
application

    ./sample_canbus_interpreter --dbc=plugin

#### Virtual CAN Bus on Linux

    ./sample_canbus_interpreter --driver=can.socket --params=device=vcan0

Run it with a virtual CAN device created with:

    $ sudo modprobe vcan
    $ sudo ip link add dev vcan0 type vcan
    $ sudo ip link set up vcan0

To send data from the console to the virtual CAN bus, run `cansend` (from the
`can-utils` package):

    $ cansend vcan0 100#fa490200
    $ cansend vcan0 200#2e920300

This sends a CAN message with the CAN ID and 2 bytes of data, containing: 0x11,
0x22

The output is similar to:

    44798857 [0x100] -> 0xfa 0x49 0x2 0x0  Car steering 0.0937563 rad at [44798857]
    64318678 [0x200] -> 0x2e 0x92 0x3 0x0  Car speed 2.3403 m/s at [64318678]

@note the virtual CAN interface is not available on QNX

#### Real CAN Device on Linux

    $ ./sample_canbus_interpreter --driver=can.socket --params=device=can0

A valid SocketCAN device must be present in the system as canX. For example, a
PCAN USB. The bitrate is set to 500 KB. Set the bitrate on the CAN device:

    $ sudo ip link set can0 type can bitrate 500000
    $ sudo ip link set can0 up

@note the steps above are not needed on QNX

The CAN interface used to listen on a real CAN bus must have SocketCAN driver
implementation in the system. A SocketCAN based driver can be identified easily
if `:> sudo ifconfig -a` returns the default network interfaces **and** a CAN
based interface (i.e., canX, slcanX, etc).

@section dwx_canbus_message_sample_output Output

The sample prints on console the interpreted CAN messages

    5652171936 [0x100] -> 0x7e 0xbb 0x2 0x0  Car steering 0.111919 rad at [5652171936]
    5652171936 [0x200] -> 0x80 0x57 0x2 0x0  Car speed 1.53472 m/s at [5652171936]
    5652174567 [0x100] -> 0xa0 0xc1 0x2 0x0  Car steering 0.1129 rad at [5652174567]
    5652174567 [0x200] -> 0x80 0x57 0x2 0x0  Car speed 1.53472 m/s at [5652174567]
    5652177205 [0x100] -> 0xa0 0xc1 0x2 0x0  Car steering 0.1129 rad at [5652177205]
    5652177205 [0x200] -> 0x80 0x57 0x2 0x0  Car speed 1.53472 m/s at [5652177205]
    5652179850 [0x100] -> 0xa0 0xc1 0x2 0x0  Car steering 0.1129 rad at [5652179850]
    5652179850 [0x200] -> 0x80 0x57 0x2 0x0  Car speed 1.53472 m/s at [5652179850]
    5652182733 [0x100] -> 0xa0 0xc1 0x2 0x0  Car steering 0.1129 rad at [5652182733]
    5652182733 [0x200] -> 0x80 0x57 0x2 0x0  Car speed 1.53472 m/s at [5652182733]
    5652185391 [0x100] -> 0x2 0xc9 0x2 0x0  Car steering 0.114081 rad at [5652185391]
    5652185391 [0x200] -> 0x10 0x65 0x2 0x0  Car speed 1.56944 m/s at [5652185391]
    5652187984 [0x100] -> 0x2 0xc9 0x2 0x0  Car steering 0.114081 rad at [5652187984]
    5652187984 [0x200] -> 0x10 0x65 0x2 0x0  Car speed 1.56944 m/s at [5652187984]
    5652190645 [0x100] -> 0x2 0xc9 0x2 0x0  Car steering 0.114081 rad at [5652190645]
    5652190645 [0x200] -> 0x10 0x65 0x2 0x0  Car speed 1.56944 m/s at [5652190645]
    5652193322 [0x100] -> 0x2 0xc9 0x2 0x0  Car steering 0.114081 rad at [5652193322]

@section dwx_canbus_message_sample_more Additional Information

For more details see @ref canbus_mainsection.
