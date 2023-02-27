# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_gps_loc_sample GPS Location Logger Sample
@tableofcontents

@section dwx_gps_loc_sample_description Description

The GPS Location Logger sample works with any serial port (UART) based
GPS sensor or with the Xsens GPS device connected over USB. The logger
requires the GPS sensor connected over serial port to deliver messages
in NMEA format, while the Xsens device can run in proprietary mode.

@subsection dwx_gps_loc_sample_description_interface Interfacing with sensors

#### Serial

Consumer grade off-the-shelf GPS sensors, also known as GPS mice (e.g.,
Garmin GPS), are usually connected over USB and implement a serial-to-USB
connection with the help of FTDI devices. The Xsens GPS device can be
connected through a serial-to-USB connection.

On Linux, these sensors can then be reached over the `/dev/ttyUSB` or
`/dev/ttyACM` devices. Before trying out the sample, ensure the user has
access to the serial device. You can do so by trying to read from the device
e.g. with `cat /dev/ttyACMx`, and if permission is denied, add the current user
to the `dialout` group and then log out/in again:

    sudo usermod -a -G dialout $USER

On QNX, prior to accessing the device, launch the `dev-serusb` driver as
follows:

    devc-serusb -b <baudrate> -F -S -d path=/dev/usb/io-usb-otg

The serial devices can the be reached over the `/dev/serusb` devices.

#### Xsens USB

The Xsens GPS device can also be connected directly over USB, but is only
supported on Linux.

@section dwx_gps_loc_sample_running Running the sample

The GPS Location Logger sample, sample_gps_logger, accepts the following parameters:

    ./sample_gps_logger --driver=[gps.virtual|gps.uart|gps.xsens|gps.novatel|gps.dataspeed|gps.custom|gps.ublox]
                        --params=[comma/separated/key/value/pairs]

Where:

    --driver=[gps.virtual|gps.uart|gps.xsens|gps.novatel|gps.dataspeed|gps.custom|gps.ublox]
        Allows to specify which GPS driver to use.
        Default value: gps.virtual

    --params=[comma/separated/key/value/pairs]
        Different parameters are available for each GPS driver.
        Default value: file=path/to/data/samples/sensors/gps/1.gps

@note For a full list of key/value pairs that can be passed to --params see @ref dwx_sensor_enum_sample .

@subsection dwx_gps_loc_sample_examples Examples

#### NMEA format

For serial devices transmitting messages in NMEA format, use the `gps.uart`
driver. For example:

    ./sample_gps_logger --driver=gps.uart --params=device=/dev/ttyACM0

Per default, if no `baud` parameter has been provided the `gps.uart`
driver assumes a baudrate of 9600. In order to change the baudrate provide
`baud` argument as:

    ./sample_gps_logger --driver=gps.uart --params=device=/dev/ttyACM0,baud=115200

On QNX, the baudrate is set when starting `devc-serusb`. The `baud`
parameter will be ignored.

#### Xsens proprietary format

The sample supports reading GPS packets from a Xsens device through the
`gps.xsens` driver. To run the sample using Xsens over USB device use:

    ./sample_gps_logger --driver=gps.xsens --params=device=0,frequency=100

Where `device=0` parameter sets the index of the Xsens device (usually 0
if only one device is installed) and `frequency=100` sets the frequency
in `[Hz]` this device should operate with.

To run the sample using Xsens over serial use:

    ./sample_gps_logger --driver=gps.xsens --params=device=/dev/ttyUSB0,frequency=100

Please note that even if the Xsens device is a shared device, like Xsens
MTi-G-700, capable of delivering GPS and IMU packets, only the GPS packets
will be parsed by the `gps.xsens` driver.

@note If the device is connected to Xavier UART you need also specify --stop-bits=1

#### Sensor sharing

The sample also demonstrates how sensor sharing can be implemented.
Two NVIDIA<sup>&reg;</sup> DriveWorks GPS sensors are created from the same
hardware device. Both sensor can be then treated independently. Both sensors
would deliver exactly the same set of packets. Each sensor is using, however,
their own FIFO hence they can be drained at different rates. The output of
both sensors is printed to the console:

    GPS[0] - 2712443194 lat: 37.38652333 lon: -122.164585 alt: 46.9 course: 233.9 speed: 0 hdop: 0.8 vdop: 0.8
    GPS[1] - 2712443194 lat: 37.38652333 lon: -122.164585 alt: 46.9 course: 233.9 speed: 0 hdop: 0.8 vdop: 0.8

The index `[0]`, `[1]` indicates what sensor produced the output. As expected the data
packets and their timestamps are equal.

@section dwx_gps_loc_sample_output Output

Any valid GPS message that is received results in an output on the
console similar to:

    GPS[0] - 2712443194 lat: 37.38652333 lon: -122.164585 alt: 46.9 course: 233.9 speed: 0 hdop: 0.8 vdop: 0.8

Where the first number indicates the timestamp of the received GPS message in
microseconds and the rest of the line indicates the geographical location of the
sensor.

If no parameters are provided, the sample starts a virtual GPS sensor and
interprets the content from the file located at `data/samples/sensors/gps/1.gps`
as GPS input.

@section dwx_gps_loc_sample_more Additional Information

For more details see @ref gps_mainsection .
