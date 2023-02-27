# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_imu_loc_sample IMU Logger Sample
@tableofcontents

@section dwx_imu_loc_sample_description Description

The IMU Logger sample works with any serial port (UART) based IMU
sensor or with the Xsens IMU device connected over USB. The logger requires
the IMU sensor connected over serial port to deliver messages in NMEA format,
while the Xsens device can run in proprietary mode.

@subsection dwx_imu_loc_sample_description_interfacing Interfacing with sensors

#### Serial

Consumer grade off-the-shelve IMU sensors are usually connected over USB
and implement a serial-to-USB connection with the help of FTDI devices.
The Xsens IMU device can be connected through a serial-to-USB connection.

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

The Xsens IMU device can also be connected directly over USB, but is only
supported on Linux.

@section dwx_imu_loc_sample_running Running the Sample

The sample requires the driver and name of the device of the IMU sensor.

The IMU Logger sample, sample_imu_logger, accepts the following parameters:

    ./sample_imu_logger --driver=[imu.virtual|imu.uart|imu.xsens|imu.novatel|imu.dataspeed|imu.xsensCan|imu.bosch|imu.continental|imu.custom]
                        --params=[comma/separated/key/value/pairs]
                        --timestamp-trace=[true|false]

Where:

    --driver=[imu.virtual|imu.uart|imu.xsens|imu.novatel|imu.dataspeed|imu.xsensCan|imu.bosch|imu.continental|imu.custom]
        Allows to specify which IMU driver to use.
        Default value: imu.virtual

    --params=[comma/separated/key/value/pairs]
        Different parameters are available for each IMU driver.
        Default value: file=path/to/data/samples/sensors/imu/imu.txt

    --timestamp-trace=[true|false]
        Allows to output only the timestamp deltas between consecutive frames.
        Default value: false

@note For a full list of key/value pairs that can be passed to --params see @ref dwx_sensor_enum_sample.

@subsection dwx_imu_loc_sample_examples Examples

#### NMEA format

For serial devices transmitting messages in NMEA format, use the `imu.uart`
driver. For example:

    ./sample_imu_logger --driver=imu.uart --params=device=/dev/ttyUSB0,baud=115200

Per default, if no `baud` parameter has been provided `imu.uart` driver assumes
a baudrate of 9600. In order to change the baudrate provide `baud` argument as:

    ./sample_imu_logger --driver=imu.uart --params=device=/dev/ttyUSB0,baud=115200

On QNX, the baudrate is set when starting `devc-serusb`. The `baud`
parameter will be ignored.

#### Xsens proprietary format

The sample supports reading IMU packets from a Xsens device through the
`imu.xsens` driver. To run the sample using Xsens over USB device use:

    ./sample_imu_logger --driver=imu.xsens --params=device=0,frequency=100

Where `device=0` parameter sets the index of the Xsens device (usually 0
if only one device is installed) and `frequency=100` sets the frequency
in `[Hz]` this device should operate with.

To run the sample using Xsens over serial use:

    ./sample_imu_logger --driver=imu.xsens --params=device=/dev/ttyUSB0,frequency=100

Please note that even if the Xsens device is a shared device, like Xsens
MTi-G-700, capable of delivering GPS and IMU packets, only the IMU packets
will be parsed by the `imu.xsens` driver.

@note If the device is connected to Xavier UART you need also specify --stop-bits=1

@section dwx_imu_loc_sample_output Output

For every valid IMU message that the sample receives, it prints to the console
data such as:

    [7156364888] Heading(True:112.07)
    [7156364959] Gyro(Z:0.0756667 )
    [7156369081] Orientation(R:-1.3 P:-0.9 ) Gyro(X:-0.01 Y:-0.05 ) Heading(True:112.1)
    [7156389724] Heading(Magnetic:112.07)
    [1475788068778749] Heading(Magnetic:112.1)
    [7156389797] Orientation(R:-1.31 P:-0.89 )
    [1475788068778919] Orientation(R:-1.30529 P:-0.893047 Y:112.078 ) Gyro(X:0.0229183 Y:-0.0687549 Z:0.120321 ) Acceleration(X:-0.1398 Y:-0.2612 Z:9.7838 ) Magnetometer(X:7.18 Y:3.056 Z:-16.16 )

Where the first number indicates the timestamp of the received IMU
message in microseconds and the rest of the line indicates the IMU information
of the sensor.

@section dwx_imu_loc_sample_more Additional Information

For more details see @ref imu_mainsection.
