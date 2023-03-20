# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_record_sample Simple Sensor Recording Sample
@tableofcontents

@section dwx_record_sample_description Description

The Simple Sensor Recording Sample allows you to record data from CAN, GPS, RADAR or LIDAR sensors.

@section dwx_record_sample_running Running the Sample

The syntax for calling the simple sensor recording sample is:

    ./sample_record --can-driver=[can.socket|can.aurix]
                    --can-params=[comma/separated/key/value/pairs]
                    --gps-driver=[gps.uart|gps.xsens|gps.novatel|gps.dataspeed]
                    --gps-params=[comma/separated/key/value/pairs]
                    --lidar-driver=[lidar.socket]
                    --lidar-params=[comma/separated/key/value/pairs]
                    --radar-driver=[radar.socket]
                    --radar-params=[comma/separated/key/value/pairs]
                    --write-file-gps=[path/to/gps/file]
                    --write-file-can=[path/to/canbus/file]
                    --write-file-lidar=[path/to/lidar/file]
                    --write-file-radar=[path/to/radar/file]

where:

    --can-driver=[can.socket|can.aurix]
        Specifies which CAN interface to use.
        Default value: can.socket

    --can-params=[comma/separated/key/value/pairs]
        Different parameters are available for each CAN driver.
        Default value: none

    --gps-driver=[gps.uart|gps.xsens|gps.novatel|gps.dataspeed]
        Specifies which GPS driver to use.
        Default value: gps.uart

    --gps-params=[comma/separated/key/value/pairs]
        Different parameters are available for each GPS driver.
        Default value: none

    --lidar-driver=[lidar.socket]
        Specifies which Lidar driver to use.
        Default value: lidar.socket

    --lidar-params=[comma/separated/key/value/pairs]
        Different parameters are available for each Lidar driver.
        Default value: none

    --radar-driver=[radar.socket]
        Specifies which Radar driver to use.
        Default value: radar.socket

    --radar-params=[comma/separated/key/value/pairs]
        Different parameters are available for each Radar driver.
        Default value: none

    --write-file-gps=[path/to/gps/file]
        Path where the recorded GPS data is going to be stored.
        Default value: none

    --write-file-can=[path/to/canbus/file]
        Path where the recorded CANBUS data is going to be stored.
        Default value: none

    --write-file-lidar=[path/to/lidar/file]
        Path where the recorded Lidar data is going to be stored.
        Default value: none

    --write-file-radar=[path/to/radar/file]
        Path where the recorded Radar data is going to be stored.
        Default value: none

@note For a full list of key/value pairs that can be passed to --[sensor]-params see @ref dwx_sensor_enum_sample.

@subsection dwx_record_sample_examples Examples

## Recording CAN ##

- Set `--can-driver` to `can.socket`.
- Set `--can-params` to `device=can0` where `can0` is the can device to live record.
- Set  `--write-file-can=filename.bin` to the recorded output file,
  where filename.bin is the output file for CAN data.

Thus, to record data from a can sensor, the following command would be used:

    ./sample_record --can-driver=can.socket --can-params=device=can0 --write-file-can=/path/to/outputfile.bin

## Recording GPS ##

GPS takes command-line options that are similar to the CAN options.

For example, the command line for recording GPS data from a UART GPS sensor is:

    ./sample_record --gps-driver=gps.uart --gps-params=device=/dev/ttyACM0 --write-file-gps=/path/to/outputfile.bin

## Recording LIDAR ##

- Set `--lidar-protocol` to `lidar.socket`
- Set `--lidar-params` to appropriate values depending on your device. Supported devices are listed in @ref dwx_lidar_replay_sample

For example, the command line for recording lidar data:

    ./sample_record --lidar-protocol=lidar.socket --lidar-params=device=[device],ip=[lidar IP address],dip=[IP address of UDP packet receiver],port=[lidar port],hres=[valid horizontal resolution],scan-frequency=[valid frequency] --write-file-lidar=/path/to/lidaroutput.bin

- dip (Destination IP Address) is applicable only for device OUSTER_OS1
- hres (Horizontal Resolution) is applicable only for device OUSTER_OS1

## Recording RADAR ##

- Set `--radar-protocol` to `radar.socket`
- Set `--radar-params` to appropriate values depending on your device. Supported devices are listed in @ref dwx_radar_replay_sample

For example, the command line for recording GPS data from a UART GPS sensor is:

    ./sample_record --radar-protocol=radar.socket --radar-params=device=[device],ip=[radar IP address],port=[radar port],scan-frequency=[valid frequency] --write-file-radar=/path/to/radaroutput.bin

@note If the radar being used is the Continental ARS430 ethernet radar, see to the prerequisite section in @ref dwx_radar_replay_sample .

## Recording Mixed Sensor Types

Different types of sensors can be combined. For example, the command for recording live GPS and LIDAR data is:

    ./sample_record --gps-driver=gps.uart --gps-params=device=/dev/ttyACM0 --write-file-gps=/path/to/gpsoutput.bin --lidar-protocol=lidar.socket \
    --lidar-params=device=[device],ip=[lidar IP address],port=[lidar port],scan-frequency=[valid frequency] --write-file-lidar=/path/to/lidaroutput.bin

@note - This sample creates output files that, per default, are put into the
current working directory. Hence, write permissions to the current working
directory are necessary if the output file arguments are not changed.

@note - Recording virtual sensors is not supported.

@section dwx_record_sample_more Additional Information

For more details see @ref sensors_mainsection.
