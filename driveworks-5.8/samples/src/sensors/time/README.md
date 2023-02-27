# SPDX-FileCopyrightText: Copyright (c) 2020-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_time_sensor_sample Time Sensor Sample
@tableofcontents

@section dwx_time_sensor_sample_description Description

The Time Sensor sample performs time synchronization with Lidar packet timestamps. The sample initializes multiple virtual Lidar sensors with different timestamp options in order to compare host, time synchronized, and raw sensor timestamps.

@section dwx_time_sensor_sample_running Running the Sample

The sample requires the driver and parameters of the Time and Lidar sensors. It accepts the following arguments:

    ./sample_timesensor --time-params=[comma/separated/key/value/pairs]
                        --lidar-params=[comma/separated/key/value/pairs]

Where:

    --time-params=[comma/separated/key/value/pairs]
        Different parameters are available for each Time driver.
        Default value: file=path/to/data/samples/sensors/time/time.bin

    --lidar-params=[comma/separated/key/value/pairs]
        Different parameters are available for each Time driver.
        Default value: file=path/to/data/samples/sensors/time/lidar_hdl32e.bin

@note For a full list of key/value pairs that can be passed to --time-params or --lidar-params see @ref dwx_sensor_enum_sample .

@note For more information about timestamp options see @ref sensors_usecase4 .

@section dwx_time_sensor_sample_output Output

For every valid Lidar packet, the sample prints to the console data such as:

```
[Lidar] timestamp synced: 1499155033600203 | original: 1499155033600797 | raw: 1499155205168587
[Lidar] timestamp synced: 1499155033600756 | original: 1499155033601349 | raw: 1499155205169140
[Lidar] timestamp synced: 1499155033601309 | original: 1499155033601902 | raw: 1499155205169693
[Lidar] timestamp synced: 1499155033601862 | original: 1499155033602454 | raw: 1499155205170246
[Lidar] timestamp synced: 1499155033602415 | original: 1499155033603010 | raw: 1499155205170799
[Lidar] timestamp synced: 1499155033602968 | original: 1499155033603562 | raw: 1499155205171352
[Lidar] timestamp synced: 1499155033603521 | original: 1499155033604113 | raw: 1499155205171905
[Lidar] timestamp synced: 1499155033604074 | original: 1499155033604669 | raw: 1499155205172458
[Lidar] timestamp synced: 1499155033604627 | original: 1499155033605218 | raw: 1499155205173011
```
