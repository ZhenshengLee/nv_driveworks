# SPDX-FileCopyrightText: Copyright (c) 2020-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_data_sensor_sample Data Sensor Sample
@tableofcontents
@section dwx_data_sensor_sample_description Description

The Data Sensor sample reads file/socket data and prints the timestamp and size of the received packet.

@section dwx_data_sensor_sample_running Running the Sample

The sample requires the driver and parameters of the data sensor. It accepts the following arguments:

    ./sample_datasensor --driver=[data.virtual|data.socket]
                        --params=[comma/separated/key/value/pairs]

Where:

    --driver=[data.virtual|data.socket]
        Allows to specify which data sensor driver to use.
        Default value: data.virtual

    --params=[comma/separated/key/value/pairs]
        Different parameters are available for each data sensor driver.
        Default value: file=path/to/data/samples/sensors/data/data_packet.bin

@note For a full list of key/value pairs that can be passed to --params see @ref dwx_sensor_enum_sample .

@section dwx_data_sensor_sample_output Output

For every valid data sensor packet, the sample prints to the console data such as:

```
[1588118153011907] received frame of size: 6
[1588118154913607] received frame of size: 6
[1588118156872613] received frame of size: 6
[1588118158846329] received frame of size: 6
[1588118163976672] received frame of size: 6
```
