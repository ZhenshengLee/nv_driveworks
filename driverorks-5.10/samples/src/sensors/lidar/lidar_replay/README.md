# SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_lidar_replay_sample Lidar Replay Sample
@tableofcontents

@section dwx_lidar_replay_sample_description Description

The Lidar Replay sample demonstrates how to connect to a Lidar and visualize the generated point cloud in 3D.

@note Custom software for the __OUSTER_OS2_128__ Lidar is required for DriveWorks support. Please contact your NVIDIA representative for more information.

@section dwx_lidar_replay_sample_running Running the Sample

The Lidar Replay sample, sample_lidar_replay, accepts the following parameters:

    ./sample_lidar_replay --protocol=[lidar.virtual|lidar.socket|lidar.custom]
                          --params=[comma/separated/key/value/pairs]
                          --show-intensity=[true|false]
Where:

    --protocol=[lidar.virtual|lidar.socket|lidar.custom]
        Specifies which Lidar driver to use.
        - `lidar.virtual`: replays from file.
        - `lidar.socket`: live Lidar replay.
        - `lidar.custom` is used for custom full sensor plugin based lidars
        Default value: lidar.virtual

    --params=[comma/separated/key/value/pairs]
        Different parameters are available for each Lidar driver.
        Default value: file=path/to/data/samples/sensors/lidar/lidar_velodyne_64.bin

    --show-intensity=[true|false]
        Enables an alternative HUE based rendering mode, which renders intensity
        proportional to wavelength. Higher intensities are in blue, while lower intensities are in red.
        In default mode, color is rendered by 2D distance from the origin where red is near and blue is far.
        Default value: false

During runtime, the following sample interactions are available:

- Mouse left button: Rotates the point cloud.
- Mouse wheel: Zooms in or out.

@note For a full list of key/value pairs that can be passed to --params see @ref dwx_sensor_enum_sample .

@subsection dwx_lidar_replay_sample_examples Examples

#### Display live Lidar point clouds

    ./sample_lidar_replay --protocol=[lidar protocol] --params=device=[type of device],ip=[lidar IP address],dip=[IP address of UDP packet receiver],port=[lidar port],hres=[valid horizontal resolution],return-mode=[valid return mode],scan-frequency=[valid frequency] (--show-intensity=[true])

- The Lidar must be up and running, and connected to the network.
- dip (Destination IP Address) is applicable only for device OUSTER_OS1 and OUSTER_OS2.
- hres (Horizontal Resolution) is applicable only for device OUSTER_OS1 and OUSTER_OS2.
- return-mode is applicable only for device Velodyne Lidars.
- Scan frequency is usually preset using the Lidar vendor tools. The exception
  is the Ouster Lidar (valid values are 10 and 20 Hz).
- For a full list of currently supported devices, see @ref dwx_sensor_enum_sample.

#### Display recorded Lidar point clouds

    ./sample_lidar_replay --protocol=lidar.virtual --params=file=[lidar file] (--show-intensity=[true])

- The Lidar file can be obtained with the provided recording tools.
- If no arguments are passed, a default Lidar file is loaded.

#### Display custom Lidar Plugin point clouds

    ./sample_lidar_replay --protocol=lidar.custom --params=decoder-path=<path_to_so>[,<custom params>]

@section dwx_lidar_replay_sample_output Output

The sample opens a window to display a 3D point cloud.

![Lidar Point Clouds Sample](sample_lidar_replay.png)

@section dwx_lidar_replay_sample_more Additional Information

For more details see @ref lidar_mainsection.

In case of packet drops see @ref lidar_usecase3.
