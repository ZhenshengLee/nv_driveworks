# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_radar_replay_sample Radar Replay Sample
@tableofcontents

@section dwx_radar_replay_sample_description Description

The Radar Replay sample demonstrates how to connect to a Radar and display the generated point cloud in 3D.
For a list of currently supported Radar devices, see the <em>Release Notes</em>.

@subsection dwx_radar_replay_sample_prerequisites Description Prerequisites

- The Radar must be up and running, and connected to the network.
- If the radar being used is the Continental ARS430 ethernet radar, the following commands need to be executed to setup the routing tables.

On Linux execute:

ARS430 Radar:
        sudo ifconfig eth0:900 10.1.0.81 || true
        sudo route add -net 224.0.0.0 netmask 240.0.0.0 dev eth0:900 || true
        sudo route add -net 10.1.0.0 netmask 255.255.0.0 dev eth0:900 || true

ARS430 RDI Radar:
        sudo ifconfig eth0:900 192.168.3.81 || true
        sudo route add -net 224.0.0.0 netmask 240.0.0.0 dev eth0:900 || true
        sudo route add -net 192.168.3.0 netmask 255.255.255.0 dev eth0:900 || true

@if QNX
On QNX execute:

ARS430 Radar:
        ifconfig eq0 alias 10.1.0.81
        route add 224.0.0.0 10.1.0.81 240.0.0.0
        route add 10.1.0.0 10.1.0.81 255.255.0.0

ARS430 RDI Radar:
        ifconfig eq0 alias 192.168.3.81
        route add 224.0.0.0 192.168.3.81 240.0.0.0
        route add 192.168.3.0 192.168.3.81 255.255.255.0
@endif

@section dwx_radar_replay_sample_running Running the sample

The Radar Replay sample, sample_radar_replay, accepts the following parameters:

    ./sample_radar_replay --protocol=[radar.virtual|radar.socket|radar.custom]
                          --params=[comma/separated/key/value/pairs]
                          --profiling=[0|1]

Where:

    --protocol=[radar.virtual|radar.socket|radar.custom]
        Allows to specify which Radar driver to use.
        radar.virtual is used for recorded file
        radar.socket is used for live IP based radars and custom decoder plugin based radars
        radar.custom is used for custom full sensor plugin based radars
        Default value: radar.virtual

    --params=[comma/separated/key/value/pairs]
        Different parameters are available for each Radar driver.
        Default value: file=path/to/data/samples/sensors/radar/conti/radar_0.bin

    --profiling=[0|1]
        When set to 1, enables sample profiling.
        Otherwise, profiling is disabled.
        Default value: 1

The following interactions with the sample are available at runtime:

- Mouse left button: rotates the point cloud.
- Mouse wheel: zooms in or out.
- SPACE: makes a pause.
- R: resets the camera view and the artificially increased or decreased frame rate.
- G: shows/hides circular and rectangular grid.
- F1 shows/hides text message hints.

@note For a full list of key/value pairs that can be passed to --params see @ref dwx_sensor_enum_sample.

@subsection dwx_radar_replay_sample_examples Examples

#### Display live Radar point clouds

    ./sample_radar_replay --protocol=radar.socket --params=device=[type of device],ip=[radar IP address],port=[radar port]

Where [type of device] is one of the following:
- CONTINENTAL_ARS430
- CONTINENTAL_ARS430_RDI
- CONTINENTAL_ARS430_RDI_V2
- DELPHI_ESR2_5
- CUSTOM

@note In case of CONTINENTAL_ARS430, the additional parameter `multicast-ip=239.0.0.1` is required.

@note For more information on using custom sensors see @ref sensorplugins_mainsection.

#### Display recorded Radar point clouds

    ./sample_radar_replay --params=file=[radar bin file]

- The Radar file can be obtained with the provided recording tools.
- If no arguments are passed, a default Radar file is loaded.

#### Display custom Radar Plugin point clouds

    ./sample_radar_replay --protocol=radar.custom --params=decoder-path=<path_to_so>[,<custom params>]

@section dwx_radar_replay_sample_output Output

The sample opens a window to display a 3D point cloud. The output contains directed unit velocity vectors, other than the points displayed.
If the velocity is 100 km/h or less, the directed vectors are red. If it is 200 km/h or more, they are green.

Worldspace axes:
- Red-OX.
- Blue-OY.
- Green-OZ.

![Radar Point Clouds Sample](sample_radar_replay.png)

@section dwx_radar_replay_sample_more Additional Information

For more details see @ref radar_mainsection.
