# SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_ipc_socketclientserver_sample Inter-process Communication (IPC) Sample

@tableofcontents

@section dwx_ipc_socketclientserver_description Description

The Socket Inter-process Communication (IPC) sample demonstrates
simple IPC functionalities using network sockets.

@section dwx_ipc_socketclientserver_running Running the Sample

The command line for the sample is:

    ./sample_socket_ipc_clientserver --role=[client|server]
                                     --ip=[address]
                                     --port=[port]


where

    --role=[client|server]
        Is either "client" or "server" (required).
        Default value: client

    --ip=[address]
        Is the server IP the client connects to (optional).
        Default value: 127.0.0.1

    --port=[port]
        Is the port the server will listen on / the client will connect to (optional).
        Default value: 49252

@subsection dwx_ipc_socketclientserver_examples Examples

Two instances of the sample are required. The server instance has to be started first.

    ./sample_socket_ipc_clientserver --role=server --port=49252

    ./sample_socket_ipc_clientserver --role=client --port=49252 --ip=127.0.0.1

@section dwx_ipc_socketclientserver_output Output

In the sample the client generates random values, sends them to the server, who echoes them back.

Server:

    nvidia@tegra-ubuntu:/usr/local/driveworks/bin$ ./sample_socket_ipc_clientserver --role=server
    Program Arguments:
    --ip=127.0.0.1
    --port=49252
    --role=server

    [9-8-2018 16:16:56] Initialize DriveWorks SDK v1.2.227
    [9-8-2018 16:16:56] Release build with GNU 4.9.4 from v1.2.0-rc6-0-g79beb2a against Vibrante PDK v5.0.10.3
    [9-8-2018 16:16:56] Platform: Detected Drive PX2 - Tegra A
    [9-8-2018 16:16:56] TimeSource: monotonic epoch time offset is 1533299678306576
    [9-8-2018 16:16:56] TimeSource: PTP ioctl returned error. Synchronized time will not be available.
    [9-8-2018 16:16:56] TimeSource: Could not detect valid PTP time source at 'eth0'. Fallback to CLOCK_MONOTONIC.
    [9-8-2018 16:16:56] Platform: number of GPU devices detected 2
    [9-8-2018 16:16:56] Platform: currently selected GPU device discrete ID 0
    [9-8-2018 16:16:56] SDK: Resources mounted from /usr/local/driveworks/data/resources
    [9-8-2018 16:16:56] SDK: Create NvMediaDevice
    [9-8-2018 16:16:56] SDK: Create NvMediaIPPManager
    [9-8-2018 16:16:56] egl::Display: found 2 EGL devices
    [9-8-2018 16:16:56] egl::Display: use drm device: drm-nvdc
    [9-8-2018 16:16:56] SocketServer: listening on 49252
    [9-8-2018 16:16:58] SocketServer: accepted 127.0.0.1:40020
    [9-8-2018 16:16:58] SocketServer: accepted 127.0.0.1:40022
    Socket Server received 1
    Socket Server send 1
    Socket Server received 2
    Socket Server send 2
    Socket Server received 3
    Socket Server send 3
    Socket Server received 4
    Socket Server send 4
    Socket Server received 5
    Socket Server send 5
    [9-8-2018 16:17:2] Driveworks SDK released

Client:

    nvidia@tegra-ubuntu:/usr/local/driveworks/bin$ ./sample_socket_ipc_clientserver --role=client
    Program Arguments:
    --ip=127.0.0.1
    --port=49252
    --role=client

    [9-8-2018 16:16:58] Initialize DriveWorks SDK v1.2.227
    [9-8-2018 16:16:58] Release build with GNU 4.9.4 from v1.2.0-rc6-0-g79beb2a against Vibrante PDK v5.0.10.30
    [9-8-2018 16:16:58] Platform: Detected Drive PX2 - Tegra A
    [9-8-2018 16:16:56] TimeSource: monotonic epoch time offset is 1533299678306576
    [9-8-2018 16:16:56] TimeSource: PTP ioctl returned error. Synchronized time will not be available.
    [9-8-2018 16:16:56] TimeSource: Could not detect valid PTP time source at 'eth0'. Fallback to CLOCK_MONOTONIC.
    [9-8-2018 16:16:56] Platform: number of GPU devices detected 2
    [9-8-2018 16:16:56] Platform: currently selected GPU device discrete ID 0
    [9-8-2018 16:16:56] SDK: Resources mounted from /usr/local/driveworks/data/resources
    [9-8-2018 16:16:56] SDK: Create NvMediaDevice
    [9-8-2018 16:16:56] SDK: Create NvMediaIPPManager
    [9-8-2018 16:16:56] egl::Display: found 2 EGL devices
    [9-8-2018 16:16:56] egl::Display: use drm device: drm-nvdc
    [9-8-2018 16:16:58] SocketClient: connected 127.0.0.1:49252
    [9-8-2018 16:16:58] SocketClient: connected 127.0.0.1:49252
    Socket Client send 1
    Socket Client received 1
    Socket Client send 2
    Socket Client received 2
    Socket Client send 3
    Socket Client received 3
    Socket Client send 4
    Socket Client received 4
    Socket Client send 5
    ^CSocket Client received 5
    [9-8-2018 16:17:2] Driveworks SDK released

@section dwx_ipc_socketclientserver_more Additional information

For more details see @ref ipc_mainsection .
