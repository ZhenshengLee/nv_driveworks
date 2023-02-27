# SPDX-FileCopyrightText: Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_camera_sample Camera Sample

@tableofcontents

@section dwx_camera_sample_description Description

The Camera sample uses the `dwSAL` and `dwSensorCamera` interface to setup physical GMSL cameras or virtual cameras, and display their data on screen. The sample can also record videos in a processed or RAW format.

@subsection dwx_camera_setup Setting Up The Cameras

For information regarding the physical location of the ports on NVIDIA DRIVE
platforms, please refer to "Camera Setup and Configuration" in the _NVIDIA DRIVE 5.1
Development Guide_.

The camera parameters located in the rig file are described in section @ref camera_mainsection_camera_creation

@section dwx_camera_sample_running Running the Sample

To launch the sample by default without parameters, connect an AR0231 SF3324 camera to csi-port A, link 0 on your NVIDIA DRIVE platform.

The Camera sample, sample_camera, accepts the following parameters:

    ./sample_camera --rig=[path/to/rig/file]
                    --write-file=[path/to/output/file]

Where:

    --rig=[path/to/rig/file]
        Specifies the rig file containing the camera descriptions.
        Default value: "{data dir}/samples/sensors/camera/camera/rig.json"
        Note: {data dir}/samples/sensors/camera/camera contains several
        example rigs for different configurations to experiment with.

    --write-file=[path/to/output/file]
        Specifies the output file, where h264, h265, mp4 and RAW formats are supported.
        If the RAW format is selected, ensure the master camera specified in the rig file
        has RAW output enabled. RAW recordings are then compatible for replay
        case (see RigReplay.json).
        If h264, h265 or mp4 formats are selected, ensure the master camera specified in the rig
        file has `processed` enabled.
        If --write-file is not provided, no file is written out on disk.
        Default value: none

The content of the rig reflects the description of protocol usage and parameter setup listed under the Sensor Camera guide.
By default the rigs provided setup the `camera.gmsl` protocol, however the sample is designed to handle other camera protocols, such as
`camera.virtual` protocol for replay of raw files. Although the rig by default lists only one camera, simply following the rules of rig
construction, it's possible to add up to 16 instance of cameras, live or virtual.

Since this sample only showcases functionality, only the camera with index 0 is recordable.

Additionally, it is possible to screenshot the current window by pressing S and saving each individual camera image as a full resolution .png image
by pressing F.

It's possible to control cameras in runtime, by pressing either 
P (dwSensor_stop) 
O (dwSensor_start) 
I (dwSensor_reset), 
followed by a number 0-9 or A-F (10-15) for the camera index
Example the sequence of keypress "P B" will stop camera 11. Note that the operation must 
be coherent with the state of the camera, ie you can't call start on a camera that is running
or stop on a camera already stopped. Pressing one of the control keys and then again a control key will toggle the previous one off