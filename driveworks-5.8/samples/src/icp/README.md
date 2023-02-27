# SPDX-FileCopyrightText: Copyright (c) 2018-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_sample_icp ICP (Iterative Closest Points) Sample
@tableofcontents

@section dwx_sample_icp_description Description

The NVIDIA<sup>&reg;</sup> DriveWorks ICP sample shows how to obtain ICP transforms via the
DriveWorks ICP module. The sample determines the relative transform between
two consecutive spins of Lidar and then chains them over a longer period. The
sample uses a point-to-plane algorithm and expects the points to be in an
order that approximately preserves their relative proximity in 3D space.

@section dwx_sample_icp_running Running the Sample

The Iterative Closest Points sample, sample_icp, accepts the following optional parameters. If none are specified, it will use a default Lidar recording.

    ./sample_icp --lidarFile=[path/to/lidar/file]
                 --plyloc=[path/to/folder]
                 --init=[integer]
                 --skip=[integer]
                 --numFrames=[integer]
                 --maxIters=[integer]

where

    --lidarFile=[path/to/lidar/file]
        Path to the Lidar file, which must be DW captured Velodyne HDL-64E file.
        Default value: path/to/data/samples/lidar/lidar_velodyne64.bin

    --plyloc=[path/to/folder]
        If specified, this os the directory where to write ICP-fused ASCII-PLY file.
        Default value: (none)

    --init=[integer]
        Number of initial spins to skip before the first pair is fed to ICP.
        Initial frames do not contain enough points for ICP. Must be > 10.
        Default value: 20

    -skip=[integer]
        Number of frames to skip before getting the second spin in the pair. The first
        spin in a pair is the second spin from the previous pair. In the case of the
        first pair, the first spin is specified by the value of --init + 1 and the second
        is N frames later, where --skip specifies N. Must be < 5.
        Default value: 0

    --numFrames=[integer]
        These many pairs are used to perform ICP before stopping.
        To process all frames, set to zero (0).
        Default value: 0

    --maxIters=[integer]
        Number of ICP iterations to run.
        Default value: 12

To pause the sample, press `SPACE`.

To rotate and move the camera while the sample is paused, select and drag the image.

@section dwx_sample_icp_output Output

![ICP Sample](sample_icp.png)

@section dwx_sample_icp_more Additional Information

For more details see @ref pointcloudprocessing_usecase5 .
