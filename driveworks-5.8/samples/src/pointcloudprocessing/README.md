# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_pointcloudprocessing_sample Point Cloud Processing Sample
@tableofcontents

@section dwx_pointcloudprocessing_sample_description Description

This sample demonstrates how to use point cloud processing APIs for primitive processing.

The sample fuses point clouds from two [VELO_HDL32E] recordings and one [VELO_HDL64E] recording. It then generates the range image and organized point cloud data. This data is used to compute the rigid transformation between two temporally adjacent point clouds.

The sample only supports recorded data as input. It does not support live data. Recordings from other types of Lidars are not guaranteed to work.

@section dwx_pointcloudprocessing_sample_running Running the Sample

    ./sample_pointcloudprocessing --rigFile=[path/to/rig file/]
                                  --numFrame=[max_frames_to_process]
                                  --maxIters=[max_icp_iterations]
                                  --displayWindowHeight=[window height in pixels]
                                  --displayWindowWidth=[window width in pixels]

Where:

    --rigFile=[path/to/rig file/]
        Path to the rig file.
        The rig file contains all the sensor configurations required for initialization.
        Default value: path/to/data/samples/pointcloudprocessing/rig.json

    --numFrame=[integer]
        If specified, the sample processes frames up to this integer. It processes all frames by default.
        Default value: 0

    --maxIters=[integer]
        The maximum number of iterations for ICP.
        Default value: 12

    --displayWindowHeight=[window height in pixels]
        Defines the sample's window height in pixels.
        Default value: 900

    --displayWindowWidth=[window width in pixels]
        Defines the sample's window width in pixels.
        Default value: 1500

The following interactions with the sample are available at runtime:

- Mouse left button: rotates the point cloud.
- Mouse wheel: zooms in or out.
- Key left arrow: switches to trajectory view.
- Key right arrow: switches to main view.

@subsection dwx_pointcloudprocessing_sample_examples Examples

#### Running with GPU pipeline

    ./sample_pointcloudprocessing --maxIters=20

In this example, the sample uses default recordings and processes up to 20 ICP iterations.

@section dwx_pointcloudprocessing_sample_output Output

The sample opens a window with several point clouds in different colors:
- Orange: Three rendered point clouds consisting of the [VELO_HDL32E] and [VELO_HDL64E] recordings, in the left column.
- Green: The rendered fused point cloud, in the top right column.
- White: The rendered range image using the fused point cloud, in the bottom right column.

Pressing the right arrow key switches to the trajectory view, which renders the fused point cloud motion trajectory over time.
![Point Cloud Processing Master Screen](PointCloudProcessingSampleMasterScreen.png)
![Point Cloud Processing Trajectory Screen](PointCloudProcessingSampleTrajectoryScreen.png)

@section dwx_pointcloudprocessing_sample_more Additional Information

For more details see @ref pointcloudprocessing_mainsection.
