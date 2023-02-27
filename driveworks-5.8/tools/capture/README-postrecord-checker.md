# SPDX-FileCopyrightText: Copyright (c) 2017-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_postrecord_checker Post-record Checker

This tool is available on the x86 Host System and NVIDIA DRIVE<sup>&trade;</sup> OS Linux.

The DriveWorks Post-record Checker verifies data integrity for each recording session.

# Usage

The tool is found in the `tools/capture` folder. Run it by executing:

    ./postrecord-checker --recording_session=<comma-separated list of paths>
                         --output=</path/to/output/report>
                         --sensor=<camera,radar,lidar,imu,gps,can,all>
                         --smoothing=<true/false>
                         --disable_decode=<true/false>

The postrecorder-checker does the following:
- Checks recording session metadata to verify that all requried files exist and
  follow the naming convention.
- Analyzes sensor timestamps.

   The postrecorder-checker outputs a sensor timestamp delta that is calculated as
   timestamps difference from consecutive sensor frames. Those deltas are then used
   to produce a delta histogram. There are 22 bins in the histogram, each of which
   has a step size that is 10% of the expected timestamp delta. The first and last
   bin corresponds to delta values that are less than 0 or larger than 2 times of
   expected delta value.

By default postrecorder-checker sets `--sensor=all`, which means that it parses
all the available sensor data. But you can also direct it to parse a subset of
the sensors. For example, setting `--sensor=camera,radar,lidar` causes the
checker to process the results for only camera, radar, and lidar sensors.

By default, postrecorder-checker sets `--smoothing=true`, which enables smoothing
for the timestamps due to the host system time jittering.

By default, postrecorder-checker sets `--disable_decode=false`, which allows the tool
to decode sensor frames/packets in order to extract the timestamps.

By default, postrecorder-checker will store the results within each session directory.
Setting `--output=</path/to/output/report>` collates the results under the specified directory.

You can find python script batch_run.py in `tools/capture/scripts` folder to process multiple recording sessions under a parent folder

    python ./batch_run.py -b <location of postrecord-checker binary> -i <input parent folder> -o <output folder to store the analysis results>

The output includes sensor timestamp files, i.e. `radar.bin.ts` and metadata report in `report.json`. To visualize timestamp delta histograms,
user can use python script plot_data.py in `tools` folder

    python ./plot_data.py -p <output folder contains the analysis results>

If user provides a single folder which contains the result produced by postrecord-checker, it plots the delta histogram inside the single folder.
If user provides a parent folder which contains multiple result folders produced by batch_run script, it traverses all the folders and
plots average delta histogram inside the parent folder.
