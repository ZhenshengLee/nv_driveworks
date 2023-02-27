# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_intrinsics_constraints Intrinsics Constraints Tool
@tableofcontents

@section dwx_intrinsics_constraints_description Description

The NVIDIA<sup>&reg;</sup> DriveWorks Intrinsics Constraints tool extracts intrinsics constraints used during
calibration for each individual camera. It takes any of the following as input:

- A [recorded video with a checkerboard pattern](@ref dwx_intrinsics_constraints_checkerboard).
- A [recorded video with an AprilTag target](@ref dwx_intrinsics_constraints_AprilTag_video) moved in front of the cameras.
- A [sequence of still frames containing the AprilTag target](@ref dwx_intrinsics_constraints_AprilTag_still) in front of the cameras. This is useful when modelling the intrinsics for DLSR cameras.

It then exports all required constraints in a JSON file, placed in the `intrinsics` subfolder in the directory structure.

@section dwx_intrinsics_constraints_prerequisites Prerequisites

This tool is available on the x86 Host System and NVIDIA DRIVE<sup>&trade;</sup> OS Linux.

This tool creates output files that are placed into the current working directory by default. Please ensure the following for your convenience:
- Write permissions are enabled for the current working directory.
- Include the tools folder in the binary search path of the system.
- Execute from your home directory.

@section dwx_intrinsics_constraints_usage Running the Tool

The Intrinsics Constraints tool accepts the following parameters. Several of these parameters are required based on the input type.<br>
For more information, please refer to the @ref dwx_intrinsics_constraints_examples.

    ./calibration-intrinsics-constraints --use-checkerboard=12x9
                                         --targetDB=[targetDB_file_path]
                                         --input-video=/[video_path]/camera-0.h264
                                         --input-folder=[folder_path]
                                         --output=/[calib_data_path]/intrinsics/camera-0.json
                                         --camera-model=[ftheta]
                                         --rig=[path/to/rig/file]
                                         [--max-constraints=0]
                                         [--skipFrameCount=0]

@subsection dwx_intrinsics_constraints_parameters Parameters

    --use-checkerboard=[NxN checkerboard grid]
            Description: The input NxN checkerboard grid.
            Example: --use-checkerboard=12x9

    --targetDB=[path to targetDB file, default=]
            Description: Path to AprilTag target database. Enables detection of AprilTag targets. Either this or [targetDB] must be supplied.

    --apriltags=[{CPU, CPU_FAST, GPU}, default=CPU]
            Description: April tag detection backend. The CPU backend is most accurate, while CPU_FAST/GPU provide faster but potentially less accurate detections.

    --input-video=[path to input video]
            Description: The file path where the input video is, for checkerboards and AprilTag videos.

    --input-folder=[path to input folder]
            Description: The file path where the input folder is, for still frames with AprilTag targets.

    --output=[path to output file]
            Description: The file path where the intrinsics constraints will be saved.
                         If this parameter is blank, they will automatically be saved to '[input-video].json'.

    --camera-model=[camera model, default=ftheta]
            Description: Camera model used for calibration. Available options are: [pinhole, ocam, ftheta]. The camera model is not used in any way for extracting the constraints. It is passed directly into the produced json file and may be used by other tools.

    --rig=[path to rig file, default=]
            Description: Rig file with initial guess of camera intrinsics. Only available for `ftheta` calibration.

    --max-constraints=[integer, default=0]
            Description: If non-zero, the maximum number of constraints after which stopping the intrinsic constraint extraction process.

    --skipFrameCount=[integer, default=0]
            Description: Number of frames to skip at the beginning of the video.

@section dwx_intrinsics_constraints_examples Examples

@subsection dwx_intrinsics_constraints_checkerboard For a 12x9 Checkerboard

    ./calibration-intrinsics-constraints --use-checkerboard=12x9
                                         --input-video=/[video_path]/camera-0.h264
                                         --output=/[calib_data_path]/intrinsics/camera-0.json

@subsection dwx_intrinsics_constraints_AprilTag_video For an AprilTag Video

    ./calibration-intrinsics-constraints --targetDB=/[calib_data_path]/targets.json
                                         --input-video=/[video_path]/camera-0.h264
                                         --output=/[calib_data_path]/intrinsics/camera-0.json

@subsection dwx_intrinsics_constraints_AprilTag_still For a Sequence of Still Frames with an AprilTag Target

    ./calibration-intrinsics-constraints --targetDB=/[calib_data_path]/targets.json
                                         --input-folder=[folder_path]
                                         --output=/[calib_data_path]/intrinsics/external.json

@section dwx_intrinsics_constraints_output Output

The tool will open a window playing back the input video and indicate with a red or green border if a new intrinsic constraint has been collected (i.e. target or checkerboard has been found)

![](intrinsic_constraints.png)

@warning For correct results, the tool must be able to find at least 30
checkerboard squares per camera.
