# SPDX-FileCopyrightText: Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_calibration_recorder Static Calibration Recorder Tool

@tableofcontents

@section dwx_calibration_recorder_description Description

The Static Calibration Recorder Tool can be used to capture the data that is required by the @ref dwx_calibration_graph_cli.
It takes a rig file as input and renders the live video streams of the rig's cameras. In "Extrinsics Mode" it allows to capture and store images from those camera video streams. In "Intrinsics Mode", the intrinsics constraints created by checkerboard marker detections are accumulated and can be stored to .json files that are required for further processing in the static camera calibration tool. If the camera's intrinsics are available from its EEPROM, the intrinsics will be read from there and stored in the corresponding .json file. In that case, it is not necessary to capture intrinsics constraints. If the camera has an intrinsics model assigned, it will be displayed in the panel of the "Intrinsics Mode". 
The rig file can contain camera parameters for live camera access, or file paths to recorded files.
Alternatively, when working with video files, a single video file can be provided instead of a rig file.

@section dwx_calibration_recorder_prerequisites Prerequisites

This tool is available on the x86 Host System and NVIDIA DRIVE<sup>&trade;</sup> OS Linux.

@section dwx_calibration_recorder_run-command Running the Tool

Run the tool by executing:

    /calibration-recorder --rig=[path to rig file]
                          --targetDB=[path to targets.json file]
                          --output=[path to directory where new directories "intrinsics" / "extrinsics" (or "extrinsics_<number>" if "external" already exists) are created to store the results]

@subsection dwx_calibration_recorder_parameters Parameters

    --rig=[path to rig file]
            Description: Rig file containing the cameras.
            Example: --rig=rig.json

    --video=[path to video file]
            Video to be loaded as a virtual camera. This parameter is ignored if a rig is provided.
            Example: --video=camera.mp4

    --targetDB=[path to target database json file]
            Description: Path to json file with apriltag target description. AprilTag detection is disabled if not provided.
            Example: --targetDB=targets.json

    --mode=[extrinsics / intrinsics]
            Description: Optional. Set mode enabled at startup. (extrinsics / intrinsics)
            Example: --mode=extrinsics

    --checkerboard=[<width>x<height>]
            Description: Optional. Set count of checkerboard squares at startup.
            Example: --checkerboard=14x9

    --output=[path to output directory]
            Description: Directory in which the 'extrinsics' / 'intrinsics' output folders are created to store results
            Example: --output=/tmp/calibration-data

    --cameraSensorFifoSize=[number of frames in camera sensor fifo]
            Description: Optional. Use a small value (minimum 1) to use less GPU memory. Use larger values to avoid frame drops. (Experimental Feature)
            Example: --cameraSensorFifoSize=4

@subsection dwx_calibration_recorder_usage Usage

Image files with frame captures and .json files with intrinsics constraints, each withing their subdirectory "extrinsics" or "intrinsics" are stored to the output directory when clicking the "Capture" button or pressing the "Enter" key. The output directory can be specified via command line parameter, by pasting a path into the corresponding UI, or by drag and dropping a directory onto the tool.

There is a UI to select the cameras to be rendered simultanously. This allows to observe and control the placement of calibration markers in the scene. When clicking on a camera while holding the control key, that camera is selected exclusively. See @ref dwx_camera_calibration_setup for more details on how to setup the scene for extrinsics.

![](tool_calibration-recorder_extrinsics.png)

In "Intrinsics Mode", intrinsics constraints are accumulated for the enabled cameras when target detection is enabled. All currently accumulated intrinsics constraint points are rendered overlayed onto the live image as red dots. The number of currently collected intrinsics constraints are printed in parentheses next to the camera name. The points of the current target detection are rendered in green. In order to collect a constraint, the target has to be held still for a moment. Once the constraints have been collected at a position, the rendered dots of the current detection turn blue. When this happens, the target can be moved to the next pose. The detection-dots turn green again once the target has been moved sufficiently far away from the previously collected constraint. See @ref dwx_camera_calibration_data_int for more details about how to capture intrinsics constraints.

![](tool_calibration-recorder_intrinsics.png)

The results are stored to .json files when the "Save" button in the UI or the "Enter" key is pressed. If there are unsaved instrinsics constraints at program exit, they get saved to disk as well.

An intrinsics calibration can be performed on selected cameras with their currently collected constraints by pressing the "Calibrate" button. The resulting camera model parameters will then be printed to both the console and in the UI panel. If a camera model is available in the camera's EEPROM, it will be read from there. The camera model will be saved to the intrinsics .json file, together with the collected intrinsics constraints, or just by itself without constraints if none have been collected. It is not required to collect constraints if a model is already available. The @ref dwx_calibration_graph_cli tool will then use both the constraints and the existing camera model in its calibration process.

There is an option to select if and which images to store together with the intrinsics constraints for debugging and archiving purposes. By default, all frames for which an instrinsics constraint is collected are stored. It is possible to additionally store images where a target has been detected, but the detection is not used as a constraint. Finally, there's an option to store all processed frames, that is including also frames where no checkerboard is recorded.

If the targetDB parameter is omitted, AprilTag detection is not available, however checkerboard detection for intrinsics and frame capturing for extrinsics images is still possible.


@section dwx_calibration_recorder_output Output

In "Extrinsics Mode" the tool outputs a screenshot for each camera listed in the rig. It creates a directory called "extrinsics" (as these captures are intended to be used for extrinsic camera calibration).
If such a directory already exists, a suffix '_\<number\>' is appended to create a non-existing directory path. This avoids overwriting existing data.
The capture images are stored inside that new directory as files named '\<camera name\>.png'.
Similarly, in "Intrinsics Mode" the tool outputs a .json file with intrinsics constraints for each camera. If no intrinsics constraints are available for a camera, no file is written. Optionally, images files are stored for each constraint in the .json file.
