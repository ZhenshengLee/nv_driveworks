# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_feature_descriptor_sample ORB Feature Descriptor Sample
@tableofcontents

@section dwx_feature_descriptor_description Description

The ORB Feature Descriptor sample demonstrates the ORB feature detection and descriptor computation
capabilities using the FAST9 detector and ORB descriptor algorithms of the @ref imageprocessing_features_mainsection module.
It loads a video stream and reads the images sequentially. For each frame, it computes the pyramids, detects feature points
on each pyramid level using the FAST9 algorithm and computes descriptors for those points using the BRIEF algorithm.
The maxFeatureCount features are divided among pyramid levels proportional to the resolution of each level.
The BRIEF descriptor implementation is only available for PVA HW and hence this sample is not supported on x86.

@section dwx_feature_descriptor_running Running the Sample

The ORB Feature Descriptor sample, sample_feature_descriptor, accepts the following optional parameters. If none are specified, it will perform detections on a supplied pre-recorded video.

    ./sample_feature_descriptor --video=[path/to/video.h264]
                                    --maxFeatureCount=[even_number]
                                    --pvaEngineNo=[0]
                                    --scoreThreshold=[fp_number]
                                    --dataType=[pixel_data_type]
                                    --pyramidLevel=[int_number]
                                    --stopFrameIdx=[int_number]
                                    --capture-end-frame=[int_number]
                                    --capture-file=[path/to/output/video]
                                    --capture-fps=[int_number]
                                    --capture-screen=[0|1]
                                    --capture-start-frame=[int_number]

where

    --video=[path/to/video.h264]
        Is the absolute or relative path of a h264.
        Containers such as MP4, AVI, MKV, etc. are not supported.
        Default value: path/to/data/samples/sfm/triangulation/video_0.h264.

    --maxFeatureCount=[should be even]
        Specifies the maximum number of features that can be stored.
        If using PVA to do fast9 feature detection, maxFeatureCount needs to
        be set as no more than 1024.
        Default value: 1000

    --pvaEngineNo=[0]
        Defines the PVA engine to run on.
        Default value: 0

    --scoreThreshold=[fp_number]
        Defines the strength for which a point is classified as a corner.
        Default value: 5.6

    --stopFrameIdx=[number_of_frame]
        When set it will process only those many frames from starting. If it is 0 then entire sequence is processed
        Default value: 0

    --dataType=[pixel_data_type]
        Defines the type of data for pixel.
        Can be float16, float32, uint16 or uint8

    --pyramidLevel[int_number]
        Specifies number of pyramid levels.

    --capture-screen=[0|1]
        Defines whether the screen will be captured as an output video.
        0: disable screen capture
        1: enable screen capture
        Default value: 0

    --capture-start-frame=[int_number]
        Defines the frame index where the screen capture starts.
        This parameter only takes effect when --capture-screen=1.
        Default value: 0

    --capture-end-frame=[int_number]
        Defines the frame index where the screen capture ends.
        --capture-end-frame=-1 to capture all frames.
        This parameter only takes effect when --capture-screen=1.
        Default value: -1

    --capture-file=[path/to/output/video]
        Defines the output filename and file type of the screen capture.
        This parameter only takes effect when --capture-screen=1.
        Default value: capture.h264

    --capture-fps=[int_number]
        Defines the screen capture framerate.
        This parameter only takes effect when --capture-screen=1.
        Default value: 15

@section dwx_feature_descriptor_output Output

The sample creates a window, displays the video, and overlays the list of features.
The feature positions from pyramid level 0 of current  frame will be overlaid by small squares.

![FAST9 Sample Output](sample_fast9.png)

For more details see @ref imageprocessing_features_usecase1.
