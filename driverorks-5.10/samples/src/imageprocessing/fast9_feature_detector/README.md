# SPDX-FileCopyrightText: Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_fast9_feature_detector_sample FAST9 Feature Detector Sample
@tableofcontents

@section dwx_fast9_feature_detector_description Description

The FAST9 Feature Detector sample demonstrates the feature detection capabilities using the
FAST9 algorithm of the @ref imageprocessing_featuredetector_mainsection module. It loads a video stream and
reads the images sequentially. For each frame, it detects feature points using the FAST9 algorithm.

@section dwx_fast9_feature_detector_running Running the Sample

The FAST9 Feature Detector sample, sample_fast9_feature_detector, accepts the following optional parameters. If none are specified, it will perform detections on a supplied pre-recorded video.

    ./sample_fast9_feature_detector --video=[path/to/video.h264]
                                    --maxFeatureCount=[even_number]
                                    --scoreThreshold=[fp_number]
                                    --NMSRadius=[0|1]
                                    --usePinnedMemory=[0|1]

where

    --video=[path/to/video.h264]
        Is the absolute or relative path of a h264 or RAW/LRAW video.
        Containers such as MP4, AVI, MKV, etc. are not supported.
        Default value: path/to/data/samples/sfm/triangulation/video_0.h264.

    --maxFeatureCount=[even_number]
        Specifies the maximum number of features that can be stored.
        Default value: 4096

    --scoreThreshold=[fp_number]
        Defines the strength for which a point is classified as a corner.
        Default value: 56

    --NMSRadius=[0|1]
        When set to 1, non-maximum suppression will be applied.
        Default value: 0

    --usePinnedMemory=[0|1]
        When set to 1, it would use of PinnedMemory for faster CudaMemcpy.
        Default value: 0

@section dwx_fast9_feature_detector_output Output

The sample creates a window, displays the video, and overlays the list of features.
The current feature positions of the current frame will be overlaid by small squares.

![FAST9 Sample Output](sample_fast9.png)

For more details see @ref imageprocessing_features_usecase1.
