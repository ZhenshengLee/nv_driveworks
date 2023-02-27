# SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_feature_tracker_sample Feature Tracker Sample
@tableofcontents

@section dwx_feature_tracker_description Description

The Feature Tracker sample demonstrates the feature detection and feature
tracking capabilities of the @ref imageprocessing_tracking_mainsection module. It loads a video stream and
reads the images sequentially. For each frame, it tracks features  from the
previous frame and detects new features in the current frame.

@section dwx_feature_tracker_running Running the Sample

The feature tracker sample, sample_feature_tracker, accepts the following optional parameters. If none are specified, it will perform detections on a supplied pre-recorded video.

    ./sample_feature_tracker --video=[path/to/video.h264]
                             --loop=[0|1]
                             --maxFeatureCount=[even_number]
                             --historyCapacity=[int_number]
                             --pyramidLevel=[int_number]
                             --detectMode=[0|1]
                             --pvaDetector=[0|1]
                             --pvaDetectorEngineNo=[0|1]
                             --detectLevel=[int_number]
                             --cellSize=[int_number]
                             --scoreThreshold=[fp_number]
                             --detailThreshold=[fp_number]
                             --numEvenDistributionPerCell=[int_number]
                             --harrisK=[fp_number]
                             --harrisRadius=[int_number]
                             --gradientSize=[3|5|7]
                             --blockSize=[3|5|7]
                             --NMSRadius=[int_number]
                             --maskType=[0|1]
                             --enableMaskAdjustment=[0|1]
                             --trackMode=[0|1]
                             --pvaTracker=[0|1]
                             --pvaTrackerEngineNo=[0|1]
                             --windowSize=[int_number]
                             --numIterTranslation=[int_number]
                             --numIterScaling=[int_number]
                             --numTranslationOnlyLevel=[int_number]
                             --nccUpdateThreshold=[fp_number]
                             --nccKillThreshold=[fp_number]
                             --displacementThreshold=[fp_number]
                             --enableAdaptiveWindowSize=[0|1]
                             --largeMotionKillRatio=[fp_number]
                             --maxScaleChange=[fp_number]
                             --stopFrameIdx=[int_number]
                             --useHalfDetector=[0|1]
                             --useHalfTracker=[0|1]
                             --useHalfRes=[0|1]

where

    --video=[path/to/video.h264]
        Is the absolute or relative path of a h264 or RAW/LRAW video.
        Containers such as MP4, AVI, MKV, etc. are not supported.
        Default value: path/to/data/samples/sfm/triangulation/video_0.h264.

    --loop=[0|1]
        Specifies the replay behavior when video reaches end of stream.
        0: exit app
        1: loop from start
        Default value: 1

    --maxFeatureCount=[even_number]
        Specifies the maximum number of features that can be stored.
        Note that only even numbers will work. If using PVA Sparse LK feature tracker,
        maxFeatureCount needs to be set as no more than 512.
        Default value: 4000

    --historyCapacity=[int_number]
        Specifies how many features are going to be stored.
        Default value: 60

    --pvaPyramid=[0|1]
        When set to 1, the pyramid is performed on PVA hardware.
        Otherwise, it is performed on GPU.

    --pvaPyramidEngineNo=[0|1]
        If pvaPyramid is set as 1, determines on which PVA pyramid should run on.
        Note, Orin only has 1 PVA engine.
        Default value: 0

    --pyramidLevel=[int_number]
        Defines the number of pyramid levels while tracking.
        Higher level provides better results for large motion, but takes more time.
        If PVA pyramid is enabled, only pyramidLevel 5 is supported.
        Default value: 5

    --detectMode=[0|1]
        Defines detecting mode
        0: Standard Harris corner detector with fixed pipeline.
        1: Extended Harris corner detector, more flexible but takes more time.
        Default value: 1

    --pvaDetector=[0|1]
        When set to 1, the feature detector algorithm is performed on PVA hardware.
        Otherwise, it is performed on GPU.
        Note that PVA can only be used in combination with --detectMode=1.
        Default value: 0

    --pvaDetectorEngineNo=[0|1]
        If pvaDetector is set as 1, determines on which PVA the detector should run on.
        Default value: 0

    --cellSize=[int_number]
        Defines the size of a cell. Input images will be split into cellSize x cellSize structures.
        This parameter only takes effect when --detectMode=0 and --pvaDetector=0.
        Default value: 64

    --scoreThreshold=[fp_number]
        Defines the minimum score for which a point is classified as a feature.
        Lower value will output more features, higher value will only keep the high frequency points.
        When --detectMode=0 and --pvaDetector=1, this should be a uint value and greater than 0.
        Default value: 4e-5

    --detailThreshold=[fp_number]
        Defines the minimum score for which a point is classified as a detail feature. A detail feature has higher priority to be output during detection.
        Higher values will output more even distribution, and lower values will have more features in high frequency area.
        This parameter only takes effect when --detectMode=0 and --pvaDetector=0.
        Default value: 0.0128

    --numEvenDistributionPerCell=[int_number]
        Defines the number of features to be selected in each cell, where the score is within [scoreThreshold, detailThreshold)
        This parameter only takes effect when --detectMode=0 and --pvaDetector=0.
        Default value: 5

    --harrisK=[fp_number]
        Defines Harris K value.
        This parameter only takes effect when --detectMode=1.
        Default value: 0.05

    --harrisRadius=[int_number]
        Defines Harris radius.
        This parameter only takes effect when --detectMode=1.
        Default value: 1

    --gradientSize=[3|5|7]
        Define gradient window size for PVA feature detector.
        This parameter only takes effect when --detectMode=0 and --pvaDetector=1.
        Default value: 3

    --blockSize=[3|5|7]
        Define block window size used to compute the Harris Corner score in PVA.
        This parameter only takes effect when --detectMode=0 and --pvaDetector=1.
        Default value: 3

    --NMSRadius=[int_number]
        Defines non-maximum suppression filter radius.
        This parameter only takes effect when --detectMode=1.
        Default value: 1

    --maskType=[0|1]
        Defines output distribution type.
        0: Uniform distribution
        1: Gaussian distribution
        This parameter only takes effect when --detectMode=1.
        Default value: 1

    --enableMaskAdjustment=[0|1]
        Defines whether the detector will update output distribution mask before each detection.
        0: disable
        1: enable
        This parameter only takes effect when --detectMode=1.
        Default value: 1

    --trackMode=[0|1|2]
        Defines tracking mode.
        0: translation-only KLT tracker or Sparse LK PVA tracker
        1: translation-scale KLT tracker
        2: translation-scale fast KLT tracker
        Default value: 2

    --pvaTracker=[0|1]
        When set to 1, the feature tracking algorithm is performed on PVA.
        Otherwise, it is performed on GPU.
        Default value: 0

    --pvaTrackerEngineNo=[0|1]
        If pvaTracker is set as 1, determines on which PVA the tracker should run on.
        Default value: 0

    --windowSize=[int_number]
        Defines the feature window size.
        --trackMode=0 supports windowSize=6,8,10,12,15
        --trackMode=1 supports windowSize=10,12 for GPU and windowSize=11 for PVA
        Default value: 10

    --numIterTranslation=[int_number]
        Defines the KLT iteration number for translation-only tracking or
        PVA Sparse LK tracking iteration number.
        Larger number gives better prediction, but takes more time.
        Default value: 8

    --numIterScaling=[int_number]
        Defines the KLT iteration number for translation-scaling tracking.
        Larger number gives better prediction, but takes more time.
        This parameter only takes effect when --trackMode=1 or 2.
        Default value: 10

    --numTranslationOnlyLevel=[int_number]
        Defines number of translation-only tracking in pyramid. The tracker will apply
        translation-only tracking on the highest numTranslationOnlyLevel level images.
        This parameter only takes effect when --trackMode=1 or 2.
        Default value: 4

    --nccUpdateThreshold=[fp_number]
        Defines the minimum ncc threshold that will cause the feature tracker to update
        the image template for a particular feature.
        This parameter only takes effect when --trackMode=1.
        Default value: 0.95

    --nccKillThreshold=[fp_number]
        Defines the minimum ncc threshold to mantain a particular feature in the tracker.
        This parameter only takes effect when --trackMode=1 or 2.
        Default value: 0.3

    --displacementThreshold=[fp_number]
        Defines the early stop threshold during translation-only tracking.
        This parameter only takes effect when --trackMode=1 or 2.
        Default value: 0.1

    --enableAdaptiveWindowSize=[0|1]
        Defines whether to use full window size at the lowest and the highest levels,
        and smaller window size at the rest of levels during tracking.
        This parameter only takes effect when --trackMode=1 or 2.
        Default value: 1

    --largeMotionKillRatio=[fp_number]
        Defines the ratio value that is used to determine if the feature's movement is
        too large. Features will be killed if the motion is larger than the template size
        times the large motion killing threshold during tracking.
        --largeMotionKillRatio=0.33 for front cameras.
        --largeMotionKillRatio=0.5 for side cameras.
        This parameter only takes effect when --trackMode=1 or 2.
        Default value: 0.33

    --maxScaleChange=[fp_number]
        Defines the maximum allowed scale change for the tracked points across consecutive frames.
        --maxScaleChange=1.8 for front cameras.
        --maxScaleChange=3.0 for side cameras.
        This parameter only takes effect when --trackMode=1 or 2.
        Default value: 1.8

    --stopFrameIdx=[int_number]
        Defines the frame index to stop the application.
        --stopFrameIdx=0 processes the entire video.
        Default value: 0

    --useHalfDetector=[0|1]
        Defines whether to use fp16 for detection.
        --useHalfDetector=0 uses fp32 for detection.
        --useHalfDetector=1 uses fp16 for detection.
        Default value: 0

    --useHalfTracker=[0|1]
        Defines whether to use fp16 for tracking.
        --useHalfTracker=0 uses fp32 for tracking.
        --useHalfTracker=1 uses fp16 for tracking.
        This parameter only takes effect when --trackMode=1 or 2.
        Default value: 0

    --useHalfRes=[0|1]
        Defines whether to use half resolution for image processing.
        --useHalfRes=0 uses full resolution for image processing.
        --useHalfRes=1 uses half resolution for image processing.
        Default value: 1

    --detectLevel=[int_number]
        Defines the pyramid level at which feature detection is performed.
        This parameter has no effect when --pvaDetector=1. PVA detector always performs
        feature detection at level 0.
        Default value: 1

@section dwx_feature_tracker_output Output

The sample creates a window, displays the video, and overlays the list of features.

There are two modes for feature drawing:
- (default) feature trajectories will be overlaid (up to previous 10 frames of history).
- Only the current feature positions will be overlaid by small squares.
You can switch drawing mode by pressing 'h' key.

Different colors represent the age for each feature. The age refers to how many frames have been tracked.
- Red: 1 <= age < 5
- Yellow: 5 <= age < 10
- Green: 10 <= age < 20
- Light blue: 20 <= age

![Tracked feature points on a single H.264 stream](sample_feature_tracker.png)

@section dwx_feature_tracker_more Additional Information

For more details see @ref imageprocessing_tracking_usecase1.
