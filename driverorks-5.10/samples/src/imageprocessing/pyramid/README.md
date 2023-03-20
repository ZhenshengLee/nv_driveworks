# Copyright (c) 2019-2022 NVIDIA CORPORATION.  All rights reserved.

@page dwx_image_pyramid_pva_sample Image Pyramid PVA Sample
@tableofcontents

@section dwx_image_pyramid_pva_description Description

The Image Pyramid PVA sample demonstrates the image pyramid creation pipeline on hardware units:
- PVA (Programmable Vision Accelerator).

The Image Pyramid PVA sample accepts  RAW videos, h264 videos, or live camera input data from an AR0231 (revision >= 4) sensor. It then applies gaussain based scaling algorithm to create images for specified number of levels.

@section dwx_image_pyramid_pva_sample_running Running the Sample

The Image Pyramid pva sample, sample_image_pyramid_pva, accepts the following optional parameters. If none are specified, it performs detections on a supplied pre-recorded video.

    ./sample_image_pyramid_pva --input-type=[video|camera]
                              --video=[path/to/video]
                              --camera-type=[camera]
                              --camera-group=[a|b|c|d]
                              --camera-index=[0|1|2|3]
                              --pyrLevel=[5]
                              --pvaNo=[0]

Where:

    --input-type=[video|camera]
            Defines if the input is from live camera or from a recorded video.
            Live camera is supported only on NVIDIA DRIVE(tm) platforms.
            It is not supported on Linux (x86 architecture) host systems.
            Default value: video

    --video=[path/to/video]
            Specifies the absolute or relative path of a raw or h264 recording.
            Only applicable if --input-type=video
            Default value: path/to/data/samples/stereo/left_1.h264

    --camera-type=[camera]
            Specifies a supported AR0231 `RCCB` sensor.
            Only applicable if --input-type=camera.
            Default value: ar0231-rccb-bae-sf3324

    --camera-group=[a|b|c]
            Is the group where the camera is connected to.
            Only applicable if --input-type=camera.
            Default value: a

    --camera-index=[0|1|2|3]
            Indicates the camera index on the given port.
            Default value: 0

    --pyrLevel=[5]
            Indicates the number of levels in a pyramid. Currently only 5 is supported.
            Default value: 5

    --pvaNo=[0]
            Indicates on which PVA engine the corresponding stage of the pipeline is to be executed.
            Currently only 0 is supported
            Default value: 0

@subsection dwx_image_pyramid_pva_sample_examples Examples

### To run the sample on a video

    ./sample_image_pyramid_pva --input-type=video --video=<video file>

### To run the sample on a camera on NVIDIA DRIVE platforms

    ./sample_image_pyramid_pva --input-type=camera --camera-type=<camera type> --camera-group=<camera group> --camera-index=<camera idx on camera group>

@subsection dwx_image_pyramid_pva_sample_more Additional information

For more information, see @ref imageprocessing_filtering_mainsection.
