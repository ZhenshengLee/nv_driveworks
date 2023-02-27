# SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_object_detector_tracker_sample Basic Object Detector and Tracker Sample
@tableofcontents

@section dwx_object_detector_tracker_description Description

The Basic Object Detector and Tracker sample demonstrates how the @ref dnn_group can be used for
object detection and the 2D object tracking capabilities of the @ref boxtracker_group module.

The sample streams a H.264 or RAW video and runs DNN inference on each frame to
detect objects using NVIDIA<sup>&reg;</sup> TensorRT<sup>&tm;</sup> model.

The interpretation of the output of a network depends on the network design. In this sample,
2 output blobs (with `coverage` and `bboxes` as blob names) are interpreted as coverage and bounding boxes.

For each frame, it detects the object locations and tracks the objects between video frames. Currently, the
object tracker resorts to image feature detection and tracking. The tracker uses feature motion to predict
the object location.

@section dwx_object_detector_tracker_sample_running Running the Sample

The Basic Object Detector and Tracker sample, `sample_object_detector_tracker`, accepts the following optional parameters. If none are specified, it performs detections on a supplied pre-recorded video.

    ./sample_object_detector_tracker --input-type=[video|camera]
                            --video=[path/to/video]
                            --camera-type=[camera]
                            --camera-group=[a|b|c|d]
                            --camera-index=[0|1|2|3]
                            --slave=[0|1]
                            --tensorRT_model=[path/to/TensorRT/model]

Where:

    --input-type=[video|camera]
            Defines if the input is from live camera or from a recorded video.
            Live camera is supported only on NVIDIA DRIVE platforms.
            Default value: video

    --video=[path/to/video]
            Specifies the absolute or relative path of a raw or h264 recording.
            Only applicable if --input-type=video
            Default value: path/to/data/samples/sfm/triangulation/video_0.h264.

    --camera-type=[camera]
            Specifies a supported AR0231 `RCCB` sensor.
            Only applicable if --input-type=camera.
            Default value: ar0231-rccb-bae-sf3324

    --camera-group=[a|b|c|d]
            Specifies the group where the camera is connected to.
            Only applicable if --input-type=camera.
            Default value: a

    --camera-index=[0|1|2|3]
            Specifies the camera index on the given port.
            Default value: 0

    --slave=[0|1]
            Setting this parameter to 1 when running the sample on Xavier B allows to access a camera that
            is being used on Xavier A. Only applicable if --input-type=camera.
            Default value: 0

    --tensorRT_model=[path/to/TensorRT/model]
            Specifies the path to the NVIDIA<sup>&reg;</sup> TensorRT<sup>&trade;</sup>
            model file.
            The loaded network is expected to have a coverage output blob named "coverage" and a bounding box output blob named "bboxes".
            Default value: path/to/data/samples/detector/<gpu-architecture>/tensorRT_model.bin, where <gpu-architecture> can be `pascal` or `volta-discrete` or `volta-integrated` or `turing`.

@note This sample loads its DataConditioner parameters from DNN metadata JSON file.
To provide the DNN metadata to the DNN module, place the JSON file in the same
directory as the model file. An example of the DNN metadata file is:

    data/samples/detector/pascal/tensorRT_model.bin.json

@subsection dwx_object_detector_tracker_sample_examples Examples

#### Default usage

     ./sample_object_detector_tracker

The video file must be a H.264 or RAW stream. Video containers such as MP4, AVI, MKV, etc. are not supported.

#### To run the sample on a video on NVIDIA DRIVE or Linux platforms with a custom TensorRT network

    ./sample_object_detector_tracker --input-type=video --video=<video file.h264/raw> --tensorRT_model=<TensorRT model file>

#### To run the sample on a camera on NVIDIA DRIVE platforms with a custom TensorRT network

    ./sample_object_detector_tracker --input-type=camera --camera-type=<rccb_camera_type> --camera-group=<camera group> --camera-index=<camera idx on camera group> --tensorRT_model=<TensorRT model file>

where `<rccb_camera_type>` is a supported `RCCB` sensor.
See @ref supported_sensors for the list of supported cameras for each platform.

@section dwx_object_detector_tracker_sample_output Output

The sample creates a window, displays the video streams, and overlays the list
of features and detected/tracked bounding boxes of the objects with IDs.

The color coding of the overlay is:

- Red bounding boxes: Indicate successfully tracked bounding boxes.
- Red points: Indicate successfully tracked 2D features.
- Yellow bounding box: Identifies the region which is given as input to the DNN.

![Object tracker on a H.264 stream](sample_object_tracker.png)

@section dwx_object_detector_tracker_sample_more Additional Information

For more information, see:
- @ref dnn_mainsection
- @ref dataconditioner_mainsection
- @ref imageprocessing_tracking_mainsection
