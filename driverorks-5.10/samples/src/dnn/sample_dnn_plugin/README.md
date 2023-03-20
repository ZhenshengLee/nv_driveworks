# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_dnn_plugin_sample DNN Plugin Sample
@tableofcontents

@section dwx_dnn_plugin_description Description

The DNN Plugin sample loads and runs an MNIST network with a custom implementation of the max pooling layer.

**PoolPlugin.cpp** implements this layer and defines the functions that are required by DriveWorks to load
a plugin. This file is then compiled as a shared library, which is then loaded by **sample_dnn_plugin** executable
at runtime.

@section dwx_dnn_plugin_running Running the Sample

The command line for the sample is:

    ./sample_dnn_plugin


Accepts the following optional parameters:

    ./sample_dnn_tensor --tensorRT_model=[path/to/TensorRT/model]

Where:

    --tensorRT_model=[path/to/TensorRT/model]
            Specifies the path to the NVIDIA<sup>&reg;</sup> TensorRT<sup>&trade;</sup>
            model file.
            The loaded network is expected to have a output blob named "prob".
            Default value: path/to/data/samples/dnn/<gpu-architecture>/mnist.bin, where <gpu-architecture> can be `volta-discrete` or `volta-integrated` or `turing`.

@note This sample loads mnist digit recognition network from:

- data/samples/dnn/volta if the detected GPU has VOLTA architecture.
- data/samples/dnn/turing if the detected GPU has TURING architecture.

@section dwx_dnn_plugin_output Output

The sample creates a window, displays a white screen where a hand-drawn digit will be recognized via aforementioned MNIST network model.

![Digit being recognized using DNN with a plugin](sample_dnn_plugin.png)

@section dwx_dnn_plugin_more Additional Information

For more information, see @ref dwx_dnn_plugins.
