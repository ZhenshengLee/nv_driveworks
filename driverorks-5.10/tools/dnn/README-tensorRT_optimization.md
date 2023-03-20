# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_tensorRT_tool TensorRT Optimizer Tool
@tableofcontents

@section dwx_tensorRT_tool_description Description

The NVIDIA<sup>&reg;</sup> DriveWorks TensorRT Optimizer Tool enables optimization for a given Caffe, UFF or ONNX model using TensorRT.

For specific examples, please refer to the following:

- @ref dwx_tensorRT_tool_examples_uff.
- @ref dwx_tensorRT_tool_examples_caffe.
- @ref dwx_tensorRT_tool_examples_onnx.

@section dwx_tensorRT_tool_prerequisites Prerequisites

This tool is available on NVIDIA DRIVE<sup>&trade;</sup> OS Linux.

This tool creates output files that are placed into the current working directory by default. Please ensure the following for your convenience:

- Write permissions are enabled for the current working directory.
- Include the tools folder in the binary search path of the system.
- Execute from your home directory.

@section dwx_tensorRT_tool_usage Running the Tool

The TensorRT Optimization tool accepts the following parameters. Several of these parameters are required based on model type. \n
For more information, please refer to the @ref dwx_tensorRT_tool_examples.

Run the tool by executing:

    ./tensorRT_optimization --modelType=[uff|caffe|onnx]
                            --outputBlobs=[<output_blob1>,<output_blob2>,...]
                            --prototxt=[path to file]
                            --caffemodel=[path to file]
                            --uffFile=[path to file]
                            --inputBlobs=[<input_blob1>,<input_blob2>,...]
                            --inputDims=[<NxNxN>,<NxNxN>,...]
                            --onxxFile=[path to file]
                            [--iterations=[int]]
                            [--batchSize=[int]]
                            [--half2=[int]]
                            [--out=[path to file]]
                            [--int8]
                            [--calib=[calibration file name]]
                            [--cudaDevice=[CUDA GPU index]]
                            [--useDLA]
                            [--useSafeDLA]
                            [--dlaLayerConfig=[path to json layer config]]
                            [--pluginConfig=[path to plugin config file]]
                            [--precisionConfig=[path to precision config file]]
                            [--testFile=[path to binary file]]
                            [--useGraph=[int]]
                            [--workspaceSize=[int]]
                            [--explicitBatch=[int]]

@subsection dwx_tensorRT_tool_params Parameters

    --modelType=[uff|caffe|onnx]
            Description: The type of model to be converted to the TensorRT network.
            Warning: uff and caffe model types are deprecated and will be dropped in the next major release.

    --outputBlobs=[<output_blob1>,<output_blob2>,...]
            Description: Names of output blobs combined with commas.
            Example: --outputBlobs=bboxes,coverage

    --prototxt=[path to file]
            Description: Deploys a file that describes the Caffe network.
            Example: --prototxt=deploy.prototxt

    --caffemodel=[path to file]
            Description: Caffe model file containing weights.
            Example: --caffemodel=weights.caffemodel

    --outputBlobs=[<output_blob1>,<output_blob2>,...]
            Description: Names of output blobs combined with commas.
            Example: --outputBlobs=bboxes,coverage

    --uffFile=[path to file]
            Description: Path to a UFF file.
            Example: --uffFile=~/myNetwork.uff

    --inputBlobs=[<input_blob1>,<input_blob2>,...]
            Description: Names of input blobs combined with commas. Ignored if the model is ONNX or Caffe.
            Example: --inputBlobs=data0,data1

    --inputDims=[<NxNxN>,<NxNxN>,...]
            Description: Input dimensions for each input blob separated by commas, given in the same
                         order as the input blobs.
                         Dimensions are separated by `x`, and given in CHW format.
            Example: --inputDims=3x480x960,1x1x10

    --onxxFile=[path to file]
            Description: Path to an ONNX file.
            Example: --onnxFile=~/myNetwork.onnx

    --iterations=[int]
            Description: Number of iterations to run to measure speed.
                         This parameter is optional.
            Example: --iterations=100
            Default value: 10

    --batchSize=[int]
            Description: Batch size of the model to be generated.
                         This parameter is optional.
            Example: --batchSize=2
            Default value: 1

    --half2=[int]
            Description: The network running in paired fp16 mode. Requires platform to support native fp16.
                         This parameter is optional.
            Example: --half2=1
            Default value: 0

    --out=[path to file]
            Description: Name of the optimized model file.
                         This parameter is optional.
            Example: --out=model.bin
            Default value: optimized.bin

    --int8
            Description: If specified, run in INT8 mode.
                         This parameter is optional.

    --calib=[calibration file name]
            Description: INT8 calibration file name.
                         This parameter is optional.
            Example: --calib=calib.cache

    --cudaDevice=[CUDA GPU index]
            Description: Index of a CUDA capable GPU device.
                         This parameter is optional.
            Example: --cudaDevice=1
            Default value: 0

    --verbose = [int]
            Description: Enable tensorRT verbose logging.
                         This parameter is optional
            Default value: 0

    --useDLA
            Description: If specified, this generates a model to be executed on DLA. This argument is only valid on platforms with DLA hardware.
                         This parameter is optional.

    --useSafeDLA
            Description: If specified, this generates a model to be executed on DLA.
                         The safe mode indicates all layers must be executable on DLA, the input/output of the DNN module
                         must be provided in the corresponding precision and format, and the input/output tensors must be provided
                         as NvMediaTensor for best performance.
                         `dwDNN` module is capable of streaming NvMediaTensors from/to CUDA and
                         converting precisions and format. For more information, please refer to `dwDNN` module's documentation.

    --dlaLayerConfig
            Descripton: If specified, specific layers to be forced to GPU are read from this json. Layers to be run on GPU can be specified by type of layer or layer number. Layer type and layer number can be obtained from logs by running with default template. This argument is valid only if --useDLA=1
                        This parameter is optional.
            Example: --dlaLayerConfig=./template_dlaconfig.json

    --pluginConfig=[path to plugin config file]
            Description: Path to plugin configuration file. See template_plugin.json for an example.
                         This parameter is optional.
            Example: --pluginConfig=template_plugin.json

    --precisionConfig=[path to precision config file]
            Description: Path to a precision configuration file for generating models with mixed
                         precision. For layers not included in the configuration file, builder mode determines the precision. For these layers, TensorRT may choose any precision for better performance. If 'output_types' is not provided for a layer, the data type of the output tensors will be set to the precision of the layer. For the layers with precision set to INT8, scaling factors of the input/output tensors should be provided. This file can also be used to set the scaling factors for each tensor by name. The values provided in this file will override the scaling factors specified in calibration file (if provided). See 'template_precision.json' for an example.
                         This parameter is optional.
            Example: --precisionConfig=template_precision.json

    --testFile=[path to binary file]
            Description: Name of a binary file for model input/output validation. This file should contain
                         flattened pairs of inputs and expected outputs in the same order as the TensorRT model expects. The file is assumed to hold 32 bit floats. The number of test pairs is automatically detected.
                         This parameter is optional.
            Example: Data with two inputs and two outputs would have a layout in the file as follows:
                     > \[input 1\]\[input 2\]\[output 1\]\[output 2\]\[input 1\]\[input 2\]\[output 1\]\[output 2\]...

    --useGraph
            Description: If specified, executes the optimized network by CUDA graph. It helps check if the optimized network
                         works with CUDA graph acceleration.
                         This parameter is optional.

    --workspaceSize=[int]
            Description: Max workspace size in megabytes. Limits the maximum size that any layer in the network
                         can use. If insufficient scratch is provided, TensorRT may not be able to find an implementation for a given layer.
                         This parameter is optional.

    --explicitBatch=[int]
            Description: Determines whether explicit batch should be enabled or not.
                         For TensorRT versions higher than or equal to 6.3, if an ONNX model is provided as
                         input, this flag will be automatically set to 1.
                         This parameter is optional.

@section dwx_tensorRT_tool_examples Examples

@subsection dwx_tensorRT_tool_examples_uff Optimizing UFF Models

    ./tensorRT_optimization --modelType=uff
                            --outputBlobs=bboxes,coverage
                            --uffFile=~/myNetwork.uff
                            --inputBlobs=data0,data1
                            --inputDims=3x480x960,1x1x10

@subsection dwx_tensorRT_tool_examples_caffe Optimizing Caffe Models

    ./tensorRT_optimization --modelType=caffe
                            --outputBlobs=bboxes,coverage
                            --prototxt=deploy.prototxt
                            --caffemodel=weights.caffemodel

@note The `--inputBlobs` and `--inputDims` parameters are ignored if you select the Caffe model type. <br>All the input blobs will be automatically marked as input.

@subsection dwx_tensorRT_tool_examples_onnx Optimizing ONNX Models

    ./tensorRT_optimization --modelType=onxx
                            --onnxFile=~/myNetwork.onnx

@note The `--inputBlobs`, `--inputDims`, and `--outBlobs` parameters are ignored if you select the ONNX model type.<br>All the input and output blobs will be automatically marked as input or output, respectively.
