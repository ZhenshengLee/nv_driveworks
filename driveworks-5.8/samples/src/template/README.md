# SPDX-FileCopyrightText: Copyright (c) 2017-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_sample_template Template for a sample
@tableofcontents

@section dwx_sample_template_description Description

This is a template sample showing how to write new samples.

@section dwx_sample_template_running Running the Sample

The occupancy grid sample, sample_template, accepts the following optional parameters.

    ./sample_template --optionA=[stuff]
                      --optionB=[stuff]

where

    --optionA=[stuff]
        Description of the parameter
        Default value:

    --optionB=[stuff]
        Description of the parameter
        Default value:

Short description of any runtime commands accepted by the sample.

@subsection dwx_sample_template_examples Examples

#### Runnign sample_template in offscreen mode

    ./sample_template --offscreen=1

#### Runnign sample_template with custom parameters

    ./sample_template --optionA=A --optionB=B

@section dwx_sample_template_output Output

Short description of the sample's output.

![Screenshot of application](sample_mytemplate.png)

@section dwx_sample_template_more Additional Information

For more details see \<add reference to the the modules used in this sample\>
