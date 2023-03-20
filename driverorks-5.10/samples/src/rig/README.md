# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_rig_sample Rig Configuration Sample
@tableofcontents

@section dwx_rig_description Description

The Rig Configuration sample demonstrates how to read the rig configuration from the
XML file produced by the NVIDIA<sup>&reg;</sup> DriveWorks Rig Configuration Tool.

@section dwx_rig_config_running Running the Sample

The Rig Configuration sample, sample_rig, accepts the path to a fig configuration file. If it is not specified, it will use a default one.

    ./sample_rig --rigconfig=[path/to/rig/file]
                 --outputrigconfig=[path/to/otuput/rig/file]

where

    --rigconfig=[path/to/rig/file]
        Points to the rig file.
        Default value: path/to/data/samples/sfm/triangulation/rig.json

    --outputrigconfig=[path/to/otuput/rig/file]
        Output rig file. This parameter is only used to demonstrate the
        deserialization API. The output rig file is a copy of the input one.
        Default value: none

@section dwx_rig_config_output Output

The Rig Configuration sample prints the content of the rig file to the console:

    Vehicle Information:
     width:       1.874000
     height:      1.455000
     length:      4.915000
    wheelbase:    2.912000
    Sensor Count: 5
    Sensor 0:
     Name:        can
     Protocol:    can.virtual
    Sensor 1:
     Name:        SVIEW_FR
     Protocol:    camera.virtual
    Sensor 2:
     Name:        SVIEW_RE
     Protocol:    camera.virtual
    Sensor 3:
     Name:        SVIEW_LE
     Protocol:    camera.virtual
    Sensor 4:
     Name:        SVIEW_RI
     Protocol:    camera.virtual

@section dwx_rig_config_more Additional information

For more details see @ref rig_mainsection.
