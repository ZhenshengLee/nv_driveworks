# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_sipl_query_tool SIPL Query Tool
@tableofcontents

@section dwx_sipl_query_tool_description Description

The NVIDIA<sup>&reg;</sup> DriveWorks SIPL Query Tool displays configuration settings for the known camera devices in the SIPL setup.

It displays the configured EEPROMs, sensors, cameras, serializers, and deserializers.
It also displays the platform configurations describing <br>
how the cameras are connected with the NVIDIA DRIVE<sup>TM</sup> platform.

This information is used to understand the devices and setups, and create new devBlock parameters for custom camera setups.

@section dwx_sipl_query_tool_prereqs Prerequisites

This tool is available on NVIDIA DRIVE<sup>&trade;</sup> OS Linux.

@section dwx_sipl_query_tool_usage Running the Tool

Run the tool by executing:

    ./sipl_query --detail=[list|full|tree]
                 --platform=[platform name]
                 --camera=[camera name]
                 --sensor=[sensor name]
                 --eeprom=[eeprom name]
                 --serializer=[serializer name]
                 --deserializer=[deserializer name]

@subsection dwx_sipl_query_tool_params Parameters

    --detail=[list|full|tree]
            Description: How much information should be displayed:
                         'list' = Just the name of the object.
                         'full' = All information about the object.
                         'tree' = All information about the object and all information about the children of each object.

    --platform=[platform name]
            Desciption: Only show platform with the name PLATFORM or \"all\" if blank.

    --camera=[camera name]
            Desciption: Only show camera with the name or \"all\" if blank.

    --sensor=[sensor name]
            Desciption: Only show sensor with the name SENSOR or \"all\" if blank.

    --eeprom=[eeprom name]
            Desciption: Only show EEPROM with name EEPROM or \"all\" if blank.

    --serializer=[serializer name]
            Desciption: Only show serializer with name SERIALIZER or \"all\" if blank.

    --deserializer=[deserializer name]
            Desciption: Only show deserializer with name DESERIALIZER or \"all\" if blank.

@section dwx_sipl_query_tool_examples Examples

@subsection dwx_sipl_query_tool_listingknownsetups To list all known platform setups

    ./sipl_query --platform

@subsection dwx_sipl_query_tool_showingdetails To show the full details for the SF3324_DPHY_x2 platform

    ./sipl_query --detail=tree
                 --platform=SF3324_DPHY_x2

@section dwx_sipl_query_tool_output Output

~~~~
$./sipl_query
NvSIPL library version: 0.0.0
NvSIPL header version: 0.0.0

Platform Detection:
CNvMPlatform: board string is e3550_t194a
CNvMPlatform: platform is found with key = e3550_t194a

Platforms:
Platform config: SF3324_DPHY_x2
Platform config: SF3324_DPHY_x2_slave
Platform config: SF3324_DPHY_x2_TPG
Platform config: SF3324_file_mode
Platform config: SF3325_DPHY_x2
Platform config: SF3325_DPHY_x2_slave
Platform config: SF3325_DPHY_x2_TPG
Platform config: SF3325_file_mode
Platform config: AR0144P_DPHY_x2
Platform config: AR0144P_DPHY_x2_slave
Platform config: AR0144P_DPHY_x2_TPG
Platform config: AR0144P_file_mode
Platform config: CONSTELLATION_2MP_DPHY_x2
Platform config: CONSTELLATION_8MP_DPHY_x4
Platform config: CONSTELLATION_2MP_DPHY_x2_SLAVE
Platform config: CONSTELLATION_8MP_DPHY_x4_SLAVE
Platform config: CONSTELLATION_2MP_TPG_DPHY_x2

Cameras:
Name: SF3324
Name: SF3325
Name: AR0144P
Name: Constellation

Sensors:
Name: AR0231
Name: AR0144
Name: Constellation

EEPROMS:
Name: N24C64

Serializers:
Serializer name: MAX96705
Serializer name: MAX96759

Serializers:
Deserializer name: MAX96712
~~~~
