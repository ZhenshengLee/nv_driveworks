<!-- Copyright (c) 2021-2022 NVIDIA CORPORATION.  All rights reserved. -->
<!-- markdownlint-disable MD041 -->

@page cgf_tools_nodedescriptor Node Descriptor Generator

The tool can be used to generate the content of the JSON node descriptor (.node.json) from implemented C++ nodes compiled into shared libraries.

Usage:

    nodedescriptor
      Shows this usage information
    nodedescriptor SHARED_LIBRARY
      List all node names registered in the shared library
    nodedescriptor SHARED_LIBRARY NODE_NAME
      Output JSON descriptor for the given node
    nodedescriptor SHARED_LIBRARY "*"
      Output JSON descriptors for all nodes separated by a record separator

The signature with passing an explicit node name can be used to integrate the tool into the build process to keep the .node.json files in sync with changes to the C++ nodes.
To run it for the dwCameraNode node for example:

`./nodedescriptor nvidia_computegraphframework_linux-amd64-ubuntu/lib/libdwframework_dwnodes.so.5.0 dw::framework::dwCameraNode`
