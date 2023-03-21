<!-- Copyright (c) 2021-2022 NVIDIA CORPORATION.  All rights reserved. -->
<!-- markdownlint-disable MD041 -->

@page cgf_tools_nodestub Node JSON to C++ stubs

A Python script in the release provides an example of JSON to C++ template function.
Before running the script, please make sure the Python package `jinja2` is installed.
The script can be used with the following syntax:

Usage:

    nodestub.py [-h] [--output-path OUTPUT_PATH]
                     [--overwrite-existing-files]
                     NODE_JSON_FILE BASE_CLASS

To run it with RenderingCGFDemoNode.node.json for example:

`./nodestub.py <path_to_release>/src/cgf/nodes/RenderingCGFDemoNode.node.json dw::framework::ExceptionSafeProcessNode`

Four files will be generated:

-   RenderingCGFDemoNode.hpp,
-   RenderingCGFDemoNode.cpp,
-   RenderingCGFDemoNodeImpl.hpp, and
-   RenderingCGFDemoNodeImpl.cpp.

Please note that when the script cannot find the proper header file for the data type defined in DATATYPE_HEADER in the script, the script will notify in the command prompt:

> Unknown data type 'dwFeatureHistoryArray', additional \#include directives might be needed

In this case, DATATYPE_HEADER in the script will need to be updated to include the correct header file.
