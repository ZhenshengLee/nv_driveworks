<!-- Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved. -->
<!-- markdownlint-disable MD041 -->

@page cgf_tools_validate JSON Descriptor Validation

A Python script check that the JSON files comply with their JSON schemas as well as additional constraints like cross references are satisfied.
Before running the script, please make sure the Python package `jsonschema` is installed.
The script can be used with the following syntax:

Usage:

    validate_cgfdescriptors.py [-h] [--ignore-order] [--ignore-indentation]
                               basepaths [basepaths ...]

If the basepath is a directory, the script crawls recursively for JSON files with known file extensions.
For each JSON file (either passed as an explicit argument or within a given directory) the script also checks recursively referenced files.
