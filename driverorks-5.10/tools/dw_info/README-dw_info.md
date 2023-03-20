# SPDX-FileCopyrightText: Copyright (c) 2020-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_info_tool DriveWorks Info Tool

This is a simple tool to output Driveworks information, such as the current version,
as an easily parsable JSON format.

# Usage

Run the tool by executing:

    ./dw_info

and you will get an output similiar to this:

```
{
    "version":
    {
        "major": 3,
        "minor": 0,
        "patch": 1,
        "hash": "ffffffffffffffffffffffffffffffffffffffff",
        "extra": ""
    }
}
```
