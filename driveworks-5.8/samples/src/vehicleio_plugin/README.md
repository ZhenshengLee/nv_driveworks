# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_vehicleio_plugin_sample VehicleIO Plugin Sample
@tableofcontents

@section dwx_vehicleio_plugin_description Description

The VehicleIO Plugin sample demonstrates how to write a custom VehicleIO backend via a
dynamically loaded library.

For simplicity, this sample only encodes and parses steering commands as
specified in a DBC file.

@section dwx_vehicleio_plugin_running Running the Sample

In order to instruct the VehicleIO module to use the plug-in backend, a rig file
should specify a VehicleIO node with `type` parameter set to `"custom"` and
`custom-lib` parameter pointing to the dynamically loaded library, e.g.

```
        "vehicleio": [
            {
                "type": "custom",
                "parent-sensor": "can:vehicle:custom",
                "custom-lib": "libsample_vehicleio_plugin.so"
            }
        ],
```

A sample rig file already provided in `<dw>/data/samples/vehicleio/rig-plugin.json`.

Once the rig file is ready, you can run sample_vehicleio:

    ./sample_vehicleio --rig=path/to/dw/data/samples/vehicleio/rig-plugin.json

@section dwx_vehicleio_plugin_more Additional information

For more information, see @ref vehicleio_mainsection.
