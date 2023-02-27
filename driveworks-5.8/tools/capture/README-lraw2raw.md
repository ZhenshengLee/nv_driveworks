# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_tools_lraw2raw LRAW to RAW Conversion Tool
@tableofcontents

This tool is available on the x86 Host System, NVIDIA DRIVE<sup>&trade;</sup> OS Linux, and NVIDIA DRIVE<sup>&trade;</sup> OS QNX.

The NVIDIA<sup>&reg;</sup> DriveWorks LRAW to RAW conversion tool decompresses
the LRAW input file and stores the result to an RCCB file.

LRAW stands for Lossless Encode of RAW RCCB data. It uses H264 Lossless Hi
Profile (244) for compression of RAW data. This format can be decoded to the
exact original, pixels of RCCB data, without any loss due to compression.

This tool decompresses the provided LRAW input file to generate the original
RAW RCCB file. The provided LRAW file can be in V1 or V2 format. LRAW encoding
differs, depending on the platform:
- NVIDIA DRIVE<sup>&trade;</sup> PX 2 (T186) encodes to the V1 format
- NVIDIA DRIVE<sup>&trade;</sup> AGX Developer Kit (T194) encodes to the V2 format.

The tool operates on pure raw data and doesn't need special support for any type of recording.

V2 offers these features over V1:
- Keeps the top and bottom lines of metadata uncompressed.
- Supports  MSB-aligned input for the new sensors.
- Supports compression of 16-bit data for new sensors.
- Supports preview encoding. Specifically, 1 frame in 30 is stored as a preview
  that the camera provides. The preview frame is encoded in lossy mode and
  embedded in the lraw file.

@section dwx_tools_lraw2raw_arguments Input Arguments

The lraw2raw tool supports the following arguments:

    --inputfile=<input.lraw file>       The fullpath of the input lraw file in LRAW V2 or V1 format.
    --outputfile=<outputfile.raw>       The fullpath of the output Raw decompressed RCCB file.
    --usePinnedMemory=1/0               The option to enable or disable use of PinnedMemory
                                        for faster CudaMemcpy.

@section dwx_tools_lraw2raw_running Running the Tool
The usage is shown below.

    sudo LD_LIBRARY_PATH=<path to CUDA Libraries> ./lraw2raw --inputfile=<input.lraw file> --outputfile=<outputfile.raw> --usePinnedMemory=1/0
