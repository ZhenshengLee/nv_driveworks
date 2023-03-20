# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_hello_world_sample Hello World Sample
@tableofcontents

@section dwx_hello_world_description Description

The Hello World sample application shows how to initialize the NVIDIA<sup>&reg;</sup> DriveWorks SDK
context and access GPU properties. This sample application prints the DriveWorks version and
GPU properties.

@section dwx_hello_world_running Running the Sample

The command line for the sample application is:

    ./sample_hello_world

@section dwx_hello_world_output Output

The sample application prints the following information on console:

    ./sample_hello_world
    *************************************************
    Welcome to Driveworks SDK
    [00-00-0000 00:00:00] Platform: Detected P3710 DDPO Platform
    [00-00-0000 00:00:00] TimeSource: monotonic epoch time offset is 0000000000000000
    [00-00-0000 00:00:00] TimeSource: Could not detect valid PTP time source at nvpps. Fallback to eth0
    [00-00-0000 00:00:00] TimeSource Eth: PTP ioctl returned error. Synchronized time will not be available from this timesource.
    [00-00-0000 00:00:00] TimeSource: Could not detect valid PTP time source at 'eth0'. Fallback to CLOCK_MONOTONIC.
    [00-00-0000 00:00:00] Platform: number of GPU devices detected 1
    [00-00-0000 00:00:00] Platform: currently selected GPU device integrated ID 0
    [00-00-0000 00:00:00] Platform: currently selected GPU device integrated ID 0
    [00-00-0000 00:00:00] Platform: currently selected GPU device integrated ID 0
    [00-00-0000 00:00:00] Context::mountResourceCandidateDataPath resource FAILED to mount from '/usr/local/driveworks/bin/data/': VirtualFileSystem: Failed to
    mount '/usr/local/driveworks/bin/data/[.pak]'
    [00-00-0000 00:00:00] Context::findDataRootInPathWalk data/DATA_ROOT found at: /usr/local/driveworks/bin/../data
    [00-00-0000 00:00:00] Context::mountResourceCandidateDataPath resource FAILED to mount from '/usr/local/driveworks/bin/../data/': VirtualFileSystem: Failed
    to mount '/usr/local/driveworks/bin/../data/[.pak]'
    [00-00-0000 00:00:00] Context::findDataRootInPathWalk data/DATA_ROOT found at: /usr/local/driveworks/data
    [00-00-0000 00:00:00] Context::mountResourceCandidateDataPath resource FAILED to mount from '/usr/local/driveworks/data/': VirtualFileSystem: Failed to mount '/usr/local/driveworks/data/[.pak]'
    [00-00-0000 00:00:00] SDK: No resources(.pak) mounted, some modules will not function properly
    [00-00-0000 00:00:00] SDK: Create NvMediaDevice
    [00-00-0000 00:00:00] SDK: Create NvMedia2D
    [00-00-0000 00:00:00] egl::Display: found 1 EGL devices
    [00-00-0000 00:00:00] egl::Display: use drm device: drm-nvdc
    [00-00-0000 00:00:00] TimeSource: monotonic epoch time offset is 0000000000000000
    [00-00-0000 00:00:00] TimeSource: Could not detect valid PTP time source at nvpps. Fallback to eth0
    [00-00-0000 00:00:00] TimeSource Eth: PTP ioctl returned error. Synchronized time will not be available from this timesource.
    [00-00-0000 00:00:00] TimeSource: Could not detect valid PTP time source at 'eth0'. Fallback to CLOCK_MONOTONIC.
    [00-00-0000 00:00:00] Initialize DriveWorks SDK v5.0.0
    [00-00-0000 00:00:00] Release build with GNU 9.3.0 from no-gitversion-build against Drive PDK v6.0.0.0
    Context of Driveworks SDK successfully initialized.
    Version: 5.0.0
    GPU devices detected: 1
    [00-00-0000 00:00:00] Platform: currently selected GPU device integrated ID 0
    ----------------------------------------------
    Device: 0, Graphics Device
    CUDA Driver Version / Runtime Version : 11.4 / 11.4
    CUDA Capability Major/Minor version number: 8.7
    Total amount of global memory in MBytes:29012.6
    Memory Clock rate Khz: 892000
    Memory Bus Width bits: 128
    L2 Cache Size: 4194304
    Maximum 1D Texture Dimension Size (x): 131072
    Maximum 2D Texture Dimension Size (x,y): 131072, 65536
    Maximum 3D Texture Dimension Size (x,y,z): 16384, 16384, 16384
    Maximum Layered 1D Texture Size, (x): 32768 num: 2048
    Maximum Layered 2D Texture Size, (x,y): 32768, 32768 num: 2048
    Total amount of constant memory bytes: 65536
    Total amount of shared memory per block bytes: 49152
    Total number of registers available per block: 65536
    Warp size: 32
    Maximum number of threads per multiprocessor: 1536
    Maximum number of threads per block: 1024
    Max dimension size of a thread block (x,y,z): 1024,1024,64
    Max dimension size of a grid size (x,y,z): 2147483647,65535,65535
    Maximum memory pitch bytes: 2147483647
    Texture alignment bytes: 512
    Concurrent copy and kernel execution: Yes, copy engines num: 2
    Run time limit on kernels: No
    Integrated GPU sharing Host Memory: Yes
    Support host page-locked memory mapping: Yes
    Alignment requirement for Surfaces: Yes
    Device has ECC support: Disabled
    Device supports Unified Addressing (UVA): Yes
    Device PCI Domain ID: 0, Device PCI Bus ID: 0, Device PCI location ID: 0
    Compute Mode: Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)
    Concurrent kernels: 1
    Concurrent memory: 0

    [00-00-0000 00:00:00] Releasing Driveworks SDK Context
    [00-00-0000 00:00:00] SDK: Release NvMediaDevice
    [00-00-0000 00:00:00] SDK: Release NvMedia2D
    Happy autonomous driving!

@section dwx_hello_world_more Additional information

For more details see:
- @ref dwx_hello_world
- @ref core_mainsection
