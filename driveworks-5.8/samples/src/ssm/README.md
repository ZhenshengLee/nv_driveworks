# SPDX-FileCopyrightText: Copyright (c) 2016-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

# SSM Demo

## Directory structure

 - The ssm directory is layed out as follows:
    ```bash

        ssm
        │
        └───setup_demo.sh
        │
        └───includes
        │    └───ssm
        │       │  #SSM library headers
        │
        └───lib
        │    │  libssm_release.so
        │
        └───samples
        │    │
        │    │  build.sh
        │    │  build_clean.sh
        │    └───demo1
        │    └───...
        │    └───demo6
        └───utils
              |  parser.py
    ```

## Building the demos

 - Copy the ssm samples directory to your home directory

    ```bash
        /usr/local/driveworks/samples/src/ssm/setup_demos.sh ${HOME}
    ```

 - Use the build.sh script and make files to build the demo applications

    ```bash
        cd ${HOME}/ssm/samples

        # Run the build script to build all the demo applications and to add the ssm library to LD_LIBRARY_PATH

        ./build.sh

        #To run parser after modifying ssm json files

        cd ${HOME}/ssm/samples/demo1
        make parser

        #To build a specific demo
        cd ${HOME}/ssm/samples/demo1
        make parser
        make

    ```

## Running the demos

 - To run the demos:

    ```bash
        cd ${HOME}/ssm/samples/demo1
        ./bin/demo1

    ```
