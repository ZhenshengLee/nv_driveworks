#!/bin/sh

################################################################################
#
# Notice
# ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS"
# NVIDIA MAKES NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR
# OTHERWISE WITH RESPECT TO THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED
# WARRANTIES OF NONINFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR
# PURPOSE.
#
# NVIDIA CORPORATION & AFFILIATES assumes no responsibility for the consequences
# of use of such information or for any infringement of patents or other rights
# of third parties that may result from its use. No license is granted by
# implication or otherwise under any patent or patent rights of NVIDIA
# CORPORATION & AFFILIATES. No third party distribution is allowed unless
# expressly authorized by NVIDIA. Details are subject to change without notice.
# This code supersedes and replaces all information previously supplied. NVIDIA
# CORPORATION & AFFILIATES products are not authorized for use as critical
# components in life support devices or systems without express written approval
# of NVIDIA CORPORATION & AFFILIATES.
#
# SPDX-FileCopyrightText: Copyright (c) 2016-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this material and related documentation without an express
# license agreement from NVIDIA CORPORATION or its affiliates is strictly
# prohibited.
#
################################################################################

PATH=/usr/bin:/bin:/usr/sbin:/sbin;
SAMPLES_SRC=${SAMPLES_SRC:-"/usr/local/driveworks-*/samples/src/ssm"};

if [ -z "$1" ]; then
    echo "Error: target directory missing";
    echo "Usage: $(basename $0) <target directory>";
    echo "       Will copy ssm release directory to <target directory>";
    exit 1;
fi

SAMPLES_DIR=$1;
echo $SAMPLES_DIR | grep -q "/$"
if [ $? -ne 0 ]; then
    SAMPLES_DIR=$SAMPLES_DIR"/";
fi
SAMPLES_DIR=$SAMPLES_DIR"ssm";

mkdir -p "$SAMPLES_DIR" >/dev/null 2>&1;

if [ -d "$SAMPLES_DIR" -a -w "$SAMPLES_DIR" ]; then
    echo "Copying ssm release direcory to $SAMPLES_DIR now...";
    cp -R $SAMPLES_SRC/* "$SAMPLES_DIR";
    echo "Finished copying samples.";
else
    echo "Do not have permissions to write to $SAMPLES_DIR";
    exit 1;
fi

exit 0;
