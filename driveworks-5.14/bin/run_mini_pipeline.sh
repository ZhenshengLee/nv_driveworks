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
# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

RUNTIME=`date +%Y%m%d_%H%M%S`
echo ${RUNTIME}

# Unify script interpreter
# Bash, Ksh and sh has some subtle syntax differences.
# To avoid confusion we force using only one Interpreter on each OS:
# QNX: KSH88
# Linux: BASH
#
# The following code changes the interpreter. /bin/sh is just a pilot interpreter
INTERP=`ps -p $$ -o args='' | cut -f1 -d' '`
OS=`uname`
ARCH=`uname -i`

if [ "$OS" = "QNX" ] && [ "$INTERP" != "/proc/boot/ksh" ]; then
    /proc/boot/ksh ${0} "$@"
    exit $?
elif [ "$OS" = "Linux" ] && [ "$INTERP" != "bash" ]; then
    /usr/bin/env bash ${0} "$@"
    exit $?
fi

__parseArgs() {
    while (( $# ))
    do
        case $1 in
        -p|--p|--product)
            PRODUCT="$2"
            shift
            ;;
        -s|--s|--schedule)
            SCHEDULE="$2"
            shift
            ;;
        --gdb)
            GDB_DEBUG=1
            ;;
        --app_parameter)
            APP_PARAMETER="$2"
            ;;
        esac
        shift
    done
}

PRODUCT=MiniPipeline
SCHEDULE=""
GDB_DEBUG=0
APP_PARAMETER=""

ALL_ARGS=$* # make a copy of the shell parameters
__parseArgs ${ALL_ARGS}

# dazel will set BUILD_WORKSPACE_DIRECTORY env var when running with dazel run command
RUN_WITH_DAZEL=0
if [[ -n ${BUILD_WORKSPACE_DIRECTORY} ]] || [[ -n ${RUNFILES_DIR} ]]; then
    RUN_WITH_DAZEL=1
    echo "Running with dazel !";
    DW_TOP_PATH=$(pwd)
    SAMPLE_TOP_PATH=${DW_TOP_PATH}/samples/minipipeline
    CGF_SYS_PATH=${SAMPLE_TOP_PATH}/applications
else
    DW_TOP_PATH=$(pwd)
    if [[ -d $DW_TOP_PATH/run_mini_pipeline.runfiles ]]; then
        RUN_WITH_DAZEL=1
        echo "Running with deploy build !"
        DW_TOP_PATH=$(pwd)/run_mini_pipeline.runfiles/av
        SAMPLE_TOP_PATH=${DW_TOP_PATH}/samples/minipipeline
        CGF_SYS_PATH=${SAMPLE_TOP_PATH}/applications
    elif [[ $DW_TOP_PATH == *"run_mini_pipeline.runfiles/av" ]]; then
        # assume pwd is run_mini_pipeline.runfiles/av
        RUN_WITH_DAZEL=1
        echo "Running with deploy build !"
        DW_TOP_PATH=$(pwd)
        SAMPLE_TOP_PATH=${DW_TOP_PATH}/samples/minipipeline
        CGF_SYS_PATH=${SAMPLE_TOP_PATH}/applications
    else
        DW_TOP_PATH=/usr/local/driveworks
        SAMPLE_TOP_PATH=${DW_TOP_PATH}/apps/minipipeline
    fi
fi

# Make sure CGF package is installed
if [[ $RUN_WITH_DAZEL -ne 1 ]]; then
    dpkg -V driveworks-cgf
    if [[ $? -ne 0 ]]; then
        echo "WARNING: This sample would be functional only when CGF Debian Package(driveworks-cgf) is installed!"
        exit 1
    fi
fi

echo "DW_TOP_PATH=${DW_TOP_PATH}"
echo "SAMPLE_TOP_PATH=${SAMPLE_TOP_PATH}"
echo "CGF_SYS_PATH=${CGF_SYS_PATH}"

CMD=$(pwd)
DATA_PATH=${DW_TOP_PATH}/data/samples/minipipeline
SMP_LOG_PATH=${SAMPLE_TOP_PATH}/LogFolder
if [[ $RUN_WITH_DAZEL -eq 1 ]]; then
    CAR=H7_0_dataset
    DATA_PATH=${DATA_PATH}/${CAR}
    BIN_TOP_PATH=${DW_TOP_PATH}/apps/roadrunner-2.0/
else
    CAR=dataset
    DATA_PATH=${DATA_PATH}/${CAR}
    BIN_TOP_PATH=${DW_TOP_PATH}/bin/
fi

echo "SMP_LOG_PATH=${SMP_LOG_PATH}"
echo "DATA_PATH=${DATA_PATH}"

LD_LIBRARY_PATH_ORIGINAL=${LD_LIBRARY_PATH}

__tuneNetworkStack() {
    echo "|--> Tuning network stack"
    if [[ "$OS" == "Linux" ]]; then
        core_wmem_max=`sysctl -n net.core.wmem_max`
        core_rmem_max=`sysctl -n net.core.rmem_max`
        core_rmem_default=`sysctl -n net.core.rmem_default`
        ipv4_tcp_wmem=`sysctl -n net.ipv4.tcp_wmem`
        ipv4_tcp_rmem=`sysctl -n net.ipv4.tcp_rmem`
        if [[ $core_wmem_max != "65011712" || \
              $core_rmem_max != "65011712" || \
              $core_rmem_default != "16777216" || \
              $ipv4_tcp_wmem != "65011712	65011712	65011712" || \
              $ipv4_tcp_rmem != "65011712	65011712	65011712" ]]; then
            echo "!!! Require larger NetworkStack !!!"
            echo "The default value of rmem_max and wmem_max is about 128 KB in Linux, which is too small to"
            echo "to fit the current sample mini pipeline (causing high-latencies)."
            echo "The following setting is going to improve networking performance."
            echo "Please enlarge NetworkStack with following commands:"
            echo "-----------------------------------------------------------------------------------"
            echo "sudo sed -i '$ a net.core.wmem_max = 65011712' /etc/sysctl.conf"
            echo "sudo sed -i '$ a net.core.rmem_max = 65011712' /etc/sysctl.conf"
            echo "sudo sed -i '$ a net.core.rmem_default = 16777216' /etc/sysctl.conf"
            echo "sudo sed -i '$ a net.ipv4.tcp_wmem = 65011712 65011712 65011712' /etc/sysctl.conf"
            echo "sudo sed -i '$ a net.ipv4.tcp_rmem = 65011712 65011712 65011712' /etc/sysctl.conf"
            echo "sudo sysctl -p"
            echo "-----------------------------------------------------------------------------------"
            exit 254
        fi
    else
        kern_sbmax=`sysctl -n kern.sbmax`
        inet_tcp_recvspace=`sysctl -n net.inet.tcp.recvspace`
        inet_tcp_sendspace=`sysctl -n net.inet.tcp.sendspace`
        inet_tcp_recvbuf_max=`sysctl -n net.inet.tcp.recvbuf_max`
        inet_tcp_sendbuf_max=`sysctl -n net.inet.tcp.sendbuf_max`
        if [[ $kern_sbmax != "65011712" || \
              $inet_tcp_recvspace != "65011712" || \
              $inet_tcp_sendspace != "65011712" || \
              $inet_tcp_recvbuf_max != "65011712" || \
              $inet_tcp_sendbuf_max != "65011712" ]]; then
            echo "!!! Require larger NetworkStack !!!"
            echo "The default value of recvbuf_max and sendbuf_max is about 4.768 MB in QNX, which is too small to"
            echo "to fit the current sample mini pipeline (causing high-latencies)."
            echo "The following setting is going to improve networking performance."
            echo "Please enlarge NetworkStack with following commands:"
            echo "-----------------------------------------------------------------------------------"
            echo "sysctl -w kern.sbmax=65011712"
            echo "sysctl -w net.inet.tcp.recvspace=65011712"
            echo "sysctl -w net.inet.tcp.sendspace=65011712"
            echo "sysctl -w net.inet.tcp.recvbuf_max=65011712"
            echo "sysctl -w net.inet.tcp.sendbuf_max=65011712"
            echo "-----------------------------------------------------------------------------------"
            exit 254
        fi
    fi
}

__tuneMqueue() {
    if [[ "$OS" == "Linux" ]]; then
        echo "|--> Tuning message queue"
        local msg_max=`cat /proc/sys/fs/mqueue/msg_max`
        if [[ msg_max -lt 4096 ]]; then
            echo "!!! Require larger message queue !!!"
            echo "STM requires at least 4096 msgs"
            echo "-------------------------------------------------------------"
            echo "sudo sed -i '$ a fs.mqueue.msg_max = 4096' /etc/sysctl.conf"
            echo "sudo sysctl -p"
            echo "-------------------------------------------------------------"
            exit 252
        fi
    fi
}

__checkNvSCI() {
    if [[ "$OS" == "Linux" ]]; then
        if [[ ! -f /etc/nvsciipc.cfg ]]; then
            echo "!!! /etc/nvsciipc.cfg not exist. STM requires the configuration the file !!!"
            exit 253
        fi

        if [[ "$ARCH" == "x86_64" ]]; then
            grep stm /etc/nvsciipc.cfg >/dev/null 2>&1 || true
            if [[ $? != 0 ]]; then
                echo "!!! STM requires the configuration file of /etc/nvsciipc.cfg to be updated !!!"
                exit 253
            fi
        fi
    fi
}

__setEnv() {
    if [[ $RUN_WITH_DAZEL -eq 1 ]]; then
        export LD_LIBRARY_PATH=${SAMPLE_TOP_PATH}/nodes:$LD_LIBRARY_PATH
    fi
    local sandbox_lib_path=$(find ${DW_TOP_PATH} -name '_solib*')
    if [[ -n ${sandbox_lib_path} ]]; then
        extra_lib_path=${DW_TOP_PATH}/$(basename ${sandbox_lib_path})
        export LD_LIBRARY_PATH=${extra_lib_path}:${LD_LIBRARY_PATH}
    fi

    if [[ "${OS}" = "Linux" ]]; then
        # for multi-process mode socket connections
        # for stm
        __tuneMqueue
        __checkNvSCI
    fi
    __tuneNetworkStack

    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${DW_TOP_PATH}/lib
    echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
}

__prepDisplay() {
    local DEFAULT_DISPLAY=0.0
    if [[ "${OS}" = "Linux" ]]; then
        # Get the release number of the current operating system.
        local UBUNTU_RELEASE=`lsb_release -rs`
        if [[ ${UBUNTU_RELEASE} == "18.04" ]]; then
            # TODO : it's just a WAR for ubuntu 1804
            export XAUTHORITY=${XDG_RUNTIME_DIR}/gdm/Xauthority

            # The X server puts its socket in /tmp/.X11-unix:
            # We are using it to check if Xorg is connected to /tmp/.X11-unix/X1
            if [[ -e /tmp/.X11-unix/X1 ]]; then
                DEFAULT_DISPLAY=1.0
            fi
        fi
    fi

    # SSH session ?
    if [[ -n ${SSH_CONNECTION} ]]; then
        export DISPLAY=:${DEFAULT_DISPLAY}
    else
        # Do nothing if DISPLAY was set
        export DISPLAY=${DISPLAY-":${DEFAULT_DISPLAY}"}
    fi

    echo "Current DISPLAY is ${DISPLAY}"
}

__cleanup() {
    if [[ -n ${LAUNCHER_PID} ]]; then
        echo
        echo "|--> Killing launcher"
        kill -TERM ${LAUNCHER_PID} > /dev/null 2>&1
        wait ${LAUNCHER_PID}
        LAUNCHER_PID=""
        export LD_LIBRARY_PATH=${LD_LIBRARY_PATH_ORIGINAL}
    fi
}

trap "__cleanup; exit 1" INT TERM QUIT

__prepDisplay
__setEnv

if [[ ! -d ${SMP_LOG_PATH} ]];then
    if [[ $RUN_WITH_DAZEL -eq 1 ]]; then
        mkdir ${SMP_LOG_PATH}
    else
        mkdir -p ${SMP_LOG_PATH}
    fi
else
    echo "${SMP_LOG_PATH} dir exists"
fi

SPEC=""
SCHED=""
if [[ -f $PRODUCT ]]; then
    SPEC=$PRODUCT
    SCHED=$SCHEDULE
else
    # CGF_SYS_PATH will be set when running in dazel
    if [[ -n ${CGF_SYS_PATH} ]]; then
        SPEC=${CGF_SYS_PATH}/${PRODUCT}.app.json
    else
        SPEC=${SAMPLE_TOP_PATH}/applications/${PRODUCT}.app.json
    fi
    SCHEDS_TAB=${PRODUCT}__*.stm
    if ! ls ${SAMPLE_TOP_PATH}/${SCHEDS_TAB} 1> /dev/null 2>&1; then
    echo "Schedule file ${SAMPLE_TOP_PATH}/${SCHEDS_TAB} doesn't exist!"
        __cleanup
        exit 1
    fi
    SCHED=$(find ${SAMPLE_TOP_PATH} -name ${SCHEDS_TAB} | paste -sd,)
fi

if [[ ! -f $SPEC ]]; then
    echo "System file ${SPEC} doesn't exist!"
    __cleanup
    exit 1
fi

if [[ ${SCHED} == *[,]* ]]; then
    for f in ${SCHED//,/ }
    do
        if [[ ! -f $f ]]; then
            echo "Schedule file ${f} doesn't exist!"
           __cleanup
            exit 1
        fi
    done
elif [[ ! -f $SCHED ]]; then
    echo "System file ${SCHED} doesn't exist!"
    __cleanup
    exit 1
fi

ARGS="--binPath=${BIN_TOP_PATH}"
ARGS="$ARGS --spec=${SPEC}"
ARGS="$ARGS --logPath=${SMP_LOG_PATH}"
ARGS="$ARGS --path=${SAMPLE_TOP_PATH}"
ARGS="$ARGS --datapath=${DATA_PATH}"
ARGS="$ARGS --dwdatapath=${DW_TOP_PATH}/data"
ARGS="$ARGS --schedule=${SCHED}"
ARGS="$ARGS --start_timestamp=0"
ARGS="$ARGS --loglevel=DW_LOG_DEBUG"
ARGS="$ARGS --fullscreen=0"
ARGS="$ARGS --winSizeW=1920"
ARGS="$ARGS --winSizeH=1200"
ARGS="$ARGS --virtual=1"
ARGS="$ARGS --gdb_debug=${GDB_DEBUG}"
ARGS="$ARGS --app_parameter=${APP_PARAMETER}"

if [[ $RUN_WITH_DAZEL -eq 1 ]]; then
    ARGS="$ARGS --useLCM=0"
fi

ls -al ${SAMPLE_TOP_PATH}

set +e
echo "Running command: ${BIN_TOP_PATH}/launcher ${ARGS} > ${SMP_LOG_PATH}/launcher.log 2>&1"
${BIN_TOP_PATH}/launcher ${ARGS} > ${SMP_LOG_PATH}/launcher.log 2>&1 &
LAUNCHER_PID=$!
wait ${LAUNCHER_PID}
MINI_PIPELINE_STATUS=$?
set -e

# reset env
echo "Check if reset NetworkStack needed"
echo "Restore LD_LIBRARY_PATH to ${LD_LIBRARY_PATH_ORIGINAL}"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH_ORIGINAL}
echo "======================================================================="
echo launcher exit status: ${MINI_PIPELINE_STATUS}
echo
