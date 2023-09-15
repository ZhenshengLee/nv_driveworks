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
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
ARCH=`uname -p`

if [ "$OS" = "QNX" ] && [ "$INTERP" != "/proc/boot/ksh" ]; then
    /proc/boot/ksh ${0} "$@"
    exit $?
elif [ "$OS" = "Linux" ] && [ "$INTERP" != "bash" ]; then
    /usr/bin/env bash ${0} "$@"
    exit $?
fi

#====================================================================
# Tracing
#====================================================================
Red='\033[31m\033[01m'      # Red
Green='\033[32m\033[01m'    # Green
Yellow='\033[33m\033[01m'   # Yellow
Blue='\033[34m\033[01m'     # Blue
Cyan='\033[36m\033[01m'     # Cyan
# Reset
Color_Off='\033[0m'         # Text Reset

__log() {
    if [[ -f ${CONSOLE_LOG} ]]; then
        echo -e $* | tee -a ${CONSOLE_LOG}
    else
        echo -e $*
    fi
}

__error() {
    __log "${Red}$*${Color_Off}" >&2
}

__die() {
    SKIP_POST=1
    __error $*
    exit 255
}

__parseArgs() {
    while (( $# ))
    do
        case $1 in
        -c|--c|--car)
            CAR="$2"
            shift
            ;;
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
            shift
            ;;
        --loglevel)
            LOG_LEVEL="$2"
            shift
            ;;
        esac
        shift
    done
}

PRODUCT=CGFDemo
SCHEDULE=""
CAR=trafficlightturning-hyperion8
GDB_DEBUG=0
APP_PARAMETER=""
LOG_LEVEL="DW_LOG_ERROR"

ALL_ARGS=$* # make a copy of the shell parameters
__parseArgs ${ALL_ARGS}

# dazel will set BUILD_WORKSPACE_DIRECTORY env var when running with dazel run command
RUN_WITH_DAZEL=0
if [[ -n ${BUILD_WORKSPACE_DIRECTORY} ]] || [[ -n ${RUNFILES_DIR} ]]; then
    RUN_WITH_DAZEL=1
    echo "Running with dazel !";
fi
if [[ ${RUN_WITH_DAZEL} -eq 0 ]] && [[ $(pwd) == *"run_cgf_demo.runfiles"* ]]; then
    RUN_WITH_DAZEL=1
    echo "Running with deploy build !"
fi

if [[ $RUN_WITH_DAZEL -eq 1 ]]; then
    if [[ -n $RUNFILES_DIR ]]; then
        DW_TOP_PATH=$(cd $(dirname ${0})/../..; pwd)
        RR_TOP_PATH=$(cd $(dirname ${0})/../roadrunner-2.0; pwd)
    else
        DW_TOP_PATH=$(pwd)
        RR_TOP_PATH=${DW_TOP_PATH}/apps/$(cd ${DW_TOP_PATH}/apps; find . -maxdepth 1 -type d -regex ".*-2.0")
        CGF_SYS_PATH=${RR_TOP_PATH}/../cgf_demo
    fi
else
    # Hardcode this in real car
    DW_TOP_PATH=/usr/local/driveworks
    RR_TOP_PATH=${DW_TOP_PATH}/apps/roadrunner-2.0
    CGF_SYS_PATH=${DW_TOP_PATH}/apps/cgf_demo
fi

echo "DW_TOP_PATH=${DW_TOP_PATH}"
echo "RR_TOP_PATH=${RR_TOP_PATH}"
echo "CGF_SYS_PATH=${CGF_SYS_PATH}"

CMD=$(pwd)
if [[ $RUN_WITH_DAZEL -eq 1 ]]; then
    RR_GRAPHS_PATH=${RR_TOP_PATH}/../cgf_demo
else
    RR_GRAPHS_PATH=${DW_TOP_PATH}/apps/cgf_demo
fi

if [[ $RUN_WITH_DAZEL -eq 1 ]]; then
    RR_LOG_PATH=${RR_TOP_PATH}/LogFolder
    RR_RUN_CFG_PATH=${RR_TOP_PATH}/RunFolder
else
    # execute this script under somewhere you can write on real car.
    RR_LOG_PATH=${PWD}/LogFolder
    RR_RUN_CFG_PATH=${PWD}/RunFolder
fi

echo "RR_LOG_PATH=${RR_LOG_PATH}"
echo "RR_RUN_CFG_PATH=${RR_RUN_CFG_PATH}"
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
            echo "to fit the current RR 2.0 (causing high-latencies)."
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
        if [[ $kern_sbmax != "650117120" || \
              $inet_tcp_recvspace != "65011712" || \
              $inet_tcp_sendspace != "65011712" || \
              $inet_tcp_recvbuf_max != "65011712" || \
              $inet_tcp_sendbuf_max != "65011712" ]]; then
            echo "!!! Require larger NetworkStack !!!"
            echo "The default value of recvbuf_max and sendbuf_max is about 4.768 MB in QNX, which is too small to"
            echo "to fit the current RR 2.0 (causing high-latencies)."
            echo "The following setting is going to improve networking performance."
            echo "Please enlarge NetworkStack with following commands:"
            echo "-----------------------------------------------------------------------------------"
            echo "sysctl -w kern.sbmax=650117120"
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
    local msg_max=$(cat /proc/sys/fs/mqueue/msg_max)
    __log " fs.mqueue.msg_max = ${msg_max}"

        # Print the mqueue size if running inside docker as incorrect size can cause
        # failures in remote exec
    if [[ msg_max -lt 4096 ]]; then
        if [[ "$ARCH" = "aarch64" ]]; then
            __trace "|--> Tuning message queue"
            sudo sed -i '$ a fs.mqueue.msg_max = 4096' /etc/sysctl.conf
            sudo sysctl -p
        else
            if [[ ${RUN_INSIDE_DOCKER} -eq 1 ]]; then
                cat <<EOF
!!! ERROR !!!
STM requires at least 4096 msgs
You should set it by adding the following arguments:
-------------------------------------------------------------
--sysctl fs.mqueue.msg_max=4096
-------------------------------------------------------------
EOF
            else
                cat <<EOF
!!! ERROR !!!
STM requires at least 4096 msgs
You could set it by using the following commands:
-------------------------------------------------------------
sudo sed -i '$ a fs.mqueue.msg_max = 4096' /etc/sysctl.conf
sudo sysctl -p
-------------------------------------------------------------
EOF
            fi
            exit 255
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
        export LD_LIBRARY_PATH=${RR_TOP_PATH}/nodes:${RR_TOP_PATH}/nodes_non_product:$LD_LIBRARY_PATH
    else
        export LD_LIBRARY_PATH=${RR_TOP_PATH}:${DW_TOP_PATH}/src/cgf/nodes:${DW_TOP_PATH}/bin:$LD_LIBRARY_PATH
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

if [[ ! -d ${RR_LOG_PATH} ]];then
  mkdir ${RR_LOG_PATH}
else
  echo "${RR_LOG_PATH} dir exists"
fi

if [[ ! -d ${RR_RUN_CFG_PATH} ]];then
  mkdir ${RR_RUN_CFG_PATH}
else
  echo "${RR_RUN_CFG_PATH} dir exists"
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
        SPEC=${RR_GRAPHS_PATH}/${PRODUCT}.app.json
    fi
    SCHEDS_TAB=${PRODUCT}__*.stm
    if ! ls ${RR_GRAPHS_PATH}/${SCHEDS_TAB} 1> /dev/null 2>&1; then
    echo "Schedule file ${RR_GRAPHS_PATH}/${SCHEDS_TAB} doesn't exist!"
        __cleanup
        exit 1
    fi
    if [ "$OS" == "QNX" ]; then
        SCHED=$(find ${RR_GRAPHS_PATH} -name ${SCHEDS_TAB} | paste -s -d "," - -)
    else
        SCHED=$(find ${RR_GRAPHS_PATH} -name ${SCHEDS_TAB} | paste -sd,)
    fi
fi

if [[ ! -f $SPEC ]]; then
    echo "System file ${SPEC} doesn't exist!"
    __cleanup
    exit 1
fi

if [[ ${SCHED} == *[,]* ]]; then
    FILE=$(echo $SCHED | tr "," "\n")
    for f in ${FILE}
    do
        if [[ ! -f $f ]]; then
            echo "Schedule file ${f} doesn't exist!"
           __cleanup
            exit 1
        fi
    done
elif [[ ! -f $SCHED ]]; then
    echo "Schedule file ${SCHED} doesn't exist!"
    __cleanup
    exit 1
fi

if [[ -f ${RR_TOP_PATH}/config/car/rig-${CAR}.json ]]; then
    # Solid car
    DATA_SOURCE="SENSOR"

    RR_RIG_NAME=rig.json
    RR_RIG_FILE=${RR_TOP_PATH}/config/car/rig-${CAR}.json
    cp $RR_RIG_FILE ${RR_RUN_CFG_PATH}/${RR_RIG_NAME}
    chmod ug+w ${RR_RUN_CFG_PATH}/${RR_RIG_NAME}

    # Update DBC file
    local dbc_file=$(grep "\"dbc-file\":" ${RR_RIG_FILE} | sed 's/^.*dbc-file":/"dbc-file":/' | sed 's/,.*//g' | sed 's/[":]//g' | awk '{print $2}')
    if [[ -n ${dbc_file} ]]; then
        __trace "|--> Found DBC file(${dbc_file})"
        local dbc_path=$(cd $(dirname ${dbc_file}); pwd)
        # This is a WAR because "dbc-file" will be overridden
        dbc_file=$(basename ${dbc_file})
        if [[ -f ${RR_TOP_PATH}/config/car/${dbc_file} ]]; then
            ln -sf  ${RR_TOP_PATH}/config/car/${dbc_file} ${RR_RUN_CFG_PATH}/${dbc_file}
        elif  [[ -f ${dbc_path}/${dbc_file} ]]; then
            ln -sf  ${dbc_path}/${dbc_file} ${RR_RUN_CFG_PATH}/${dbc_file}
        else
            __die "Can NOT find dbc file ${RR_TOP_PATH}/config/car/${dbc_file} or ${dbc_path}/${dbc_file} !!!"
        fi
    fi
else
    # Virtual car
    DATA_SOURCE="RECORDINGS"
    if [[ -d ${DW_TOP_PATH}/data/apps/roadrunner/${CAR} ]]; then
        RR_DATASET_PATH=${DW_TOP_PATH}/data/apps/roadrunner/${CAR}
    elif [[ -d ${DW_TOP_PATH}/data/samples/cgf/${CAR} ]]; then
        RR_DATASET_PATH=${DW_TOP_PATH}/data/samples/cgf/${CAR}
    else
        __die "Can NOT find ${DW_TOP_PATH}/data/apps/roadrunner/${CAR} or ${DW_TOP_PATH}/data/samples/cgf/${CAR} !!!"
    fi
fi

if [[ -f ${RR_TOP_PATH}/launcher ]]; then
    BIN_PATH=${RR_TOP_PATH}
elif [[ -f ${DW_TOP_PATH}/bin/launcher ]]; then
    BIN_PATH=${DW_TOP_PATH}/bin
else
    __die "Can NOT find ${RR_TOP_PATH}/launcher or ${DW_TOP_PATH}/bin/launcher !!!"
fi

ARGS="--binPath=${BIN_PATH}"
ARGS="$ARGS --spec=${SPEC}"
ARGS="$ARGS --logPath=${RR_LOG_PATH}"
ARGS="$ARGS --dwdatapath=${DW_TOP_PATH}/data"
ARGS="$ARGS --schedule=${SCHED}"
ARGS="$ARGS --start_timestamp=0"
ARGS="$ARGS --loglevel=${LOG_LEVEL}"
ARGS="$ARGS --fullscreen=1"
ARGS="$ARGS --winSizeW=1280"
ARGS="$ARGS --winSizeH=800"
ARGS="$ARGS --disableStmControlLogger=1"
ARGS="$ARGS --gdb_debug=${GDB_DEBUG}"
ARGS="$ARGS --app_parameter=${APP_PARAMETER}"
ARGS="$ARGS --useLCM=0"

if [[ ${DATA_SOURCE} = "SENSOR" ]]; then
    NVM_PATH="/storage/driveworks/data/vehicle_cfg/rig"
    RR_CALIBRATION_OVERLAY="${NVM_PATH}/${CAR}_dynamic.calibration.json"
    ARGS="$ARGS --calibrationOverlayFile=${RR_CALIBRATION_OVERLAY}"
    ARGS="$ARGS --datapath=${RR_RUN_CFG_PATH}"
    ARGS="$ARGS --rig=${RR_RIG_NAME}"
    ARGS="$ARGS --path=${RR_RUN_CFG_PATH}"
    ARGS="$ARGS --virtual=0"
else
    ARGS="$ARGS --datapath=${RR_DATASET_PATH}"
    ARGS="$ARGS --mapPath=maps/sample/sanjose_loop"
    ARGS="$ARGS --path=${RR_TOP_PATH}"
    ARGS="$ARGS --virtual=1"
fi

ls -al ${RR_TOP_PATH}

set +e
echo "Running command: ${BIN_PATH}/launcher ${ARGS} > ${RR_LOG_PATH}/launcher.log 2>&1"
${BIN_PATH}/launcher ${ARGS} > ${RR_LOG_PATH}/launcher.log 2>&1 &
LAUNCHER_PID=$!
wait ${LAUNCHER_PID}
RR_STATUS=$?
set -e

# reset env
echo "Check if reset NetworkStack needed"
echo "Restore LD_LIBRARY_PATH to ${LD_LIBRARY_PATH_ORIGINAL}"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH_ORIGINAL}
echo "======================================================================="
echo launcher exit status: ${RR_STATUS}
echo
