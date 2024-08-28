#!/bin/sh

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

# zs:
PROJECT=cgf_custom_nodes
TEAM=example
PRODUCT=DWCGF
APP_TYPE=Helloworld
SCHEDULE=""
CAR=trafficlightturning-hyperion8
GDB_DEBUG=0
APP_PARAMETER=""
LOG_LEVEL="DW_LOG_VERBOSE"

ALL_ARGS=$* # make a copy of the shell parameters
__parseArgs ${ALL_ARGS}

DW_TOP_PATH=$(cd $(dirname ${0})/../../../; pwd)
RR_TOP_PATH=$(cd /usr/local/driveworks/bin; pwd)

echo "DW_TOP_PATH=${DW_TOP_PATH}"
echo "RR_TOP_PATH=${RR_TOP_PATH}"
echo "CGF_SYS_PATH=${CGF_SYS_PATH}"

CMD=$(pwd)
RR_GRAPHS_PATH=${DW_TOP_PATH}/graphs/${PROJECT}

DATA_PATH=${DW_TOP_PATH}/data/${PROJECT}
RR_LOG_PATH=${CMD}/LogFolder/${PROJECT}/${APP_TYPE}
SCHEDULE_MANAGER_PATH=${DW_TOP_PATH}/bin/xplatform_schedule_manager
XPLATFORM_SSM_PATH=${DW_TOP_PATH}/bin/xplatform_ssm
XPLATFORM_COMMON_PATH=${DW_TOP_PATH}/bin/common_cgf_channel

RR_RUN_CFG_PATH=${CMD}/RunFolder/${PROJECT}/${APP_TYPE}
TCP_NODELAY_PRELOAD_SO=${DW_TOP_PATH}/bin/${PROJECT}/libnodelay.so

echo "RR_LOG_PATH=${RR_LOG_PATH}"
echo "RR_RUN_CFG_PATH=${RR_RUN_CFG_PATH}"
echo "DATA_PATH=${DATA_PATH}"

LD_LIBRARY_PATH_ORIGINAL=${LD_LIBRARY_PATH}

__releaseNvSCIIPC() {
    if [[ "$OS" == "Linux" ]]; then
        while read line
        do
            needLine=`echo $line | grep INTER_PROCESS | grep stm_`
            [ ! "$needLine" ] && continue
            channel1=`echo $line | awk -F " " '{print $2}'`
            channel2=`echo $line | awk -F " " '{print $3}'`
            [ "$channel1" ] && ${DW_TOP_PATH}/bin/$PROJECT/nvsciipc_reset -c $channel1 >/dev/null 2>&1
            [ "$channel2" ] && ${DW_TOP_PATH}/bin/$PROJECT/nvsciipc_reset -c $channel2 >/dev/null 2>&1
        done < /etc/nvsciipc.cfg
    fi
}

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

    if [ -f "${TCP_NODELAY_PRELOAD_SO}" ]; then
        export LD_PRELOAD=${TCP_NODELAY_PRELOAD_SO}
        echo "export LD_PRELOAD=${TCP_NODELAY_PRELOAD_SO}"
    fi
    export LD_LIBRARY_PATH=${RR_TOP_PATH}/lib:$LD_LIBRARY_PATH

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
        __releaseNvSCIIPC
    fi
    __tuneNetworkStack

    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${DW_TOP_PATH}/lib

    # source ros2 env
    # source /opt/ros/humble/setup.bash

    # all folders
    # for GW_LIB_FOLDER in `ls -d ${DW_TOP_PATH}/lib/*/`
    #     do
    #         export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${GW_LIB_FOLDER}
    #     done
    # for 3rdparty
    MACHINE_ARCH=$(uname -m)
    if [ ${MACHINE_ARCH} == "x86_64" ]; then
        GW_3RD_LIB_PATH="${DW_TOP_PATH}/../../xlab/x86/lib"
    fi
    if [ ${MACHINE_ARCH} == "aarch64" ]; then
        GW_3RD_LIB_PATH="${DW_TOP_PATH}/../../xlab/sysroot/lib"
    fi
    export LD_LIBRARY_PATH=${GW_3RD_LIB_PATH}:${LD_LIBRARY_PATH}
    echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
    ## copy xplatform_schedule_manager && xplatform_stm_master
    # cp ${SCHEDULE_MANAGER_PATH}/* ${RR_TOP_PATH}
    # cp ${XPLATFORM_SSM_PATH}/* ${RR_TOP_PATH}
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
        ps -ef | grep -E 'stm|ssm|ScheduleManager|schdule_manager' | grep -v 'grep' | awk -F " " '{print $2}' | xargs kill -9
        # rm ${RR_TOP_PATH}/xplatform_schedule_manager ${RR_TOP_PATH}/xplatform_stm_master
        # rm ${RR_TOP_PATH}/xplatform_ssm_demo1
        export LD_LIBRARY_PATH=${LD_LIBRARY_PATH_ORIGINAL}
    fi
}

trap "__cleanup; exit 1" INT TERM QUIT

__prepDisplay
__setEnv

if [[ ! -d ${RR_LOG_PATH} ]];then
  mkdir -p ${RR_LOG_PATH}
else
  echo "${RR_LOG_PATH} dir exists"
fi

# zs: delete codes from 5.14 with RR_RUN_CFG_PATH

SPEC=""
SCHED=""
if [[ -f $PRODUCT ]]; then
    SPEC=$PRODUCT
    SCHED=$SCHEDULE
else
    SPEC=${RR_GRAPHS_PATH}/apps/${TEAM}/app$APP_TYPE/${PRODUCT}${APP_TYPE}.app.json
    SCHEDS_TAB=${PRODUCT}${APP_TYPE}__*.stm
    if ! ls ${DW_TOP_PATH}/bin/${PROJECT}/${SCHEDS_TAB} 1> /dev/null 2>&1; then
    echo "Schedule file ${DW_TOP_PATH}/bin/${PROJECT}/${SCHEDS_TAB} doesn't exist!"
        __cleanup
        exit 1
    fi
    if [ "$OS" == "QNX" ]; then
        SCHED=$(find ${DW_TOP_PATH}/bin/${PROJECT} -name "${SCHEDS_TAB}" | paste -s -d "," - -)
    else
        SCHED=$(find ${DW_TOP_PATH}/bin/${PROJECT} -name "${SCHEDS_TAB}" | paste -sd,)
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

# zs: deleted some codes from 5.14 with DATA_SOURCE

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
ARGS="$ARGS --path=${RR_TOP_PATH}"
ARGS="$ARGS --datapath=${DATA_PATH}"
ARGS="$ARGS --dwdatapath=${DW_TOP_PATH}/data"
ARGS="$ARGS --schedule=${SCHED}"
ARGS="$ARGS --start_timestamp=0"
ARGS="$ARGS --mapPath=maps/sample/sanjose_loop"
ARGS="$ARGS --loglevel=${LOG_LEVEL}"
ARGS="$ARGS --fullscreen=1"
ARGS="$ARGS --winSizeW=1280"
ARGS="$ARGS --winSizeH=800"
ARGS="$ARGS --virtual=0"
ARGS="$ARGS --disableStmControlLogger=1"
ARGS="$ARGS --gdb_debug=${GDB_DEBUG}"
ARGS="$ARGS --app_parameter=${APP_PARAMETER}"
ARGS="$ARGS --useLCM=0"

# dwtrace params
ARGS="$ARGS --memTraceEnabled=1"
ARGS="$ARGS --stmControlTracing=0"
ARGS="$ARGS --traceChannelMask=0xFFFFFFFF" ## trace all channel
ARGS="$ARGS --traceFilePath=${RR_LOG_PATH}"
ARGS="$ARGS --traceLevel=0" ## 0: close dwtrace

# zs: delete some codes from 5.14 with more args

ls -al ${RR_TOP_PATH}

set +e
echo "Running command: ${BIN_PATH}/launcher ${ARGS} > ${RR_LOG_PATH}/launcher.log 2>&1"
${BIN_PATH}/launcher ${ARGS} > ${RR_LOG_PATH}/launcher.log 2>&1 &
LAUNCHER_PID=$!
wait ${LAUNCHER_PID}
RR_STATUS=$?
set -e

# reset env
# rm ${RR_TOP_PATH}/xplatform_schedule_manager ${RR_TOP_PATH}/xplatform_stm_master
# rm ${RR_TOP_PATH}/xplatform_ssm_demo1
echo "Check if reset NetworkStack needed"
echo "Restore LD_LIBRARY_PATH to ${LD_LIBRARY_PATH_ORIGINAL}"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH_ORIGINAL}
echo "======================================================================="
echo launcher exit status: ${RR_STATUS}
echo
