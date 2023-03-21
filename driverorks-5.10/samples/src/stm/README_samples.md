/////////////////////////////////////////////////////////////////////////////////////////
//
// Notice
// ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS" NVIDIA MAKES
// NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//
// NVIDIA CORPORATION & AFFILIATES assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of NVIDIA CORPORATION & AFFILIATES. No third party distribution is allowed unless
// expressly authorized by NVIDIA. Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// NVIDIA CORPORATION & AFFILIATES products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// NVIDIA CORPORATION & AFFILIATES.
//
// SPDX-FileCopyrightText: Copyright (c) 2018-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

Pre-requisites:

1. Make sure that /etc/nvsciipc.cfg on target contains the entries in /usr/local/driveworks/targets/aarch64-Linux(or qnx)/config/nvsciipc.cfg (can append to existing /etc/nvsciipc.cfg file if they are not present). Ensure that the entries are unique in /etc/nvsciipc.cfg. Please reboot the system after this step.
NOTE: Ensure that there no newlines at the end of the file in /etc/nvsciipc.cfg. Run "sudo service nv_nvsciipc_init status" after the reboot. If this command returns an error, please re-check the contents of /etc/nvsciipc.cfg

2. Mqueue length of at least 4096 needs to be supported. On Linux, do either of the following:
     i. Change the contents of file /proc/sys/fs/mqueue/msg_max to 4096 (does not persist across reboots)
    ii. Add fs.mqueue.msg_max = 4096 to /etc/sysctl.conf and restart (persists across reboot)

3. Besides the minimum mqueue number of messages, the total mqueue size (ulimit -q) needs to be increased since this build uses larger sized messages. Either run as sudo or add these lines to /etc/security/limits.conf
       <user>          soft    msgqueue        unlimited
       <user>          hard    msgqueue        unlimited
Allows the <user>(change it to appropriate name) to have unlimited sized mqueue

To run the sample binaries directly on x86:

1. `ps -ef | grep -e framesync -e stm_ | grep -v grep | awk '{print $2}' | xargs -rt sudo kill -s KILL || true`

2. `sudo rm -rf /dev/shm/* /dev/mqueue/*`

   Note: The above command must be run if PDK < 6.0.5.0 only

3. `export CUDA_VISIBLE_DEVICES=1`

4. `export LD_LIBRARY_PATH=/usr/local/driveworks/targets/x86_64-Linux/lib:/usr/local/cuda-11.4/lib:/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH`

5. Commands for each sample:

    i. `sudo /usr/local/driveworks/bin/stm_master -s /usr/local/driveworks/bin/cpu_simple.stm -l x.log -e 50 & sudo /usr/local/driveworks/bin/stm_test_cpu_simple`

    ii. `sudo /usr/local/driveworks/bin/stm_master -s /usr/local/driveworks/bin/gpu_multistream_multiprocess.stm -l x.log -e 50 & sudo /usr/local/driveworks/bin/stm_test_gpuX & sudo /usr/local/driveworks/bin/stm_test_gpuY`

6. STM packages a sample schedule manager in the release. This sample schedule manager switches between schedule IDs 101 and 102 (cpu_gpu1.stm and cpu_gpu2.stm respectively) to demonstrate the schedule switch functionality. Execute the following commands in order and in different terminals

    i. Run the stm_master along with list of schedules

        sudo /usr/local/driveworks/bin/stm_master -s /usr/local/driveworks/bin/cpu_gpu1.stm,/usr/local/driveworks/bin/cpu_gpu2.stm -l x.log -e 500 -i 2 -N default

    ii. Run the test schedule manager binary.

        sudo /usr/local/driveworks/bin/stm_sample_manager default -v

    iii. Run client binaries

        sudo /usr/local/driveworks/bin/stm_sample_gpuX & sudo /usr/local/driveworks/bin/stm_sample_gpuY

    Each cycle of execution has 1 schedule switch (one switch between the two schedules passed as input to stm_master) and by default the schedules will switch with a time period of 1000 milliseconds.
    There should be 10 cycles of execution for the above commands.
    The schedule switches can be seen in the logs of `stm_sample_manager`. Use `-v` with `stm_sample_manager` for verbose outputs.

To run the sample binaries directly on target:

1. `ps -ef | grep -e framesync -e stm_ | grep -v grep | awk '{print $2}' | xargs -rt sudo kill -s KILL || true`

2. `sudo rm -rf /dev/shm/* /dev/mqueue/*`

   Note: The above command must be run if PDK < 6.0.5.0 only

3. `export CUDA_VISIBLE_DEVICES=1`

4. [For linux] : `export LD_LIBRARY_PATH=/usr/local/driveworks/targets/aarch64-Linux/lib:/usr/local/cuda-11.4/lib:/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH`

    [For QNX] : `export LD_LIBRARY_PATH=/usr/local/driveworks/targets/aarch64-qnx/lib:/usr/local/cuda-11.4/lib:/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH`

5. Commands for each sample:

    i. `sudo /usr/local/driveworks/bin/stm_master -s /usr/local/driveworks/bin/cpu_simple.stm -l x.log -e 50 & sudo /usr/local/driveworks/bin/stm_test_cpu_simple`

    ii. `sudo /usr/local/driveworks/bin/stm_master -s /usr/local/driveworks/bin/gpu_multistream_multiprocess.stm -l x.log -e 50 & sudo /usr/local/driveworks/bin/stm_test_gpuX & sudo /usr/local/driveworks/bin/stm_test_gpuY`

    iii. `sudo /usr/local/driveworks/bin/stm_master -s /usr/local/driveworks/bin/dla_simple.stm -l x.log -e 50 & sudo /usr/local/driveworks/bin/stm_test_dla`

    (Note: The dla_simple sample only works till PDK 6.0.3.0)

    iv. `sudo /usr/local/driveworks/bin/stm_master -s /usr/local/driveworks/bin/vpu_simple.stm -l x.log -e 50 & sudo /usr/local/driveworks/bin/stm_test_vpu`

    - (Note: The vpu_simple app is only available for PDK 6.0.4.0+ and requires the presence of cuPVA SDK v2.0.0 libraries)
    - To run the STM samples on a linux target, disable the cuPVA signature by running the following commands:

            sudo su
            echo 0 > /sys/kernel/debug/pva0/vpu_app_authentication
            exit

6. Commands to check STM's schedule switch. Execute the following commands in order in different terminals

    i. Run the stm_master along with list of schedules

        sudo /usr/local/driveworks/bin/stm_master -s /usr/local/driveworks/bin/cpu_gpu1.stm,/usr/local/driveworks/bin/cpu_gpu2.stm -l x.log -e 500 -i 2 -N default

    ii. Run the test schedule manager binary.

        sudo /usr/local/driveworks/bin/stm_sample_manager default -v

    iii. Run client binaries

        sudo /usr/local/driveworks/bin/stm_sample_gpuX & sudo /usr/local/driveworks/bin/stm_sample_gpuY

    Each cycle of execution has 1 schedule switch (one switch between the two schedules passed as input to stm_master) and by default the schedules will switch with a time period of 1000 milliseconds.
    There should be 10 cycles of execution for the above commands.
    The schedule switches can be seen in the logs of `stm_test_manager`. Use `-v` with `stm_test_manager` for verbose outputs.
    This sample schedule manager switches between schedule IDs 101 and 102 (cpu_gpu1.stm and cpu_gpu2.stm respectively) to demonstrate the schedule switch functionality.

To use the tools given by STM on x86:

1. STMCompiler :  `/usr/local/driveworks/tools/stmcompiler -i /path/to/input_file.yml -o /path/to/output_file.stm`

2. STMVizschedule : `/usr/local/driveworks/tools/stmvizschedule -i /path/to/input_file.stm -o /path/to/output_file.html`

3. STMVizGraph : `/usr/local/driveworks/tools/stmvizgraph -i /path/to/input_file.yml  -o /path/to/output_file.svg`

        NOTE: Needs GraphViz installed on the system (sudo apt install graphviz)

4. STM Analytics :

    i. `/usr/local/driveworks/tools/stmanalyze -s /path/to/input_file.stm -l /path/to/log_file -f html`

    ii. If a schedule manager was used to execute multiple schedules, the list of schedules must be passed to stmanalyze
    `/usr/local/driveworks/tools/stmanalyze -s /path/to/input_file1.stm,/path/to/input_file2.stm,... -l /path/to/log_file -f html`

NOTE: The log file is obtained after running the sample binaries above

To compile and run samples from src, follow these steps :

`cd /usr/local/driveworks/samples/src/stm/src/`

STM Compiler Step:

1a. `/usr/local/driveworks/tools/stmcompiler -i test_cpu_gpu_simple/gpu_multistream_multiprocess.yml -o gpu_multistream_multiprocess.stm`

1b. `/usr/local/driveworks/tools/stmcompiler -i test_cpu_simple/cpu_simple.yml -o cpu_simple.stm`

1c. `/usr/local/driveworks/tools/stmcompiler -i  test_dla_simple/dla_simple.yml -o dla_simple.stm`

1d. `/usr/local/driveworks/tools/stmcompiler -i  sample_complete_swap/cpu_gpu1.yml -o cpu_gpu1.stm`

1e. `/usr/local/driveworks/tools/stmcompiler -i  sample_complete_swap/cpu_gpu2.yml -o cpu_gpu2.stm`

1f. `/usr/local/driveworks/tools/stmcompiler -i   test_vpu_simple/vpu_simple.yml -o vpu_simple.stm`

STM Runtime Step:

NOTE: For cross compilation, ensure that driveworks_stm_cross.deb is installed

`cd /usr/local/driveworks/samples/src/stm/src/`

1. `mkdir stm-build & cd stm-build`

2.
    - To cross-compile for Linux: `cmake -DCMAKE_BUILD_TYPE=Release .. -DCMAKE_TOOLCHAIN_FILE=cmake/Toolchain-V5L.cmake -DVIBRANTE_PDK:STRING=/drive/drive-linux -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda -DSTM_BASE_DIR=/usr/local/driveworks/targets/aarch64-Linux/ -DVIBRANTE_PDK_FOUNDATION:STRING=/drive/drive-foundation`
    - For x86 : `cmake -DCMAKE_BUILD_TYPE=Release .. -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda -DSTM_BASE_DIR=/usr/local/driveworks/targets/x86_64-Linux/`
    - To cross-compile for QNX: `cmake -DCMAKE_BUILD_TYPE=Release .. -DCMAKE_TOOLCHAIN_FILE=cmake/Toolchain-V5Q.cmake -DVIBRANTE_PDK:STRING=/drive/drive-qnx -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-safe-11.4 -DSTM_BASE_DIR=/usr/local/driveworks/targets/aarch64-QNX/ -DVIBRANTE_PDK_FOUNDATION:STRING=/drive/drive-foundation`

3. `make install -j <number of jobs>`

To run the built samples on x86:

1. `ps -ef | grep -e framesync -e stm_ | grep -v grep | awk '{print $2}' | xargs -rt sudo kill -s KILL || true`

2. `sudo rm -rf /dev/shm/* /dev/mqueue/*`

   Note: The above command must be run if PDK < 6.0.5.0 only

3. `export CUDA_VISIBLE_DEVICES=1`

4. `export LD_LIBRARY_PATH=/usr/local/driveworks/targets/x86_64-Linux/lib:/usr/local/cuda-11.4/lib:/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH`

5. Commands for each sample:

    i. `sudo /usr/local/driveworks/bin/stm_master -s /usr/local/driveworks/samples/src/stm/src/cpu_simple.stm -l x.log -e 50 & sudo /usr/local/driveworks/samples/src/stm/src/stm-build/test_cpu_simple/client/stm_test_cpu_simple`

    ii. `sudo /usr/local/driveworks/bin/stm_master -s /usr/local/driveworks/samples/src/stm/src/gpu_multistream_multiprocess.stm -l x.log -e 50 & sudo /usr/local/driveworks/samples/src/stm/src/stm-build/test_cpu_gpu_simple/clientX/stm_test_gpuX & sudo /usr/local/driveworks/samples/src/stm/src/stm-build/test_cpu_gpu_simple/clientY/stm_test_gpuY`

6. Commands to check STM's schedule switch. Execute the following commands in order and in different terminals

    i. Run the stm_master along with list of schedules

        sudo /usr/local/driveworks/bin/stm_master -s /usr/local/driveworks/samples/src/stm/src/cpu_gpu1.stm,/usr/local/driveworks/samples/src/stm/src/cpu_gpu2.stm -l x.log -e 500 -i 2 -N default

    ii. Run the test schedule manager binary.

        sudo /usr/local/driveworks/samples/src/stm/src/stm-build/sample_complete_swap/schedule_manager/stm_sample_manager default -v

    iii. Run client binaries

        sudo /usr/local/driveworks/samples/src/stm/src/stm-build/sample_complete_swap/clientX/stm_sample_gpuX & sudo /usr/local/driveworks/samples/src/stm/src/stm-build/sample_complete_swap/clientY/stm_sample_gpuY

    Each cycle of execution has 1 schedule switch (one switch between the two schedules passed as input to stm_master) and by default the schedules will switch with a time period of 1000 milliseconds.
    There should be 10 cycles of execution for the above commands.
    The schedule switches can be seen in the logs of `stm_sample_manager`. Use `-v` with `stm_sample_manager` for verbose outputs.
    This sample schedule manager switches between schedule IDs 101 and 102 (cpu_gpu1.stm and cpu_gpu2.stm respectively) to demonstrate the schedule switch functionality.

To run the built samples on Target:

NOTE: Rsync the built samples to the equivalent folder in Target

1. `ps -ef | grep -e framesync -e stm_ | grep -v grep | awk '{print $2}' | xargs -rt sudo kill -s KILL || true`

2. `sudo rm -rf /dev/shm/* /dev/mqueue/*`

   Note: The above command must be run if PDK < 6.0.5.0 only

3. `export CUDA_VISIBLE_DEVICES=1`

4. [For linux] : `export LD_LIBRARY_PATH=/usr/local/driveworks/targets/aarch64-Linux/lib:/usr/local/cuda-11.4/lib:/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH`

   [For QNX] : `export LD_LIBRARY_PATH=/usr/local/driveworks/targets/aarch64-qnx/lib:/usr/local/cuda-11.4/lib:/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH`

5. Commands for each sample:

    i. `sudo /usr/local/driveworks/bin/stm_master -s /usr/local/driveworks/samples/src/stm/src/cpu_simple.stm -l x.log -e 50 & sudo /usr/local/driveworks/samples/src/stm/src/stm-build/test_cpu_simple/client/stm_test_cpu_simple`

    ii. `sudo /usr/local/driveworks/bin/stm_master -s /usr/local/driveworks/samples/src/stm/src/gpu_multistream_multiprocess.stm -l x.log -e 50 & sudo /usr/local/driveworks/samples/src/stm/src/stm-build/test_cpu_gpu_simple/clientX/stm_test_gpuX & sudo /usr/local/driveworks/samples/src/stm/src/stm-build/test_cpu_gpu_simple/clientY/stm_test_gpuY`

    iii. `sudo /usr/local/driveworks/bin/stm_master -s /usr/local/driveworks/samples/src/stm/src/dla_simple.stm -l x.log -e 50 & sudo /usr/local/driveworks/samples/src/stm/src/stm-build/test_dla_simple/dla_simple/stm_test_dla`

6. Commands to check STM's schedule switch. Execute the following commands in order in different terminals

    i. Run the stm_master along with list of schedules

        sudo /usr/local/driveworks/bin/stm_master -s /usr/local/driveworks/samples/src/stm/src/cpu_gpu1.stm,/usr/local/driveworks/samples/src/stm/src/cpu_gpu2.stm -l x.log -e 500 -i 2 -N default

    ii. Run the test schedule manager binary.

        sudo /usr/local/driveworks/samples/src/stm/src/stm-build/sample_complete_swap/schedule_manager/stm_sample_manager default -v

    iii. Run client binaries

        sudo /usr/local/driveworks/samples/src/stm/src/stm-build/sample_complete_swap/clientX/stm_sample_gpuX & sudo /usr/local/driveworks/samples/src/stm/src/stm-build/sample_complete_swap/clientY/stm_sample_gpuY

    Each cycle of execution has 1 schedule switch (one switch between the two schedules passed as input to stm_master) and by default the schedules will switch with a time period of 1000 milliseconds.
    There should be 10 cycles of execution for the above commands.
    The schedule switches can be seen in the logs of `stm_test_manager`. Use `-v` with `stm_test_manager` for verbose outputs.
    This sample schedule manager switches between schedule IDs 101 and 102 (cpu_gpu1.stm and cpu_gpu2.stm respectively) to demonstrate the schedule switch functionality.

Use the stmanalyze tool given by STM on x86, to obtain the final performance of the logs produced by these steps.
