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
# SPDX-FileCopyrightText: Copyright (c) 2017-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys
import getopt
import os
from os import listdir
from os.path import join
from genericpath import isdir

def print_usage():
    print 'Usage: batch_run.py -b <postrecord-checker binary> -i <input path> -o <output path>'

def parse_arg(argv):
    try:
        opts, args = getopt.getopt(argv, "hb:i:o:", ["path="])
    except getopt.GetoptError:
        print_usage()
        sys.exit(2)

    bin_path = ''
    in_path = ''
    out_path = ''

    for opt, arg in opts:
        if opt == '-h':
            print_usage()
            sys.exit()
        elif opt in ("-i", "--input"):
            in_path = arg
        elif opt in ("-o", "--output"):
            out_path = arg
        elif opt in ("-b", "--binary"):
            bin_path = arg

    return bin_path, in_path, out_path

def get_all_dirs(p, dirs):
    dnames = [f for f in listdir(p) if isdir(join(p,f))]
    if len(dnames) == 0:
        dirs.append(p)
    else:
        for d in dnames:
            get_all_dirs(p+'/'+d, dirs)


def main(argv):
    bin_dir, in_dir, out_dir = parse_arg(argv)
    dirs = []
    get_all_dirs(in_dir, dirs)
    finished_count = 0
    for d in dirs:
        # a typical example of recording session path is
        #      av/recording-sessions/hyperion/2017/11/23/00-29-14.000
        pos = d.find("hyperion")
        if pos != -1:
            folder_name = d[pos:]
            folder_name = folder_name.replace("hyperion/","")
            folder_name = folder_name.replace("/","-")

            folder_name = out_dir + '/' + folder_name
        else:
            folder_name = out_dir + '/' + os.path.basename(d)

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        log_name = folder_name + '/log'
        recording_session = ' --recording_session=' + d
        output = ' --output=' + folder_name
        sensors = ' --sensor=all'
        cmd = bin_dir + recording_session + output + sensors + ' >>' + log_name + ' 2>&1'
        os.system(cmd)
        finished_count = finished_count + 1
        print finished_count, '/', len(dirs), ': ', folder_name, 'done'



if __name__ == "__main__":
    main(sys.argv[1:])
