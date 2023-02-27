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

import sys, getopt
import csv
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import os.path
from collections import defaultdict
from os import listdir
from os.path import join
from genericpath import isdir
from operator import add
from audioop import avg
from IN import INT64_C

bin_labels = []

def print_usage():
    print 'Usage: plot_data -p <path to directory containing timestamp data>'

def parse_arg(argv):
    try:
        opts, args = getopt.getopt(argv, "hp:", ["path="])
    except getopt.GetoptError:
        print_usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print_usage()
            sys.exit()
        elif opt in ("-p", "--path"):
            return arg

def get_stats(vec):
    return max(vec), vec.index(max(vec)), min(vec), vec.index(min(vec)), reduce(lambda x, y: x + y, vec) / len(vec)

def get_bin_labels():
    # the output histogram has 22 bins :
    # | <-100% | -100% | -90% | ... | -10% | 10% | ... | 90% | 100% | >100% |

    bin_labels.append('')
    start = -100
    for x in range(0,21):
        bin_labels.append(str(start + x*10)+"%")
    bin_labels.append('')

def calc_hist(vec, step):
    bin_boundaries = []
    bin_data = [0]*22

    for x in range(0,21):
        bin_boundaries.append(x*step)

    for x in vec:
        if x < 0:
            bin_data[0] = bin_data[0] + 1
        if x >= bin_boundaries[20]:
            bin_data[21] = bin_data[21] + 1
        for y in range(0,20):
            if x >= bin_boundaries[y] and x < bin_boundaries[y+1]:
                bin_data[y+1] = bin_data[y+1] + 1
                break

    bin_data = map(float, bin_data)
    bin_data[:] = [x * 100.0 / float(len(vec)) for x in bin_data]

    return bin_data

def bar_plot(ax, vec, max_val, max_idx, min_val, min_idx, frame_rate):
    plt.sca(ax)

    x_pos = np.arange(len(vec))
    barlist = plt.bar(x_pos, vec, align='edge')
    barlist[10].set_color('r')
    barlist[11].set_color('r')

    plt.xticks(fontsize=8, rotation=70)
    x_pos = np.arange(len(bin_labels))
    plt.xticks(x_pos, bin_labels)
    plt.ylabel('Percentage(%)')
    plt.annotate('max = '+str(max_val)+' (frame: '+str(max_idx)+')', xy=(0.6, 0.9), xycoords='axes fraction')
    plt.annotate('min =  '+str(min_val)+' (frame: '+str(min_idx)+')', xy=(0.6, 0.8), xycoords='axes fraction')
    if frame_rate > 0:
        plt.annotate('frame_rate = '+str(frame_rate)+'/s', xy=(0.6, 0.7), xycoords='axes fraction')
    else:
        plt.annotate('frame_rate: unknown', xy=(0.6, 0.7), xycoords='axes fraction')

def plot_sensor_ts(ts, frame_rate, fname):
    ts_val = map(int, ts)
    ts_delta = [t - s for s, t in zip(ts_val, ts_val[1:])]
    max_val, max_idx, min_val, min_idx, avg_val = get_stats(ts_delta)

    bin_size = int(frame_rate)
    if bin_size > 0:
        bin_size = 100000 / bin_size
    else:
        bin_size = avg_val / 10
    bin_data = calc_hist(ts_delta, bin_size)


    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    plt.sca(ax1)
    plt.plot(ts_delta)
    plt.ylabel('Timestamp Delta (us)')

    ax2 = fig.add_subplot(2,1,2)
    bar_plot(ax2, bin_data, max_val, max_idx, min_val, min_idx, frame_rate)

    fname = os.path.splitext(fname)[0]+'.png'
    fig.savefig(fname)
    plt.close(fig)


def plot_host_sensor_ts(host_ts, sensor_ts, frame_rate, fname):
    host_ts_val = map(int, host_ts)
    host_ts_delta = [t - s for s, t in zip(host_ts_val, host_ts_val[1:])]
    host_max, host_max_idx, host_min, host_min_idx, host_avg = get_stats(host_ts_delta)

    sensor_ts_val = map(int, sensor_ts)
    sensor_ts_delta = [t - s for s, t in zip(sensor_ts_val, sensor_ts_val[1:])]
    sensor_max, sensor_max_idx, sensor_min, sensor_min_idx, sensor_avg = get_stats(sensor_ts_delta)

    host_bin_size = 0
    sensor_bin_size = 0
    if int(frame_rate) > 0:
        host_bin_size = 100000 / int(frame_rate)
        sensor_bin_size = 100000 / int(frame_rate)
    else:
        host_bin_size = host_avg / 10
        sensor_bin_size = sensor_avg / 10

    host_bin_data = calc_hist(host_ts_delta, host_bin_size)
    sensor_bin_data = calc_hist(sensor_ts_delta, sensor_bin_size)

    fig = plt.figure(figsize=(15,10))
    ax1 = fig.add_subplot(2,2,1)
    plt.sca(ax1)
    plt.plot(host_ts_delta)
    plt.ylabel('Timestamp Delta (us)')
    plt.title('Host')

    ax2 = fig.add_subplot(2,2,2)
    plt.sca(ax2)
    plt.plot(sensor_ts_delta)
    plt.ylabel('Timestamp Delta (us)')
    plt.title('Sensor')

    ax3 = fig.add_subplot(2,2,3)
    bar_plot(ax3, host_bin_data, host_max, host_max_idx, host_min, host_min_idx, frame_rate)

    ax4 = fig.add_subplot(2,2,4)
    bar_plot(ax4, sensor_bin_data, sensor_max, sensor_max_idx, sensor_min, sensor_min_idx, frame_rate)

    fname = os.path.splitext(fname)[0]+'.png'
    fig.savefig(fname)
    plt.close(fig)

def sensor_plot(fname, frame_rate):
    if os.path.exists(fname):
        with open(fname) as tsv:
            print fname
            cols = zip(*[line for line in csv.reader(tsv, dialect="excel-tab")])
            if len(cols) == 2:
                plot_sensor_ts(cols[1], frame_rate, fname)
            else:
                plot_host_sensor_ts(cols[1], cols[2], frame_rate, fname)


# the function loop through all folders to do plots for
# files inside each individual folder
def plot_figure(path):
    dnames = [f for f in listdir(path) if isdir(join(path,f))]
    if len(dnames) == 0:
        result_json = join(path, 'post-record-check.json')
        if not os.path.exists(result_json):
            result_json = join(path, 'report.json')
        json_data = json.load(open(result_json))
        for sensor in json_data["sensors"]:
            sensor_plot(join(path, sensor["name"]+".ts"), sensor["frame_rate (frames/s)"])
    else:
        for d in dnames:
            plot_figure(path+d)


def main(argv):
    path = parse_arg(argv)
    get_bin_labels()
    plot_figure(path)

if __name__ == "__main__":
    main(sys.argv[1:])
