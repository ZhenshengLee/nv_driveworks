#!/usr/bin/env python3

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
# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import socket, struct
import time, datetime
import os, sys, shutil
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.realpath(sys.argv[0])) + "/scripts/")
import recorder_utils as utils

def printOverride(func, log):
    def printStdoutFile(*args,**kwargs):
        func(*args,**kwargs, flush=True)
        func(*args,**kwargs, file=log, flush=True)
    return printStdoutFile

class BBR:
    def __init__(self, logFile):
        self.can_frame_fmt = "=IB3x8s"
        self.can_frame_size = struct.calcsize(self.can_frame_fmt)

        self.DATASPEED_ID_BRAKE_REPORT = 0x061
        self.DATASPEED_ID_THROTTLE_REPORT = 0x063
        self.DATASPEED_ID_STEERING_REPORT = 0x065

        self.epoch_s = 30.0 # 30 seconds epoch
        self.can = "can0"  # To Do: get it from rig, if not use can0
        self.fileCheckEpoch_s = max(self.epoch_s / 10, 0.5)

        self.vehicleState = {"steeringEnabled": False, "throttleEnabled": False, "brakeEnabled": False,
                        "lastLatCtrlEn": False, "lastLongCtrlEn": False,
                        "lastDisengagementTime": datetime.datetime.fromtimestamp(time.time() - (self.epoch_s + 1))}
        self.rigStorageMap = dict()

        self.startTime = datetime.datetime.fromtimestamp(time.time() - (self.epoch_s + 1))
        self.fileCheckTime = datetime.datetime.fromtimestamp(time.time() - (self.fileCheckEpoch_s + 1))
        self.recording = False
        self.log = open(logFile, "w")
        global print
        print = printOverride(print, self.log)

    def __del__(self):
        self.log.close()

    def detectPathChange(self, line, rigStorageMap):
        if line and len(line) == 4:
            if line[2] == "NewSink:":
                rigName = line[1]
                if line[3] != "/dev/null":
                    dirname = os.path.dirname(line[3])

                    if dirname not in rigStorageMap[rigName]["mntPaths"]:
                        rigStorageMap[rigName]["mntPaths"].append(dirname)
                        rigStorageMap[rigName]["pathQueue"][dirname] = deque()
                        rigStorageMap[rigName]["disengagementQueue"][dirname] = deque()

                    rigStorageMap[rigName]["pathQueue"][dirname].append(line[3])
                    print("BBR:", rigName, rigStorageMap[rigName]["pathQueue"][dirname], "\n")
                    return line[3]

        return ""

    def readBackendInput(self, line, isErr=False):
        if isErr:
            print(b"BBR: Error: ", line)
            return ""

        return self.detectPathChange(line.decode().strip().split(), self.rigStorageMap)

    def canParse(self, frame, currentTimeStamp):
        disengagement = False

        frame_id, can_dlc, frame_data = struct.unpack(self.can_frame_fmt, frame)
        frame_data = frame_data[:can_dlc]

        if frame_id == self.DATASPEED_ID_BRAKE_REPORT or frame_id == self.DATASPEED_ID_THROTTLE_REPORT or frame_id == self.DATASPEED_ID_STEERING_REPORT:
            mask = 0x01  # last bit is enable

            if len(frame_data) != 8:
                return disengagement

            value = mask & frame_data[7]

            if frame_id == self.DATASPEED_ID_BRAKE_REPORT:
                self.vehicleState["brakeEnabled"] = bool(value)

            elif frame_id == self.DATASPEED_ID_THROTTLE_REPORT:
                self.vehicleState["throttleEnabled"] = bool(value)

            elif frame_id == self.DATASPEED_ID_STEERING_REPORT:
                self.vehicleState["steeringEnabled"] = bool(value)

            lat_en = self.vehicleState["steeringEnabled"]
            long_en = self.vehicleState["brakeEnabled"] and self.vehicleState["throttleEnabled"]

            if self.vehicleState["lastLatCtrlEn"] and (not lat_en):
                if (currentTimeStamp - self.vehicleState["lastDisengagementTime"]).total_seconds() > self.epoch_s:
                    print("BBR: Lateral Control Disengagement")
                    disengagement = True

            self.vehicleState["lastLatCtrlEn"] = lat_en

            if self.vehicleState["lastLongCtrlEn"] and (not long_en):
                if (currentTimeStamp - self.vehicleState["lastDisengagementTime"]).total_seconds() > self.epoch_s:
                    print("BBR: Longitudinal Control Disengagement")
                    disengagement = True

            self.vehicleState["lastLongCtrlEn"] = long_en

            if disengagement:
                timeNow = datetime.datetime.now()
                print("BBR: Disengagement time =", currentTimeStamp.strftime("%Y_%m_%d_%H:%M:%S"), "!!", "timeNow =",
                      timeNow.strftime("%Y_%m_%d_%H:%M:%S"), "diff_s = ", (timeNow - currentTimeStamp).total_seconds())

                self.vehicleState["lastDisengagementTime"] = currentTimeStamp

                for rigName in self.rigStorageMap:
                    for mnt in self.rigStorageMap[rigName]["mntPaths"]:
                        self.rigStorageMap[rigName]["disengagementQueue"][mnt].append(currentTimeStamp)
                        print("BBR: ", rigName, "disengagementQueue =", self.rigStorageMap[rigName]["disengagementQueue"][mnt])

        return disengagement

    def setRigMap(self, rigs):
        for rigName in rigs:
            rigBaseName = os.path.basename(rigName)
            self.rigStorageMap[rigBaseName] = {}
            mnts = utils.getStoragePaths(rigName)

            self.rigStorageMap[rigBaseName]["mntPaths"] = mnts
            self.rigStorageMap[rigBaseName]["pathQueue"] = {}
            self.rigStorageMap[rigBaseName]["disengagementQueue"] = {}

            for mnt in mnts:
                self.rigStorageMap[rigBaseName]["pathQueue"][mnt] = deque()
                self.rigStorageMap[rigBaseName]["disengagementQueue"][mnt] = deque()

        print("BBR: rigMap\n", self.rigStorageMap)

    def initialize(self, rigs):
        try:
            SO_TIMESTAMPNS = 35
            s_CAN = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
            s_CAN.setsockopt(socket.SOL_SOCKET, SO_TIMESTAMPNS, 1)
            s_CAN.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s_CAN.bind((self.can,))
        except Exception as e:
            print("BBR:", e)
            raise Exception("BBR: Local CAN socket cannot be opened for device " + self.can)

        self.setRigMap(rigs)

        self.vehicleState = {"steeringEnabled": False, "throttleEnabled": False, "brakeEnabled": False,
                             "lastLatCtrlEn": False, "lastLongCtrlEn": False,
                             "lastDisengagementTime": datetime.datetime.fromtimestamp(time.time() - (self.epoch_s + 1))}

        # self.startTime = datetime.datetime.fromtimestamp(time.time() - (self.epoch_s + 1))
        # self.fileCheckTime = datetime.datetime.fromtimestamp(time.time() - (self.fileCheckEpoch_s + 1))
        self.recording = False

        return s_CAN

    def readCAN(self, rfd):
        frame, ancdata, _, _ = rfd.recvmsg(self.can_frame_size, 1024)
        tmp = struct.unpack("iiii", ancdata[0][2])
        canTS = datetime.datetime.fromtimestamp(tmp[0] + tmp[2] * 1e-10)

        if frame:
            return self.canParse(frame, canTS)

        return False

    def movefiles(self):
        if (datetime.datetime.now() - self.fileCheckTime).total_seconds() > self.fileCheckEpoch_s:
            self.fileCheckTime = datetime.datetime.now()
        else:
            return

        for rigName in self.rigStorageMap:
            for mnt in self.rigStorageMap[rigName]["mntPaths"]:
                if len(self.rigStorageMap[rigName]["disengagementQueue"][mnt]) > 0:
                    time_t = self.rigStorageMap[rigName]["disengagementQueue"][mnt][0]
                    idx = -1  # find idx of pathQueue where this time_t fits

                    for recordPath in self.rigStorageMap[rigName]["pathQueue"][mnt]:
                        recordName = os.path.basename(recordPath)
                        recordTime = datetime.datetime.strptime("_".join(recordName.split('_')[1:7]), utils.TimeFormat)

                        if time_t > recordTime:
                            idx += 1
                        else:
                            break

                    # idx < len(self.rigStorageMap[rigName]["pathQueue"][mnt]) this will be always true
                    if idx != -1 and idx < len(self.rigStorageMap[rigName]["pathQueue"][mnt]):
                        pass

                    else:
                        if idx == -1:
                            # time_t < record time of 0th idx
                            # this case will occur only if there are consecutive disengagement with in two epochs
                            print("BBR:", rigName, "disengagementQueue popped without storing")
                            self.rigStorageMap[rigName]["disengagementQueue"][mnt].popleft()
                        continue

                    while idx > 1:
                        delPath = self.rigStorageMap[rigName]["pathQueue"][mnt].popleft()
                        if delPath and delPath != "/dev/null":
                            shutil.rmtree(delPath)
                            print("BBR:", rigName, "idx =", idx, "delpath = ", delPath)

                        idx -= 1

                    # B - before disengagement, A - after disengagement, D - disengagement, C - current recording
                    # now path Queue can have two case B, D, or D. Also we may have A or A, C in both cases
                    if not (len(self.rigStorageMap[rigName]["pathQueue"][mnt]) > (2 + idx)):
                        continue

                    # now path Queue has B, D, A, C or D, A, C
                    print("BBR:", rigName, "movefiles disengagementQueue =", self.rigStorageMap[rigName]["disengagementQueue"][mnt])
                    kernelDisengageTS = self.rigStorageMap[rigName]["disengagementQueue"][mnt].popleft()
                    disengageDir = "/Disengagement_" + kernelDisengageTS.strftime(utils.TimeFormat)

                    # store the recording in the Disengagement folder, it has same path as that of dw recording
                    # ex: *dw_recording*/../Disengagement*/ will be that path of the Disengagement folder
                    destPath = os.path.abspath(self.rigStorageMap[rigName]["pathQueue"][mnt][0] + "/.." + disengageDir)

                    print("BBR: storing files in", destPath)
                    os.makedirs(destPath)

                    for count in range(2 + idx):
                        storePath = self.rigStorageMap[rigName]["pathQueue"][mnt].popleft()
                        if len(storePath) and storePath != "/dev/null":
                            os.rename(storePath, destPath + "/" + os.path.basename(storePath))

                else:  # no disengagements
                    # to be safe we can check for greater than 3; last 2 pathQueue values are needed and one is for slack time
                    if len(self.rigStorageMap[rigName]["pathQueue"][mnt]) > 3:
                        delPath = self.rigStorageMap[rigName]["pathQueue"][mnt].popleft()
                        if delPath and delPath != "/dev/null":
                            shutil.rmtree(delPath)
                            print("BBR: delpath = ", delPath)

    def changeSinks(self):
        if not self.recording:
            return False

        elapsed_s = (datetime.datetime.now() - self.startTime).total_seconds()  # diff in seconds

        if elapsed_s > self.epoch_s:
            print("BBR: elapsed_s = ", elapsed_s)
            self.startTime = datetime.datetime.now()
            return True

        return False

    def start(self):
        self.recording = True

    def stop(self):
        self.recording = False
