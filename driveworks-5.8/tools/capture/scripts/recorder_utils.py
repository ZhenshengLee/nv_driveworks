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
# SPDX-FileCopyrightText: Copyright (c) 2018-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import glob
import os
import json
import sys, socket, fcntl, struct, array
import subprocess

import hashlib
from Crypto.Cipher import AES
from Crypto.Cipher import PKCS1_OAEP
from Crypto.PublicKey import RSA

minSpace = 5 * 1024 * 1024 * 1024
TimeFormat = "%Y_%m_%d_%H_%M_%S"

def getMountsInfo():
    infos = []
    with open('/proc/self/mounts') as f:
        for l in f:
            l = l.strip()
            cols = l.split(" ")
            if cols[2] != 'ext4': continue
            if cols[1] == '/' or cols[1] == '/driveota': continue
            if not cols[0].startswith('/dev/sd'): continue
            if not os.access(cols[1], os.W_OK): continue
            s = os.statvfs(cols[1])
            tot = s.f_bsize * s.f_blocks
            avail = s.f_bsize * s.f_bavail
            infos += [(cols[1], tot, avail)]
    infos = sorted(infos, key=lambda x: x[2], reverse=True)
    return infos

def getLargestMount(infos):
    if not infos: return []
    if infos[0][2] <= minSpace: return []
    return [infos[0][0]]

def getAvailableSpace(path):
    try:
        stat = os.statvfs(path)
        return stat.f_bavail * stat.f_frsize
    except:
        return 0

def hasMinimumSpace(ipaths):
    paths = ipaths
    if type(paths) ==  str:
        paths = [paths]
    for path in paths:
        if getAvailableSpace(path) < minSpace:
            return False
    return True

def getNumPeersForRig(rig):
    return max(len(getStoragePaths(rig)), 1)

def getRigMount(rigfile, path="storage_path"):
    rigmnt = ""
    with open(rigfile) as f:
        rig = json.load(f)["rig"]
        if "properties" in rig and path in rig["properties"]:
            rigmnt = rig["properties"][path]

    return rigmnt

def getStoragePaths(rigfile, path="recorder_storage_paths"):
    with open(rigfile) as f:
        rig = json.load(f)["rig"]
        if path in rig:
            return rig[path]
    #backwards complatablity
    mnt = getRigMount(rigfile)
    if mnt: return [mnt]
    return []

def getSensorStorage(rigfile, sensor):
    storage = getStoragePaths(rigfile)
    if not storage:
        return ""
    idx = 0
    with open(rigfile) as f:
        rig = json.load(f)["rig"]
        if "recorder_storage_map" in rig and rig["recorder_storage_map"][sensor]:
            idx = int(rig["recorder_storage_map"][sensor])

    return storage[idx]
def removeFileIfPresent(path):
    try:
        os.remove(path)
    except:
        pass

def printNlines(fileName, N):
    with open(fileName, "rb") as f:
        i = 0
        line = f.readline()
        while (i < N) and line:
            line = line.decode("utf-8").strip()
            if line:
                print(line)
                i += 1
            line = f.readline()

# Try to fetch the VIN
# Returns valid VIN digits or 00000000000000000
def tryFetchVIN():
    try:
        with open('/tmp/car_vin', 'r') as f:
            vin = f.read().strip()
            if vin.isalnum() and len(vin) == 17:
                return vin
    except:
        pass
    return '00000000000000000'

def checkLocalIP(IP):
    # check whether system is 32 or 64 bits
    if sys.maxsize == 2**32-1:
        outputStructSize = 32
    else:
        outputStructSize = 40

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # maximum interfaces supported
    maximumInterfaces = 128
    maxSize = maximumInterfaces * outputStructSize

    # array of ifreq structs
    nameArray = array.array('B', bytes(('\0' * maxSize), 'utf-8'))

    # create a struct to pass data in and out of ioctrl
    ifreqStruct = struct.pack('iL', maxSize, nameArray.buffer_info()[0])

    # invoke ioctl
    try:
        outputSize = struct.unpack('iL', fcntl.ioctl(s.fileno(), 0x8912, ifreqStruct))[0]
        nameString = nameArray.tostring()

        for i in range(0, outputSize, outputStructSize):
            if socket.inet_ntoa(nameString[i+20:i+24]) == IP:
                return True
    except:
        return False

    return False

def generateAESKey(length_bits):
    return os.urandom(int(length_bits / 8))

def padDataForAESEncryption(s):
    return s + (AES.block_size - len(s) % AES.block_size) * chr(AES.block_size - len(s) % AES.block_size)

def encryptAES(key, b):
    if type(key) == str:
        key = bytes.fromhex(key)
    cipher = AES.new(key, AES.MODE_CBC, b'\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0')
    return cipher.encrypt(padDataForAESEncryption(b)).hex()

def encryptRSA(keyFile, b):
    rsaKey = RSA.importKey(open(keyFile).read())
    cipher = PKCS1_OAEP.new(rsaKey)
    return cipher.encrypt(b)

def getRSAMD5(keyFile):
    return hashlib.md5(open(keyFile,'rb').read()).hexdigest()

def getVersionInfo(tmpDir, recordingDir, dwInfoPath, buildInfoPaths,
                    buildInfoOverlayPaths):
    versionJson = {}
    out = {}
    versionCheckerOutPath = os.path.join(tmpDir, "version_checker_out.json")
    try:
        subprocess.call(["version_checker", "--output-file", versionCheckerOutPath],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(versionCheckerOutPath, "r") as versionCheckerFile:
            out = json.load(versionCheckerFile)
    except:
        print("recorder-tui: Failed to get DW version info from version_checker")
        pass
    finally:
        versionJson["version_checker"] = out
        removeFileIfPresent(versionCheckerOutPath)

    out = {}
    try:
        out = json.loads(subprocess.check_output([dwInfoPath]).decode("utf-8"))
    except:
        print("recorder-tui: Failed to get DW info from dw_info")
        pass

    versionJson["dw_info"] = out

    out = {}
    overlay_out = {}
    try:
        for buildInfoPath in buildInfoPaths:
            if os.path.exists(buildInfoPath):
                with open(buildInfoPath, "r") as buildInfoFile:
                    out = json.load(buildInfoFile)
        # Glob to find "custom" build_info files whose fields should be overlaid
        # onto contents of buildInfoPath used above
        for overlayPath in buildInfoOverlayPaths:
            # Glob returns either empty list (custom info missing, or legacy
            # flash was used), or a list with at most 1 entry (from
            # Bazel-generated image).
            candidatePaths = glob.glob(overlayPath)
            if candidatePaths:
                with open(candidatePaths[0], "r") as buildInfoOverlayFile:
                    overlay_out = json.load(buildInfoOverlayFile)
        # Combine overlay and buildInfo contents
        out.update(overlay_out)
    except:
        print("recorder-tui: Failed to get PDK info from build_info.txt")
        pass

    versionJson["build_info"] = out

    versionFilePath = os.path.join(recordingDir, "software_versions.json")
    with open(versionFilePath, "w+") as versionFile:
        json.dump(versionJson, versionFile, indent=4)
