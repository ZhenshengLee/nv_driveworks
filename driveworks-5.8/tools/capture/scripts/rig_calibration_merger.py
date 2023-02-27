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
# SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json, sys, getopt
import os, os.path
import tempfile
from functools import reduce

def getSectionFromJson(rigNames, section):
    for rigName in rigNames:
        with open(rigName, 'r') as inf:
            rig = json.load(inf)
            try:
                return rig[section]
            except KeyError as e:
                continue
    return {}

def getSectionFromRig(rigNames, section):
    for rigName in rigNames:
        with open(rigName, 'r') as inf:
            rig = json.load(inf)
            try:
                return rig["rig"][section]
            except (KeyError, TypeError):
                continue
    return {}

def getSensors(rigName):
    sensors = {}

    with open(rigName, 'r') as inf:
        rig = json.load(inf)
        for sensor in rig["rig"]["sensors"]:
            name = sensor["name"]
            sensors[name] = sensor

    return sensors

def getSensorsSection(calibratedRigName, inputRigs, useAbsolutePaths):
    calibratedSensors = {}
    if calibratedRigName:
        calibratedSensors = getSensors(calibratedRigName)

    allSensors = {}
    for rigName in inputRigs:
        rigSensors = getSensors(rigName)

        for name, value in rigSensors.items():
            if name in allSensors:
                raise Exception("Duplicate sensor name " + name + ", quitting")
            else:
                allSensors[name] = value

                # Try to get calibration from the calibrated rig
                try:
                    allSensors[name]["correction_sensor_R_FLU"] = calibratedSensors[name]["correction_sensor_R_FLU"]
                # KeyError handles missing data. TypeError handles incorrect data types (i.e. subscripting an int)
                except (KeyError, TypeError):
                    print("No correction_sensor_R_FLU found for sensor " + name)

                try:
                    allSensors[name]["correction_rig_T"] = calibratedSensors[name]["correction_rig_T"]
                except (KeyError, TypeError):
                    print("No correction_rig_T found for sensor " + name)

                try:
                    allSensors[name]["properties"]["self-calibration"] = calibratedSensors[name]["properties"]["self-calibration"]
                except (KeyError, TypeError):
                    print("No properties.self-calibration found for sensor " + name)

                if useAbsolutePaths:
                    newParamString = ""
                    params = allSensors[name]["parameter"].split(",")

                    for param in params:
                        if param.startswith("file=") or param.startswith("video=") or param.startswith("timestamp="):
                            fileSplit = param.split("=")
                            pathToRig = os.path.abspath(rigName)
                            pathToRig = os.path.dirname(pathToRig)
                            param = fileSplit[0] + "="
                            param += pathToRig + "/"

                            # WAR for cases where virtual rig is inside the recorder
                            # config directory
                            if os.path.basename(pathToRig) == "config":
                                param += "../"

                            param += fileSplit[1]

                        if newParamString:
                            newParamString += ","

                        newParamString += param

                    allSensors[name]["parameter"] = newParamString

    sensorList = []
    for name, value in allSensors.items():
        sensorList.append(value)

    return sensorList


def getVehicleioSection(rigNames):
    for rigName in rigNames:
        with open(rigName, 'r') as inf:
            rig = json.load(inf)
            try:
                # We look for first non-empty vehicleio section
                vehicleio = rig['rig']['vehicleio']
                if len(vehicleio):
                    return vehicleio
            except KeyError as e:
                continue
    return []


def getPropertiesSection(rigNames):
    properties = {}
    for rigName in rigNames:
        with open(rigName, 'r') as inf:
            rig = json.load(inf)
            try:
                # Update with all values from the given rig
                properties.update(rig['rig']['properties'])
            except KeyError as e:
                continue
    return properties

# Yields a generator so wrap in dict()
def merge_two_dicts_gen(dict1, dict2):
    for k in set(dict1.keys()).union(dict2.keys()):
        if k in dict1 and k in dict2:
            if isinstance(dict1[k], dict) and isinstance(dict2[k], dict):
                yield (k, dict(merge_two_dicts_gen(dict1[k], dict2[k])))
            # If there is nothing to merge, use the value from dict2, same behavior as dict.update()
            else:
                yield (k, dict2[k])
        elif k in dict1:
            yield (k, dict1[k])
        else:
            yield (k, dict2[k])

def getVehicleSection(rigInputList, calibratedRig):
    base = {}
    for rigName in rigInputList:
        with open(rigName, 'r') as inf:
            rig = json.load(inf)
            base = dict(merge_two_dicts_gen(base, rig['rig'].get('vehicle', {})))
    with open(calibratedRig, 'r') as inf:
        rig = json.load(inf)
        base = dict(merge_two_dicts_gen(base, rig['rig'].get('vehicle', {})))
    return base

def generateOutput(calibratedRigName, inputRigList, outputRigName, useAbsolutePaths=False):
    # We search for common properties in priority order,
    # starting w/ calibrated rigs and then input rigs

    sortedRigs = []
    if calibratedRigName:
        sortedRigs.append(calibratedRigName)
    sortedRigs.extend(inputRigList)

    def loadRig(rigFilename):
        with open(rigFilename, 'r') as inf:
            return json.load(inf)

    # The jsons for each rig file
    rigs = list(map(loadRig, sortedRigs))
    topLevelKeys = reduce(lambda x,y: x.union(y), map(dict.keys, rigs), set())
    # The .rig json value in each rig file
    rigObjs = list(map(lambda x: x.get("rig", {}), rigs))
    rigKeys = reduce(lambda x,y: x.union(y), map(dict.keys, rigObjs), set())

    # Sensor merging is a special case
    if "sensors" in rigKeys:
        rigKeys.remove("sensors")
    # Vehicleio merging is a special case
    if "vehicleio" in rigKeys:
        rigKeys.remove("vehicleio")
    # Properties merging is a special case
    if "properties" in rigKeys:
        rigKeys.remove("properties")
    # We copy the values in .rig one by one, not the entire .rig value
    if "rig" in topLevelKeys:
        topLevelKeys.remove("rig")
    outputRig = {}
    outputRig["rig"] = {}
    # In general, we merge by copying the first found value for each key
    for k in topLevelKeys:
        outputRig[k] = getSectionFromJson(sortedRigs, k)
    for k in rigKeys:
        outputRig["rig"][k] = getSectionFromRig(sortedRigs, k)
    # Special sensor merging that uses the calibration rig values
    outputRig["rig"]["sensors"] = getSensorsSection(calibratedRigName, inputRigList, useAbsolutePaths)
    # Special vehicleio section, which might only exist in a single input file
    outputRig["rig"]["vehicleio"] = getVehicleioSection(inputRigList)
    # Special properties section which might have additional values in some files that all need to be consolidated
    outputRig["rig"]["properties"] = getPropertiesSection(inputRigList)
    # Special vehicle body section.
    if calibratedRigName:
        outputRig["rig"]["vehicle"] = getVehicleSection(inputRigList, calibratedRigName)
    
    with open(outputRigName, 'w') as outf:
        json.dump(outputRig, outf, indent=4, sort_keys=True)

def getCorrectedCalibrationFile(originalCalibratedRig):
    with open(originalCalibratedRig, 'r') as inf:
        rig = json.load(inf)
        try:
            if rig["calibration"]:
                rig["rig"] = rig["calibration"]
                del rig["calibration"]
        except KeyError:
            print('Can not find "calibration", just make a copy')
    fd, path = tempfile.mkstemp()
    with os.fdopen(fd, 'w') as tmp:
        json.dump(rig, tmp, indent = 4)

    return path

def usage():
    print("Combines multiple recorded rigs into a single rig.\n"
          "The tool can also superimposes calibration corrections from a calibrated rig, if provided\n\n"
          "Usage: " + sys.argv[0] + "\n"
          "     -i <rig-in0.json>                 # Input Rig 0\n"
          "     -i <rig-in1.json>                 # Input Rig 1\n"
          "     -i ...                            # Additional Input Rig\n"
          "     -o <rig-out.json>                 # Output Rig\n"
          "     [-c <rig-calibrated.json>]        # Calibrated Rig\n"
          "     [-a]                              # Creates absolute paths in filenames\n")

if __name__ == "__main__":
    # Check validity of inputs
    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:o:c:a", [])
    except getopt.GetoptError as err:
        print(str(err))

    outputRigName = ""
    calibratedRigName = ""
    inputRigList = []

    useAbsolutePaths = False

    for option, value in opts:
        if option in ("-i", "--input-rigs"):
            if not os.path.basename(value).endswith('.json'):
                print("Error: input is not a json file...skipping")
                continue

            inputRigList.append(value)
        elif option in ("-o", "--output-rig"):
            if os.path.basename(value).endswith('.json'):
                outputRigName = value
        elif option in ("-c", "--calibrated-rigs"):
            calibratedRigName = getCorrectedCalibrationFile(value)
        elif option in ("-a", "--absolute"):
            useAbsolutePaths = True

    if not outputRigName or not inputRigList:
        usage()
        exit(1)

    generateOutput(calibratedRigName, inputRigList, outputRigName, useAbsolutePaths)

    # Remove the temp file
    if calibratedRigName:
        os.remove(calibratedRigName)
