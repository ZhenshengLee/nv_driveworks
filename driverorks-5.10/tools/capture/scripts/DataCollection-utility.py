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
# SPDX-FileCopyrightText: Copyright (c) 2020-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import datetime
from datetime import timedelta,timezone
from functools import reduce
import glob
import json, sys, getopt, time
import logging
import os, os.path
from shutil import copyfile
import subprocess
import tkinter.font as Font
from tkinter import *
import tkinter
from StorageCheck import *
import PreChecks

class DataMapping():
    def __init__(self):
        self.vin2config = {}
        self.BomId2config = {}
        self.AllAvailableConfigs=[]
        self.populateVinMappings()
        self.populateBomIdMappings()
        accumulated_configs = []
        for value in self.BomId2config.values():
            accumulated_configs.extend(value)
        for value in self.vin2config.values():
            accumulated_configs.extend(value)
        self.AllAvailableConfigs = list(set(accumulated_configs))

    def populateBomIdMappings(self):
        filelist = glob.glob("/usr/local/driveworks/tools/capture/configs/campaign/config_mapping/*_bom_recording_config_mapping.json")
        for file in filelist:
            fd=open(file,"r")
            JsonStructure=json.load(fd)
            self.BomId2config.update(JsonStructure["Bom2ConfigMapping"])
        accumulated_configs = []
        for value in self.BomId2config.values():
            accumulated_configs.extend(value)
        self.BomId2config['None'] = list(set(accumulated_configs))

    def populateVinMappings(self):
        ##############################################################################################
        #  VINs are not supposed to be populated anymore, Only use BOM IDs                           #
        #  VINs are left only for backward compatability (i.e. Cars without BomIDs)                  #
        ##############################################################################################
        vinfilelist = glob.glob("/usr/local/driveworks/tools/capture/configs/campaign/config_mapping/*_vin_recording_config_mapping.json")
        vinConfigs={}
        for file in vinfilelist:
            fd=open(file,"r")
            JsonStructure=json.load(fd)
            self.vin2config.update(JsonStructure["Vin2ConfigMapping"])
        ##############################################################################################
        #  VINs are not supposed to be populated anymore, Only use BOM IDs                           #
        #  VINs are left only for backward compatability (i.e. Cars without BomIDs)                  #
        ##############################################################################################
        accumulated_configs = []
        for value in self.vin2config.values():
            accumulated_configs.extend(value)
        self.vin2config['None'] = list(set(accumulated_configs))
        return

    def getConfigByVin(self,vin):
        if vin in self.vin2config:
            return self.vin2config[vin]
        else:
            return self.AllAvailableConfigs

    def getConfigByBomId(self,bomId):
        if bomId in self.BomId2config:
            config = self.BomId2config[bomId]
            if config == []:
                return self.AllAvailableConfigs
            else:
                return config
        else:
            return self.AllAvailableConfigs

    def get_configs(self,bomId,vin):
        if (bomId == "Un-defined") or (bomId == ""):

            return (self.getConfigByVin(vin))
        else:
            return (self.getConfigByBomId(bomId))

class RenderPopUp():
    def __init__(self, recorder_config):
        # Create a window with size 600x400
        self.window = tkinter.Tk()
        self.window.title('Choose Recorder Configuration')
        self.window.geometry('600x400')

        self.recorder_options = recorder_config
        self.tag_options = getTagOptions()
        if len(self.tag_options) is 1 and "CALIBRATION" in self.tag_options:
            DisplayCalibrationTimeMessageBox()

        self.chosenConfig = None
        self.chosenTag = None

    def Render(self):

        # Add a grid
        mainframe = Frame(self.window)
        mainframe.grid(column=100,row=0, sticky=(N,W,E,S))
        mainframe.columnconfigure(0, weight = 1)
        mainframe.rowconfigure(0, weight = 1)
        mainframe.pack(pady = 100, padx = 100)
        text = tkinter.Text(self.window)
        text.configure(font=('Helevetica',20))

        # Create a variable for storing recorder config
        self.recorder_config = StringVar(self.window)
        self.recorder_config.set(self.recorder_options[0])

        # Create a variable for storing tag
        self.tag = StringVar(self.window)
        self.tag.set(self.tag_options[0])

        # Create a dropdown for recorder configs
        popupMenu1 = OptionMenu(mainframe, self.recorder_config, *self.recorder_options)
        label1 = Label(mainframe, text="Choose a recorder config")
        label1.grid(row = 1, column = 1)
        label1.config(font=('Helvetica', 15))
        popupMenu1.grid(row = 3, column =1)
        popupMenu1.config(font=('Helvetica', 20))
        widget = self.window.nametowidget(popupMenu1.menuname)
        widget.config(font=('Helvetica', 20))

        # Create a dropdown for tag
        popupMenu2 = OptionMenu(mainframe, self.tag, *self.tag_options)
        label2 = Label(mainframe, text="Choose a tag")
        label2.grid(row = 10, column = 1)
        label2.config(font=('Helvetica', 15))
        popupMenu2.grid(row = 15, column =1)
        popupMenu2.config(font=('Helvetica', 20))
        widget = self.window.nametowidget(popupMenu2.menuname)
        widget.config(font=('Helvetica', 20))

        # Create a Launch button
        button = tkinter.Button(self.window, text='Launch', width=25, command=self.LaunchConfig)
        button.pack()
        button.config(font=('Helevetica',20))

        self.recorder_config.trace('w', self.change_dropdown1)
        self.window.mainloop()
        return [self.chosenConfig,self.chosenTag]

    def change_dropdown1(self, *args):
        self.chosenConfig = self.recorder_config.get()

    def change_dropdown2(self, *args):
        self.chosenTag = self.tag.get()

    def LaunchConfig(self):
        self.chosenConfig = self.recorder_config.get()
        self.chosenTag = self.tag.get()
        self.window.destroy()

def DisplayStorageCheckerMessageBox(Msg):
    rootwindow = tkinter.Tk()
    T = tkinter.Text(rootwindow, height=30, width=160)
    T.pack()
    T.insert(tkinter.END,Msg)
    tkinter.messagebox.showerror("Storage Checker Tests", "Storage Checker Tests Failed")
    tkinter.mainloop()

def DisplayCalibrationTimeMessageBox():
    tkinter.messagebox.showwarning("Calibration Time Check", "Calibration missing or old. Please perform a drive with the Calibration procedure")

def getTagOptions():
    log.info("Checking time of last calibration...")
    cal_time_file = '/storage/recorder/calibration_time'
    current_time = datetime.datetime.now(timezone.utc)
    cal_time_hours = 24

    try:
        with open(cal_time_file) as f:
            out = f.readlines()
            out = out[0]
        print(out)
        calibration_time = datetime.datetime.fromtimestamp(float(out), tz=timezone.utc)
    except:
        calibration_time = datetime.datetime.fromtimestamp(0, tz=timezone.utc)

    log.info("Last Calibration time: {}".format(calibration_time))
    if current_time - timedelta(hours=cal_time_hours) > calibration_time:
        tag_list = ["CALIBRATION"]
    else:
        tag_list = ["DATA-COLLECTION", "CALIBRATION", "DEV"]

    return tag_list

def CheckVINOverlays(config):
    log.info("Check if VIN overlays are picked successfully")
    # Create dummy folder to run rig_json2json tool
    os.system('mkdir -p /tmp/vin-overlay/')
    configpath = '/usr/local/driveworks/tools/capture/configs/campaign/'
    src = os.path.join(configpath, config, '192.168.0.30_'+config+'.json')
    copyfile(src, '/tmp/vin-overlay/temp.json')

    p1 = subprocess.Popen(['/usr/local/driveworks/tools/sensors/rig_json2json', '/tmp/vin-overlay/temp.json'], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(["grep", "Rig overlay applied"], stdin=p1.stdout, stdout=subprocess.PIPE)
    output = p2.communicate()[0]
    output = output.decode("utf-8")
    if not output:
        log.warning("VIN overlays NOT picked, using default")
        return False
    else:
        log.info("VIN overlays applied successfully")
        return True

if __name__ == "__main__":

    # Create a logger for debugging
    format_string = ' %(asctime)-20s: %(levelname)-10s: %(message)s'
    logging.basicConfig(level=logging.DEBUG,
                            filename="/tmp/DataCollect_Utility.log",
                            format=format_string,
                            filemode='w')
    terminal_logger = logging.StreamHandler()
    terminal_logger.setFormatter(logging.Formatter(format_string))
    logging.getLogger().addHandler(terminal_logger)
    log = logging.getLogger()


    # Run recorder-prechecks before attempting to launch recorder
    log.info('Launching Recorder Pre-checks')
    if not PreChecks.main():
        log.error('Recorder Prechecks Failed. Cannot launch Recorder-TUI')
        sys.exit()

    VIN = "123456" # Add a default VIN
    BomId = "Un-defined"    # Un-defined
    if os.path.exists('/storage/version/brbct_data.json'):
        BrbctHandler = open('/storage/version/brbct_data.json', 'r')
        Brbct= json.load(BrbctHandler)
        BomId = Brbct['bomId']
        VinFull = Brbct['VIN']
        VIN = VinFull.strip()[-6:]
        BrbctHandler.close()

    # Read VIN if available
    elif os.path.exists('/etc/car_vin'):
        identifier = open('/etc/car_vin', 'r')
        VIN = identifier.read().strip()[-6:]
        identifier.close()

    log.info("Chosen VIN is {}".format(VIN))
    log.info("Chosen BomId is {}".format(BomId))
    data_mapper = DataMapping()
    config_options = data_mapper.get_configs(BomId,VIN)
    # Get Config, Tag from user
    Render = RenderPopUp(config_options)
    config,tag = Render.Render()
    log.info('Chosen config is {} and chosen tag is {}'.format(config,tag))
    Recorder_Path = '/usr/local/driveworks/tools/capture/recorder-qtgui'
    Recorder_Config_Path = '/usr/local/driveworks/tools/capture/configs/campaign/'

    # Check if VIN overlays are picked
    if not CheckVINOverlays(config):
        log.warning("VIN Overlays are not applied, using default")

    log.info("Launching {}".format([Recorder_Path, Recorder_Config_Path+config+'/', '--tag', tag]))
    StorageCheckRet,StorageCheckMsg=StorageCheck(Recorder_Config_Path+config+'/')
    if(StorageCheckRet != True):
        DisplayStorageCheckerMessageBox(StorageCheckMsg)
        log.info(StorageCheckMsg)
        log.error("Storage Checks failed, cannot launch recorder")
        sys.exit()
    try:
        subprocess.Popen([Recorder_Path, Recorder_Config_Path+config+'/', '--tag', tag])
    except OSError as e:
        log.error(e)

    log.info("RecorderQT Launch Completed")
