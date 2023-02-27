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

import subprocess
import shlex
import json
import glob
import os
import argparse
import errno


# TODO: Check on how to remove Magic numbers for command output parsing
# TODO: use Logger to log status of each step.
# TODO: Report if no connection to Remote.
# TODO: Handle errors for sh commands.
# TODO: Check on Maximum size to be checked.
    #Provide a Warning Level and an Error Level.
    # Less than 1 TB shall trhow an error for SSD

##############################################################################
##############################################################################
#       Module Parameters                                                    #
##############################################################################
##############################################################################
SSD_FAILURE_LIMIT_GB=500
SYSTEM_FAILURE_LIMIT_GB=3

# Default Configuration in case module is used directly
HYP8PROTOPLUS_DEFAULT_IPS_MNTPATH_DICT={'192.168.0.30': [u'/mnt/SSD1', u'/mnt/SSD2'],
'192.168.0.31': [u'/mnt/SSD1', u'/mnt/SSD2', u'/mnt/SSD3'],
'192.168.0.40': [u'/mnt/SSD1', u'/mnt/SSD2'],
'192.168.0.41': [u'/mnt/SSD1', u'/mnt/SSD2']
}


HYP7_DEFAULT_IPS_MNTPATH_DICT={'192.168.0.11': [u'/mnt/SSD1', u'/mnt/SSD2'],
}

##############################################################################
##############################################################################
# Local Helper Functions                                                     #
##############################################################################
##############################################################################

##############################################################################
#    #Function Name#: run_local_command():                                   #
#                    Python Function to run sh command locally and returns   #
#                    stdout and stderror                                     #
#    #Inputs:  <cmd>: string of command to be executed                       #
#                                                                            #
#    #outputs: <out>: stdout of the command run                              #
#              <err>: stderr of the command run                              #
##############################################################################
def run_local_command(cmd):
    cmd = shlex.split(cmd)
    process= subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
    out, err= process.communicate()

    return (out, err)
##############################################################################
#    #Function Name#: run_remote_command()                                   #
#                    Python Function to run sh command on a remote target and#
#                    returns stdout and stderror                             #
#    #Inputs:  <cmd>: string of command to be executed                       #
#              <ip> : IPv4 of the remote machine                             #
#              <username>: Valid username on remote machine to be used       #
#              <password>: Password of the username on the remote machine    #
#                                                                            #
#    #outputs: <out>: stdout of the command run                              #
#              <err>: stderr of the command run                              #
##############################################################################
def run_remote_command(cmd, ip, username, password):
    RemoteSSH= "sshpass -p "+ password +" ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "+username+"@"+ip+" -tt"
    out,err = run_local_command(RemoteSSH+" "+cmd)
    return (out, err)


##TODO: Rework for Formal Logging
def Log(LogMsg):
    print(LogMsg)



##############################################################################
##############################################################################
#  Class::StorageChecker                                                     #
#         A Self Cotnained Class that identifies storage chceks needed       #
#         to be performed  on a target address identified as local or remote #
##############################################################################
##############################################################################
class StorageChecker():

    ##########################################################################
    #    #Class Member Func#: Constructor()                                  #
    #                                                                        #
    #                                                                        #
    #    #Inputs:  <target>: string defining if Storage Checks are local or  #
    #                        on remote machine whose IP address defined by   #
    #                        the passed value#                               #
    #              <MountPaths> : Expected Mount Paths to be checked versus  #
    ##########################################################################
    def __init__(self, MountPaths,target="Local"):
        self.StorageStatus="UnKnown";
        self.disksOnSys=[]
        self.SsdCountOnSys=0
        self.SsdMountedCorrectly=0
        self.expectedMntPaths=MountPaths
        self.expectedCountOfSsd=len(MountPaths)
        if (target == "Local"):
            self.machine=target
        else:
            self.machine=target
            self.ip=target
            self.username = "nvidia"
            self.password = "nvidia"

        self.fill_DisksListOnSystem()
    ##########################################################################
    #    #Class Member Func#: fill_DisksListOnSystem()                       #
    #                         Responsible for extracting the information of  #
    #                         disks (SSDs and System) on the target machine  #
    #                         defined, filling a listed dictionary with info #
    #                         needed for testing                             #
    #                         (Disks Info extracted using "df --o")          #
    ##########################################################################
    #TODO: Find solution for Magic Numbers in extraction of data from "df" command
    def fill_DisksListOnSystem(self):
        cmd = "df --o "
        if self.machine == 'Local':
            out,err = run_local_command(cmd)
        else:
            out,err = run_remote_command(cmd,self.ip,self.username,self.password)


        for line in out.splitlines():
            line = line.split(" ")
            file_system = line[0]
            mount_path = line[-1].rstrip("\n")
            #Extract Consumed% and Avaialble Bytes from the line
            i=-2
            while ( (line[i] == '') or (line[i] == '-')):
                  i=i-1;
            consumed_percentage = line[i]
            i=i-1
            while ( (line[i] == '') or (line[i] == '-')):
                  i=i-1;
            avaialable_bytes = line[i]


            if(mount_path == "/") :
                diskformat=line[1]
                self.disksOnSys.append({"FileSystem":file_system,
                                        "MountPath":mount_path,
                                        "ConsumedPercentage":consumed_percentage,
                                        "AvailableBytes":avaialable_bytes,
                                        "Format":diskformat,
                                        "Status":" "})
            elif (mount_path.find("/mnt/SSD") != -1):
                diskformat=line[7]
                self.SsdCountOnSys+=1
                self.disksOnSys.append({"FileSystem":file_system,
                                        "MountPath":mount_path,
                                        "ConsumedPercentage":consumed_percentage,
                                        "AvailableBytes":avaialable_bytes,
                                        "Format": diskformat,
                                        "Status":" "})

        return()

    ##########################################################################
    #    #Class Member Func#: Test_Disks()                                   #
    #                         Runs following Checks on the extracted file    #
    #                         system list                                    #
    #                         1- Verify the space available on disk & format #
    #                            (based on type SSD / System)                #
    #                         2- Verify Mount path of the disk vs expected   #
    #                            paths received from Rig                     #
    #                            (Disk Consdired as mounted correct iff mount#
    #                             path matched expected)                     #
    #                                                                        #
    #                         Pre-Requisite                                  #
    #                      self.disksOnSys filled by "fill_DisksListOnSystem"#
    ##########################################################################
    def Test_Disks(self):
        for disk in self.disksOnSys:
            disk["Status" ] = "Pass"
            disk["ConsumedPercentage"]= int(disk["ConsumedPercentage"].rstrip("%"))
            disk["AvailableBytes"]= int(disk["AvailableBytes"])

            ret1 = self.verify_space_format(disk["AvailableBytes"],disk["MountPath"],disk["Format"])
            #Verify Mount path, Return is not used to verify since DDPS might have extra SSDs not for Recording
            #Check that all SSDs needed and mounted Correctly is done by verify_SsdCount
            ret2 = self.verify_MntPath(disk["MountPath" ])
            if (ret1 == False):
                disk["Status"] = "Failure"
                self.StorageStatus ="Failure"

            Log("Disk:" + disk["FileSystem"] + " mounted at: " + disk["MountPath" ] + " is consumed @: " + str(disk["ConsumedPercentage"]) + "%"+ ", available space of "+ str(int(disk["AvailableBytes"]/(1024*1024)))+ "GB, formatted as " + disk["Format" ]+ " and its Status is " + disk["Status" ])
            Log("Space and Format Check Status: "+str(ret1))
            Log("Mount Path Status: "+str(ret2) +"\n")

    ##########################################################################
    #    #Class Member Func#: verify_space_format()                          #
    #                         Verifies Space and format of passed disk based #
    #                         on whether it is SSD/System and return status  #
    ##########################################################################
    def verify_space_format(self,spaceavailable,mntpath,expectedformat):
        ret=False
        if( (mntpath  in self.expectedMntPaths) and (spaceavailable < (SSD_FAILURE_LIMIT_GB*1024*1024)) and (expectedformat == "ext4")  ):
            ret=False
        elif ( (mntpath == "/") and (spaceavailable < (SYSTEM_FAILURE_LIMIT_GB*1024*1024)) and (expectedformat == "ext4" )):
            ret=False
        else:
            ret=True
        return(ret)
    ##########################################################################
    #    #Class Member Func#: verify_MntPath()                               #
    #                         Verify Mount path of the disk vs expected      #
    #                            paths received from Rig                     #
    #                            (Disk Consdired as mounted correct iff mount#
    #                             path matched expected)                     #
    ##########################################################################
    def verify_MntPath(self, mntpath):
        ret=False
        if( (mntpath  in self.expectedMntPaths) ):
            ret=True
            self.SsdMountedCorrectly+=1
        elif( mntpath == "/"):
            ret=True
        else:
            ret=False
        return(ret)

    ##########################################################################
    #    #Class Member Func#: Test_SsdCount()                                #
    #                         Tests if the Number of correctly Mounted SSDs  #
    #                         is meeting expected count from the passed      #
    #                         list of mounts                                 #
    ##########################################################################
    def Test_SsdCount(self):
        ret=False
        if(self.SsdMountedCorrectly == self.expectedCountOfSsd):
            ret=True
        else:
            ret=False
            self.StorageStatus ="Failure"
        Log("Machine SSD Count Status: "+str(ret))
        return(ret)
    ##########################################################################
    #    #Class Member Func#: getSummaryMessage()                            #
    #                         Builds a textual summary for the status of the #
    #                         disks, could be returned to class driver       #
    ##########################################################################

    def getSummaryMessage(self):
        Message = "DDPX: ("+self.ip+  ") Storage Checks Status: "+self.StorageStatus +", with "+str(self.SsdMountedCorrectly)+ " SSDs mounted in correct path\n"
        for disk in self.disksOnSys:
            if disk["MountPath" ] == "/":
                Message = Message+ "        FileSystem: mounted at: " + disk["MountPath" ] + " with available space of "+ str(int(disk["AvailableBytes"]/(1024*1024)))+ "GB and its Status is " + disk["Status" ] + "\n"
            else:
                Message = Message+ "        Disk:" + disk["FileSystem"] + " mounted at: " + disk["MountPath" ] + " with available space of "+ str(int(disk["AvailableBytes"]/(1024*1024)))+ "GB, formatted as " + disk["Format" ]+ " and its Status is " + disk["Status" ] + "\n"

        return (Message)
    ###########################################################################
    #   #Class Member Func#: StorageTests()                                   #
    #                        Class main Function used to drive all the Storage#
    #                        Tests, and evaluate the status of the tests      #
    ###########################################################################
    def StorageTests(self):
        Log("\n******************************")
        Log("******************************")
        Log("On Machine: " + self.machine)
        self.Test_Disks()
        self.Test_SsdCount()


        if (self.StorageStatus != "Unknown") and (self.StorageStatus != "Failure"):
            self.StorageStatus = "Pass"
        Log("******************************")
        Log("******************************\n")




##############################################################################
##############################################################################
#  Class::ConfigurationHandler                                               #
#         A Self Cotnained Class that extracts List of IPs and SSDs mount    #
#         path based on a rig folder with Rig files                          #
#            -Rig files needs to match the pattern (IPv4_Whatevername.JSON)  #
#            -SSD mount paths need to be under the path of:                  #
#                    "rig:recorder_storage_paths" inside JSON file           #
##############################################################################
##############################################################################
#TODO: Handle Errors of Rig Files
#    (Not existing i.e. not meeting name standard)
#TODO:
class ConfigurationsHandler:


    ##########################################################################
    #    #Class Member Func#: constructor()                                  #
    ##########################################################################
    def __init__(self,path):
        self.IP_List = list()
        self.RigFilesList=list()
        self.SsdMntPath = list()
        self.RigFolderPath = path
        self.getJsonFileList()
        self.IpListExtractor()
        self.SsdCountAndMount()

    ##########################################################################
    #    #Class Member Func#: getJsonFileList()                              #
    #                         Builds list of Rig Files based on JSON files in#
    #                         Rig Folder                                     #
    ##########################################################################
    def getJsonFileList(self):
        #Build a list of all *.json files in Rig Folder
        JsonFilesList=glob.glob(self.RigFolderPath+"*.json")
        for file in JsonFilesList:
            if not file.endswith('_init.json'):
                self.RigFilesList.append(file)
                print("Rig Files are:"+file)

    ##########################################################################
    #    #Class Member Func#: IpListExtractor()                              #
    #                         Builds IP list based on JSON Files in Rig      #
    #                         folder                                         #
    #                                                                        #
    #                         Pre-Requisite                                  #
    #                         self.RigFilesLis filled by "getJsonFileList"   #
    ##########################################################################
    def IpListExtractor(self):
        for file in self.RigFilesList:
            file =file.replace(self.RigFolderPath,"")
            ip=file.split("_")
            self.IP_List.append(ip[0])

    ##########################################################################
    #    #Class Member Func#: SsdCountAndMount()                             #
    #                         Builds Mount paths list based on JSON Files in #
    #                         Rig folder                                     #
    #                                                                        #
    #                         Pre-Requisite                                  #
    #                         self.RigFilesLis filled by "getJsonFileList"   #
    ##########################################################################
    def SsdCountAndMount(self):
        for file in self.RigFilesList:
            f=open(file,"r")
            jsonfile=json.load(f)
            self.SsdMntPath.append(jsonfile["rig"]["recorder_storage_paths"])
    ##########################################################################
    #    #Class Member Func#: getIpsAndMntPaths()                            #
    #                         Returns a Buit Dictionary of IPs vs List of SSD#
    #                         Mount Paths                                    #
    #                                                                        #
    #                         Pre-Requisite                                  #
    #                         SsdCountAndMount() filled  self.SsdMntPath     #
    #                         IpListExtractor()  filled  self.IP_List        #
    ##########################################################################
    def getIpsAndMntPaths(self):
        return(dict(zip(self.IP_List,self.SsdMntPath)) )




##############################################################################
##############################################################################
#  Module Main Functions:                                                    #
##############################################################################
##############################################################################




##############################################################################
#    #Function Name#: StorageCheck(path):                                    #
#                     Main function to be called by External Modules         #
#    #Inputs:  <path>: Rig Folder path to include the rig files              #
#                                                                            #
#    #outputs: <status>: Boolean indicating the pass of all checks or fail of#
#                        one check                                           #
#              <Message>: Summary Message to be used by Module Driver        #
##############################################################################
def StorageCheck(path):

    StorageStatus=list()
    MachineInstance=list()
    #Use Rig Folder passed to Know List of IPs on which Sotorage Tests shall be
    # run on and expected mounted SSD paths in each IP
    Configurations = ConfigurationsHandler(path)
    Ips_MntPath_Dict=Configurations.getIpsAndMntPaths()

    # For each IP create an instance of StorageChecker Class, run the tests and
    # store the instance to get further details if needed.
    for Ip,MntPaths in Ips_MntPath_Dict.items():
        instance=StorageChecker(MntPaths,Ip)
        instance.StorageTests()
        MachineInstance.append(instance)
        StorageStatus.append(instance.StorageStatus)
    # Build an overall status for all tests to know if Recorder can start or not.
    if ( ("Failure" in StorageStatus) or ( "UnKnown" in StorageStatus)):
        status = False
    else:
        status = True


    # Build a Summary Message to be returned and to be viewed to used if needed.
    Message="Storage Checks have been perfromed on Following DDPXs, checking for:\
            \n 1- SSDs needed by Chosen RIG Folder ("+ path + ") are connected to DDPXs and mounted in correct path \
            \n 2- "+str(SSD_FAILURE_LIMIT_GB)+"GB of space available on SSDs and "+str(SYSTEM_FAILURE_LIMIT_GB)+"GB on System Memory\
            \n 3- SSDs are in the correct Format \n \n"
    for instance in MachineInstance:
        Message += instance.getSummaryMessage()
        Message += "\n"


    return (status,Message)

##############################################################################
#    #Function Name#: ():                      #
#                                                                            #
##############################################################################
def RunHardCodedCheck(config):
    StorageStatus=list()
    MachineInstance=list()
    status = False
    if (config == "Hyp8+"):
        DicToBeUsed= HYP8PROTOPLUS_DEFAULT_IPS_MNTPATH_DICT
    elif (config == "Hyp7"):
        DicToBeUsed= HYP7_DEFAULT_IPS_MNTPATH_DICT
    else:
        DicToBeUsed= HYP8PROTOPLUS_DEFAULT_IPS_MNTPATH_DICT

    for Ip,MntPaths in DicToBeUsed.items():
        instance=StorageChecker(MntPaths,Ip)
        instance.StorageTests()
        MachineInstance.append(instance)
        StorageStatus.append(instance.StorageStatus)
        print("Machine Status is: "+instance.StorageStatus)
    # Build an overall status for all tests to know if Recorder can start or not.
    if ( ("Failure" in StorageStatus) or ( "UnKnown" in StorageStatus)):
        status = False
    else:
        status = True

    return (status)



##############################################################################
#    #Function Name#: Main():                                                #
#                     Default Function to be called in case of Module called #
##############################################################################
def main():
    status = False
    message = " "
    CommandLineArgs = argparse.ArgumentParser("Storage Checker based on Rigs provided")
    CommandLineArgs.add_argument('-p','--path',\
                                  help = 'Path to Folder with JSON\
                                  Configurations to use for\
                                  Storage Checks, should be\
                                  the same as used for the recording\
                                  (Hyp8 can be used as an example).')
    CommandLineArgs.add_argument('-o','--OutStatus',\
                                  help = 'Status File Used to store Storage Test Status\
                                  [preferable located in temp folder]         \
                                  [used for Inter-Process Communication].')

    CommandLineArgs.add_argument('-c','--config',\
                                 help = 'Use Default Hardcoded Confg (<Hyp8+> or <Hyp7>)\
                                 [cannot be used with -p as Config of Json\
                                 File takes precedence].')

    arguments = CommandLineArgs.parse_args()

    if (arguments.path != '') and (arguments.path != None):
        print("Rig path used for Stoarage Check is: "+arguments.path )
        status,message = StorageCheck(arguments.path)
    else:
        if (arguments.config == "Hyp8+"):
            status = RunHardCodedCheck("Hyp8+")
        elif(arguments.config == "Hyp7"):
            status= RunHardCodedCheck("Hyp7")
        else:
            status = RunHardCodedCheck("Hyp8+")

    if (arguments.OutStatus != '') and (arguments.OutStatus != None):
        try:
            fd = open(arguments.OutStatus,"w")
            if status==True:
                fd.write("True\n")
            else:
                fd.write("False\n")
            print (message)
            fd.close()
        except OSError as e:
            if (e.errno != errno.EEXIST):
                print("Error:"+os.strerror(e))


if __name__ == '__main__':
    main()
