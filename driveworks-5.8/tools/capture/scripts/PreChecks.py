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
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tkinter import *
from tkinter import messagebox
import collections
import logging
import os
import sys
import filecmp
import subprocess
from datetime import datetime
import shlex

class Logger():
    def __init__(self):
        format_string = ' %(asctime)-20s: %(levelname)-10s: %(message)s'
        logging.basicConfig(level=logging.DEBUG, filename="/tmp/Recorder-precheck_results.log",
							format=format_string,
                            filemode='w')
        terminal_logger = logging.StreamHandler()
        terminal_logger.setFormatter(logging.Formatter(format_string))
        logging.getLogger().addHandler(terminal_logger)


class PreChecker():
    def __init__(self):
        self.testlist = []
        self.testresults = collections.defaultdict()
        self.logger = Logger()


    def PrepTestList(self):
        self.VINtests = [
                {"VIN-FILE":"Check if VIN file exists in /tmp and /etc"},
                {"VIN-CONST": "Check if vin file content is same in /tmp and /etc"},
                {"VIN-VALID":"Check if VIN is all 0"},
                {"VIN-LEN":"Check VIN length is 17 chars"}
        ]
        self.BOMtests = [
                {"BOM-FILE":"Check if BRBCT json exists in persistent storage"},
                {"BOM-VALID": "Check if BRBCT json has a valid/non-empty BOMid"},
                {"BOM-LEN":"Validate BOMid length"}
        ]
        self.datetimetests = [
                {"DATE-GPS":"Acquire UTC date/time from GPS packets"},
                {"DATE-PTP": "Compare PTP synced system datetime with GPS time"},
                {"DATE-RTC": "Compare RTC datetime with GPS time"}
        ]
        self.testlist = self.VINtests + self.BOMtests + self.datetimetests
        return self.testlist

    def ExecuteTests(self, testlist):
        self.testlist = testlist
        for index in range(len(self.testlist)):
            for key,value in (self.testlist[index]).items():
                if "VIN" in key or "BOM" in key or "DATE" in key:
                    testid = key
                    testdescription = value

                    # Main instance where test cases are launched
                    runner = TestRunner(testid, testdescription, self.testresults)
                    runner.StartTest()
        # Populating summary for displaying on screen
        summary = runner.populate_test_summary()

        # Printing test summary on console and saving to log file
        runner.print_test_summary()
        return summary

    def tabulate_results(self, summary):
        self.summary = summary
        # take the data
        lst = [('Test Case', 'Result')]
        lst.extend(self.summary)
        fail_flag=0

        # find total number of rows and
        # columns in list
        total_rows = len(lst)
        total_columns = len(lst[0])

        # create root window
        root = Tk()
        root.title("Recorder Pre-Checker Summary")
        #root.geometry("500x500")  # Size of the window
        t = TestResultSummary(root,lst,total_rows,total_columns,fail_flag)
        is_fail = t.draw_result()

        if is_fail:
            messagebox.showerror("ERROR!!!", "Recorder Pre-checks Failed! Cannot launch Recorder")
            root.withdraw()
            root.destroy()
            ret = False
        else:
            root.withdraw()
            root.destroy()
            ret = True
        return ret


class Utils():
    def __init__(self):
        self.log = logging.getLogger()

    def run_subprocess(self, args=list, mute=False):
        if not mute:
            self.log.info("Executing command : {}".format(" ".join(args)))
        result = subprocess.run(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return [result.stdout.decode('ascii', 'ignore'),
                result.stderr.decode('ascii', 'ignore')]
    def run_pmc(self,ip):
        ssh_command = "sshpass -p nvidia ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null nvidia@"+ip
        pmc_command = ' sudo ~/drive-t186ref-linux/samples/nvavb/daemons/pmc -u -b 0'

        command=shlex.split(ssh_command+pmc_command)
        command.append('"GET TIME_STATUS_NP"')

        output_lines= subprocess.run(command,universal_newlines=True,stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
        output_lines = output_lines.split("\n")

        self_mac="0:0:0:0:0:0"
        gm_mac="ff:ff:ff:ff:ff:ff"
        ingress_time="0"
        #assuming ptp_service running till proven else
        ptp_service_running=True

        for line in output_lines:
            if "No such file or directory" in line:
                ptp_service_running=False
            if "RESPONSE" in line:
                self_mac = line.split(" ")[0].split("-")[0].lstrip("\t")
            if "ingress_time" in line:
                ingress_time = line.split(" ")[-1]
            if "gmIdentity" in line:
                gm_mac = line.split(" ")[-1]

        return ptp_service_running, self_mac, gm_mac, ingress_time

class TestCases():
    def __init__(self, utils_class):
        self.dwpath="/usr/local/driveworks"
        self.vinpath_primary="/tmp/car_vin"
        self.vinpath_secondary="/etc/car_vin"
        self.brbct_json="/storage/version/brbct_data.json"
        self.args = []
        self.utils = utils_class
        self.log = logging.getLogger()
        self.dateptp = None
        self.dategps = None
        self.datertc = None

    def test_VINFILE(self):
        ret = True
        if not (os.path.exists(self.vinpath_primary)):
            self.log.error("VIN file missing in " + self.vinpath_primary)
            ret = False

        if not (os.path.exists(self.vinpath_secondary)):
            self.log.error("VIN file missing in " + self.vinpath_secondary)
            ret = False
        return ret
    def test_VINCONST(self):
        if not filecmp.cmp(self.vinpath_primary, self.vinpath_secondary):
            self.log.error("VIN contents not same in /tmp/car_vin and /etc/car_vin")
            return False
        return True

    def test_VINVALID(self):
        if not (os.path.getsize(self.vinpath_primary)):
            self.log.error("VIN files empty")
            return False
        else:
            with open(self.vinpath_primary) as f:
                if "000000" in f.read():
                    self.log.error("VIN file is all zeroes")
                    return False
        return True

    def test_VINLEN(self):
        with open(self.vinpath_primary) as f:
                if len(f.read().rstrip('\n')) != 17:
                    self.log.error("VIN length not equal to 17 chars")
                    return False
        return True

    def test_BOMFILE(self):
        ret = True
        if not (os.path.exists(self.brbct_json)):
            self.log.error("BRBCT version file missing in " + self.brbct_json)
            ret = False
        return ret

    def test_BOMVALID(self):
        if not (os.path.getsize(self.brbct_json)):
            self.log.error("BRBCT file is empty")
            return False
        try:
            output = subprocess.check_output("grep bomId /storage/version/brbct_data.json | cut -d '\"' -f4", shell=True)
            output = output.decode("utf-8").rstrip("\n")
            if not output:
                self.log.error("bomId string empty")
                return False
        except subprocess.CalledProcessError as e:
            self.log.error(e.output)
            return False
        return True

    def test_BOMLEN(self):
        output = subprocess.check_output("grep bomId /storage/version/brbct_data.json | cut -d '\"' -f4", shell=True)
        output = output.decode("utf-8")
        if (len(output.rstrip("\n")) != 29) and (len(output.rstrip("\n")) != 31):
            self.log.error('bomId length incorrect')
            return False
        return True

    def test_DATEGPS(self):
        # TODO: Using utcnow as ref, to replace with acquiring UTC time from GPS packets
        self.dategps = datetime.utcnow().strftime("%a %B %d %H%M")
        self.log.info("GPS datetime: " + self.dategps)
        return True

    def test_DATEPTP(self):
        gm_ip=[]
        slave_ip=[]
        ip_list=["192.168.0.30","192.168.0.31","192.168.0.40","192.168.0.41"]
        ret = True
        for ip in ip_list:
            ptp_running_1, self_mac1, gm_mac1, ingress_time1 = self.utils.run_pmc(ip)
            ptp_running_2,  self_mac2, gm_mac2, ingress_time2 = self.utils.run_pmc(ip)

            if ptp_running_1==True and ptp_running_2==True :
                #Ingress Time is not a zero and increasing, then PTP is synched
                #Not relying on gmPresent and gmIdentiy, as they are not consistent
                #in behavior always.
                if (int(ingress_time1) !=0 and int(ingress_time2)!=0):
                    if (int(ingress_time1) < int(ingress_time2)):
                            self.log.info("PTP slave:"+ip+" synched")
                            slave_ip.append(ip)
                #Ingress Time is zero or not incrementing, Is the node a GM?
                elif(self_mac1!=gm_mac1):
                    # Ingress Time Equal to Zero and the GM is not self
                    # it is case of ptp synched then lost (ingress_time =0 &Gm identified)
                    # GM identified could be old one or new slave announced as GM
                    # but still no synch
                    self.log.error("PTP slave:"+ip +" not synched")
                    ret = False
                elif (self_mac1 ==gm_mac1):
                    self.log.info("Grandmaster detected on Ip:"+ip)
                    if (len(gm_ip) ==0):
                        gm_ip.append(ip)
                    else:
                        #MultipleGM, which means possible
                        #Bad PTP
                        self.log.error("Multiple Grandmaster, PTP dysfunction")
                        ret = False
            else:
                self.log.error("PTP Service not running on Ip: "+ ip)
                ret = False

        if ( (len(gm_ip) >1) or (len(slave_ip) < 3) or (ret ==False) ):
            ret = False
        return (ret)


    def test_DATERTC(self):
        output, error = self.utils.run_subprocess(['sudo', 'bash', '/usr/local/sbin/canrtc.sh', 'get'])
        if error !="":
            self.log.error(error)
            return False

        output = output.split('\n')
        output = output[-2]

        self.datertc = output
        self.test_DATEGPS()
        self.datertc = datetime.strptime(self.datertc, '%Y/%m/%d %H:%M:%S')
        self.datertc = self.datertc.strftime("%a %B %d %H%M")
        self.log.info("RTC datetime: " + self.datertc)
        if self.datertc != self.dategps:
            self.log.error("RTC datetime doesn't match datetime in GPS packets")
            return False
        return True


class TestRunner():
    def __init__(self, t_id, t_description, t_results):
        self.t_id = t_id
        self.t_description = t_description
        self.testresults = t_results
        self.utils = Utils()
        self.log = logging.getLogger()
        self.testcase = TestCases(self.utils)
        self.summary = []


    def StartTest(self):
        self.log.info("Starting test ID: %s Description: :%s " % (self.t_id, self.t_description))
        self.testresults[self.t_id] = "FAIL"

        if self.t_id == "VIN-FILE":
            if self.testcase.test_VINFILE():
                 self.testresults[self.t_id] = "PASS"
        if self.testresults["VIN-FILE"] == "PASS":
            if self.t_id == "VIN-CONST":
                if self.testcase.test_VINCONST():
                    self.testresults[self.t_id] = "PASS"
            if self.t_id == "VIN-VALID":
                if self.testcase.test_VINVALID():
                    self.testresults[self.t_id] = "PASS"
            if self.t_id == "VIN-LEN":
                if self.testcase.test_VINLEN():
                    self.testresults[self.t_id] = "PASS"
        if self.t_id == "BOM-FILE":
            if self.testcase.test_BOMFILE():
                self.testresults[self.t_id] = "PASS"
        if self.t_id == "BOM-VALID":
            if self.testresults["BOM-FILE"] == "PASS":
                if self.testcase.test_BOMVALID():
                    self.testresults[self.t_id] = "PASS"
        if self.t_id == "BOM-LEN":
            if self.testresults["BOM-FILE"] == "PASS":
                if self.testcase.test_BOMLEN():
                    self.testresults[self.t_id] = "PASS"
        if self.t_id == "DATE-GPS":
            if self.testcase.test_DATEGPS():
                 self.testresults[self.t_id] = "PASS"
        if self.t_id == "DATE-PTP":
            if self.testcase.test_DATEPTP():
                self.testresults[self.t_id] = "PASS"
        if self.t_id == "DATE-RTC":
            if self.testcase.test_DATERTC():
                 self.testresults[self.t_id] = "PASS"

    def populate_test_summary(self):
        for item in self.testresults.keys():
            tup = (item, self.testresults[item])
            self.summary.append(tup)
        return self.summary


    def print_test_summary(self):
        self.log.info("=======================================================")
        self.log.info("PRINTING TEST SUMMARY")
        self.log.info("=======================================================")
        for item in self.testresults.keys():
            self.log.info('{0:35} : {1:20} '.format(
                item, self.testresults[item]))
        return

class TestResultSummary:
    def __init__(self,root,lst,total_rows,total_columns,fail_flag):
        self.is_fail = fail_flag
        self.root = root
        self.lst = lst
        self.total_rows = total_rows
        self.total_columns = total_columns

    def draw_result(self):

        # code for creating table
        for i in range(self.total_rows):
            for j in range(self.total_columns):
                if self.lst[i][1] in 'PASS':
                    fg_color='green'
                elif self.lst[i][1] in 'FAIL':
                    self.is_fail = 1
                    fg_color='red'
                else:
                    fg_color='black'
                self.e = Entry(self.root, width=25, fg='black', bg='white',
                               font=('Calibri',12), justify='center')
                self.e1 = Entry(self.root, width=15, fg=fg_color, bg='white',
                               font=('Calibri',12), justify='center')
                if j == 0:
                    self.e.grid(row=i, column=j)
                    self.e.insert(END, self.lst[i][j])
                else:
                    self.e1.grid(row=i,column=j)
                    self.e1.config(fg=fg_color)
                    self.e1.insert(END, self.lst[i][j])
        return self.is_fail

def main():
    result = False
    check = PreChecker()
    testlist = check.PrepTestList()
    results = check.ExecuteTests(testlist)
    result = check.tabulate_results(results)
    return result

if __name__ == '__main__':
    main()
