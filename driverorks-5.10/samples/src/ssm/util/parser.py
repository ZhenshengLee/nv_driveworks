#!/usr/bin/env python3

'''
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
// SPDX-FileCopyrightText: Copyright (c) 2016-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.
//
// Parser Version: 0.7.5
// SSM Version:    0.9.3
//
/////////////////////////////////////////////////////////////////////////////////////////
'''

import json
from pydoc import classname
import shutil
import os
import sys
import argparse
from collections import namedtuple
from collections import defaultdict
import re

from pprint import pprint

# DO NOT remove the spaces in the version line
PARSER_VERSION = "0.7.5"
SSM_VERSION = "0.9.1"

SMs = {}
g_hierarchies = {}
Hierarchy = namedtuple("Hierarchy", ["ParentState", "ChildSM"])
HierarchialTransitions = namedtuple("HierarchialTransitions", ["ParentState", "ChildSM", "ChildStateSrc", "ChildStateDst"])
g_head = ""
swcList = []
lsCommandArr = [];
SwcIgnoreFile = ""
SwcListFile = ""
StartPort = ""
CreateHeadSMChannels = "true"
SwcDebug = False
namespace = ""
cloneName="SSMClone"
copyrightData=""
SPD="    "
globalLabelArray = []
jsonFileName = ""
subcomponent_include_directory = ""
subcomponent_src_directory = ""

class StateMachine(object):
    """
    Custom data-structure to store the state machine attributes.

    Attributes
    ----------
    m_name:
        Name of the satte machine
    m_ipaddress:
        Default IP address in multi-soc environment
    m_states:
        List of state names
    m_transitions:
        Dict of teansitions.
        key: from_state
        value: list of to_states
    m_hierarchicalTransition:
        List of transitions that a parent can initiate over children
    m_isHead: boolean
        m_isHead is set to true if the state machine is the head state machine
    overrideInit: boolean
        Set to true if the state machine should skip the lock stepped init process
    hasClone: boolean
        hasClone is true when the State machine has a clone
    """

    def __init__(self, name, states, transistions, startState, ipaddress, overrideInit, hasClone):

        global globalLabelArray

        globalLabelArray.append("\n")
        globalLabelArray.append("/// Labels for "+name)
        globalLabelArray.append("#define SSM_"+name+"_str \""+name +"\"")
        for st in states:
            globalLabelArray.append("#define SSM_"+name+"_"+st+"_str \""+st +"\"")
        globalLabelArray.append("#define SSM_"+name+"_STATES_COUNT " +str(len(states)))
        

        self.m_name = name
        self.m_ipaddress = ipaddress
        self.m_states = sorted(set(states))
        self.m_transitions = defaultdict(list)
        self.m_hierarchies = {}
        self.m_hierarchicalTransition = []
        self.m_isHead = False
        self.overrideInit = overrideInit
        self.hasClone = hasClone

        for transistion in transistions:
            if "from" not in transistion or "to" not in transistion:
                raise Exception("From or to tags not found ")
            if transistion["from"] not in self.m_states:
                raise Exception("Invalid src state found: " + transistion["from"])

            for destination_state in transistion["to"]:
                if destination_state not in self.m_states:
                    raise Exception("Invalid dst state found: " + destination_state)
                self.m_transitions[transistion["from"]].append(destination_state)

        if(startState not in self.m_states):
            raise Exception("Invalid start state found: " + startState)
        else:
            self.m_startState = startState

    def printState(self):
        print ("name: " + self.m_name)
        print ("startState: " + self.m_startState)
        print ("States: ")
        for st in self.m_states:
            print (" " + st)
        print ("Transitions: ")
        for t in self.m_transitions:
            print (" [" + t + "]: " + self.m_transitions[t])
        print ("-----")

def parseJson(file):
    """
    This function parses the JSON file and creates StateMachine objects.
    The function also updates global variables.

    Parameters
    ----------
        file: JSON file with the state machine framework definition
    """
    global namespace
    global swcList
    global SwcIgnoreFile
    global SwcListFile
    global StartPort
    global CreateHeadSMChannels
    global SwcDebug
    global SMs
    global lsCommandArr
    global g_head

    smListByName = []
    with open(file) as f:
        object = json.load(f)

    if 'SwcList' in object:
        for swc in object["SwcList"]:
            swcList.append(swc)

    if 'SwcIgnoreFile' in object:
        SwcIgnoreFile = object["SwcIgnoreFile"]

    if 'SwcListFile' in object:
        SwcListFile = object["SwcListFile"]

    if 'StartPort' in object:
        StartPort = object["StartPort"];

    if 'CreateHeadSMChannels' in object:
        CreateHeadSMChannels = object["CreateHeadSMChannels"];

    if 'SwcDebug' in object:
        SwcDebug = object["SwcDebug"]

    if 'StateMachine' in object:
        for sm in object["StateMachine"]:
            overrideInit=False
            hasClone = False
            if ('overrideInit' in sm):
                if (sm['overrideInit'] == "true" or sm['overrideInit'] == "true" ):
                    overrideInit = True
                elif (sm['overrideInit'] == "false" or sm['overrideInit'] == "False" ):
                    overrideInit=False
                else:
                    raise ValueError("Invalid value for overrideInit; must be true or false")
            if 'hasClone' in sm:
                if(sm['hasClone']=="true" or sm['hasClone']=="True"):
                    hasClone = True
                elif(sm['hasClone']=="false" or sm['hasClone']=="False"):
                    hasClone = False
                else:
                    raise ValueError("Invalid value for hasClone; must be true or false")

            #check transition definitions in JSON file
            for transition in sm["transitions"]:
                if type(transition["to"]) is not list or type(transition["from"]) is not str:
                    raise ValueError(
                                    f"Invalid transition definition for {transition}\n"
                                    f"Please correct transition definition to:\n"
                                    f"\"from\":\"state\", \"to\":\"[list, of, states]\""
                                    )

            SMs[sm['name']] = StateMachine(sm['name'], sm['states'], sm['transitions'], sm['startState'], sm['ipaddress'], overrideInit, hasClone)
            if ('disableByDefault' in sm and sm['disableByDefault'] == 'true'):
                SMs[sm['name']].disableByDefault = True;
            else:
                SMs[sm['name']].disableByDefault = False;

            if 'head' in sm and sm['head'] == 'true':
                namespace = sm['name']
                if (SMs[sm['name']].overrideInit == True):
                    raise Exception("Invalid overrideInit setting; head State machine can't override init process")
                SMs[sm['name']].m_isHead = True
                g_head = sm['name']

            SMs[sm['name']].m_isGlobalStateAware = False
            SMs[sm['name']].m_isStateGloballyRelevant = False

            if ('subscribeToStateChanges' in sm):
                SMs[sm['name']].m_isGlobalStateAware = True

            if ('broadcastStateChange' in sm):
                SMs[sm['name']].m_isStateGloballyRelevant = True

            smListByName.append(sm['name'])
    else:
        raise Exception("Invalid Json file, key StateMachine not found")

    #validate hierarchy
    if 'Hierarchy' in object:
        for h in object['Hierarchy']:
            if(h["parent"] not in smListByName):
                raise Exception("Invalid parent statemachine: " + h["parent"])
            parent = h["parent"]
            for childMapping in h['childMapping']:
                if childMapping['parentState'] not in SMs[parent].m_states:
                    raise Exception("Invalid parent state: " + SMs[childMapping['parentState']])
                for childSM in childMapping['children']:
                    if childSM not in smListByName:
                        raise Exception("Invalid children SM name:" + childSM)
                    else:
                        if not h["parent"] in g_hierarchies:
                            g_hierarchies[h["parent"]] = []
                        g_hierarchies[h["parent"]].append(Hierarchy(ParentState=childMapping['parentState'], ChildSM=childSM))
    else:
        raise Exception("Key Hierarchy not found in json file")

    # validate Hierarchial transistions
    if 'HierarchialTransitions' in object:
        for h in object['HierarchialTransitions']:
            if (h["parent"] not in smListByName):
                raise Exception("Invalid parent statemachine: " + h["parent"])
            if h['parentState'] not in SMs[h['parent']].m_states:
                raise Exception("Invalid parent state " + h['parentState'])
            child = h['child']
            if child not in smListByName:
                raise Exception("Child statemachine not found " + child)
            for transition in h['childStates']:
                if 'from' not in transition or 'to' not in transition:
                    raise Exception("from or to tag not found")
                if transition['from'] not in SMs[child].m_states:
                    raise Exception("Invalid child source state " + transition['from'])
                if transition['to'] not in SMs[child].m_states:
                    raise Exception("Invalid child dst state " + transition['to'])

                #if not h["parent"] in SMs[h["parent"]].m_hierarchicalTransition:
                #    SMs[h["parent"]].m_hierarchicalTransition[h["parent"]] = []
                SMs[h["parent"]].m_hierarchicalTransition.append(HierarchialTransitions(ParentState=h['parentState'], ChildSM=child, ChildStateSrc= transition['from'], ChildStateDst=transition['to']))
    else:
        raise Exception("Key HierarchialTransitions not found in json file")

    #validate Notifications
    if 'Notifications' in object:
        for notification in object['Notifications']:
            origin = notification['origin']
            target = notification['target']
            if origin not in smListByName or target not in smListByName:
                raise Exception("Invalid src or dst state. origin: " + origin + " target: " + target)

    # Validate LockSteppedCommands
    if 'LockSteppedCommands' in object:
        for lsCommand in object['LockSteppedCommands']:
            lsCommandArr.append(lsCommand);


#WIP for ClassBuilder
class ClassBuilder:
    def __init__(self, name):
        self.m_name = name
        self.strVar = ""
        self.m_constrVaructor = ""
        self.m_publicFunctions = ""
        self.m_privateFunctions = ""
        self.m_privateVariables = ""
        self.m_publicVariables = ""

    def addIncludeFiles(self, headerFiles):
        self.m_headerFiles+= headerFiles+"i\n"

    def addConstrVaructor(self):
        self.m_constrVaructor+= SPD+"{classname}();\n"

    def addPublicFunction(self, func):
        self.m_publicVariables += SPD + func + "\n"

    def addPrivateFunctions(self,func):
        self.m_privateFunctions += SPD + func + "\n"

    def addPublicVariables(self, var):
        self.m_publicVariables += SPD + var + "\n"

    def addPrivateVariables(self, var):
        self.m_privateVariables += SPD + var + "\n"

    def getClass(self):
        return strVar

def generateDynamicNamesFile():
    """
    The function generates the "SSMNames.hpp" file.
    SSMNames.hpp has #define macros to define common strings for state machines and their transitions
    """

    strVar= "\n"

    # Adding header gaurds
    strVar+= "#pragma once\n\n"

    strVar+= "#include <stdint.h>\n"
    strVar+= "#include <map>\n"
    strVar+= "#define FixedMap std::map\n\n"
    strVar+= "#define MAX_DATA_TYPE_SIZE              50\n\n"

    strVar+= 'namespace SystemStateManager\n{\n'

    cmdIndex = 0

    for label in globalLabelArray:
        strVar+= label + '\n'

    strVar+= "\n"

    strVar += "//Number of State Machines\n"
    strVar += "#define SSM_{config}_SM_COUNT {sm_count}\n".format(config=jsonFileName.upper(),sm_count=len(SMs.keys()))

    strVar+= "\n"

    strVar+= "namespace "+jsonFileName.upper() + "\n{\n\n"

    strVar+= "enum class StateMachines;\n"
    strVar+= "typedef FixedMap<std::string, StateMachines> StateMachineStrEnumMap;\n\n"

    strVar += "enum class LockSteppedCommands : int {" + '\n'
    for a in lsCommandArr:
        if (cmdIndex == 0):
            strVar += SPD + a + " = 0," + '\n'
        else:
            strVar += SPD + a + "," + '\n'
        cmdIndex += 1;

    if (cmdIndex == 0):
        strVar += SPD + "MAX_CMD = 0," + '\n'
    else:
        strVar += SPD + "MAX_CMD," + '\n'
    strVar += "};" + '\n\n'

    strVar += "enum class StateMachines : int\n"
    strVar += "{\n"
    strVar += SPD + "INVALID_STATEMACHINE,\n"
    for sm_name in SMs.keys():
        strVar += SPD + "{0},\n".format(sm_name)
    strVar += "};\n\n"

    for sm_name,sm_object in SMs.items():
        strVar += "enum class {0}States : int\n".format(sm_name)
        strVar += "{\n"
        strVar += SPD + "NULL_STATE,\n"
        for state in sm_object.m_states:
            strVar += SPD + "{0},\n".format(state)
        strVar += "};\n\n"

    strVar += "constexpr int MAX_USER_DATA_PKT_SIZE = 2048;\n\n"

    strVar += "typedef struct _userDataPkt {\n"
    strVar += SPD + "char dataType[MAX_DATA_TYPE_SIZE];\n"
    strVar += SPD + "uint64_t timestamp;\n"
    strVar += SPD + "StateMachines targetStateMachine;\n"
    strVar += SPD + "StateMachines sourceStateMachine;\n"
    strVar += SPD + "int size;\n"
    strVar += SPD + "char data[MAX_USER_DATA_PKT_SIZE]; //2048 is the Maximum Data Size\n"
    strVar += "} UserDataPkt;\n\n"

    strVar+="}\n\n"

    strVar+="}\n"

    f = open(os.path.join(subcomponent_include_directory,"SSMNames.hpp"), "w")
    for a in copyrightData:
        f.write(a)
    f.write("\n")
    f.write(strVar)
    f.close()

def generateDynamicMappingHeaderFile():
    """
    The function generates the "SSMMappings.hpp" file.
    SSMMappings.hpp has the string to enum mapping and helper methods to get states as enums
    """

    strVar= "\n"

    # Adding header gaurds
    strVar+= "#pragma once\n\n"

    strVar+= "#include <stdint.h>\n"
    strVar+= "#include <map>\n"
    strVar+= "#include <iostream>\n"
    strVar+= "#include <ssm/SMBaseClass.hpp>\n"
    strVar+= "#include <"+jsonFileName+"/SSMNames.hpp>\n\n"

    strVar+= 'namespace SystemStateManager\n{\n'

    strVar+= "namespace "+jsonFileName.upper() + "\n{\n\n"

    for sm_name in SMs.keys():
        strVar += "// Helper methods for {sm} that returns state enums for strings\n".format(sm=sm_name)
        strVar += "{sm}States {sm}_strToEnum(std::string state);\n\n".format(sm=sm_name)
        strVar += "{sm}States get_{sm}State(SMBaseClass *obj);\n\n".format(sm=sm_name)
        strVar += "// Helper method for {sm} that returns strings for state enums\n".format(sm=sm_name)
        strVar += "const char* {sm}_enumToStr({sm}States e);\n\n".format(sm=sm_name)
        strVar += "//Overload << operator to help pretty print state enums\n"
        strVar += "std::ostream& operator<<(std::ostream& out, const {sm}States state_enum);\n\n".format(sm=sm_name)
    strVar += "//Helper method to return string for a state machine enum\n"
    strVar += "const char* SmEnumToStr(StateMachines e);\n\n"
    strVar +=" // Helper method: Returns an enum for state machine string passed as an argument\n"
    strVar += "StateMachines getSmEnum(std::string sm);\n"

    strVar += "\n}\n"
    strVar += "}\n"

    f = open(os.path.join(subcomponent_include_directory,"SSMMappings.hpp"), "w")
    for a in copyrightData:
        f.write(a)
    f.write("\n")
    f.write(strVar)
    f.close()

def generateDynamicMappingCppFile():
    """
    The function generates the "SSMMappings.cpp" file.
    SSMMappings.cpp has the implementation for SSMMappings.hpp
    """

    strVar= "\n"

    strVar+= '#include <'+jsonFileName+'/SSMMappings.hpp>\n\n'

    strVar+= 'namespace SystemStateManager\n{\n'

    strVar+= "namespace "+jsonFileName.upper() + "\n{\n\n"

    for sm_name in SMs.keys():
        strVar += generatePrettyPrintOverload(sm_name)

        strVar += "\n"
        strVar += "const char* {classname}_enumToStr({classname}States e)\n".format(classname=sm_name)
        strVar += "{\n"

        strVar += SPD + "switch(e)\n"
        strVar += SPD + "{\n"

        for state in SMs[sm_name].m_states:
            strVar += SPD + SPD + "case {classname}States::{enum_name}: return SSM_{classname}_{enum_name}_str;\n".format(classname=sm_name, enum_name=state)
        strVar += SPD + "}\n"

        strVar += SPD + "return \"NULL_STATE\";\n"

        strVar+= '}\n'
        
        strVar += "\n"

        strVar += generateStrToEnumFunction(sm_name)


        strVar += "{sm}States get_{sm}State(SMBaseClass *obj)\n".format(sm=sm_name)
        strVar += "{\n"
        strVar += SPD + "return {sm}_strToEnum(obj->getCurrentState(SSM_{sm}_str));\n".format(sm=sm_name)
        strVar += "}\n\n"

    strVar += generateSmEnumToStrFunction()
    strVar += generateGetSmEnumFunction()


    strVar += "\n}\n"
    strVar += "}\n"

    f = open(os.path.join(subcomponent_src_directory,"SSMMappings.cpp"), "w")
    for a in copyrightData:
        f.write(a)
    f.write("\n")
    f.write(strVar)
    f.close()


def generateDynamicHeaderFile(sm_name):
    """
    This function generates the header file for the state machine.
    The file has the class declaration for the state machine.
    The state machine class is derived from the SMBaseClass.
    """

    filename = sm_name
    strVar= ""

    # Adding header gaurds
    strVar+= "#pragma once\n\n"

    strVar+= '#include <'+jsonFileName+'/SSMNames.hpp>\n'
    strVar+= '#include <ssm/SMBaseClass.hpp>\n'
    strVar+= '#include <ssm/SSMHistogram.hpp>\n'
    strVar+= '#include <'+jsonFileName+'/SSMMappings.hpp>\n'
    strVar+= '\n'
    strVar+= 'namespace SystemStateManager\n{\n\n';
    strVar+= "namespace "+jsonFileName.upper() + "\n{\n\n";
    strVar+= 'extern bool setupHierarchy(StateMachineVector& smv, StateMachinePtr& hPtr);\n\n'
    strVar+= "class {classname} : public SMBaseClass\n".format(classname=sm_name)
    strVar+= "{\n"
    strVar+= "public:\n"

    #add destrVaructor
    strVar+= SPD + "~{classname}() {{}};\n".format(classname=sm_name)

    # add constrVaructor
    # If initHierachy calls initLockSteppedCommands,
    # we cannot register locked step commands without calling initHierarchy.
    # So initLockSteppedCommands is added to constructor.
    if (sm_name == cloneName):
        strVar+= SPD + '{classname}(std::string myname)\n'.format(classname=sm_name);
        strVar+= SPD + '{\n'
        strVar+= SPD + SPD + 'name = myname;\n'
        strVar+= SPD + SPD + 'initLockSteppedCommands();\n'
        strVar+= SPD + '}\n'
    else:
        strVar+= SPD + '{classname}()\n'.format(classname=sm_name);
        strVar+= SPD + '{\n'
        strVar+= SPD + SPD + 'name = "{classname}";\n'.format(classname=sm_name)
        strVar+= SPD + SPD + 'initLockSteppedCommands();\n'
        strVar+= SPD + '}\n'

    #add virtual methods
    strVar+= SPD + "virtual void initStateMachine() {};\n"
    strVar+= SPD + "virtual void ENTER() {}\n"
    strVar+= SPD + "virtual void EXIT() {}\n"
    strVar+= SPD + "virtual void PRE_INIT() {}\n"
    strVar+= SPD + "virtual void INIT() {}\n"
    strVar+= SPD + "virtual void POST_INIT() {}\n"
    strVar+= SPD + "virtual void PRE_READY() {}\n"
    strVar+= SPD + "virtual void READY() {}\n"

    sname = sm_name;
    if (sname == cloneName):
        sname = g_head;
    for state in SMs[sname].m_states:
        strVar+= SPD + "virtual void {function_name}() {{}};\n".format(function_name=state)

    strVar+= SPD + "virtual void preSwitchHandler() {}\n"
    strVar+= SPD + "virtual void postSwitchHandler() {}\n"
    strVar+= SPD + "void executeStateMachine() override;\n"
    strVar+= SPD + "bool setupTreeHierarchy(StateMachineVector& smv, StateMachinePtr& hPtr) override;\n";
    strVar+= SPD + "void documentStateChangeExternally() override;\n"

    if sm_name == g_head:
        strVar+= SPD + "void initialize() override;\n"
        strVar+= SPD + 'bool executeLockedCommand(SystemStateManager::{head_name}::LockSteppedCommands command);\n'.format(head_name=jsonFileName.upper())
    else:
        strVar+= SPD + "void runInitPhase(int micros);\n\n"

    strVar += SPD + "bool changeState({sm_enum_name}States e);\n".format(sm_enum_name=sm_name)
    strVar += SPD + "int sendDataByID(StateMachines e, void *data, int datalen, std::string dataStructID);\n"
    strVar += SPD + "{sm_enum_name}States getState();\n".format(sm_enum_name=sm_name)

    strVar += "\n"
    strVar += SPD + "StateMachines stateMachineId {{StateMachines::{sm_enum_name}}};\n".format(sm_enum_name=sm_name)

    strVar+= SPD + '// API to register LockedStepCommands\n'
    strVar+= SPD + 'bool registerLockedCommand(SystemStateManager::{head_name}::LockSteppedCommands command, SSMFunctionHandler ptr);\n\n'.format(head_name=jsonFileName.upper())
    strVar+= SPD + "bool getUserSpecificData(UserDataPkt &pkt);\n\n"

    strVar+= "private:\n"
    strVar+= SPD + "void printClientHistogram();\n";
    strVar+= SPD + "void initLockSteppedCommands();\n";
    strVar+= SPD + "SSMHistogram runnableHist{};\n\n";


    strVar+= "};\n\n"

    f = open(os.path.join(subcomponent_include_directory ,( filename+".hpp")), "w")
    strVar+="}\n}\n"

    for a in copyrightData:
        f.write(a);
    f.write("\n");
    f.write(strVar)
    f.close()

def addResetHandlerExecutionRoutine(spc):
    """
    Returns
    -------
    str()
        The C++ block of code which calls resetFunctionHandlerPtr() based on
        executeResetHandlerFlag
    """

    strVar = "";
    strVar+= spc + 'if (executeResetHandlerFlag)\n'
    strVar+= spc + '{\n'
    strVar+= spc + SPD + 'if (resetFunctionHandlerPtr)\n'
    strVar+= spc + SPD + '{\n'
    strVar+= spc + SPD + SPD + 'resetFunctionHandlerPtr(this);\n'
    strVar+= spc + SPD + '}\n'
    strVar+= spc + SPD + 'executeResetHandlerFlag = false;\n'
    strVar+= spc + '}\n'
    return strVar;

def generateDynamicCPPFile(sm_name):
    """
    This function generates the implementation file for the state machine.
    The file has the class definitions for the state machine.
    """

    filename = sm_name;
    strVar = ""
    strVar += '#include <'+jsonFileName+'/{classname}.hpp>\n'.format(classname=sm_name)
    strVar+= '\n'
    strVar += 'namespace SystemStateManager\n{\n\n';
    strVar += "namespace "+jsonFileName.upper() + "\n{\n\n";

    if sm_name == g_head:
        strVar+= 'void {classname}::initialize()\n'.format(classname=sm_name)
        strVar+= '{\n';
        strVar+= SPD + 'SSM_LOG("Initializing '+jsonFileName.upper()+'");\n'
        strVar+= SPD + 'isStateMachineActive = true;\n'
        strVar+= SPD + 'initStateMachine();\n'
        strVar+= SPD + 'ENTER();\n'
        strVar+= '}\n\n'

        strVar+= 'bool {classname}::executeLockedCommand(SystemStateManager::{head_name}::LockSteppedCommands command)\n'.format(classname=sm_name, head_name=jsonFileName.upper())
        strVar+= '{\n';
        strVar+= SPD + 'return executeLockSteppedCommand((int)command);\n'
        strVar+= '}\n\n'

    strVar+= 'bool {classname}::registerLockedCommand(SystemStateManager::{head_name}::LockSteppedCommands command, SSMFunctionHandler ptr)\n'.format(classname=sm_name, head_name=jsonFileName.upper())
    strVar+= '{\n'
    strVar+= SPD + 'return registerLockSteppedFunction((int)command, ptr);\n'
    strVar+= '}\n\n'

    strVar+= 'void {classname}::executeStateMachine()\n'.format(classname=sm_name)
    strVar+= '{\n'
    strVar+= SPD + 'runnableHist.startTimer();\n'
    strVar+= SPD + 'if (mySMPtr->getCurrentState() == ENTER_STATE)\n'
    strVar+= SPD + '{\n'
    strVar+= SPD + SPD + 'bool isStateChangeValid = false;\n'
    #strVar+= SPD + SPD + 'runnableHist.addTimestamp(11110);\n'
    strVar+= addResetHandlerExecutionRoutine(SPD + SPD);
    strVar+= SPD + SPD + 'ENTER();\n'
    #strVar+= SPD + SPD + 'runnableHist.addTimestamp(11111);\n'
    strVar+= SPD + SPD + 'isStateChangeValid = mySMPtr->changeStateByString(mySMPtr->getStartState());\n'
    strVar+= SPD + SPD + 'if (!isStateChangeValid)\n'
    strVar+= SPD + SPD + '{\n'
    strVar+= SPD + SPD + SPD + 'runnableHist.endTimer(SSM_LATENCY_THRESHOLD);\n'
    strVar+= SPD + SPD + SPD + 'SSM_ERROR("Invalid state transition requested: " + mySMPtr->getCurrentState() + " -> " + mySMPtr->getStartState());\n'
    strVar+= SPD + SPD + SPD + 'return;\n'
    strVar+= SPD + SPD + '}\n'
    strVar+= SPD + SPD + 'broadCastStateChange(name, mySMPtr->getStartState());\n'
    #strVar+= SPD + SPD + 'runnableHist.addTimestamp(11112);\n'
    strVar+= SPD + SPD + 'activateChildren(mySMPtr);\n'
    #strVar+= SPD + SPD + 'runnableHist.addTimestamp(11113);\n'
    strVar+= SPD + '}\n\n'
    #strVar+= SPD + SPD + 'runnableHist.addTimestamp(22220);\n'
    strVar+= SPD + 'if (mySMPtr->getCurrentState() == EXIT_STATE)\n'
    strVar+= SPD + '{\n'
    strVar+= SPD + SPD + 'EXIT();\n'
    #strVar+= SPD + SPD + 'runnableHist.addTimestamp(22221);\n'
    strVar+= SPD + '}\n\n'

    strVar+= SPD + 'if (mySMPtr->getCurrentState() == PRE_INIT_STATE)\n'
    strVar+= SPD + '{\n'
    strVar+= SPD + SPD + 'PRE_INIT();\n'
    #strVar+= SPD + SPD + 'runnableHist.addTimestamp(22222);\n'
    strVar+= SPD + '}\n\n'

    strVar+= SPD + 'if (mySMPtr->getCurrentState() == INIT_STATE)\n'
    strVar+= SPD + '{\n'
    strVar+= SPD + SPD + 'INIT();\n'
    #strVar+= SPD + SPD + 'runnableHist.addTimestamp(22223);\n'
    strVar+= SPD + '}\n\n'

    strVar+= SPD + 'if (mySMPtr->getCurrentState() == POST_INIT_STATE)\n'
    strVar+= SPD + '{\n'
    strVar+= SPD + SPD + 'POST_INIT();\n'
    #strVar+= SPD + SPD + 'runnableHist.addTimestamp(22224);\n'
    strVar+= SPD + '}\n\n'

    strVar+= SPD + 'if (mySMPtr->getCurrentState() == PRE_READY_STATE)\n'
    strVar+= SPD + '{\n'
    strVar += SPD + SPD + 'PRE_READY();\n'
    #strVar+= SPD + SPD + 'runnableHist.addTimestamp(22225);\n'
    strVar+= SPD + '}\n\n'

    strVar+= SPD + 'if (mySMPtr->getCurrentState() == READY_STATE)\n'
    strVar+= SPD + '{\n'
    strVar += SPD + SPD + 'READY();\n'
    #strVar+= SPD + SPD + 'runnableHist.addTimestamp(22226);\n'
    strVar+= SPD + '}\n\n'

    ELSE_strVar = ""

    sname = sm_name;
    if (sname == cloneName):
        sname = g_head;
    ind = 0;
    for state in SMs[sname].m_states:
        strVar+= SPD + '{else_strVar}if (mySMPtr->getCurrentState() == SSM_{classname}_{statename}_str)\n'.format(else_strVar=ELSE_strVar, classname=sm_name, statename=state)
        strVar+= SPD + "{\n"
        strVar+= SPD + SPD + '{statename}();\n'.format(statename=state)
        #strVar+= SPD + SPD + "runnableHist.addTimestamp(22225" + str(ind) + ");\n";
        ind = ind + 1;
        strVar+= SPD + "}\n"
        ELSE_strVar="else "

    strVar+= SPD + 'std::string currState = mySMPtr->getCurrentState();\n'
    strVar+= SPD + 'bool printLatency     = true;\n'
    strVar+= SPD + 'if (currState == "PRE_INIT_STATE" ||\n'
    strVar+= SPD + SPD + 'currState == "INIT_STATE" ||\n'
    strVar+= SPD + SPD + 'currState == "POST_INIT_STATE" ||\n'
    strVar+= SPD + SPD + 'currState == "PRE_READY_STATE" ||\n'
    strVar+= SPD + SPD + 'currState == "READY_STATE")\n'
    strVar+= SPD + '{\n'
    strVar+= SPD + SPD + 'printLatency = false;\n'
    strVar+= SPD + '}\n'
    strVar+= SPD + 'runnableHist.endTimer(SSM_LATENCY_THRESHOLD, printLatency);\n'

    strVar+= '}\n'
    if (sm_name != g_head):
        strVar+= generateInitFunction(sm_name)
    strVar+= generateSetupTreeHierarchyFunction(sm_name)
    strVar+= generatePrintHistogramFunction(sm_name)
    strVar+= generateLockedSteppedFunctions(sm_name)
    strVar+= generateDocumentStateChangeExternallyFunction(sm_name)
    strVar += generateChangeStateFunction(sm_name)
    strVar += generateSendDataByIdFunction(sm_name)
    strVar += generateGetStateFunction(sm_name)
    strVar += generateGetUserSpecificDataFunction(sm_name)



    strVar+="}\n}\n"
    f = open(os.path.join(subcomponent_src_directory , (filename+".cpp")), "w")
    for a in copyrightData:
        f.write(a);
    f.write("\n");
    f.write(strVar)
    f.close()

def generateDynamicCloneHeaderFile(sm_name):
    """
    This function generates the header for a clone state machine. The clone class is derived from SMBaseClass but does not have any states
    To generate a clone state machine the state machine definition in the json file must have 'hasClone' set to true
    """

    strVar= "\n"
    # Adding header gaurds
    strVar+= "#pragma once\n\n"

    strVar+= '#include <'+jsonFileName+'/SSMNames.hpp>\n'
    strVar+= '#include <ssm/SMBaseClass.hpp>\n'
    strVar+= '#include <ssm/SSMHistogram.hpp>\n'
    strVar+= "\n"
    strVar+= 'namespace SystemStateManager\n{\n\n'
    strVar+= "namespace "+jsonFileName.upper() + "\n{\n\n"
    strVar+= "class {classname} : public SMBaseClass\n".format(classname=sm_name)
    strVar+= "{\n"
    strVar+= "public:\n"

    #add destrVaructor
    strVar+= SPD + "~{classname}(){{}};\n".format(classname=sm_name)

    # add constrVaructor
    strVar+= SPD + '{classname}(std::string myname)\n'.format(classname=sm_name)
    strVar+= SPD + '{\n'
    strVar+= SPD + SPD + 'name = myname;\n'
    strVar+= SPD + SPD + 'runnableHist.setName(name);\n'
    strVar+= SPD + SPD + 'initLockSteppedCommands();\n'
    strVar+= SPD + '}\n'

    #add virtual methods
    strVar+= SPD + "virtual void initStateMachine(){};\n"
    strVar+= SPD + "void executeStateMachine() override;\n"
    strVar+= SPD + "bool registerFunctionHandler(std::string name, SSMFunctionHandler ptr);\n";
    strVar+= SPD + "void runInitPhase(int micros);\n"
    strVar+= SPD + 'void printClientHistogram() override;\n'
    strVar+= SPD + 'void setRunnableId(const char* runnableId)\n'
    strVar+= SPD + '{\n'
    strVar+= SPD + SPD + 'm_runnableId = runnableId;\n'
    strVar+= SPD + '}\n'
    strVar+= SPD + 'const std::string& getRunnableId()\n'
    strVar+= SPD + '{\n'
    strVar+= SPD + SPD + 'return m_runnableId;\n'
    strVar+= SPD + '}\n'
    strVar+= SPD + '// API to register LockedStepCommands\n'
    strVar+= SPD + 'bool registerLockedCommand(SystemStateManager::{head_name}::LockSteppedCommands command, SSMFunctionHandler ptr);\n\n'.format(head_name=jsonFileName.upper())
    strVar+= "\nprivate:\n"
    strVar+= SPD + "SSMHistogram runnableHist{};\n";
    strVar+= SPD + "int totalFuncs{0};\n"
    strVar+= SPD + "std::string m_runnableId;\n"
    strVar+= SPD + "SSMFunctionHandlerVector funcVector;\n"
    strVar+= SPD + "SSMFunctionHandler enterHandlerPtr{};\n"
    strVar+= SPD + "SSMFunctionHandler exitHandlerPtr{};\n"
    strVar+= SPD + "SSMFunctionHandler pre_initHandlerPtr{};\n"
    strVar+= SPD + "SSMFunctionHandler initHandlerPtr{};\n"
    strVar+= SPD + "SSMFunctionHandler postHandlerPtr{};\n"
    strVar+= SPD + "SSMFunctionHandler preReadyHandlerPtr{};\n"
    strVar+= SPD + "SSMFunctionHandler readyHandlerPtr{};\n"
    strVar+= SPD + "SSMFunctionHandler preSwitchHandlerPtr{};\n"
    strVar+= SPD + "SSMFunctionHandler postSwitchHandlerPtr{};\n"
    strVar+= SPD + "bool setupTreeHierarchy(StateMachineVector& smv, StateMachinePtr& hPtr) override;\n";
    strVar+= SPD + "void initLockSteppedCommands();\n";
    strVar+= SPD + "void generateLockedSteppedFunctions();\n";
    strVar+= "};\n"
    strVar+="}\n}\n"


    f = open(os.path.join(subcomponent_include_directory,(sm_name+".hpp")), "w")
    for a in copyrightData:
        f.write(a);
    f.write("\n");
    f.write(strVar)
    f.close()

def generateDynamicCloneCPPFile(sm_name):
    """
    This function generates the implementation file for a clone state machine.
    To generate a clone state machine the state machine definition in the json file must have 'hasClone' set to true
    """

    filename = sm_name;
    strVar = ""
    strVar += '#include <'+jsonFileName+'/{classname}.hpp>\n'.format(classname=sm_name)
    strVar+= '\n'
    strVar+= 'namespace SystemStateManager\n{\n\n';
    strVar+= "namespace "+jsonFileName.upper() + "\n{\n\n";
    strVar+= 'extern bool setupHierarchy(StateMachineVector& smv, StateMachinePtr& hPtr);\n\n'
    if sm_name == g_head:
        strVar+= 'void SSM::initialize()\n';
        strVar+= '{\n';
        strVar+=  SPD + 'isStateMachineActive = true;\n';
        strVar+=  SPD + 'initStateMachine();\n';
        strVar+=  SPD + 'ENTER();\n';
        strVar+=  '}\n'

    strVar+= 'void {classname}::executeStateMachine()\n'.format(classname=sm_name)
    strVar+= '{\n'
    strVar+= SPD + 'runnableHist.startTimer();\n'
    strVar+= SPD + 'if (mySMPtr->getCurrentState() == ENTER_STATE)\n'
    strVar+= SPD + '{\n'
    strVar+= SPD +  SPD + 'bool isStateChangeValid = false;\n'
    strVar+= addResetHandlerExecutionRoutine(SPD + SPD);
    strVar+= SPD +  SPD + 'if (enterHandlerPtr)\n'
    strVar+= SPD +  SPD + '{\n'
    #strVar+= SPD +  SPD + SPD + 'runnableHist.addTimestamp(11110);\n'
    strVar+= SPD +  SPD + SPD + 'enterHandlerPtr(this);\n'
    #strVar+= SPD +  SPD + SPD + 'runnableHist.addTimestamp(11111);\n'
    strVar+= SPD +  SPD + '}\n'
    #strVar+= SPD +  SPD + 'runnableHist.addTimestamp(22220);\n'
    strVar+= SPD +  SPD + 'isStateChangeValid = mySMPtr->changeStateByString(mySMPtr->getStartState());\n'
    strVar+= SPD +  SPD + 'if (!isStateChangeValid)\n'
    strVar+= SPD +  SPD + '{\n'
    strVar+= SPD +  SPD + SPD + 'runnableHist.endTimer(SSM_LATENCY_THRESHOLD);\n'
    strVar+= SPD +  SPD + SPD + 'SSM_ERROR("Invalid state transition requested: " + mySMPtr->getCurrentState() + " -> " + mySMPtr->getStartState());\n'
    strVar+= SPD +  SPD + SPD + 'return;\n'
    strVar+= SPD +  SPD + '}\n'
    #strVar+= SPD +  SPD + 'runnableHist.addTimestamp(33330);\n'
    strVar+= SPD +  SPD + 'broadCastStateChange(name, mySMPtr->getStartState());\n'
    #strVar+= SPD +  SPD + 'runnableHist.addTimestamp(33331);\n'
    strVar+= SPD +  SPD + 'activateChildren(mySMPtr);\n'
    strVar+= SPD + '}\n\n'
    #strVar+= SPD + 'runnableHist.addTimestamp(44440);\n'
    strVar+= SPD + 'if (mySMPtr->getCurrentState() == EXIT_STATE && exitHandlerPtr)\n'
    strVar+= SPD + '{\n'
    #strVar+= SPD +  SPD + 'runnableHist.addTimestamp(55550);\n'
    strVar+= SPD +  SPD + 'exitHandlerPtr(this);\n'
    strVar+= SPD + '}\n\n'
    #strVar+= SPD +  'runnableHist.addTimestamp(66660);\n'
    strVar+= SPD + 'if (mySMPtr->getCurrentState() == PRE_INIT_STATE && pre_initHandlerPtr)\n'
    strVar+= SPD + '{\n'
    strVar+= SPD +  SPD + 'pre_initHandlerPtr(this);\n'
    #strVar+= SPD +  SPD + 'runnableHist.addTimestamp(66661);\n'
    strVar+= SPD + '}\n\n'

    strVar+= SPD + 'if (mySMPtr->getCurrentState() == INIT_STATE && initHandlerPtr)\n'
    strVar+= SPD + '{\n'
    strVar+= SPD +  SPD + 'initHandlerPtr(this);\n'
    #strVar+= SPD +  SPD + 'runnableHist.addTimestamp(66662);\n'
    strVar+= SPD + '}\n\n'

    strVar+= SPD + 'if (mySMPtr->getCurrentState() == POST_INIT_STATE && postHandlerPtr)\n'
    strVar+= SPD + '{\n'
    strVar+= SPD +  SPD + 'postHandlerPtr(this);\n'
    #strVar+= SPD +  SPD + 'runnableHist.addTimestamp(66663);\n'
    strVar+= SPD + '}\n\n'

    strVar+= SPD + 'if (mySMPtr->getCurrentState() == PRE_READY_STATE && preReadyHandlerPtr)\n'
    strVar+= SPD + '{\n'
    strVar+= SPD +  SPD + 'preReadyHandlerPtr(this);\n'
    #strVar+= SPD +  SPD + 'runnableHist.addTimestamp(66664);\n'
    strVar+= SPD + '}\n\n'

    strVar+= SPD + 'if (mySMPtr->getCurrentState() == READY_STATE && readyHandlerPtr)\n'
    strVar+= SPD + '{\n'
    strVar+= SPD +  SPD + 'readyHandlerPtr(this);\n'
    #strVar+= SPD +  SPD + 'runnableHist.addTimestamp(66665);\n'
    strVar+= SPD + '}\n\n'

    strVar+= SPD + 'if (mySMPtr->getCurrentState() == INIT_CLONE_STATE)\n'
    strVar+= SPD + '{\n'
    strVar+= SPD +  SPD + 'changeStateByString(headPtr->getCurrentState());\n'
    #strVar+= SPD +  SPD + 'runnableHist.addTimestamp(66666);\n'
    strVar+= SPD + '}\n\n'
    #strVar+= SPD + 'runnableHist.addTimestamp(77770);\n'
    ELSE_strVar = ""

    strVar+= SPD + 'std::string cs = mySMPtr->getCurrentState();\n'
    strVar+= SPD + 'for (int index = 0; index < totalFuncs; index++)\n'
    strVar+= SPD + '{\n';
    strVar+= SPD +  SPD + 'if (funcVector.getObject(index).stateName == cs)\n'
    strVar+= SPD +  SPD + '{\n'
    #strVar+= SPD +  SPD + SPD + 'runnableHist.addTimestamp(88880);\n'
    if (SwcDebug) :
        strVar+= SPD +  SPD + SPD + 'cout << cs << \" : \" << this->getName() << endl;\n'
    strVar+= SPD +  SPD + SPD + 'funcVector.getObject(index).handler(this);\n'
    #strVar+= SPD +  SPD + SPD + 'runnableHist.addTimestamp(99990);\n'
    strVar+= SPD +  SPD + SPD + 'break;\n'
    strVar+= SPD +  SPD + '}\n'
    strVar+= SPD + '}\n'
    strVar+= SPD + 'std::string currState = mySMPtr->getCurrentState();\n'
    strVar+= SPD + 'bool printLatency     = true;\n'
    strVar+= SPD + 'if (currState == "PRE_INIT_STATE" ||\n'
    strVar+= SPD + SPD + 'currState == "INIT_STATE" ||\n'
    strVar+= SPD + SPD + 'currState == "POST_INIT_STATE" ||\n'
    strVar+= SPD + SPD + 'currState == "PRE_READY_STATE" ||\n'
    strVar+= SPD + SPD + 'currState == "READY_STATE")\n'
    strVar+= SPD + '{\n'
    strVar+= SPD + SPD + 'printLatency = false;\n'
    strVar+= SPD + '}\n'
    strVar+= SPD + 'runnableHist.endTimer(SSM_LATENCY_THRESHOLD, printLatency);\n'
    strVar+= '}\n\n'
    strVar+= 'bool {classname}::registerFunctionHandler(std::string name, SSMFunctionHandler ptr)\n'.format(classname=sm_name)
    strVar+= '{\n'
    strVar+= SPD + 'isFunctionHandlerUpdated = true;\n'
    strVar+= SPD + 'if (registerStdFunctionHandler(name, ptr))\n'
    strVar+= SPD + '{\n'
    strVar+= SPD +  SPD + 'return true;\n'
    strVar+= SPD + '}\n'

    # TODO: Move the standard handlers to SMBaseClass
    strVar+= SPD + 'if (name == ENTER_STATE)\n'
    strVar+= SPD + '{\n'
    strVar+= SPD +  SPD + 'enterHandlerPtr = ptr;\n'
    strVar+= SPD + '}\n'
    strVar+= SPD + 'else if (name == EXIT_STATE)\n'
    strVar+= SPD + '{\n'
    strVar+= SPD + SPD + 'exitHandlerPtr = ptr;\n'
    strVar+= SPD + '}\n'
    strVar+= SPD + 'else if (name == PRE_INIT_STATE)\n'
    strVar+= SPD + '{\n'
    strVar+= SPD + SPD + 'pre_initHandlerPtr = ptr;\n'
    strVar+= SPD + '}\n'
    strVar+= SPD + 'else if (name == INIT_STATE)\n'
    strVar+= SPD + '{\n'
    strVar+= SPD + SPD + 'initHandlerPtr = ptr;\n'
    strVar+= SPD + '}\n'
    strVar+= SPD + 'else if (name == SHUTDOWN_STATE)\n'
    strVar+= SPD + '{\n'
    strVar+= SPD + SPD + 'shutdownHandlerPtr = ptr;\n'
    strVar+= SPD + '}\n'
    strVar+= SPD + 'else if (name == PRE_SHUTDOWN_STATE)\n'
    strVar+= SPD + '{\n'
    strVar+= SPD + SPD + 'preShutdownHandlerPtr = ptr;\n'
    strVar+= SPD + '}\n'
    strVar+= SPD + 'else if (name == POST_INIT_STATE)\n'
    strVar+= SPD + '{\n'
    strVar+= SPD + SPD + 'postHandlerPtr = ptr;\n'
    strVar+= SPD + '}\n'
    strVar+= SPD + 'else if (name == PRE_READY_STATE)\n'
    strVar+= SPD + '{\n'
    strVar+= SPD + SPD + 'preReadyHandlerPtr = ptr;\n'
    strVar+= SPD + '}\n'
    strVar+= SPD + 'else if (name == READY_STATE)\n'
    strVar+= SPD + '{\n'
    strVar+= SPD + SPD + 'readyHandlerPtr = ptr;\n'
    strVar+= SPD + '}\n'
    strVar+= SPD + 'else\n'
    strVar+= SPD + '{\n'
    strVar+= SPD + SPD + 'for (int index = 0; index < totalFuncs; index++)\n'
    strVar+= SPD + SPD + '{\n'
    strVar+= SPD + SPD + SPD + 'if (funcVector.getObject(index).stateName == name)\n'
    strVar+= SPD + SPD + SPD + '{\n'
    strVar+= SPD + SPD + SPD + SPD + 'return false;\n'
    strVar+= SPD + SPD + SPD + '}\n'
    strVar+= SPD + SPD + '}\n'
    strVar+= SPD + SPD + 'SSMFunctionHandlerStruct st;\n'
    strVar+= SPD + SPD + 'st.stateName = name;\n'
    strVar+= SPD + SPD + 'st.handler   = ptr;\n'
    strVar+= SPD + SPD + 'funcVector.push_back(st);\n'
    strVar+= SPD + SPD + 'totalFuncs++;\n'
    strVar+= SPD + '}\n'
    strVar+= SPD + 'return true;\n'
    strVar+= '}\n'
    strVar+= generateInitFunction(sm_name);
    strVar+= generateSetupTreeHierarchyFunction(sm_name);
    strVar+= generatePrintHistogramFunction(sm_name);
    strVar+= generateLockedSteppedFunctions(sm_name);

    strVar+= 'bool {classname}::registerLockedCommand(SystemStateManager::{head_name}::LockSteppedCommands command, SSMFunctionHandler ptr)\n'.format(classname=sm_name, head_name=jsonFileName.upper())
    strVar+= '{\n'
    strVar+= SPD + 'return registerLockSteppedFunction((int)command, ptr);\n'
    strVar+= '}\n'

    strVar+="}\n}"
    f = open(os.path.join(subcomponent_src_directory,(filename+".cpp")), "w")
    for a in copyrightData:
        f.write(a);
    f.write("\n");
    f.write(strVar)
    f.close()


def generateInitFunction(sm_name):
    """
    Returns
    -------
        The runInitPhase() function for a the state machine definition file (<state machine name>.cpp)
    """

    strVar = '\n'
    strVar+= 'void {classname}::runInitPhase(int micros)\n'.format(classname=sm_name)
    strVar+= '{\n'
    strVar+= SPD + 'initHierarchy();\n'
    #strVar+= SPD + 'startServer();\n'
    strVar+= SPD + 'while (!isStartStateMachineMsgReceived())\n'
    strVar+= SPD + '{\n'
    if (SwcDebug):
        strVar+= SPD + SPD + 'cout << "SM: waiting for " + name + " to start" << endl;\n'
    strVar+= SPD + SPD + 'sleep(1);\n'
    strVar+= SPD + '}\n'
    strVar+= SPD + 'initClientComm();\n'
    strVar+= SPD + 'while (!isSMReadyForScheduler())\n'
    strVar+= SPD + '{\n'
    strVar+= SPD + SPD + 'runStateMachine();\n'
    strVar+= SPD + SPD + 'usleep(micros);\n'
    strVar+= SPD + '}\n'
    strVar+= '}\n'
    return strVar;

def generateLockedSteppedFunctions(sm_name):
    """
    Returns
    -------
        The initLockedInfrastrcuture() function initializes the state machine
        to be able to allow users to register function handlers to executed lock stepped commands
        across the system.
    """

    strVar = '\n'
    strVar+= 'void {classname}::initLockSteppedCommands()\n'.format(classname=sm_name)
    strVar+= '{\n'
    strVar+= SPD + 'for (int index = 0; index < (int)SystemStateManager::{classname}::LockSteppedCommands::MAX_CMD; index++)\n'.format(classname=jsonFileName.upper())
    strVar+= SPD + '{\n'
    strVar+= SPD + SPD + 'addLockSteppedCommand(index);\n'
    strVar+= SPD + '}\n'
    strVar+= '}\n'
    return strVar;

def generateSendDataByIdFunction(sm_name):
    """
    Returns
    -------
        the sendDataByID() function that calls sendData()
    """

    strVar = "\n"
    strVar += "int {classname}::sendDataByID(StateMachines e, void *data, int datalen, std::string dataStructID)\n".format(classname=sm_name)
    strVar += "{\n"
    strVar += SPD + "return sendData(SmEnumToStr(e), data, datalen, dataStructID);\n".format(classname=sm_name)
    strVar += "}\n"

    return strVar

def generateChangeStateFunction(sm_name):
    """
    Returns
    -------
        the changeState() function that calls changeState(std::string) internally
    """

    strVar = "\n"
    strVar += "bool {classname}::changeState({classname}States e)\n".format(classname=sm_name)
    strVar += "{\n"
    strVar += SPD + "return changeStateByString({classname}_enumToStr(e));\n".format(classname=sm_name)
    strVar += "}\n"

    return strVar


def generateEnumToStrFunction(sm_name):
    """
    Returns
    -------
        the enumToStr() function that returns a string for an enum
    """

    strVar = "\n"
    strVar += "const char* {classname}::enumToStr({classname}States e)\n".format(classname=sm_name)
    strVar += "{\n"

    strVar += SPD + "switch(e)\n"
    strVar += SPD + "{\n"

    for state in SMs[sm_name].m_states:
        strVar += SPD + SPD + "case {classname}States::{enum_name}: return SSM_{classname}_{enum_name}_str;\n".format(classname=sm_name, enum_name=state)
    strVar += SPD + "default:\n"
    strVar += SPD + SPD + "break;\n"
    strVar += SPD + "}\n"

    strVar += SPD + "return \"\";\n"

    strVar+= '}\n'
    return strVar

def generateDocumentStateChangeExternallyFunction(sm_name):
    """
    Returns
    -------
        documentStateChangeExternally() updates the external logger (Roadcast) with
        the current state of the state machine
    """

    strVar = "\n"
    strVar += "void {classname}::documentStateChangeExternally()\n".format(classname=sm_name)
    strVar += "{\n"
    strVar += SPD + "documentStateChange((int)stateMachineId, (int)getState());\n"
    strVar += "}\n\n"
    return strVar

def generateSmEnumToStrFunction():
    """
    Returns
    -------
        the SmEnumToStr() function that returns a string for a State Machine enum
    """

    strVar = "\n"
    strVar += "const char* SmEnumToStr(StateMachines e)\n"
    strVar += "{\n"

    strVar += SPD + "switch(e)\n"
    strVar += SPD + "{\n"

    for state_machine in SMs.keys():
        strVar += SPD + SPD + "case StateMachines::{enum_name}: return SSM_{enum_name}_str;\n".format( enum_name=state_machine)
    strVar += SPD + "}\n"

    strVar += SPD + "return \"\";\n"

    strVar+= '}\n'
    return strVar

def generateGetStateFunction(sm_name):
    """
    Returns
    -------
        the getState() function that returns an enum for the current state of the state machine
    """

    strVar = "\n"
    strVar += "{classname}States {classname}::getState()\n".format(classname=sm_name)
    strVar += "{\n"
    strVar += SPD + "return {classname}_strToEnum(mySMPtr->getCurrentState());\n".format(classname=sm_name)
    strVar += "}\n\n"

    return strVar

def generateGetUserSpecificDataFunction(sm_name):
    """
    Returns
    -------
        the getUserSpecificData() function that returns the UserDataPkt which uses enums
    """

    strVar = "\n"
    strVar += "bool {classname}::getUserSpecificData(UserDataPkt &pkt)\n".format(classname=sm_name)
    strVar += "{\n"
    strVar += SPD + "UserDataPktInternal internalPkt;\n\n"
    strVar += SPD + "bool status = getUserSpecificDataInternal(internalPkt);\n"
    strVar += SPD + "if(status != false)\n"
    strVar += SPD + "{\n"
    strVar += SPD + SPD + "if(internalPkt.dataType.size() > MAX_DATA_TYPE_SIZE)\n"
    strVar += SPD + SPD + "{\n"
    strVar += SPD + SPD + SPD + "SSM_ERROR(\"Data type size is greater than \"+ std::to_string(MAX_DATA_TYPE_SIZE) + \" characters\");\n"
    strVar += SPD + SPD + SPD + "return false;\n"
    strVar += SPD + SPD + "}\n"
    strVar += SPD + SPD + "strcpy(pkt.dataType,internalPkt.dataType.c_str());\n"
    strVar += SPD + SPD + "pkt.timestamp=internalPkt.timestamp;\n"
    strVar += SPD + SPD + "pkt.targetStateMachine=getSmEnum(internalPkt.targetStateMachine);\n"
    strVar += SPD + SPD + "pkt.sourceStateMachine=getSmEnum(internalPkt.sourceStateMachine);\n"
    strVar += SPD + SPD + "pkt.timestamp=internalPkt.timestamp;\n"
    strVar += SPD + SPD + "pkt.size=internalPkt.size;\n"
    strVar += SPD + SPD + "memcpy(pkt.data,internalPkt.data,sizeof(internalPkt.data));\n"
    strVar += SPD + "}\n"
    strVar += SPD + "return status;\n"
    strVar += "}\n\n"

    return strVar


def generatePrettyPrintOverload(sm_name):
    """
    Returns
    -------
        the function which overloads << operator to pretty print state enums
    """

    strVar = "\n"
    strVar += "std::ostream& operator<<(std::ostream& out, const {classname}States state_enum)\n".format(classname=sm_name)
    strVar += "{\n"
    strVar += SPD + "return out << {classname}_enumToStr(state_enum);\n".format(classname=sm_name)
    strVar += "}\n"

    return strVar


def generateGetSmEnumFunction():
    """
    Returns
    -------
        the getSmEnum() function that returns an enum for a state machine string
    """

    strVar = "\n"
    strVar += "StateMachines getSmEnum(std::string sm)\n"
    strVar += "{\n"

    strVar += SPD + "static FixedMap<std::string, StateMachines> sm_string_to_enum_map = \n"
    strVar += SPD + "{\n"
    for sm_name in SMs.keys():
        strVar += SPD + SPD+"{{SSM_{sm_name}_str, StateMachines::{sm_name}}},\n".format(sm_name=sm_name)
    strVar += SPD + "};\n\n"

    strVar += SPD + "auto it = sm_string_to_enum_map.find(sm);\n"
    strVar += SPD + "if(it != sm_string_to_enum_map.end())\n"
    strVar += SPD + "{\n"
    strVar += SPD + SPD+ "return it->second;\n"
    strVar += SPD + "}\n"
    strVar += SPD+ "return StateMachines::INVALID_STATEMACHINE;\n"
    strVar += "}\n\n"

    return strVar

def generateStrToEnumFunction(sm_name):
    """
    Returns
    -------
        the strToEnum() function that returns an enum for a string
    """

    strVar = "\n"
    strVar += "{classname}States {classname}_strToEnum(std::string state)\n".format(classname=sm_name)
    strVar += "{\n"

    strVar += SPD + "static FixedMap<std::string, {classname}States> stringToState\n".format(classname=sm_name)
    strVar += SPD + "{\n"
    for state in SMs[sm_name].m_states:
        strVar += SPD+SPD+"{{SSM_{classname}_{enum_name}_str, {classname}States::{enum_name}}},\n".format(classname=sm_name, enum_name=state)
    strVar += SPD + "};\n"
    strVar += "\n"
    strVar += SPD + "auto it = stringToState.find(state);\n"
    strVar += SPD + "if(it != stringToState.end())\n"
    strVar += SPD + "{\n"
    strVar += SPD + SPD+ "return it->second;\n"
    strVar += SPD + "}\n"
    strVar += SPD + "return {classname}States::NULL_STATE;\n".format(classname=sm_name)
    strVar+= '}\n\n'
    return strVar

def generatePrintHistogramFunction(sm_name):
    """
    Returns
    -------
        The printClientHistogram() function for the state machine definition file (<state machine name>.cpp)
    """

    strVar = '\n'
    strVar+= 'void {classname}::printClientHistogram()\n'.format(classname=sm_name)
    strVar+= '{\n'
    strVar+= SPD + 'runnableHist.printHistogram(\"executeStateMachine\");\n'
    strVar+= '}\n'
    return strVar;

def generateSetupTreeHierarchyFunction(sm_name):
    """
    Returns
    -------
        The setupTreeHierarchy() function for the state machine definition file (<state machine name>.cpp)
    """

    strVar = '\n'
    strVar+= 'bool {classname}::setupTreeHierarchy(StateMachineVector& smv, StateMachinePtr& hPtr)\n'.format(classname=sm_name)
    strVar+= '{\n'
    strVar+= SPD + 'return setupHierarchy(smv, hPtr);\n'
    strVar+= '}\n'
    return strVar;

def generateHierarchyCppFile(sm_name):
    """
    This function generates the source file for a hierachy_<state machine framework>.cpp.
    """

    strVar= ""
    strVar+= '#include <ssm/StateMachine.hpp>\n'
    strVar+= "\n"
    strVar+= 'namespace SystemStateManager\n{\n\n';
    strVar+= "namespace "+jsonFileName.upper() + "\n{\n\n";
    if StartPort != "":
        strVar+= "#define START_PORT " + StartPort + "\n\n";
    strVar+= "static std::atomic<bool> showHeadSpecs {true};\n\n";
    strVar+= "static std::mutex showHeadSpecsLock;\n\n";
    strVar+= "bool setupHierarchy(StateMachineVector &stateMachineVector, StateMachinePtr &headPtr)\n"
    strVar+= "{\n"
    strVar+= SPD + "int smStartPort = " + ("0" if StartPort == "" else "START_PORT") + ";\n"
    strVar+= SPD + "int cloneStartPort = 0;\n"

    # Add the list of all the SWCs listed in the json file
    strVar+= SPD + "SWCVector swcVector;\n"
    for swc in swcList:
        swcName = swc["name"];
        swcIP = swc["ipaddress"];
        strVar+= SPD + "swcVector.push_back({\""+swcName+"\", \""+swcIP+"\"});\n"
    strVar+="\n";

    strVar+= SPD + "std::string startPort = getKeyFromMasterQAFile(\"startport\");\n";
    strVar+= SPD + "if (startPort != \"\")\n";
    strVar+= SPD + "{\n";
    strVar+= SPD + SPD + "overrideBasePort = std::stoi(startPort);\n";
    strVar+= SPD + SPD + "setPorts();\n";
    strVar+= SPD + "}\n";
    strVar+= SPD + "std::string remoteSSMIP = getKeyFromMasterQAFile(\"remoteSSMIP\");\n";
    strVar+= SPD + "if (remoteSSMIP != \"\")\n";
    strVar+= SPD + "{\n";
    strVar+= SPD + SPD + "ssmMasterIPAddr = remoteSSMIP;\n";
    strVar+= SPD + "}\n\n";
    if StartPort == "":
        strVar+=SPD + "smStartPort = startSocketPort;\n"
    strVar+=SPD + "cloneStartPort = smStartPort + MAX_SMs_ALLOWED;\n"

    #create StateMachinePtr for the head first
    strVar+= SPD + 'StateMachinePtr {classname}_sm = StateMachinePtr(new StateMachine("{classname}", ssmMasterIPAddr.c_str(), smStartPort++, {createHeadSMChannels}));\n'.format(classname=g_head,
            SM=g_head, createHeadSMChannels=CreateHeadSMChannels);

    #create StateMachinePtr for each state machine
    for sm_name in SMs:
        if (sm_name != g_head):
            strVar+= SPD + 'StateMachinePtr {classname}_sm = StateMachinePtr(new StateMachine("{classname}", ssmMasterIPAddr.c_str(), smStartPort++, {createHeadSMChannels}));\n'.format(classname=sm_name,
                SM=sm_name, createHeadSMChannels=CreateHeadSMChannels);

    #add States
    for sm_name in SMs:
        strVar+= SPD + "{SM}_sm->addStates({{".format(SM=sm_name)
        for idx, val in enumerate(SMs[sm_name].m_states):
            strVar+= '"{st}"'.format(st=val)
            if idx != len(SMs[sm_name].m_states) - 1:
                strVar+= ", "
        strVar+= "});\n"
        strVar+= SPD + "{SM}_sm->finalizeStates();\n".format(SM=sm_name);
        if (SMs[sm_name].overrideInit == True):
            strVar+= SPD + "{SM}_sm->overRideInitPhase();\n".format(SM=sm_name);

        if (SMs[sm_name].m_isGlobalStateAware == True):
            strVar+= SPD + "{SM}_sm->setGlobalAwareness();\n".format(SM=sm_name);

        if (SMs[sm_name].m_isStateGloballyRelevant == True):
            strVar+= SPD + "{SM}_sm->setStateGloballyRelevant();\n".format(SM=sm_name);

        #disable the state machine if required
        if (SMs[sm_name].disableByDefault == True):
            strVar+= SPD + '{classname}_sm->setDisabledByDefaultFlag();\n'.format(classname=sm_name)

        strVar+= "\n"

        #add Transitions
        source_states = sorted(SMs[sm_name].m_transitions)
        for s in source_states:
            destination_states = sorted(SMs[sm_name].m_transitions[s])
            for d in destination_states:
                strVar+= SPD + 'if (!{classname}_sm->addTransition("{src}", "{dst}")) return false;\n'.format(classname=sm_name, src=s, dst=d)

        #set start state
        strVar+= SPD + 'if (!{classname}_sm->setStartState("{start_state}")) return false;\n'.format(classname=sm_name, start_state=SMs[sm_name].m_startState)

        strVar+= "\n"

    #Add Child
    for key in g_hierarchies:
        for t in g_hierarchies[key]:
            strVar+= SPD + 'if (!{classname}_sm->addChild("{state}", {childSM}_sm)) return false;\n'.format(classname=key, state=t.ParentState, childSM=t.ChildSM)
        strVar += "\n"

    #Pushback all statemachine pointers in SMVector
    for sm_name in SMs:
        strVar+= SPD + "stateMachineVector.push_back({classname}_sm);\n".format(classname=sm_name)

    # TODO: rename setIgnoreClonesFile and fix https://jirasw.nvidia.com/browse/NVSTM-1106
    # Add SWC clone state machines
    if (SwcIgnoreFile != ""):
        strVar+= SPD + "{head}_sm->setIgnoreClonesFile(\"".format(head=g_head) + SwcIgnoreFile + "\");\n"

    if (SwcListFile != ""):
        strVar+= SPD + "{head}_sm->setSWCClonesListFile(\"".format(head=g_head) + SwcListFile + "\");\n"

    strVar+= SPD + "{head}_sm->addClones(swcVector, stateMachineVector, cloneStartPort);\n".format(head=g_head);
    strVar+= "\n";

    strVar+= SPD + "headPtr = {head}_sm;\n\n".format(head=g_head)

    strVar+= SPD + "////////// CRITICAL SECTION //////////////\n";
    strVar+= SPD + "SSMLock lg(showHeadSpecsLock);\n";
    strVar+= SPD + "if (showHeadSpecs) {\n";
    strVar+= SPD + SPD + "showHeadSpecs = false;\n";
    strVar+= SPD + SPD + "SSM_LOG(\"Setup Hierarchy for "+jsonFileName.upper()+"\");\n"
    strVar+= SPD + SPD + "headPtr->logStateMachineSpecs();\n";
    strVar+= SPD + "}\n";
    strVar+= SPD + "/////////////////////////////////////////\n";
    strVar+= SPD + "finalMaxPort = smStartPort > finalMaxPort ? smStartPort : finalMaxPort;\n";
    strVar+= SPD + "finalMaxPort = cloneStartPort > finalMaxPort ? cloneStartPort : finalMaxPort;\n";
    strVar+= SPD + "return true;\n"
    strVar+= "}"
    strVar+= "}\n}\n"

    #write file
    f = open(os.path.join(subcomponent_src_directory, ("hierarchy_" + jsonFileName + ".cpp")), "w");
    for a in copyrightData:
        f.write(a);
    f.write("\n");
    f.write(strVar)
    f.close()

def changeVersions(subcomponent_directory):
    """
    Replace the version parser and SSM version number in autogenerated files
    """

    #change file is the directory
    for filename in os.listdir(subcomponent_directory):
        with open(os.path.join(subcomponent_directory,filename), "r") as sources:
            lines = sources.readlines()
        with open(os.path.join(subcomponent_directory, filename), "w") as sources:
            for line in lines:
                if(re.match("// Parser Version",line)):
                    sources.write(re.sub("// Parser Version: (\S{9})","// Parser Version: " + PARSER_VERSION, line))
                elif(re.match("// SSM Version",line)):
                    sources.write(re.sub("// SSM Version:    (\S{9})","// SSM Version:    " + SSM_VERSION, line))
                else:
                    sources.write(line)



def cleanup_directories():
    """
    Remove existing include and src directories for state machine frameworks.
    """

    if os.path.exists(subcomponent_src_directory):
        shutil.rmtree(subcomponent_src_directory)

    if os.path.exists(subcomponent_include_directory):
        shutil.rmtree(subcomponent_include_directory)


def main():

    global SMs
    global subcomponent_src_directory
    global subcomponent_include_directory
    global sm_name
    global copyrightData
    global jsonFileName

    strVar="";

    parser = argparse.ArgumentParser("Arg parser for json parser")
    parser.add_argument("-i", "--input", required=True,
                        help="input json file to parse")
    parser.add_argument("-n", "--noclone", required=False,  action='store_true',
                        help="do not create SSMClone")
    parser.add_argument("-t", "--testclone", required=False,  action='store_true',
                        help="Create a test clone")
    parser.add_argument("-src","--src", required=True,
                        help="src directory for subcomponent")
    parser.add_argument("-inc","--include", required=True,
                        help="include directory for subcomponent")
    args = parser.parse_args()

    jsonlocation = args.input
    parseJson(jsonlocation)

    dir_path = os.path.dirname(os.path.abspath(__file__))
    with open (dir_path+"/copyright.txt", "r") as myfile:
        copyrightData=myfile.readlines()

    jsonFileName = os.path.splitext(os.path.basename(args.input))[0];

    subcomponent_src_directory = args.src
    subcomponent_include_directory = args.include

    #clean up existing directories
    cleanup_directories()

    #create include dir for subcomponent
    if not os.path.exists(subcomponent_include_directory):
        os.makedirs(subcomponent_include_directory)
    #create src dir for subcomponent
    if not os.path.exists(subcomponent_src_directory):
        os.makedirs(subcomponent_src_directory)

    if (args.testclone):
        cloneName="SSMTestClone";

    # Create the dynamic CPP files
    for sm_name in SMs:
        generateDynamicHeaderFile(sm_name);
        generateDynamicCPPFile(sm_name);
        if SMs[sm_name].hasClone :
            cloneName = sm_name + "Clone"
            generateDynamicCloneHeaderFile(cloneName);
            generateDynamicCloneCPPFile(cloneName);

    generateDynamicNamesFile()
    generateDynamicMappingHeaderFile()
    generateDynamicMappingCppFile()
    generateHierarchyCppFile(sm_name)

    #change version of ssm
    changeVersions(subcomponent_src_directory)
    changeVersions(subcomponent_include_directory)

if __name__ == "__main__":
    main()
