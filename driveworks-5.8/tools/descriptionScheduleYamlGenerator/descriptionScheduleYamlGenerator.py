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
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import bisect
import json
import os
import pathlib
import sys

# only used in release to find packaged dependencies
lib_path = pathlib.Path(__file__).parent / "lib"
if lib_path.exists():
    sys.path.insert(0, str(lib_path))

import yaml
import copy
from collections import defaultdict
from enum import IntEnum
from sys import setrecursionlimit
from yaml.representer import Representer

# =============  stm_core BEGIN =============
from enum import Enum
#from typeguard import typechecked
from typing import List, Dict, Any
import warnings

class ArgsHelper():
    """
    Record the global vars needed here
    """
    def __init__(self):
        self._check_optional = False

    @property
    def check_optional(self) -> bool:
        return self._check_optional

    @check_optional.setter
    def check_optional(self, other: bool):
        if not isinstance(other, bool):
            raise TypeError("Incorrect check_optional type")
        self._check_optional = other

argsHelper = ArgsHelper()

class TimingHelper():

    def add_to_group(module_list):
        stmGroups = []
        groupDict = []
        runnableList = []
        groupList = []
        for count, module in enumerate(module_list):
            if module is None:
                continue

            # TODO(hongwang): the deepcopy limit
            sys.setrecursionlimit(5000)
            runnables = copy.deepcopy(module.getRunnables())

            # Currently empty step is not supported
            if len(runnables) == 0:
                return []

            runnables.pop(0)
            runnables.pop(len(runnables)-1)
            runnableList.append(runnables)
            resourceType = None
            typeCount = 0
            groupDict.append([{}])
            for c, r in enumerate(runnables):
                if isinstance(r, Submitter):
                    runnables.insert(c + 1, r.submittee)
                if resourceType == None:
                    resourceType = r.resourceType
                    groupDict[count][typeCount]['type'] = r.resourceType
                    groupDict[count][typeCount]['startIndex'] = c
                else:
                # Change arg to comment in the future when each group allow multiple runnables
                # elif r.resourceType != resourceType:
                    resourceType = r.resourceType
                    groupDict[count][typeCount]['endIndex'] = c
                    typeCount+=1
                    groupDict[count].append({})
                    groupDict[count][typeCount]['startIndex'] = c
                    groupDict[count][typeCount]['type'] = r.resourceType

            groupDict[count][typeCount]['endIndex'] = len(runnables)

        count = 0
        while True:
            resCount = {CPU: 0, GPU: 0, DLA: 0, VPI: 0}
            for g in groupDict:
                if len(g):
                    resCount[g[0]['type']]+=1
            res = max(resCount, key = lambda k: resCount[k])
            if resCount[res] == 0:
                break
            # TODO(hongwang): the `id`: top.objectPerception2.objectDetectorNode
            # we need `objectDetectorNode`
            name = '_'.join([module_list[0].id.split('.')[-1], res.__name__, str(count)])
            group = AliasGroup(name, len(module_list))
            count += 1
            for c, g in enumerate(groupDict):
                if len(g) and g[0]['type'] == res:
                    runnable = runnableList[c][g[0]['startIndex']]#:g[0]['endIndex']]
                    group.appendRunnable(c, runnable)
                    g.pop(0)
                else:
                    group.appendRunnable(c, None)
            groupList.append(group)

        return groupList

def typechecked(func):
    """
    Adding NOOP decorator until typeguard reflected in GVS
    """
    return func

class TimeInterval:
    """
    Class to define time intervals.
    All intervals are internally stored as ns.
    Methods are provided to read and write in different units.
    """
    to_ns = {
        "ns": 1,
        "us": 1e3,
        "ms": 1e6,
        "s" : 1e9
    }
    @typechecked
    def __init__(self, ns:float=0, us:float=0, ms:float=0, s:float=0):
        self._ns = s*TimeInterval.to_ns['s'] +\
                   ms*TimeInterval.to_ns['ms'] +\
                   us*TimeInterval.to_ns['us'] +\
                   ns*TimeInterval.to_ns['ns']

    @property
    def ns(self):
        return self._ns/TimeInterval.to_ns['ns']

    @ns.setter
    @typechecked
    def ns(self, other: int):
        self._ns = other*TimeInterval.to_ns['ns']

    @property
    def us(self):
        return self._ns/TimeInterval.to_ns['us']

    @us.setter
    @typechecked
    def us(self, other: int):
        self._ns = other*TimeInterval.to_ns['us']

    @property
    def ms(self):
        return self._ns/TimeInterval.to_ns['ms']

    @ms.setter
    @typechecked
    def ms(self, other: int):
        self._ns = other*TimeInterval.to_ns['ms']

    @property
    def s(self):
        return self._ns/TimeInterval.to_ns['s']

    @s.setter
    @typechecked
    def s(self, other: int):
        self._ns = other*TimeInterval.to_ns['s']

    def __str__(self):
        """
        Convert time to the appropriate unit for pretty printing
        """
        if self._ns >= TimeInterval.to_ns['s']:
            return '{:.2f}'.format(self.s) + ' s'
        elif self._ns >= TimeInterval.to_ns['ms']:
            return '{:.2f}'.format(self.ms) + ' ms'
        elif self._ns >= TimeInterval.to_ns['us']:
            return '{:.2f}'.format(self.us) + ' us'
        else:
            return '{:.2f}'.format(self.ns) + ' ns'

class SoC(Enum):
    TegraA = 0
    TegraB = 1

class Resource:
    """
    Base class to represent hardware and software resources.
    This class must not be instantiated.
    Instantiate one of the inherited classes.
    """
    @typechecked
    def __init__(self, name: str):
        self.name = name
        self._client = None
        self._hyperepoch = None

    @property
    def global_id(self):
        id = self.name
        if self.soc is not None:
            id = self.soc.name + "." + id
        if self.client is not None:
            id = self.client.name + "." + id

        return id

    @property
    def client(self) -> 'Client':
        return self._client

    @client.setter
    def client(self, other: 'Client'):
        if self.client is not None:
            raise ValueError("Resource already added to another client")
        self._client = other

    @property
    def hyperepoch(self) -> 'HyperEpoch':
        return self._hyperepoch

    @hyperepoch.setter
    def hyperepoch(self, other: 'HyperEpoch'):
        if self.hyperepoch is not None:
            raise ValueError("Resource already added to another HyperEpoch")
        self._hyperepoch = other

    def yml(self) -> str:
        return self.global_id

class CPU(Resource):
    """
    Class to represent CPU resources.
    example instantiation in timing file
        cpu4 = CPU("CPU4")
    """
    name = "CPU"
    @typechecked
    def __init__(self, name: str, soc: SoC = None):
        super().__init__(name)
        self.resource_type = "CPU"
        self.soc = soc

class GPU(Resource):
    """
    Class to represent GPU resources.
    example instantiation in timing file
        iGPU = GPU("iGPU")
    """
    name = 'GPU'
    @typechecked
    def __init__(self, name: str, soc: SoC):
        super().__init__(name)
        self.resource_type = "GPU"
        self.soc = soc

class DLA(Resource):
    """
    Class to represent DLA resources.
    example instantiation in timing file
        dla0 = DLA("DLA0")
    """
    name = 'DLA'
    @typechecked
    def __init__(self, name: str, soc: SoC):
        super().__init__(name)
        self.resource_type = "DLA"
        self.soc = soc

class VPI(Resource):
    """
    Class to represent VPI resources.
    example instantiation in timing file
        vpi0 = VPI("VPI0")
    """
    name = 'VPI'
    def __init__(self, name: str, soc: SoC):
        super().__init__(name)
        self.resource_type = "VPI"
        self.soc = soc

class MUTEX(Resource):
    """
    Class to represent MUTEX resources.
    example instantiation in timing file
        cuda_context_lock = MUTEX("CUDA_CONTEXT_LOCK")
    """
    @typechecked
    def __init__(self, name: str, soc: SoC = None):
        super().__init__(name)
        self.resource_type = "MUTEX"
        self.soc = soc

class Serializer(Resource):
    """
    Class to represent serializers viz.
    cuda stream, dla handle and vpi handle
    Instantiate one of the inherited classes
    """
    def yml(self) -> str:
        return {self.name : self.serializee.name}

    @property
    def soc(self) -> SoC:
        return self.serializee.soc

class CUDAStream(Serializer):
    """
    Class to represent CUDA Streams.
    example instantiation in timing file
        stream1 = CUDAStream("STREAM1", iGPU)
        # where iGPU is an already instantiated GPU
    This creates a stream resource and associates it
    with a particular GPU. This association
    is used for serialization of runnables.
    """
    name = 'CUDA_STREAM'
    @typechecked
    def __init__(self, name: str, gpu: GPU):
        super().__init__(name)
        self.resource_type = "CUDA_STREAM"
        self.serializee = gpu

class CUDLAStream(Serializer):
    """
    Class to represent CUDLA Streams.
    example instantiation in timing file
        cudla_stream_1 = CUDLAStream("CUDLA_STREAM1", dla0)
        # where dla0 is an already instantiated DLA
    This creates a cudla stream resource and associates it
    with a particular DLA resource. This association
    is used for serialization of runnables.
    """
    name = 'CUDLA_STREAM'
    @typechecked
    def __init__(self, name: str, dla: DLA):
        super().__init__(name)
        self.resource_type = "CUDLA_STREAM"
        self.serializee = dla

class DLAHandle(Serializer):
    """
    Class to represent DLA Handles.
    example instantiation in timing file
        dla_handle_1 = DLAHandle("DLA_HANDLE1", dla0)
        # where dla0 is an already instantiated DLA
    This creates a dla handle resource and associates it
    with a particular DLA resource. This association
    is used for serialization of runnables.
    """
    name = 'DLA_HANDLE'
    @typechecked
    def __init__(self, name: str, dla: DLA):
        super().__init__(name)
        self.resource_type = "DLA_HANDLE"
        self.serializee = dla

class VPIHandle(Serializer):
    """
    Class to represent VPI Handles.
    example instantiation in timing file
        vpi_handle_1 = VPIHandle("VPI_HANDLE1", vpi0)
        # where vpi0 is an already instantiated VPI
    This creates a vpi handle resource and associates it
    with a particular VPI resource. This association
    is used for serialization of runnables.
    """
    name = 'VPI_HANDLE'
    @typechecked
    def __init__(self, name: str, vpi: VPI):
        super().__init__(name)
        self.resource_type = "VPI_HANDLE"
        self.serializee = vpi

class PipelinePhase(Enum):
    PIPELINE_PHASE_UNCHANGED = 0
    PIPELINE_PHASE_1 = 1
    PIPELINE_PHASE_2 = 2

class Runnable:
    """
    Base class for runnables.
    This class must not be instantiated in timing file.
    Instantiate one of the inherited classes viz. CPURunnable, GPURunnable etc
    """
    @typechecked
    def __init__(self, name: str):
        self._name = name
        self._resources = []
        self._wcet = None
        self._dependencies = set()
        self._extraDependencies = set()
        self._parent = None
        self._start_time = None
        self._deadline = None # Not implemented
        self._client = None
        self._epoch = None
        self._extern = name.startswith("External.")
        self._pipelinePhase = None
        self._priority = None

    @property
    def parent(self) -> 'dwNode':
        return self._parent

    @parent.setter
    def parent(self, other: 'dwNode'):
        self._parent = other

    @property
    def name(self) -> str:
        if self.parent is None: # ssm pass in Client
            return self._name
        if self.parent.is_dwnode:
            return '_'.join([
                self.parent.id.replace('.','_'),
                self._name
            ])
        else: # runnable in module, i.e. ssm pass
            return '_'.join([
                self.parent.id.replace('.','_').replace('[', '').replace(']', ''),
                self._name
            ])

    @property
    def client(self) -> 'Client':
        return self._client

    @client.setter
    @typechecked
    def client(self, client: 'Client'):
        # if self.client is not None:
        #     raise ValueError("runnable already added to another client")
        self._client = client

    @property
    def epoch(self) -> 'Epoch':
        return self._epoch

    @epoch.setter
    @typechecked
    def epoch(self, epoch: 'Epoch'):
        if self.epoch is not None:
            raise ValueError("runnable already added to another epoch")
        self._epoch = epoch

    @property
    def global_id(self) -> str:
        return self.client.name + '.' + self.name


    @typechecked
    def depends(self, other: 'Runnable'):
        if isinstance(other, GPUSubmitter) and self is not other.submittee:
            self._dependencies.add(other.submittee)
        elif isinstance(other, CUDLASubmitter) and self is not other.submittee:
            self._dependencies.add(other.submittee)
        else:
            self._dependencies.add(other)

    @typechecked
    def dependsAsync(self, other: 'Runnable'):
        self._dependencies.add(other)

    def __rshift__(self, other):
        self.depends(other)

    @property
    def pipelinePhase(self) -> PipelinePhase:
        return self._pipelinePhase

    @pipelinePhase.setter
    @typechecked
    def pipelinePhase(self, pipelinePhase: 'PipelinePhase'):
        self._pipelinePhase = pipelinePhase
        if pipelinePhase == PipelinePhase.PIPELINE_PHASE_2:
            # When pipelining is enabled, prioritize the completion of runnables in phase 2
            # which may mean a slightly suboptimal schedule e2e but prevents their completion
            # times from drifting out to the end of the frame (almost 2 frame latency from when
            # they send their outputs, important for delays from active safety -> control)
            # note: we only want to do this for Phase 2 passes.
            self.priority = 1

    @property
    def priority(self) -> int:
        return self._priority

    @priority.setter
    def priority(self, priority: int):
         # 1 is the highest priority, 10 is the default / lowest (defined by STM)
        if 1 <= priority <= 10 :
            self._priority = priority
        else:
            raise ValueError("runnable priority should be between 1 (highest) and 10 (lowest)")

    @property
    def wcet(self) -> TimeInterval:
        return self._wcet

    @wcet.setter
    @typechecked
    def wcet(self, wcet: TimeInterval):
        self._wcet = wcet

    @property
    def start_time(self) -> TimeInterval:
        return self._start_time

    @start_time.setter
    @typechecked
    def start_time(self, start_time: TimeInterval):
        self._start_time = start_time

    @property
    def deadline(self) -> TimeInterval:
        return self._deadline

    @deadline.setter
    @typechecked
    def deadline(self, deadline: TimeInterval):
        self._deadline = deadline

    @property
    def resources(self) -> List[Resource]:
        if not self._resources:
            return [self._resourceType]
        else:
            return self._resources

    @property
    def resourceType(self):
        return self._resourceType

    @resources.setter
    @typechecked
    def resources(self, other: List[Resource]):
        if not all(isinstance(resource, (self._resourceType, MUTEX)) for resource in other):
            raise TypeError("Incorrect resource type")
        self._resources = other

    # TODO(yel, ryanw): sync with stm team
    @typechecked
    def appendResources(self, other: List[Resource]):
        resourcesHelper = self.resources + other
        self._resources.extend(resourcesHelper)

    def getResources(self):
        return self._resources

    def yml(self) -> Dict[str, Dict]:
        ymlDict = {}
        if self.wcet is None:
            #warnings.warn("{} wcet not found. Setting 1 us".format(self.global_id))
            self.wcet = TimeInterval(us=1)
        if self.epoch is None:
            raise ValueError("runnable epoch not set: " + self.name)
        if self.client is None:
            raise ValueError("runnable client not set: " + self.name)
        ymlDict["WCET"] = str(self.wcet)
        if self.start_time is not None:
            ymlDict["start_time"] = str(self.start_time)
        if self.deadline is not None:
            ymlDict["deadline"] = str(self.deadline)
        if self._extern:
            ymlDict["Extern"] = str(self.name[len("External."):])
        if self._dependencies:
            ymlDict["Dependencies"] = sorted([
                dep.global_id for dep in self._dependencies
            ])
        if self._extraDependencies:
            ymlDict["Dependencies"] = sorted(ymlDict.get("Dependencies", []) + self._extraDependencies)
        ymlDict["Resources"] = sorted(resource.name for resource in self.resources)
        if self._priority:
            ymlDict["Priority"] = self._priority
        return {self.name : ymlDict}

class CPURunnable(Runnable):
    """
    Class for CPU Runnables.
    """
    @typechecked
    def __init__(self, name: str):
        super().__init__(name)
        self._resourceType = CPU

class Submitter(Runnable):
    """
    Base class to represent submitters.
    A virtual submittee runnable is instantiated for every Submitter.
    The assumption is that every submitter has an associated submittee.
    Setting the serializer for the submitter automatically sets the
    corresponding resources for the submittee.
    Instantiate one of the inherited classes
    """
    @typechecked
    def __init__(self, name: str):
        super().__init__(name)
        self._resourceType = CPU
        self._serializer = None

    @property
    def resources(self) -> List[Resource]:
        resources = super().resources
        resources.append(self.serializer)
        return resources

    @resources.setter
    def resources(self, other: List[Resource]):
        Runnable.resources.fset(self, other)

    @property
    def serializer(self) -> Serializer:
        if self._serializer is None:
            return self.serializerType
        else:
            return self._serializer

    @serializer.setter
    @typechecked
    def serializer(self, serializer: Serializer):
        if not self.submittee.resources:
            raise ValueError("Submittee resources already specified")
        self._serializer = serializer

        if not self.submittee._resources:
            self.submittee._resources = [serializer.serializee]
            return

        for i, resource in enumerate(self.submittee._resources):
            if isinstance(resource, self.submittee._resourceType):
                self.submittee._resources[i] = serializer.serializee

    def yml(self) -> Dict[str, Dict]:
        ymlDict = super().yml()
        ymlDict[self.name]["Submits"] = self.submittee.global_id
        return ymlDict

class GPUSubmitter(Submitter):
    """
    Class to represent GPU Submitters
    """
    def __init__(self, name: str):
        super().__init__(name)
        self.submittee = GPUSubmittee(self._name + "_submittee")
        self.submittee.submitter = self
        self.submittee.depends(self)
        self.serializerType = CUDAStream

    @property
    def stream(self) -> CUDAStream:
        return super().serializer

    @stream.setter
    def stream(self, other: CUDAStream):
        self.serializer = other

class DLASubmitter(Submitter):
    """
    Class to represent DLA Submitters
    """
    def __init__(self, name: str):
        super().__init__(name)
        self.submittee = DLASubmittee(self._name + "_submittee")
        self.submittee.submitter = self
        self.submittee.depends(self)
        self.serializerType = DLAHandle

    @property
    def handle(self) -> DLAHandle:
        return super().serializer

    @handle.setter
    def handle(self, other: DLAHandle):
        self.serializer = other

class CUDLASubmitter(Submitter):
    """
    Class to represent CUDLA Submitters
    """
    def __init__(self, name: str):
        super().__init__(name)
        self.submittee = CUDLASubmittee(self._name + "_submittee")
        self.submittee.submitter = self
        self.submittee.depends(self)
        self.serializerType = CUDLAStream

    @property
    def handle(self) -> CUDLAStream:
        return super().serializer

    @handle.setter
    def handle(self, other: CUDLAStream):
        self.serializer = other

class VPISubmitter(Submitter):
    """
    Class to represent VPI Submitters
    """
    def __init__(self, name: str):
        super().__init__(name)
        self.submittee = VPISubmittee(self._name + "_submittee")
        self.submittee.submitter = self
        self.submittee.depends(self)
        self.serializerType = VPIHandle

    @property
    def handle(self) -> VPIHandle:
        return super().serializer

    @handle.setter
    def handle(self, other: VPIHandle):
        self.serializer = other

class Submittee(Runnable):
    """
    Class for GPU submits.
    This class must not be instantiated in timing file.
    This class is automatically instantiated for every GPU Submitter.
    """
    def __init__(self, name: str):
        super().__init__(name)
        self._resourceType = GPU
        self.submitter = None

    @property
    def resources(self) -> List[Resource]:
        return super().resources

    @resources.setter
    def resources(self, other):
        if not self.submitter._serializer is None:
            raise ValueError("Submitter stream already set")
        Runnable.resources.fset(self, other)

    @property
    def parent(self) -> 'dwNode':
        return self.submitter.parent

    @property
    def client(self) -> 'Client':
        return self.submitter.client

    @property
    def epoch(self) -> 'Epoch':
        return self.submitter.epoch

class GPUSubmittee(Submittee):
    def __init__(self, name: str):
        super().__init__(name)
        self._resourceType = GPU

class DLASubmittee(Submittee):
    def __init__(self, name: str):
        super().__init__(name)
        self._resourceType = DLA

class VPISubmittee(Submittee):
    def __init__(self, name: str):
        super().__init__(name)
        self._resourceType = VPI

class CUDLASubmittee(Submittee):
    def __init__(self, name: str):
        super().__init__(name)
        self._resourceType = DLA

class Client:
    """
    Class to define STM Clients
    Instantiate this class in timing file and add runnables to it.
    """
    @typechecked
    def __init__(self,
                 name: str,
                 soc: SoC,
                 runnables: List[Runnable],
                 externalRunnables: List[Runnable],
                 resources: List[Resource]):
        self.name = name
        self.soc = soc
        self.ssm_pass = CPURunnable(name + "_ssm_pass_0")
        self.ssm_pass.wcet = TimeInterval(ns=100)
        # handle the case where shadow only has external runnables
        self.runnables = []
        if runnables:
            self.runnables.append(self.ssm_pass)
        self.runnables.extend(runnables)
        for runnable in self.runnables:
            runnable.client = self
        self.externalRunnables = externalRunnables
        for externalRunnable in self.externalRunnables:
            externalRunnable.client = self

        self.resources = resources
        for resource in self.resources:
            resource.client = self

    def handleSSMPassDependency(self):
        # put the ssm pass at first epoch
        # handle the case where client only has external runnables
        if len(self.runnables) < 2:
            if not self.externalRunnables:
                raise ValueError("Client " + self.name + " has no runnable.")
        else:
            firstRunnable = self.runnables[1]
            self.ssm_pass.epoch = firstRunnable.epoch
            for runnable in self.runnables[1:]:
                if runnable.epoch is self.ssm_pass.epoch:
                    runnable >> self.ssm_pass

    def ymlResources(self) -> Dict:
        resourceTree = defaultdict(lambda: list())
        for resource in sorted(self.resources, key=lambda x: x.resource_type):
            if resource.resource_type == "MUTEX":
                bisect.insort(resourceTree[resource.resource_type], resource.name)
            ## bisect insort could not compare the order of a dict. This issue will met when having more than one
            ## same resource_type resources, WAR to use append for CUDLA_STREAM directly
            elif resource.resource_type == "CUDLA_STREAM":
                resourceTree[resource.resource_type].append(resource.yml())
            else:
                bisect.insort(resourceTree[resource.resource_type], resource.yml())
        return resourceTree

    def yml(self) -> Dict:
        # at this point, epoch entry of self.runnables should all be filled in
        self.handleSSMPassDependency()
        ymlEpochs = defaultdict(lambda: {"Runnables": list()})
        # handle the case where client only has external runnables
        if not self.runnables and not self.externalRunnables:
            raise ValueError("Client " + self.name + " has no runnable.")
        for runnable in sorted(self.runnables, key=lambda x: x.name):
            if runnable.epoch is None:
                continue
            ymlEpochs[runnable.epoch.global_id]["Runnables"].append(runnable.yml())
            if isinstance(runnable, Submitter):
                ymlEpochs[runnable.epoch.global_id]["Runnables"].append(runnable.submittee.yml())
        for externalRunnable in sorted(self.externalRunnables, key=lambda x: x.name):
            yml = externalRunnable.yml()
            key = list(yml.keys())[0]
            new_key = key[len("External."):]
            yml[new_key] = yml.pop(key)
            ymlEpochs[externalRunnable.epoch.global_id]["Runnables"].append(yml)
        return {
            self.name: {
                "SOC": self.soc.name,
                "Resources" : self.ymlResources(),
                "Epochs" : [{key: ymlEpochs[key]} for key in sorted(ymlEpochs.keys())],
            }
        }

    def findRunnable(self, inputRunnable: Runnable) -> Runnable:
        ret = None
        for runnable in self.runnables:
            if isinstance(runnable, Submitter):
                if runnable.submittee.name == inputRunnable.name:
                    ret = runnable.submittee
                    break
            if runnable.name == inputRunnable.name:
                ret = runnable
                break

        return ret

class AliasGroup:
    def __init__(self,
                 name: str,
                 steps: int,
                 runnables: List[Runnable] = None):
        self.name = name
        self.runnables = runnables
        self.steps = steps
        if self.runnables is None:
            self.runnables = [None for i in range(self.steps)]

    def appendRunnable(self,
                        step: int,
                        runnable: Runnable):
        if step >= self.steps:
            raise ValueError("step out of bound, current: " + step + ", maximum: " + self.steps)
        self.runnables[step] = runnable

    def yml(self) -> Dict:
        names = []
        for r in self.runnables:
            if r is None:
                names.append("None")
            else:
                names.append(r.global_id)
        return {self.name: {'Steps': [name for name in names]}}

    def deepcopy(self, clients: List[Client]):
        newRunnables = []
        for r in self.runnables:
            if r is None:
                newRunnables.append(None)
                continue
            newR = None
            for c in clients:
                newR = c.findRunnable(r)
                if newR is not None and newR.global_id == r.global_id:
                    break
            newRunnables.append(newR)

        return AliasGroup(self.name, self.steps, newRunnables)
class Epoch:
    """
    Class to define Epochs
    Instantiate this class in timing file and add Clients to it.
    """
    def __init__(self,
                 name: str,
                 runnables: List[Runnable],
                 externalRunnables: List[Runnable],
                 period: TimeInterval,
                 groups: List[AliasGroup] = None,
                 frames: int = None,
                 deadline: TimeInterval = None):
        self.name = name
        self.period = period
        self.frames = frames
        self.deadline = deadline
        self.runnables = runnables
        self.externalRunnables = externalRunnables
        for runnable in self.runnables:
            runnable.epoch = self
        for externalRunnable in self.externalRunnables:
            externalRunnable.epoch = self
        self._hyperepoch = None
        self.groups = groups
        if groups is None:
            self.groups = []

    @property
    def hyperepoch(self) -> 'HyperEpoch':
        return self._hyperepoch

    @hyperepoch.setter
    @typechecked
    def hyperepoch(self, other: 'HyperEpoch'):
        if self.hyperepoch is not None:
            raise ValueError("Epoch already assisgned to another HyperEpoch")
        self._hyperepoch = other

    @property
    def global_id(self) -> str:
        return self.hyperepoch.name + '.' + self.name

    def setGroups(self, groups: List[AliasGroup]):
        self.groups = groups

    def yml(self) -> Dict:
        ret = {self.name: {"Period": str(self.period)}}
        if self.frames is not None:
            ret[self.name]["Frames"] = self.frames
        if self.groups is not None and len(self.groups) > 0:
            ret[self.name]["AliasGroups"] = [group.yml() for group in self.groups]
        return ret

class HyperEpoch:
    """
    Class to define HyperEpochs
    """
    @typechecked
    def __init__(self,
                 name: str,
                 epochs: List[Epoch],
                 resources: List[Resource] = [],
                 period: TimeInterval = None,
    ):
        self.name = name
        self.period = period
        self.resources = resources
        for resource in self.resources:
            resource.hyperepoch = self

        self.epochs = epochs
        for epoch in self.epochs:
            epoch.hyperepoch = self

    def yml(self) -> Dict:
        ret = {self.name: {
            "Epochs" : [epoch.yml() for epoch in sorted(self.epochs, key=lambda x: x.name)],
            "Resources" : sorted(resource.global_id for resource in self.resources)
        }}
        if self.period is not None:
            ret[self.name]["Period"] = str(self.period)
        return ret

class STM_graph:
    """
    Top level graph to define STM timing graph.
    Instantiate this graph and add epochs to it.
    """
    @typechecked
    def __init__(self,
                 name: str,
                 resources: List[Resource],
                 hyperepochs: List[HyperEpoch],
                 clients: List[Client],
                 identifier: int = 0,
                 wcet_file: str = None,
                 extern: bool = False):
        self.name = name
        self.resources = resources
        self.hyperepochs = hyperepochs
        self.clients = clients
        self.wcets = None
        self.identifier = identifier
        self.extern = extern
        # TODO(yel): rcsPorts and rcsPassMapping are for RoadCastService which is not in DAG but still scheduled
        #            by STM for now
        self.rcsPorts = {}
        self.rcsUpstreams = set() # no mailbox/indirect check
        self.rcsPassMapping = {}
        if wcet_file:
            wcet_file = FileValidation.wcet_validate(wcet_file)
            with open(wcet_file) as infile:
                self.wcets = yaml.safe_load(infile)
            for runnable in self.runnables:
                if runnable.global_id in self.wcets:
                    runnable.wcet = TimeInterval(ns=self.wcets[runnable.global_id])
                if isinstance(runnable, Submitter):
                    if runnable.submittee.global_id in self.wcets:
                        runnable.submittee.wcet = TimeInterval(ns=self.wcets[runnable.submittee.global_id])

    @property
    def runnables(self) -> List[Runnable]:
        return {runnable for client in self.clients
                for runnable in client.runnables}

    @property
    def dwNodes(self) -> List['dwNode']:
        return {runnable.parent for runnable in self.runnables if runnable.parent is not None and runnable.parent.is_dwnode}

    def findRunnable(self, inputRunnable: Runnable) -> Runnable:
        for client in self.clients:
            r = client.findRunnable(inputRunnable)
            if r is not None and r.global_id == inputRunnable.global_id:
                return r
        return None

    def setRcsPorts(self, ports):
        self.rcsPorts = ports

    def setRcsPassMapping(self, passMapping):
        self.rcsPassMapping = passMapping

    @typechecked
    def getFrameworkPass(self) -> List[Runnable]:
        # Create framework pass for each client's each Epoch.
        # To support application layer service injection
        # Now only support sync(at the beginning of a Epoch) and RCS

        # get virtualPortName to dwNode dict from graph
        virtualPort2Node = {}
        moduleSet = {node.parent for node in self.dwNodes if node.parent is not None}
        for module in moduleSet:
            for portName, portInstance in module.outputPorts.items():
                # skip dangling virtual port. actually this is invalid in description
                if portInstance.upstream == None:
                    continue
                virtualPort2Node[module.id + "." + portInstance.name] = portInstance.upstream.parent.id
        # get all producers for RCS
        def convertGdlPortName2Cgf(gdlPortName):
            portName = gdlPortName[gdlPortName.rfind('.') + 1 + len('output'):]
            # fix index at the end
            if '__' in portName:
                portName = portName[:portName.find('__')] + '[' + portName[portName.find('__') + 2:] + ']'
            fixedPortName = ""
            for c in portName:
                if c.isupper():
                    fixedPortName = fixedPortName + '_' + c
                else:
                    fixedPortName = fixedPortName + c
            return gdlPortName[:gdlPortName.rfind('.') + 1] + fixedPortName[1:].upper() # remove first "_" and make it upper case
        for portFullName in self.rcsPorts.keys():
            fixedPortName = convertGdlPortName2Cgf(portFullName)
            if fixedPortName in virtualPort2Node:
                self.rcsUpstreams.add(virtualPort2Node[fixedPortName])

        # create standalone framework_pass "rcs" for RoadCastService.
        # now only cameraEpoch and ImuEpoch are active for both rcs framework_pass
        rcsEpochs = []
        if "lowFrequency" in self.rcsPassMapping:
            rcsEpochs.append(self.rcsPassMapping["lowFrequency"])
        if "highFrequency" in self.rcsPassMapping:
            rcsEpochs.append(self.rcsPassMapping["highFrequency"])
        # create framework_pass "sync" for the cameraEpoch only now. No sync task needed for other epochs yet.
        # the sync framework_pass is shceduled at the beginning of an epoch and all nodes in the epoch
        # are supposed to depends on the sync framework_pass.
        syncEpochs =["cameraEpoch"]
        frameworkPassList = []
        # TODO(yel): For CGFDemo, there are a lot clients, this means total clientCount * epochCount sync passes
        #            are generated. Once GDL is totally deprecated, new design needed for the case.
        for client in self.clients:
            epochList = []
            for runnable in client.runnables:
                # skip non-node runnables and marked epoch
                if not runnable.name.endswith("_ssm_pass_0") \
                   and not runnable.name.endswith("_framework_pass_rcs") \
                   and not runnable.name.endswith("_framework_pass_sync") \
                   and runnable.epoch not in epochList:
                    epochList.append(runnable.epoch)

            for epoch in epochList:
                if not epoch:
                    continue
                syncFrameworkPass = None
                rcsFrameworkPass = None
                frameworkPassTmpList = []
                # instantiate sync pass
                if epoch.name in syncEpochs:
                    syncFrameworkPass = CPURunnable(client.name + "_" + epoch.name + "_framework_pass_sync")
                    frameworkPassTmpList.append(syncFrameworkPass)
                # instantiate rcs pass
                if epoch.name in rcsEpochs:
                    rcsFrameworkPass = CPURunnable(client.name + "_" + epoch.name + "_framework_pass_rcs")
                    frameworkPassTmpList.append(rcsFrameworkPass)
                if syncFrameworkPass == None and rcsFrameworkPass == None:
                    continue
                # set default values for passes
                for frameworkPass in frameworkPassTmpList:
                    # assign epoch & client, set wcet, push into client
                    frameworkPass.epoch  = epoch
                    frameworkPass.client = client
                    frameworkPass.wcet = TimeInterval(ns=100)
                    if self.wcets and frameworkPass.global_id in self.wcets:
                        frameworkPass.wcet = TimeInterval(ns=self.wcets[frameworkPass.global_id])
                    client.runnables.append(frameworkPass)
                if syncFrameworkPass and rcsFrameworkPass:
                    # syncFrameworkPass is always scheduled at the beginning of an epoch
                    rcsFrameworkPass.depends(syncFrameworkPass)
                # set dependency for all runnables in this epoch instead client.
                checked_module = set()
                for runnable in epoch.runnables:
                    if runnable.client is None:
                        continue
                    # skip ssm_pass_0
                    if runnable.name.endswith("ssm_pass_0"):
                        continue
                    # special case for framework_pass
                    if "_framework_pass_" in runnable.name:
                        continue
                    moduleName = runnable.parent.id
                    if moduleName in checked_module:
                        continue
                    checked_module.add(moduleName)
                    if syncFrameworkPass:
                        # add sync pass as all recv runnables' dependency
                        # find parent's recvPass. Ignore other passes
                        # This helps to better compare performace for pipeline vs non-pipeline mode & dataset vs in car too
                        recvPass = self.findRunnable(runnable.parent.passes[0])
                        recvPass.depends(syncFrameworkPass)
                    if rcsFrameworkPass:
                        # add all producers to rcsFrameworkPass dependency
                        # don't add dependency if no connection between RCS and the node
                        if not moduleName in self.rcsUpstreams:
                            continue
                        # find parent's sendPass. Ignore other passes
                        # special for cameraNode
                        outputPass = runnable.parent.passes[-1]
                        if 'cameraNode' in runnable.parent.id:
                            # Exception, depend on native process pass of cameraNode
                            # if only consuming native process image output
                            outputPass = runnable.parent.passes[1]
                        sendPass = self.findRunnable(outputPass)
                        # don't add dependency if the pass belongs to pipeline phase 1 !!!
                        if not sendPass.pipelinePhase == PipelinePhase.PIPELINE_PHASE_1:
                            rcsFrameworkPass.depends(sendPass)
                # append frameworkPassList
                if rcsFrameworkPass:
                    frameworkPassList.append(rcsFrameworkPass)
                if syncFrameworkPass:
                    frameworkPassList.append(syncFrameworkPass)
        return frameworkPassList

    def ymlResources(self) -> Dict:
        socResources = []
        tegraAResources = [r for r in self.resources if r.soc == SoC.TegraA]
        tegraBResources = [r for r in self.resources if r.soc == SoC.TegraB]
        if len(tegraAResources) > 0:
            resourceTree = defaultdict(lambda: list())
            for resource in sorted(tegraAResources, key=lambda x: x.resource_type):
                bisect.insort(resourceTree[resource.resource_type], resource.name)
            socResources.append({"TegraA": {"Resources": resourceTree}})
        if len(tegraBResources) > 0:
            resourceTree = defaultdict(lambda: list())
            for resource in sorted(tegraBResources, key=lambda x: x.resource_type):
                bisect.insort(resourceTree[resource.resource_type], resource.name)
            socResources.append({"TegraB": {"Resources": resourceTree}})
        else:
            # this is WAR that multi-SoC STM assume there must be tegraB resources
            # will remove once the stmcompiler support there is no tegraB resources
            # add a dummy tegraB resources
            socResources.append({"TegraB": {"Resources": {"CPU": ["CPU0"]}}})

        return socResources

    def yml(self) -> Dict:
        # insert framework pass before dump. This is mainly for RoadCastService for now.
        self.getFrameworkPass()
        ymlTree = {}
        ymlTree["Version"] = "3.0.0"
        if not self.extern:
            ymlGraph = ymlTree[self.name] = {}
            ymlGraph["Identifier"] = self.identifier
            ymlGraph["SOC"] = self.ymlResources()
            ymlGraph["Hyperepochs"] = [hyperepoch.yml() for hyperepoch in sorted(self.hyperepochs, key=lambda x: x.name)]
            ymlGraph["Clients"] = [client.yml() for client in sorted(self.clients, key=lambda x: x.name)]
        else:
            for client in self.clients:
                client.handleSSMPassDependency()
                ymlGraph = ymlTree[client.name] = {}
                ymlGraph["Runnables"] = [runnable.yml() for runnable in sorted(client.runnables, key=lambda x: x.name)]
                for item in ymlGraph["Runnables"]:
                    for value in item.values():
                        if "Dependencies" in value.keys():
                            value["Dependencies"] = sorted([x.split(".",1)[1] for x in value["Dependencies"]])
        return ymlTree

def createStartPass() -> Runnable:
    """
    Returns a CPURunnable startNode
    """
    return CPURunnable("pass_0")

def createResetPass(idx) -> Runnable:
    """
    Returns a CPURunnable startNode
    """
    return CPURunnable("pass_" + str(idx))
# =============  stm_core END =============

class FileValidation:
    """
    FileValidation is used for file sanity check, and if fileName does not exist, check if offseted path exists.
    """
    path_offset = ""

    # try to insert path offset to find file name, working up the tree.
    def searchInsertPathOffset(filename):
        tokens = filename.split("/")
        for i in range(len(tokens) - 1, -1, -1):
            searchTokens = tokens[:i]
            searchTokens.append(FileValidation.path_offset)
            searchTokens.extend(tokens[i:])
            searchString = "/".join(searchTokens)
            if os.path.isfile(searchString):
                return searchString
        raise ValueError("File not found " + filename + " with path offset " + FileValidation.path_offset)


    def validate(fileName, tag0, tag1 = "gdl-graphlet") -> str:
        # tag1 for the second input of validation
        # e.g. *.node.json / *.gdl-node.json should be both fine
        if not isinstance(fileName, str) or \
           (not fileName.endswith(tag0 + '.json') and not fileName.endswith(tag1 + '.json')):
            raise ValueError('Wrong descriptionType for generating Description: ' + fileName)
        if not os.path.isfile(fileName):
            fileName0 = fileName
            if FileValidation.path_offset in fileName:
                fileName1 = fileName.replace(FileValidation.path_offset, "")
                if not os.path.isfile(fileName1):
                    raise ValueError('Description file does not exist in both paths:\n' + 'fileName0: ' + fileName0 + '\n' + 'fileName1: ' + fileName1 + '\n')
                return fileName1
            fileName = FileValidation.searchInsertPathOffset(fileName0)
        return fileName

    def wcet_validate(fileName) -> str:
        if not fileName.endswith(".yaml"):
            raise ValueError('Wrong file extension for wcet: ' + fileName)
        if not os.path.isfile(fileName):
            if FileValidation.path_offset in fileName:
                fileName = fileName.replace(FileValidation.path_offset, "")
            # WAR(hongwang): The yaml file relative path is hardcoded now
            if not os.path.isfile(fileName):
                fileName = fileName.replace("_systemdescription/", "")
            if not os.path.isfile(fileName):
                raise ValueError('Description file does not exist: ' + fileName)
        return fileName

def keyValidation(keyList, jsonObj, tag, fileName):
    for key in keyList:
        if not key in jsonObj:
            raise KeyError('Missing required key(s) ' + key + ' in description for generating ' + tag + ' Description: ' + fileName)

def fixPath(fileName, baseDir):
    if not fileName.startswith('/'):
        return os.path.abspath(os.path.join(baseDir, fileName))
    else:
        return fileName

class HardwareResources:
    def __init__(self, resourceDict, soc):
        self.resourcesDict = {}
        self.resourcesDict['CPU'] = [] if not 'CPU' in resourceDict else [CPU(x, soc) for x in resourceDict['CPU']]
        self.resourcesDict['GPU'] = [] if not 'GPU' in resourceDict else [GPU(x, soc) for x in resourceDict['GPU']]
        self.resourcesDict['DLA'] = [] if not 'DLA' in resourceDict else [DLA(x, soc) for x in resourceDict['DLA']]
        self.resources = [resource for resourceList in self.resourcesDict.values() for resource in resourceList]
    def getResourceByType(self, resType):
        return [] if not resType in self.resourcesDict.keys() else self.resourcesDict[resType]
    def getResources(self):
        return self.resources

class PortType(IntEnum):
    INPUT   = 0,
    OUTPUT  = 1,
    VIRTUAL = 2,
    INVALID = 3

class Port:
    def __init__(self, name):
        self.name   = name
        self.type   = PortType.INVALID
        self.parent = None
        self.singleton = False

class InputPort(Port):
    def __init__(self, name, parent):
        self.name   = name
        self.type   = PortType.INPUT
        self.parent = parent
        self.peer   = None # unique
        self.singleton = False

class OutputPort(Port):
    def __init__(self, name, parent):
        self.name     = name
        self.type     = PortType.OUTPUT
        self.parent   = parent
        self.peers    = [] # list
        self.indirect = []
        self.singleton = False

class VirtualPort(Port):
    def __init__(self, name, parent):
        self.name          = name
        self.type          = PortType.VIRTUAL
        self.parent        = parent
        self.upstream      = None # unique
        self.downstreams   = []
        self.indirectFlags = []
        self.singleton = False

class DescriptionType(IntEnum):
    NODE     = 0
    GRAPHLET = 1
    APP      = 2
    SCHEDULE = 3
    INVALID  = 4

class ScheduleDescription:
    def __init__(self, baseDir, scheduleJson, scheduleKey, identifier, isDefaultSchedule, systemDescription):
        self.type = DescriptionType.SCHEDULE
        self.baseDir  = baseDir
        self.scheduleKey = scheduleKey
        self.identifier = identifier
        self.isDefaultSchedule = isDefaultSchedule
        self.extern = False
        self.scheduleJson = scheduleJson
        # required key validation
        HYPEREPOCHS = 'hyperepochs'
        WCET_FILE   = 'wcet'
        PASS_DEPENDENCIES = 'passDependencies'
        # wcet file
        self.wcetFile = None
        if self.scheduleJson[WCET_FILE]:
            self.wcetFile = os.path.join(self.baseDir, self.scheduleJson[WCET_FILE])
        # reference to system DAG
        self.systemDescription = systemDescription
        self.graphlet = GraphletDescription(self.systemDescription.getId(), self.systemDescription.getGraphletFile(), self.systemDescription) # use system.id as top graphlet id.
        clientInfo = ScheduleDescription.__gatherClientInfo(self.systemDescription.getProcesses())
        resourcesConfig = ScheduleDescription.__parseResources(clientInfo, self.scheduleJson[HYPEREPOCHS])
        if not list(resourcesConfig["hyperepochResources"].values())[0]:
            self.extern = True
        # ===== resources =====
        # TODO(xingpengd) remove hard coded machine name machine0 and machine1
        # TODO(xingpengd) remove hard coded mapping between machine name and SoC name
        self.hardwareResourcesA = HardwareResources({} if not "machine0" in resourcesConfig["globalResources"] else resourcesConfig["globalResources"]["machine0"], SoC.TegraA)
        self.hardwareResourcesB = HardwareResources({} if not "machine1" in resourcesConfig["globalResources"] else resourcesConfig["globalResources"]["machine1"], SoC.TegraB)
        self.hardwareResources  = self.hardwareResourcesA.getResources() + self.hardwareResourcesB.getResources()
        self.softwareResources  = {} # created in self.parseClients. Dict of clientName: list(Resource)
        self.cpu                = CPU("CPU") # TODO(yel): what is this for?
        # ====== clients ======
        self.clients = self.parseClients(clientInfo, resourcesConfig["clientResources"], scheduleJson[HYPEREPOCHS])
        # ==== hyperepochs ====
        self.hyperepochs = self.parseHyperepochs(self.scheduleJson[HYPEREPOCHS], resourcesConfig["hyperepochResources"])
        # ==== dependency  ====
        self.extraDeps = self.scheduleJson[PASS_DEPENDENCIES] if PASS_DEPENDENCIES in self.scheduleJson else None

        self.generateDependency()

    @staticmethod
    def __gatherClientInfo(processes):
        dwPrograms = {}
        for proc, config in processes.items():
            if "subcomponents" in config and "Loader" in config["executable"]:
                if not "runOn" in config:
                    raise RuntimeError("Process " + proc + " is not specified on which machine it will be run")
                dwPrograms[proc] = config
        return dwPrograms

    @staticmethod
    def __determineResourceType(resourceName):
        type = ""
        if resourceName.startswith("CPU"):
            type = "CPU"
        elif "iGPU" == resourceName:
            type = "GPU"
        elif "dGPU" == resourceName:
            type = "GPU"
        elif resourceName.startswith("DLA"):
            type = "DLA"
        elif resourceName.startswith("VPI"):
            type = "VPI"
        elif resourceName.startswith("CUDA_STREAM"):
            type = "CUDA_STREAM"
        elif resourceName.startswith("CUDLA_STREAM"):
            type = "CUDLA_STREAM"
        elif resourceName.startswith("DLA_HANDLE"):
            type = "DLA_HANDLE"
        elif resourceName.startswith("VPI_HANDLE"):
            type = "VPI_HANDLE"
        else:
            type = "MUTEX"
        return type

    @staticmethod
    def __getGlobalResource(hyperepochs, machineName):
        ret = {}
        for hyperepoch, hyperepochDesc in hyperepochs.items():
            for res in hyperepochDesc.get("resources", {}).keys():
                ids = res.split(".")
                # ids[0] could be machine name, client name or resource name
                if ids[0] == machineName:
                    resourceName = ids[1]
                    if ":" in resourceName:
                        raise RuntimeError("Hardware resource name cannot be mapped")
                    resType = ScheduleDescription.__determineResourceType(resourceName)
                    if not resType in ("CPU", "GPU", "VPI", "DLA"):
                        raise RuntimeError("Only hardware resources can be specified with machine name")
                    if resType in ret:
                        if resourceName in ret[resType]:
                            raise RuntimeError("The same resource is specified for two hyperepochs")
                        else:
                            ret[resType].add(resourceName)
                    else:
                        ret[resType] = set()
                        ret[resType].add(resourceName)
        return {resType: list(ret[resType]) for resType in ret.keys()}

    @staticmethod
    def __getClientResource(hyperepochs, clientName):
        ret = {}
        for hyperepoch, hyperepochDesc in hyperepochs.items():
            for res, runnables in hyperepochDesc.get("resources", {}).items():
                ids = res.split(".")
                # ids[0] could be machine name, client name or resource name
                if ids[0] == clientName:
                    resourceNameWithMapping = ids[1]
                    ret["resources"] = ret.get("resources", {})
                    resType = ScheduleDescription.__determineResourceType(resourceNameWithMapping)
                    if resType in ("CPU", "GPU", "DLA", "VPI"):
                        raise RuntimeError("Only software resources can be spcified with client name")
                    if resType in ("CUDA_STREAM", "CUDLA_STREAM", "DLA_HANDLE", "VPI_HANDLE"):
                        if not ":" in resourceNameWithMapping:
                            raise RuntimeError("CUDA_STREAM, CUDLA_STREAM, DLA_HANDLE, VPI_HANDLE resource instance need to be mapped to a hardware resource")
                    elif resType == "MUTEX":
                        pass
                    else:
                        raise RuntimeError("Unsupported resources type")
                    ret["resources"][resType] = ret["resources"].get(resType, [])
                    ret["resources"][resType].append(resourceNameWithMapping)
                    if len(runnables) > 0:
                        resourceName = resourceNameWithMapping.split(":")[0]
                        ret["resourcesAssignment"] = ret.get("resourcesAssignment", {})
                        ret["resourcesAssignment"][resourceName] = runnables
        return ret

    @staticmethod
    def __getHyperepochResource(hyperepochs, clientInfo, hyperepochName):
        ret = []
        hyperepochDesc = hyperepochs[hyperepochName]
        for res in hyperepochDesc.get("resources", {}).keys():
            ids = res.split(".")
            if ids[0] in clientInfo:
                # client resources
                ret.append(ids[0] + "." + clientInfo[ids[0]]["runOn"] + "." + ids[1].split(":")[0])
            else:
                ret.append(res) # hardware resources
        return ret

    @staticmethod
    def __parseResources(clientInfo, hyperepochs):
        # in the form of {"machineName": {"resourcesType": [<resource list>]}}
        machines = set()
        for _, client in clientInfo.items():
            machines.add(client["runOn"])
        globalResources = {
            machineName: ScheduleDescription.__getGlobalResource(hyperepochs, machineName)
            for machineName in machines
        }
        # in the form of {
        #    "clientName": {
        #        "resources": {
        #            "resourceType": [<name:hardwareResource>]
        #        },
        #        "resourcesAssignment": {
        #            "<resourceName>": [<runnables>]
        #        }
        #    }
        # }
        clientResources = {
            clientName: ScheduleDescription.__getClientResource(hyperepochs, clientName)
            for clientName in clientInfo.keys()
        }
        # in the form of {"hyperepoch": [<resource name>}
        hyperepochResources = {
            hyperepochName: ScheduleDescription.__getHyperepochResource(hyperepochs, clientInfo, hyperepochName)
            for hyperepochName in hyperepochs.keys()
        } # key is hyperepoch name, value is resources
        return {
            "globalResources": globalResources,
            "clientResources": clientResources,
            "hyperepochResources": hyperepochResources
        }

    def parseHyperepochResourcePrefix(self, hyperEpoch: dict) -> set:
        """
        Example:
             For cameraHyperepoch, resource keys are:
                 camera_master.CUDA_MUTEX_LOCK
                 camera_master.CUDA_STREAM0:iGPU
                 camera_master.CUDLA_STREAM0:DLA0
                 machine0.CPU1
                 machine0.CPU2
                 machine0.CPU3
                 machine0.iGPU
             This function returns {"camera_master", "machine0"}
        """
        assert isinstance(hyperEpoch, dict)
        keys = list(hyperEpoch['resources'].keys())
        key_prefix_list = [e.split('.')[0] for e in keys]
        key_prefix_set = set(key_prefix_list)
        has_machine0 = ("machine0" in key_prefix_set)
        has_machine1 = ("machine1" in key_prefix_set)
        if has_machine0 and has_machine1:
            raise RuntimeError("find both machine0 and machine1 in hyper epoch resources")
        elif not has_machine0 and not has_machine1:
            raise RuntimeError("cannot find machine0/machine1 in hyper epoch resources")
        else:
            # We have either machine0 or machine1 in key_prefix_set
            pass
        return key_prefix_set

    def isTegraAHyperEpoch(self, hyperEpoch: dict):
        return 'machine0' in self.parseHyperepochResourcePrefix(hyperEpoch)

    def isTegraBHyperEpoch(self, hyperEpoch: dict):
         return 'machine1' in self.parseHyperepochResourcePrefix(hyperEpoch)

    def parseClients(self, clientsDict, clientResources, hyperepochsDict):
        leafNodes = self.graphlet.getLeafNodes()
        clients = []
        # Each stmExternalRunnables have its own client
        externalRunnables = self.systemDescription.getExternalRunnables()
        for externalRunnable in externalRunnables:
            externalRunnable._client = externalRunnable._name[len("External."):]
            clients.append(Client(
                name      = externalRunnable._client,
                soc       = self.findExternalRunnableSOC(externalRunnable),
                runnables = [],
                externalRunnables = [externalRunnable],
                resources = []
            ))

        # If the app is running on dual machines,
        # the systemDescription.getSubComponent(subId).getRunnables()
        # cannot distinguish the passes belonging to.
        # so here first check if the dual machines
        runOnMachines = set()
        for clientContent in clientsDict.values():
            runOnMachines.add(clientContent["runOn"])

        for clientName in clientsDict.keys():
            # TODO(xingpengd) remove hard coded machine name machine0 and machine1
            # TODO(xingpengd) remove hard coded mapping between machine name and SoC name
            soc = None
            if "runOn" in clientsDict[clientName]:
                if clientsDict[clientName]["runOn"] == "machine0":
                    soc = SoC.TegraA
                elif clientsDict[clientName]["runOn"] == "machine1":
                    soc = SoC.TegraB
                else:
                    raise RuntimeError("Machine name has to be machine0 or machine1")
            else:
                soc = SoC.TegraA
            subcomponentIds = clientsDict[clientName]['subcomponents']
            runnables = []
            externalRunnables = []

            if len(runOnMachines) > 1:
                # Collect the runnables based on the hyperepoch info
                # instead of the systemDescription.getSubComponent(subId).getRunnables(),
                # which cannot distinguish the passes belonging to TA or TB.
                sysSubComponents = self.getSubComponents()
                for hyperepochId, hyperepochDict in hyperepochsDict.items():
                    if soc == SoC.TegraA and self.isTegraAHyperEpoch(hyperepochDict):
                        for epochId, epochContent in hyperepochDict['epochs'].items():
                            epochRunnablesLists   = epochContent['passes']
                            for epochRunnables in epochRunnablesLists:
                                for runnablePath in epochRunnables:
                                    layers = runnablePath.split('.')[1:]
                                    runnables.extend(self.parseHyperepochsGetRunnables(layers, sysSubComponents))

                    if soc == SoC.TegraB and self.isTegraBHyperEpoch(hyperepochDict):
                        for epochId, epochContent in hyperepochDict['epochs'].items():
                            epochRunnablesLists   = epochContent['passes']
                            for epochRunnables in epochRunnablesLists:
                                for runnablePath in epochRunnables:
                                    layers = runnablePath.split('.')[1:]
                                    runnables.extend(self.parseHyperepochsGetRunnables(layers, sysSubComponents))
            else:
                for subId in subcomponentIds:
                    if self.getSubComponent(subId):
                        runnables.extend(self.getSubComponent(subId).getRunnables())
                    else:
                        externalRunnable = self.systemDescription.getExternalRunnable(subId)
                        if externalRunnable is None:
                            raise ValueError('Can not find external runnable: ', subId)
                        externalRunnable._client = clientName
                        externalRunnables.append(externalRunnable)
            # create software resources per process
            resourcesDict         = clientResources[clientName].get('resources', {})
            softwareResources     = []
            softwareResourcesDict = {}
            for resourceType, resourceList in resourcesDict.items():
                for resourceParam in resourceList:
                    if resourceType == 'MUTEX':
                        resource = MUTEX(resourceParam, soc)
                        softwareResources.append(resource)
                        softwareResourcesDict[resourceParam] = resource
                    elif resourceType == 'CUDA_STREAM':
                        layers      = resourceParam.split(':')
                        gpuResource = [g for g in self.getHardwareResourceSoc(soc).getResourceByType('GPU') if g.name == layers[1]]
                        resource    = CUDAStream(layers[0], gpuResource[0])
                        softwareResources.append(resource)
                        softwareResourcesDict[layers[0]] = resource
                    elif resourceType == 'CUDLA_STREAM':
                        layers      = resourceParam.split(':')
                        dlaResource = [d for d in self.getHardwareResourceSoc(soc).getResourceByType('DLA') if d.name == layers[1]]
                        resource = CUDLAStream(layers[0], dlaResource[0])
                        softwareResources.append(resource)
                        softwareResourcesDict[layers[0]] = resource
                    else:
                        raise KeyError('SoftwareResource only support type "MUTEX", "CUDASTREAM and "CUDLA_STREAM": ' + resourceType)
            self.softwareResources[clientName] = softwareResources
            # assgin some resources to specified runnables
            resourcesAssignment = clientResources[clientName].get("resourcesAssignment", {})
            for resourceName, runnableList in resourcesAssignment.items():
                if resourceName not in softwareResourcesDict.keys():
                    raise KeyError('Resouce name' + resourceName + ' not in softwareResources keys(): ' + softwareResourcesDict.keys())
                # Get the resource
                resource = softwareResourcesDict[resourceName]
                for runnablePath in runnableList:
                    # remove the PASS info, get the Node obj and get the Pass obj.
                    # e.g.   top.render.renderingNode.PASS_ACQUIRE_FRAME_CPU_SYNC
                    #    ->  top.render.renderingNode
                    runnablePathArr = runnablePath.split('.')
                    if not runnablePathArr[-1].isupper(): # runnablePath is a component ID
                        for runnable in self.getSubComponent(runnablePath).getRunnables():
                            runnable.appendResources([resource])
                    else: # runnablePath is a pass ID
                        componentName = '.'.join(runnablePathArr[:-1])
                        for runnable in self.getSubComponent(componentName).getRunnables():
                            # prefixing "PASS_" is because enumName is PASS_ + pass name
                            # please refer to function parsePasses
                            if runnable.enumName == runnablePathArr[-1] or runnable.enumName == "PASS_" + runnablePathArr[-1]:
                                # Assign MUTEX to runnable
                                if isinstance(resource, MUTEX):
                                    runnable.appendResources([resource])
                                # Assign CUDLA STREAM to runnable
                                elif isinstance(resource, CUDLAStream):
                                    runnable.handle = resource
                                else:
                                    raise TypeError('Resource ' + type(resource) + ' is not supported to assgin to runnable yet. Please add support if needed.')

            clients.append(Client(
                name      = clientName,
                soc       = soc,
                runnables = runnables,
                externalRunnables = externalRunnables,
                resources = softwareResources
            ))
        # TODO(yel): fix this WAR. add CUDA_MUTEX_LOCK to all runnables which use CUDA_STREAM or CUDLA_STREAM
        for client in clients:
            for runnable in client.runnables:
                for resource in runnable.resources:
                    if 'CUDA_STREAM' in resource.name or 'CUDLA_STREAM' in resource.name:
                        for softResource in self.softwareResources[client.name]:
                            if softResource.resource_type == "MUTEX":
                                cuda_mutex = softResource
                        runnable.resources = [self.cpu, cuda_mutex]

        return clients
    def parseHyperepochs(self, hyperepochsDict, hyperepochResources):
        hyperepochs = {}
        for hyperepochId, hyperepochDict in hyperepochsDict.items():
            # resources
            resourceList = hyperepochResources[hyperepochId]
            resources    = []
            for resource in resourceList:
                layouts = resource.split('.')
                # software resources. layouts[0] == clientName
                if layouts[0] in self.softwareResources.keys():
                    resources.extend([sr for sr in self.softwareResources[layouts[0]] if sr.name == layouts[2]])
                else:
                    # hardware resources. layouts[0] == soc
                    if layouts[0] == "machine0":
                        resSoc = self.getHardwareResourceSoc(SoC.TegraA)
                    else:
                        resSoc = self.getHardwareResourceSoc(SoC.TegraB)
                    resources.extend([hr for hr in resSoc.getResources() if hr.name == layouts[1]])
            # epochs
            epochsDict = hyperepochDict['epochs']
            epochs     = []
            for epochId, epochContent in epochsDict.items():
                epochPeriod      = epochContent['period']
                epochFrames      = 1 if 'frames' not in epochContent else epochContent['frames']
                epochRunnables   = epochContent['passes']
                sysSubComponents = self.getSubComponents()
                runnables        = []
                externalRunnables = []
                for phaseIndex, phaseRunnables in enumerate(epochRunnables): # Get the phaseIndex and apply to each runnables
                    if not isinstance(phaseRunnables, list):
                        raise RuntimeError('epoch does not contain phase info in runnables. Please add phases to the runnables in all the epoch :' + epochId)
                    for runnablePath in phaseRunnables:
                        if runnablePath.startswith("External."):
                            externalRunnable = self.systemDescription.getExternalRunnable(runnablePath)
                            if externalRunnable is None:
                                raise ValueError('Can not find external runnable: ', runnablePath)
                            externalRunnables.append(externalRunnable)
                        else:
                            # The first is top graphlet name and subcomponent is from the second layer
                            layers = runnablePath.split('.')[1:]
                            # 'Pass' in the runnablePath means only the pass specified, otherwise each pass in the graphlet/node are included.
                            if 'PASS' in runnablePath:
                                idx = runnablePath.index('PASS')
                                runnablePath = runnablePath[idx:]
                                # Note that `layers[:-1]` represents that searching to the node level, skip pass level
                                runnables.extend(self.parseHyperepochsGetRunnables(layers[:-1], sysSubComponents, PipelinePhase(phaseIndex), runnablePath))
                            else:
                                runnables.extend(self.parseHyperepochsGetRunnables(layers, sysSubComponents, PipelinePhase(phaseIndex)))
                epoch = Epoch(
                    epochId,
                    runnables = runnables,
                    externalRunnables = externalRunnables,
                    period    = TimeInterval(ns=epochPeriod),
                    frames    = epochFrames
                )
                # set groups
                if 'aliasGroups' in epochContent.keys():
                # TODO(hongwang): AliasGroup should support both node and runnable level
                    # epoch.setGroups([group])
                    groupNodes = []
                    for groupName in epochContent['aliasGroups']:
                        # Node level
                        if 'PASS' in groupName:
                            raise ValueError('Only Node level AliasGroup assignment is supported')
                        groupNodes.append(self.getSubComponent(groupName))
                    # TODO(hongwang): figure out why the groups must be add by twice
                    groups = TimingHelper.add_to_group(groupNodes[:4])
                    groups.extend(TimingHelper.add_to_group(groupNodes[4:8]))
                    groups.extend(TimingHelper.add_to_group(groupNodes[8:]))
                    epoch.setGroups(groups)

                epochs.append(epoch)
            # Hyperepoch.period
            if not 'period' in hyperepochDict:
                if len(epochs) == 1:
                    period = epochs[0].period
                else:
                    raise RuntimeError('Hyperepoch.period can only be omitted when it has only one epoch.')
            else:
                period = TimeInterval(ns=hyperepochDict['period'])
            hyperepoch = HyperEpoch(
                name      = hyperepochId,
                epochs    = epochs,
                resources = resources,
                period    = period,
            )
            hyperepochs[hyperepochId] = hyperepoch
        return hyperepochs
    def parseHyperepochsGetRunnables(self, layers, subComponents, pipelinePhase = PipelinePhase.PIPELINE_PHASE_UNCHANGED, runnablePath = None):
        if not layers[0] in subComponents:
            raise RuntimeError('Failed traverse module path ', layers, ' for scheduleDescription')
        subComponent = subComponents[layers[0]]
        if len(layers) == 1:
            # Graphlet and Node assignment
            subComponent.setPipelinePhase(pipelinePhase)
            if not runnablePath:
                return subComponent.getRunnables()
            # pass assignment
            res = []
            for runnable in subComponent.getRunnables():
                if runnablePath == runnable.enumName:
                    res.append(runnable)
            return res
        else:
            return self.parseHyperepochsGetRunnables(layers[1:],subComponent.getSubComponents(), pipelinePhase, runnablePath)
    def generateDependency(self):
        '''
        1. EpochCounter dependency defined in description.
        2. Sepcify each runnable's Epoch in description for MultiEpochs node
        '''
        def returnRealConsumerPort(port):
            if port.type == PortType.VIRTUAL:
                # dangling port
                if port.downstreams == []:
                    return []
                else:
                    ports = []
                    for index in range(len(port.downstreams)):
                        if port.indirectFlags[index]:
                            continue
                        ports.extend(returnRealConsumerPort(port.downstreams[index]))
                    return ports
            elif port.type == PortType.INPUT:
                return [port]
            else:
                raise TypeError('Invalid downstream type: ' + port.type.name)

        def setDependency(srcNode, dstNode, srcPort, destPort, dependsAsync ):

            # If pipelining is set and the connection between the producer -> consumer are cross pipeline boundary
            # we must do the following:
            #  if the producer is phase 2 and consumer phase 1, raise an error / crash. Pipelining does
            #    not support backward edges today.
            #  if the producer is phase 1 and consumer phase 2,
            #    if the channel is singleton, invert the dependencies between the nodes.
            #    else, remove the dependency (don't add it)
            if dstNode.getPipelinePhase() == PipelinePhase.PIPELINE_PHASE_2 and srcNode.getPipelinePhase() == PipelinePhase.PIPELINE_PHASE_1:
                if srcPort.singleton == True or destPort.singleton == True and dstNode.passes[-1] is not None:
                    srcNode.passes[0].depends(dstNode.passes[-1])
                    print(f"Pipelining: inverting dep to {dstNode.name}/{dstNode.id} from {srcNode.name}/{srcNode.id}. Port {destPort.name}")
                else:
                    print(f"Pipelining: skipping dep to  {dstNode.name}/{dstNode.id} from {srcNode.name}/{srcNode.id}. Port {destPort.name}")
                return
            elif dstNode.getPipelinePhase() == PipelinePhase.PIPELINE_PHASE_2 and srcNode.getPipelinePhase() == PipelinePhase.PIPELINE_PHASE_1:
                raise RuntimeError(f"Pipelining Error: backwards edge from phase 2 to phase 1. {dstNode.name}/{dstNode.id} to {srcNode.name}/{srcNode.id}. Port {inputPort.name}")

            outputPass = srcNode.passes[-1]
            if dependsAsync :
                outputPass = srcNode.passes[1]

            # add dependencies
            if dstNode.passes[0].epoch is not None     \
                and outputPass.epoch is not None         \
                and dstNode.passes[0].epoch.name == outputPass.epoch.name:

                if dependsAsync:
                    dstNode.passes[0].dependsAsync(outputPass)
                    # TODO(hongwang): to check why
                    if dstNode.passes[0].parent.name == 'dw::framework::dwLocalizationCameraNode':
                        dstNode.passes[0].depends(srcNode.passes[-1])
                else:
                    dstNode.passes[0].depends(outputPass)

        leafNodes = self.graphlet.getLeafNodes()
        for leafNode in leafNodes.values():
            for p in leafNode.passes:
                if p.epoch is None:
                    # Disable throwing because sensor service graphlet does not have an epoch
                    print('Unassigned Pass found! pass id: ' + p.name)
                    # throw if any pass without epoch info.
                    # raise RuntimeError('Unassigned Pass found! pass id: ' + p.name)
            for outputPort in leafNode.getOutputPorts().values():
                for downstream in outputPort.peers:
                    if downstream in outputPort.indirect:
                        continue
                    finalInputPorts = returnRealConsumerPort(downstream)
                    dependsAsync = False
                    for finalInputPort in finalInputPorts:
                        dependsAsync = True if argsHelper.check_optional \
                                        and (leafNode.name == 'dw::framework::dwCameraNode' \
                                        and 'IMAGE_NATIVE_PROCESSED' == outputPort.name \
                                        or 'IMAGE_TIMESTAMP' == outputPort.name) else False
                        setDependency(leafNode, finalInputPort.parent, outputPort, finalInputPort, dependsAsync)

        # Parse passDependencies
        if self.extraDeps:
            # passDict and dependencyDict are introduced to store the name-obj pairs
            passDict = {}
            dependencyDict = {}
            for passName, dependencyNameList in self.extraDeps.items():
                passDict[passName] = None
                for dep in dependencyNameList:
                    dependencyDict[dep] = None
            # Assign objects to passDict and dependencyDict
            for moduleId, moduleObj in self.graphlet.getSubComponents().items():
                for nodeId, nodeObj in moduleObj.getSubComponents().items():
                    for passObj in nodeObj.getRunnables():
                        currName = self.systemDescription.getId() + '.' + moduleId + '.' + nodeId + '.' + passObj.enumName
                        if currName in passDict.keys():
                            passDict[currName] = passObj
                        if currName in dependencyDict.keys():
                            dependencyDict[currName] = passObj
            # Add dependency
            for passName, dependencyNameList in self.extraDeps.items():
                for dep in dependencyNameList:
                    if passDict[passName] and dependencyDict[dep]:
                        passDict[passName].depends(dependencyDict[dep])
                    elif passDict[passName] is None:
                        warnings.warn(f'The pass object {passName} is not available and ignored here, plz check why!')
                    else:
                        warnings.warn("The dependency object", dep, "is not available and ignored here, plz check why!")

    def getHardwareResourceSoc(self, soc):
        return self.hardwareResourcesA if soc == SoC.TegraA else self.hardwareResourcesB
    def getHardwareResources(self):
        return self.hardwareResources
    def getClients(self):
        return self.clients
    def getHyperepochs(self):
        return self.hyperepochs
    def getWcetFile(self):
        return self.wcetFile
    def getIsDefaultSchedule(self):
        return self.isDefaultSchedule
    def getScheduleName(self):
        return self.scheduleKey
    def getScheduleIdentifier(self):
        return self.identifier
    def getSubComponent(self, componentId):
        componentIdList = componentId.split('.')
        if self.graphlet.id != componentIdList[0]:
            # Runables with "External" subcomponent name is a stmExternalRunnable
            if componentIdList[0] == "External":
                return None
            else:
                raise ValueError('componentId invalid. graphlet id: ', self.graphlet.id, ' componentId: ', componentIdList)
        if len(componentIdList) == 1:
            return self.graphlet
        else:
            return self.graphlet.getSubComponent(componentIdList[1:])
    def getSubComponents(self):
        return self.graphlet.getSubComponents()
    def getRunnables(self):
        return self.graphlet.getRunnables()
    def isExtern(self):
        return self.extern
    def findExternalRunnableSOC(self, externalRunnable):
        for hyperepoch, hyperepochDesc in self.scheduleJson["hyperepochs"].items():
            for epoch, epochDesc in hyperepochDesc["epochs"].items():
                for passList in epochDesc["passes"]:
                    if externalRunnable._name in passList:
                        for key in hyperepochDesc["resources"].keys():
                            if key.startswith("machine0"):
                                return SoC.TegraA
                            elif key.startswith("machine1"):
                                return SoC.TegraB

class NodeDescription:
    def __init__(self, nodeId, nodeFile, parent):
        self.type   = DescriptionType.NODE
        self.parent = parent
        # file validation
        nodeFile = FileValidation.validate(nodeFile, self.type.name.lower(), "gdl-node")
        with open(nodeFile, 'r') as gf:
            nodeJson = json.load(gf)
        # required key validation
        NAME   = 'name'
        PASSES = 'passes'
        keyValidation([NAME, PASSES], nodeJson, self.type.name.lower(), nodeFile)
        # parse required
        self.id     = nodeId
        self.name   = nodeJson[NAME]
        self.passes = self.parsePasses(nodeJson[PASSES])
        # parse optional
        self.inputPorts  = self.parsePorts({} if not 'inputPorts' in nodeJson else nodeJson['inputPorts'], True)
        self.outputPorts = self.parsePorts({} if not 'outputPorts' in nodeJson else nodeJson['outputPorts'], False)
        # TODO(yel): remove this
        self.is_dwnode = True
        self.downstreamNodes = []
        self.pipelinePhase = PipelinePhase.PIPELINE_PHASE_UNCHANGED
    def parsePasses(self, passList):
        # default dependency between each pass inside the node
        passes = []
        passNamePrefix = 'PASS_'
        for index in range(len(passList)):
            passName = passList[index]['name']
            #TODO(yel): move this check logic to yaml generation
            if argsHelper.check_optional and self.check_optional(passName):
                continue
            dependencyNames = passList[index].get('dependencies')
            dependencyPasses = []
            if dependencyNames is not None:
                for dependencyName in dependencyNames:
                    dependencyPass = [r for r in passes if r.enumName[len(passNamePrefix):] == dependencyName]
                    assert len(dependencyPass) == 1, 'Pass dependency not found'
                    dependencyPasses.append(dependencyPass)
            elif index > 0:
                # use the previous pass as the implicit dependency
                dependencyPasses.append(passes[-1])

            if index == 0:
                # setupRun pass
                runnable = createStartPass()
            elif index < len(passList) - 1:
                runnable = self.generateRunnable(passList[index]['processorTypes'], index)
            else:
                # tearDown pass
                runnable = createResetPass(len(passList) - 1)
            runnable.parent = self
            runnable.enumName = passNamePrefix + passName
            assert runnable.enumName not in [r.enumName for r in passes], 'Pass with duplicate name'
            for dependencyPass in dependencyPasses:
                runnable.depends(dependencyPass)
            passes.append(runnable)
        return passes
    def generateRunnable(self, resources, index):
        if not isinstance(resources, list):
            raise TypeError('Runnable resources is support to be list type, but ' + str(type(resources)))
        if len(resources) != len(set(resources)):
            raise ValueError('Assign one resource to runnable multiple times is invalid.')
        hardware_set = {'CPU', 'GPU', 'DLA', 'VPI'}
        hardware = hardware_set.intersection(set(resources))
        if len(hardware) != 1:
            raise ValueError('Do not assign none or more than one valid hardware to the runnable! ' + str(hardware))

        if 'GPU' in hardware:
            return GPUSubmitter('pass_' + str(index))
        elif 'DLA' in hardware:
            # TODO(azou): DLA could be used for both DLASubmitter and CUDLASubmitter
            # For in node information, it could not distinguish CUDLASubmitter/DLASubmitter
            # Now dierectly Create CUDLASubmitter.
            return CUDLASubmitter('pass_' + str(index))
        elif 'VPI' in hardware:
            return VPISubmitter('pass_' + str(index))
        elif 'CPU' in hardware:
            return CPURunnable('pass_' + str(index))
        else:
            raise ValueError('Runnable requires hardware CPU/GPU/DLA/VPI but ' + str(resources))
    # TODO(yel): move this check logic to yaml generation
    # TODO(hongwang): maintain check_optional along with core.py
    def check_optional(self, passName):
        if self.name == 'dw::framework::dwObjectTrackerNode' and passName == 'RUN_ALL':
            return True
        if self.name == 'dw::framework::dwCameraNode' and passName == 'RAW_OUTPUT' or passName == 'PROCESSED_RGBA_OUTPUT':
            return True
        return False
    def parsePorts(self, portsDict, isInput):
        def instantiatePort(portName, isInput, parent):
            if isInput:
                return InputPort(portName, parent)
            else:
                return OutputPort(portName, parent)

        ports = {}
        for portName in portsDict.keys():
            # Parse ports array if find key word `array`
            if 'array' in portsDict[portName].keys():
                for i in range(portsDict[portName]['array']):
                    arrayItemName = portName + '[' + str(i) + ']'
                    ports[arrayItemName] = instantiatePort(arrayItemName, isInput, self)
            else:
                ports[portName] = instantiatePort(portName, isInput, self)
        return ports
    def getId(self):
        return self.id
    def getDescriptionType(self):
        return self.type
    def getRunnables(self):
        return self.passes
    def getInputPorts(self):
        return self.inputPorts
    def getOutputPorts(self):
        return self.outputPorts
    def getPipelinePhase(self):
        return self.pipelinePhase
    def setPipelinePhase(self, pipelinePhase: PipelinePhase):
        for runnable in self.passes:
            runnable.pipelinePhase = pipelinePhase
        self.pipelinePhase = pipelinePhase

class GraphletDescription:
    def __init__(self, graphletId, graphletFile, parent):
        self.type   = DescriptionType.GRAPHLET
        self.parent = parent
        graphletFile = FileValidation.validate(graphletFile, self.type.name.lower(), "gdl-graphlet")
        with open(graphletFile, 'r') as gf:
            graphletJson = json.load(gf)
        self._graphletFile = os.path.abspath(graphletFile)
        # required key validation
        NAME          = 'name'
        SUBCOMPONENTS = 'subcomponents'
        CONNECTIONS   = 'connections'
        keyValidation([NAME, SUBCOMPONENTS, CONNECTIONS], graphletJson, self.type.name.lower(), graphletFile)
        # parse required
        self.id            = graphletId
        self.name          = graphletJson[NAME] # name is classname.
        self.subcomponents = self.parseSubComponents(graphletJson[SUBCOMPONENTS])
        self.connections   = graphletJson[CONNECTIONS]
        # parse optional
        self.inputPorts  = self.parsePorts({} if not 'inputPorts' in graphletJson else graphletJson['inputPorts'])
        self.outputPorts = self.parsePorts({} if not 'outputPorts' in graphletJson else graphletJson['outputPorts'])
        # self.soc defined in ScheduleDescription
        # TODO(yel): remove this
        self.is_dwnode = False
        # process
        self.pipelinePhase = PipelinePhase.PIPELINE_PHASE_UNCHANGED
        # connect all connections
        self.connectSubcomponent()

    @property
    def baseDir(self):
        return os.path.dirname(self._graphletFile)

    def parseSubComponents(self, subcomponentsDict):
        subcomponents = {}
        for componentId in subcomponentsDict.keys():
            componentType   = subcomponentsDict[componentId]['componentType']
            componentFullId = self.getId() + '.' + componentId
            if componentType.endswith(DescriptionType.GRAPHLET.name.lower() + '.json') or componentType.endswith('gdl-graphlet.json'):
                component = GraphletDescription(componentFullId, fixPath(componentType, self.baseDir), self)
            elif componentType.endswith(DescriptionType.NODE.name.lower() + '.json') or componentType.endswith('gdl-node.json'):
                component = NodeDescription(componentFullId, fixPath(componentType, self.baseDir), self)
            else:
                raise ValueError('Subcomponents of Graphlet are required to be Node or Graphlet: ' + componentType)
            subcomponents[componentId] = component
        return subcomponents
    def parsePorts(self, portsDict):
        ports = {}
        for portName in portsDict.keys():
            # Parse ports array if find key word `array`
            if 'array' in portsDict[portName].keys():
                for i in range(portsDict[portName]['array']):
                    arrayItemName = portName + '[' + str(i) + ']'
                    ports[arrayItemName] = VirtualPort(arrayItemName, self)
            else:
                ports[portName] = VirtualPort(portName, self)
        return ports
    def connectSubcomponent(self):
        def connectInstance(connectionFromStr, connectionToStr, indirectFlag, singletonFlag = False):
            # skip INBOUND
            if connectionFromStr == "":
                return
            # skip OUTBOUND
            if connectionToStr == "":
                return
            # get instance
            connectionFrom = connectionFromStr.split('.')
            connectionTo   = connectionToStr.split('.')
            if len(connectionFrom) == 1:
                upstreamPort = self.inputPorts[connectionFrom[0]]
            elif len(connectionFrom) == 2:
                upstreamComponent = self.subcomponents[connectionFrom[0]]
                upComOutPorts     = upstreamComponent.getOutputPorts()
                if not connectionFrom[1] in upComOutPorts:
                    raise KeyError(f"Cannot find 'src' port '{connectionFromStr}' in graphlet '{self._graphletFile}'")
                upstreamPort      = upComOutPorts[connectionFrom[1]]
            else:
                raise ValueError('Invalid "from" value for connection: ' + str(connectionFrom))
            if len(connectionTo) == 1:
                downstreamPort = self.outputPorts[connectionTo[0]]
            elif len(connectionTo) == 2:
                downstreamComponent = self.subcomponents[connectionTo[0]]
                downComInPorts      = downstreamComponent.getInputPorts()
                if not connectionTo[1] in downComInPorts:
                    raise KeyError(f"Cannot find 'dest' port '{connectionToStr}' in graphlet '{self._graphletFile}'")
                downstreamPort      = downComInPorts[connectionTo[1]]
            else:
                raise ValueError('Invalid "to" value for connection: ' + str(connectionTo))
            # connect with VirtualPort
            if singletonFlag:
                upstreamPort.singleton = singletonFlag
                downstreamPort.singleton = singletonFlag
            if upstreamPort.type == PortType.OUTPUT:
                upstreamPort.peers.append(downstreamPort)
                if indirectFlag:
                    upstreamPort.indirect.append(downstreamPort)
            elif upstreamPort.type == PortType.VIRTUAL:
                upstreamPort.downstreams.append(downstreamPort)
                upstreamPort.indirectFlags.append(indirectFlag)
            else:
                raise RuntimeError('InputPort as producer is not allowed')
            if downstreamPort.type == PortType.INPUT:
                downstreamPort.peer = upstreamPort
            elif downstreamPort.type == PortType.VIRTUAL:
                downstreamPort.upstream = upstreamPort
            else:
                raise RuntimeError('OutputPort as consumer is not allowed')

        for connection in self.connections:
            if "src" in connection:
                connectionSrcStr = connection['src']
                connectionDests  = connection['dests']
                indirectFlag     = connection.get('params', {}).get('indirect', False)
                singletonFlag    = connection.get('params', {}).get('singleton', False)
                for connectionDestStr, connectionDest in connectionDests.items():
                    connectInstance(connectionSrcStr, connectionDestStr, connectionDest.get('indirect', indirectFlag), singletonFlag)
            elif "from" in connection:
                connectionFromStr = connection['from']
                connectionToStr  = connection['to']
                indirectFlag   = False
                singletonFlag    = connection.get('params', {}).get('singleton', False)
                if 'parameters' in connection and 'indirect' in connection['parameters'] and connection['parameters']['indirect']:
                    indirectFlag = True
                connectInstance(connectionFromStr, connectionToStr, indirectFlag, singletonFlag)
            else:
                assert False, connection
    def getId(self):
        return self.id
    def getDescriptionType(self):
        return self.type
    def getSubComponent(self, componentIdList):
        if not componentIdList[0] in self.subcomponents:
            raise ValueError('Can not find subComponent: ', componentIdList, ' in ', self.id)
        if len(componentIdList) == 1:
            return self.subcomponents[componentIdList[0]]
        else:
            return self.subcomponents[componentIdList[0]].getSubComponent(componentIdList[1:])
    def getSubComponents(self):
        return self.subcomponents
    def getLeafNodes(self):
        leafs = {}
        for subcomponentId, subcomponent in self.subcomponents.items():
            if subcomponent.getDescriptionType() == DescriptionType.NODE:
                leafs[subcomponent.id] = subcomponent
            elif subcomponent.getDescriptionType() == DescriptionType.GRAPHLET:
                leafs.update(subcomponent.getLeafNodes())
            else:
                raise TypeError('Subcomponents of Graphlet are required to be Node or Graphle: ' + subcomponent.getDescriptionType().name)
        return leafs
    def getRunnables(self):
        runnables = []
        for subcomponent in self.subcomponents.values():
            runnables.extend(subcomponent.getRunnables())
        return runnables
    def getInputPorts(self):
        return self.inputPorts
    def getOutputPorts(self):
        return self.outputPorts
    def getPipelinePhase(self):
        return self.pipelinePhase
    def setPipelinePhase(self, pipelinePhase: PipelinePhase):
        for subcomponent in self.subcomponents.values():
            subcomponent.setPipelinePhase(pipelinePhase)
        self.pipelinePhase = pipelinePhase

class SystemDescription:
    def __init__(self, systemFile):
        self.type = DescriptionType.APP
        # file validation
        systemFile = FileValidation.validate(systemFile, self.type.name.lower())
        with open(systemFile, 'r') as sj:
            systemJson = json.load(sj)
        self.baseDir = os.path.dirname(os.path.abspath(systemFile))
        # required key validation
        NAME          = 'name'
        SUBCOMPONENTS = 'subcomponents'
        STATES  = 'states'
        STMSCHEDULES  = 'stmSchedules'
        stmExternalRunnables = 'stmExternalRunnables'
        PROCESSES = "processes"
        COMPTYPE  = "componentType"
        PARAMS    = "parameters"
        keyValidation([NAME, SUBCOMPONENTS, STATES, STMSCHEDULES, PROCESSES], systemJson, self.type.name.lower(), systemFile)
        # get the first subcomponents
        if len(systemJson[SUBCOMPONENTS]) > 1:
            raise ValueError('Right now only support one single subcomponent in application description')
        rootGraphletID = list(systemJson[SUBCOMPONENTS].keys())[0]
        keyValidation([COMPTYPE], systemJson[SUBCOMPONENTS][rootGraphletID], self.type.name.lower(), systemFile)
        self.processes = systemJson[PROCESSES]
        # parse required
        self.id       = rootGraphletID
        self.graphletFile = fixPath(systemJson[SUBCOMPONENTS][rootGraphletID][COMPTYPE], self.baseDir)
        states = systemJson[STATES]
        DEFAULT = "default"
        defaultStmScheduleKey = None
        stmScheduleKeys = []
        for state, stmSchedule in states.items():
            if stmSchedule.get(DEFAULT):
                if defaultStmScheduleKey is not None:
                    raise KeyError('More than one default STM Schedule in states ' + STATES + ' Description: ' + systemFile)
                defaultStmScheduleKey = stmSchedule["stmScheduleKey"]
            stmScheduleKeys.append( str(stmSchedule["stmScheduleKey"]))
        if defaultStmScheduleKey is None:
            raise KeyError('Missing required key(s) in description for generating ' + STATES + ":" + DEFAULT + ' Description: ' + systemFile)
        stmSchedules = systemJson[STMSCHEDULES]
        keyValidation(stmScheduleKeys, stmSchedules, self.type.name.lower(), systemFile)

        # Parse stmExternalRunnables if exists
        self.externalRunnables = []
        WCET = "wcet"
        extra_dependencies = 'passDependencies'
        if stmExternalRunnables in systemJson:
            externalRunnables = systemJson[stmExternalRunnables]
            for key, externalRunnable in externalRunnables.items():
                keyValidation([WCET], externalRunnable, self.type.name.lower(), systemFile)
                runnable = self.generateExternalRunnable(key)
                runnable._wcet = externalRunnable[WCET]
                if extra_dependencies in externalRunnable.keys():
                    runnable._extraDependencies = externalRunnable[extra_dependencies]
                self.externalRunnables.append(runnable)

        # schedule id are created and stored in sorted order
        self.schedules = [
            ScheduleDescription(self.baseDir, stmSchedule, stmScheduleKey, identifier, stmScheduleKey == defaultStmScheduleKey, self)
            for identifier, (stmScheduleKey, stmSchedule) in  enumerate(sorted(stmSchedules.items()))
        ]
        # instantiate scheduleDescription at last to generate correct runnable dependecies
        #no self.name for app.json. As system is not an actual class, we don't need classname for it.

    def getId(self):
        return self.id
    def getDescriptionType(self):
        return self.type
    def getSchedules(self):
        return self.schedules
    def generateExternalRunnable(self, name):
        return CPURunnable(name)
    def getExternalRunnable(self, name):
        for externalRunnable in self.externalRunnables:
            if name == externalRunnable.name:
                return externalRunnable
        return None
    def getExternalRunnables(self):
        return self.externalRunnables
    def getProcesses(self):
        return self.processes
    def getGraphletFile(self):
        return self.graphletFile

def main(argv=None):
    parser = argparse.ArgumentParser(description='Generate YAML schedule file from systemDAG.')
    parser.add_argument('--app',  type = str, required = True,  help = 'system DAG json description file: *.app.json')
    parser.add_argument('--output', type = str, required = False, help = 'output files containing path for schedule yaml file.', default="/tmp/demo.yaml")
    parser.add_argument('--check_optional', type = str, required = False, help = 'check optional.', default="False")
    parser.add_argument('--path-offset', type = str, required = False, help = 'denotes the relative path between generated files and source files.', default="")
    parser.add_argument('--roadCastService', type = str, required = False,  help = 'system DAG json description file.', default="")
    argvs = parser.parse_args()

    # use cwd as default dir
    currentPath = os.getcwd()

    # use argvs.graph directly if it is abspath
    # otherwise, use ndasPath as base directory
    if argvs.app.startswith('/'):
        appFile = argvs.app
    else:
        appFile = os.path.abspath(os.path.join(currentPath, argvs.app))
    if not appFile.endswith('.app.json'):
        print('Input description: ', appFile,' is not .app.json file!')
        return

    if argvs.output.startswith('/'):
        outputFile = argvs.output
    else:
        outputFile = os.path.abspath(os.path.join(currentPath, argvs.output))

    outputDir = os.path.split(outputFile)[0]
    if not os.path.exists(outputDir):
        print('Output schedule file dir is invalid! Please specify an exist directory path.')
        return
    outputDir = os.path.join(outputDir, '')                         #adds a / at the end if it is not there

    if argvs.check_optional == "True":
        argsHelper.check_optional = True

    if argvs.path_offset:
        FileValidation.path_offset = argvs.path_offset

    print('Input Description file path: ' + appFile)
    print('Output schedule directory path: ' + outputDir)

    print('=============================')
    print('Analysis...')
    # System and schedule description
    systemDescription = SystemDescription(appFile)

    lastPos = appFile.rfind("/")
    appFileName = appFile[lastPos + 1:]  if lastPos >= 0 else appFile
    appFileName = appFileName.split('.')[0]

    print('=============================')
    print('Generating...')

    # roadCastService
    rcs_file_path = argvs.roadCastService
    if rcs_file_path != "":
        with open(rcs_file_path, "r") as rcs_file:
            rcs = json.load(rcs_file)
            rcs = rcs["RoadRunner"] # remove top layer RoadRunner for CGF

    for schedule in systemDescription.getSchedules():
        # === stmGraph  ===
        stmGraph = STM_graph(
            name        = schedule.getScheduleName(),
            resources   = schedule.getHardwareResources(),
            hyperepochs = schedule.getHyperepochs().values(), # covert dict to list
            clients     = schedule.getClients(),
            identifier  = schedule.getScheduleIdentifier(),
            wcet_file   = schedule.getWcetFile(),
            extern      = schedule.isExtern(),
        )
        # === roadCastService  ===
        if rcs_file_path != "":
            stmGraph.setRcsPorts(rcs["RoadCastService"]["ports"])
            stmGraph.setRcsPassMapping(rcs["RoadCastService"]["passMapping"])
        # === generate  ===
        yaml.add_representer(defaultdict, Representer.represent_dict)
        outputString = '#Autogenerated using json2yaml backend\n' + yaml.dump(stmGraph.yml())
        if schedule.isExtern():
            currentFile = outputDir + appFileName + "_sub__" + schedule.getScheduleName() + ".yaml"
        else:
            currentFile = outputDir + appFileName + "__" + schedule.getScheduleName() + ".yaml"

        with open(currentFile, "w") as f:
            print(outputString, file=f)
        print('Yaml schedule generated : ' + schedule.getScheduleName() + ' : ' + str(schedule.getScheduleIdentifier()) + ' : '+ currentFile)

    print('=============================')
    print('Json description to Yaml schedule file done.')

if __name__ == '__main__':
    main()
    sys.exit(0)
