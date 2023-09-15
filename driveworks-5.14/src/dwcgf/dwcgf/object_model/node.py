#########################################################################################
# This code contains NVIDIA Confidential Information and is disclosed
# under the Mutual Non-Disclosure Agreement.
#
# Notice
# ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS" NVIDIA MAKES
# NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
# THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED WARRANTIES OF NONINFRINGEMENT,
# MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
#
# NVIDIA Corporation assumes no responsibility for the consequences of use of such
# information or for any infringement of patents or other rights of third parties that may
# result from its use. No license is granted by implication or otherwise under any patent
# or patent rights of NVIDIA Corporation. No third party distribution is allowed unless
# expressly authorized by NVIDIA. Details are subject to change without notice.
# This code supersedes and replaces all information previously supplied.
# NVIDIA Corporation products are not authorized for use as critical
# components in life support devices or systems without express written approval of
# NVIDIA Corporation.
#
# Copyright (c) 2022-2023 NVIDIA Corporation. All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property and proprietary
# rights in and to this software and related documentation and any modifications thereto.
# Any use, reproduction, disclosure or distribution of this software and related
# documentation without an express license agreement from NVIDIA Corporation is
# strictly prohibited.
#
#########################################################################################
"""Data structures for node instance."""
from copy import deepcopy
from typing import Dict
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from dwcgf.descriptor import NodeDescriptor
from dwcgf.descriptor import ParameterDefinition
from dwcgf.descriptor import PassDefinition

from .component import Component
from .object_model_channel import object_model_channel
from .port import Port
from .process import Process

if TYPE_CHECKING:
    from .graphlet import Graphlet  # noqa: F401


class Pass:
    """class for node pass."""

    @object_model_channel.ignore
    def __init__(
        self,
        *,
        name: str,
        processor_types: List[str],
        dependencies: Dict[str, List["Pass"]],
        wcet: Optional[int] = None,
        process: Optional[Process] = None,
    ):
        """Create Node instance.

        @param name             name of the pass.
        @param processor_types  a list of processors used by this pass.
        @param dependencies     the passes on which this pass depends, if the array is empty
                                which means the pass has no dependencies, if it's None, it
                                means the dependencies are not specified, and use default
                                behavior which means depend on previous pass.
        @param wcet             wcet of the pass
        """
        self._name = name
        self._processor_types = processor_types
        self._dependencies = dependencies
        self._wcet = wcet
        self._process = process

    @property
    def name(self) -> str:
        """Return the name of the pass."""
        return self._name

    @property
    def processor_types(self) -> List[str]:
        """Return the processor_types of the pass."""
        return self._processor_types

    @property
    def dependencies(self) -> Dict[str, List["Pass"]]:
        """Return the dependencies of the pass."""
        return self._dependencies

    @dependencies.setter
    def dependencies(self, value: Dict[str, List["Pass"]]) -> None:
        """Set the dependencies of the pass."""
        self._dependencies = value

    @property
    def wcet(self) -> Optional[int]:
        """Return the wcet of the pass."""
        return self._wcet

    @wcet.setter
    def wcet(self, value: int) -> None:
        """Set the wcet of the pass."""
        self._wcet = value

    @property
    def process(self) -> Optional[Process]:
        """Return the process of the pass."""
        return self._process

    @process.setter
    def process(self, value: Process) -> None:
        """Set the process of the pass."""
        self._process = value

    @property
    def process_name(self) -> Optional[str]:
        """Return the name of the process to which the pass belongs."""
        return self._process.name if self._process is not None else None

    @staticmethod
    def from_descriptor(desc: List[PassDefinition]) -> List["Pass"]:
        """Create a list of Pass instance from a list PassDefinition."""
        ret: list = []
        for definition in desc:
            ret.append(
                Pass(
                    name=deepcopy(definition.name),
                    processor_types=deepcopy(definition.processor_types),
                    dependencies={},  # very interesting ...
                    # if not pass an empty list,
                    # all attr:dependencies of Pass instances in ret reference the same list ...
                )
            )

        return ret


class Node(Component):
    """class for Node instances in DAG."""

    @object_model_channel.ignore
    def __init__(
        self,
        *,
        name: str,
        comment: Optional[str] = None,
        generated: Optional[bool] = None,
        library: Optional[str] = None,
        parameters: Optional[Dict[str, ParameterDefinition]] = None,
        input_ports: Optional[Dict[str, Port]] = None,
        output_ports: Optional[Dict[str, Port]] = None,
        passes: Optional[List[Pass]] = None,
    ):
        """Create Node instance."""
        super().__init__(
            name=name,
            comment=comment,
            parameters=parameters,
            input_ports=input_ports,
            output_ports=output_ports,
        )
        self._generated = generated
        self._library = library
        self._passes = passes

    @property
    def passes(self) -> Optional[List[Pass]]:
        """Return passes of this node."""
        return self._passes

    @property
    def generated(self) -> Optional[bool]:
        """Return if the descriptor is generated."""
        return self._generated

    @property
    def library(self) -> Optional[str]:
        """Return the library name containing this node."""
        return self._library

    @property
    def children(self) -> List:
        """Nodes are the leaves so this returns empty."""
        return []

    @staticmethod
    def from_descriptor(name: str, desc: NodeDescriptor) -> "Node":
        """Create a Node instance from descriptor.

        @param name   name of the newly created instance
        @param desc   NodeDescriptor
        """

        node = Node(
            name=name,
            comment=desc.comment,
            generated=desc.generated,
            library=desc.library,
            parameters=desc.parameters,
            input_ports=Port.from_descriptor(desc.input_ports, True),
            output_ports=Port.from_descriptor(desc.output_ports, False),
            passes=Pass.from_descriptor(desc.passes),
        )

        node.descriptor_path = desc.file_path

        return node
