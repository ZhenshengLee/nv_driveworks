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
# Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property and proprietary
# rights in and to this software and related documentation and any modifications thereto.
# Any use, reproduction, disclosure or distribution of this software and related
# documentation without an express license agreement from NVIDIA Corporation is
# strictly prohibited.
#
#########################################################################################
"""For Node descriptor."""
from collections import OrderedDict
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from .component_descriptor import ComponentDescriptor
from .descriptor_factory import DescriptorFactory
from .descriptor_factory import DescriptorType


class PassDefinition:
    """Class for passes of node descriptor."""

    def __init__(
        self,
        *,
        name: str,
        processor_types: List[str],
        dependencies: Optional[List[str]],
    ):
        """Create a PassDefinition instance for an entry in passes section.

        @param name            name of the pass
        @param processor_types a list of processors used by this pass
        @param dependencies    the passes on which this pass depends, if the array is empty
                               which means the pass has no dependencies, if it's None, it
                               means the dependencies are not specified, and use default
                               behavior which means depend on previous pass.
        """
        self._name = name
        self._processor_types = processor_types
        self._dependencies = dependencies

    @property
    def name(self) -> str:
        """Return name of this pass."""
        return self._name

    @property
    def processor_types(self) -> List[str]:
        """Return the list of processors used by this pass."""
        return self._processor_types

    @property
    def dependencies(self) -> Optional[List[str]]:
        """Return the passes on which this pass depends."""
        return self._dependencies


@DescriptorFactory.register(DescriptorType.NODE)
class NodeDescriptor(ComponentDescriptor):
    """class for node descriptors."""

    def __init__(
        self,
        file_path: Path,
        *,
        passes: Optional[List[PassDefinition]],
        generated: bool = False,
        library: Optional[str] = None,
        **kwargs: Any,
    ):
        """Create a NodeDescriptor instance.

        @param file_path path of this node descriptor file
        """
        super().__init__(file_path, **kwargs)
        self._passes = passes if passes is not None else []
        self._generated = generated
        self._library = library

    @property
    def passes(self) -> List[PassDefinition]:
        """Return the pass definition of this node."""
        return self._passes

    @property
    def library(self) -> Optional[str]:
        """Return the library containing this node.

        If the node is statically linked, 'static' will be returned.
        If this descriptor has no real code, None will be returned.
        """
        return self._library

    @property
    def generated(self) -> bool:
        """Indicates if the descriptor is generated."""
        return self._generated

    @classmethod
    def from_json_data(cls, content: Dict, path: Union[str, Path]) -> "NodeDescriptor":
        """Create NodeDescriptor from JSON data."""

        path = Path(path)

        passes_raw = content.get("passes", [])

        passes = [
            PassDefinition(
                name=pass_raw["name"],
                processor_types=pass_raw["processorTypes"],
                dependencies=pass_raw.get("dependencies", None),
            )
            for pass_raw in passes_raw
        ]

        comp = ComponentDescriptor.from_json_data(content, path)
        return NodeDescriptor(
            path,
            name=comp.name,
            parameters=comp.parameters,
            input_ports=comp.input_ports,
            output_ports=comp.output_ports,
            comment=comp.comment,
            passes=passes,
            generated=content.get("generated", False),
            library=content["library"],
        )

    @classmethod
    def empty_desc(cls, name: str, path: Union[str, Path]) -> "NodeDescriptor":
        """Create empty NodeDescriptor and bind to a desc path."""
        path = Path(path)
        return NodeDescriptor(
            path,
            name=name,
            parameters={},
            input_ports={},
            output_ports={},
            comment=None,
            passes=[],
            generated=False,
            library=None,
        )

    def to_json_data(self) -> OrderedDict:
        """Dump NodeDescriptor to JSON data."""

        def dump_passes(passes: List[PassDefinition]) -> List:
            node_passes = []
            for p in passes:
                pass_json = OrderedDict(name=p.name, processorTypes=p.processor_types)
                if p.dependencies is not None:
                    pass_json["dependencies"] = p.dependencies
                node_passes.append(pass_json)
            return node_passes

        comp_json_data = super().to_json_data()
        node_json: OrderedDict = OrderedDict()

        if self.comment is not None:
            node_json["comment"] = self.comment

        if self.generated:
            node_json["generated"] = self.generated

        if self.library is not None:
            node_json["library"] = self.library

        node_json["name"] = comp_json_data["name"]
        node_json["inputPorts"] = comp_json_data["inputPorts"]
        node_json["outputPorts"] = comp_json_data["outputPorts"]
        node_json["parameters"] = comp_json_data["parameters"]

        node_json["passes"] = dump_passes(self.passes)

        return node_json
