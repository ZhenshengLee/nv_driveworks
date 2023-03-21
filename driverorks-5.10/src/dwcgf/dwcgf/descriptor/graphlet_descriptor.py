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
"""For Graphlet descriptor."""
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


class ConnectionDefinition:
    """Class for connection of Graphlet or Application descriptor."""

    def __init__(self, *, src: str, params: Optional[Dict], dests: Dict[str, Dict]):
        """Create ConnectionDefinition instance for an entry in connections section.

        @param src    the source port of the connection
        @param params the connection parameters for all connected ports
        @param dests  the destination ports of the src port
                      It's a dict where the keys are destination port name,
                      and the value is connection parameter specific for the dest port
        """
        self._src = src
        self._params = params if params is not None else {}
        self._dests = dests

    @property
    def src(self) -> str:
        """Return src port name."""
        return self._src

    @property
    def dests(self) -> Dict[str, Dict]:
        """Return dest port name and param pair."""
        return self._dests

    @property
    def params(self) -> Optional[Dict[str, Any]]:
        """Return connection parameters."""
        return self._params

    @params.setter
    def params(self, value: Dict[str, Any]) -> None:
        """Set connection parameters."""
        self._params = value


class SubcomponentDefinition:
    """Class for subcomponent of Graphlet or Application descriptor."""

    ParameterValueType = Union[
        bool,
        int,
        float,
        str,
        List[Union[str, int]],  # parameter mapping may happen in elements
        List[Union[str, float]],
        List[Union[str, bool]],
        List[str],
    ]

    def __init__(
        self,
        *,
        name: str,
        component_type: Path,
        parameters: Optional[Dict[str, ParameterValueType]] = None,
        comment: Optional[str] = None,
    ):
        """Create Subcomponent descriptor instance for an entry in subcomponents section."""
        self._name = name
        self._component_type = component_type
        self._parameters = parameters
        self._comment = comment

    @property
    def component_type(self) -> Path:
        """Return component type."""
        return self._component_type

    @property
    def parameters(self) -> Optional[Dict[str, "ParameterValueType"]]:
        """Return subcomponent parameter mappings."""
        return self._parameters

    @property
    def name(self) -> str:
        """Return the name of the subcomponent."""
        return self._name

    @property
    def comment(self) -> Optional[str]:
        """Return the comment."""
        return self._comment


@DescriptorFactory.register(DescriptorType.GRAPHLET)
class GraphletDescriptor(ComponentDescriptor):
    """class for node descriptors."""

    def __init__(
        self,
        file_path: Path,
        *,
        subcomponents: Optional[Dict[str, SubcomponentDefinition]] = None,
        connections: Optional[List[ConnectionDefinition]] = None,
        devviz_components: Optional[Dict] = None,
        generated_from_gdl: Optional[bool] = False,
        **kwargs: Any,
    ):
        """Create a GraphletDescriptor instance.

        @param file_path path of this graphlet descriptor file
        """
        super().__init__(file_path, **kwargs)
        self._subcomponents = subcomponents if subcomponents is not None else {}
        self._connections = connections if connections is not None else []
        self._devviz_components = devviz_components
        self._generated_from_gdl = (
            generated_from_gdl if generated_from_gdl is not None else False
        )

    @property
    def connections(self) -> List[ConnectionDefinition]:
        """Return connections of this graphlet instance."""
        return self._connections

    @property
    def subcomponents(self) -> Dict[str, SubcomponentDefinition]:
        """Return subcomponents of this graphlet instance."""
        return self._subcomponents

    @property
    def devviz_components(self) -> Optional[Dict]:
        """For migration. Should be deprecated after migration."""
        return self._devviz_components

    @property
    def generated_from_gdl(self) -> bool:
        """For migration. Should be deprecated after migration."""
        return self._generated_from_gdl

    @property
    def referenced_descriptors(self) -> List[Path]:
        """Return the descriptor files referenced by this descriptor.

        Return value is an array of referenced file path
        """
        ret = []
        for desc in self.subcomponents.values():
            file_path = self.dirname / desc.component_type
            ret.append(file_path)
        return ret

    @classmethod
    def from_json_data(
        cls, content: Dict, path: Union[str, Path]
    ) -> "GraphletDescriptor":
        """Create GraphletDescriptor from JSON data."""

        path = Path(path)

        subcomponents = {
            name: SubcomponentDefinition(
                name=name,
                component_type=Path(subcomp_raw["componentType"]),
                parameters=subcomp_raw.get("parameters", {}),
                comment=subcomp_raw.get("comment", None),
            )
            for name, subcomp_raw in content.get("subcomponents", {}).items()
        }

        connections = [
            ConnectionDefinition(
                src=conn_raw["src"],
                params=conn_raw.get("params", {}),
                dests=conn_raw["dests"],
            )
            for conn_raw in content.get("connections", [])
        ]

        comp = super().from_json_data(content, path)

        devviz_components = content.get("devvizComponents", None)

        return GraphletDescriptor(
            path,
            name=comp.name,
            parameters=comp.parameters,
            input_ports=comp.input_ports,
            output_ports=comp.output_ports,
            comment=comp.comment,
            subcomponents=subcomponents,
            connections=connections,
            devviz_components=devviz_components,
            generated_from_gdl=content.get("generatedFromGDL", False),
        )

    @classmethod
    def empty_desc(cls, name: str, path: Union[str, Path]) -> "GraphletDescriptor":
        """Create empty GraphletDescriptor and bind to a desc path."""
        path = Path(path)
        return GraphletDescriptor(
            path,
            name=name,
            parameters={},
            input_ports={},
            output_ports={},
            comment=None,
            subcomponents={},
            connections=[],
            devviz_components={},
            generated_from_gdl=False,
        )

    def to_json_data(self) -> OrderedDict:
        """Dump GraphletDescriptor to JSON data."""

        comp_json_data = super().to_json_data()
        graphlet_json: OrderedDict = OrderedDict()

        if self.comment is not None:
            graphlet_json["comment"] = self.comment

        graphlet_json["name"] = comp_json_data["name"]
        graphlet_json["inputPorts"] = comp_json_data["inputPorts"]
        graphlet_json["outputPorts"] = comp_json_data["outputPorts"]
        graphlet_json["parameters"] = comp_json_data["parameters"]

        def dump_subcomponents(
            subcomponents: Dict[str, SubcomponentDefinition]
        ) -> OrderedDict:
            """Dump subcomponents."""
            subcomp_json: OrderedDict = OrderedDict()

            for k, v in subcomponents.items():
                subcomp_json[k] = OrderedDict(componentType=str(v.component_type))
                if v.parameters is not None and len(v.parameters) > 0:
                    subcomp_json[k]["parameters"] = v.parameters
                if v.comment is not None:
                    subcomp_json[k]["comment"] = v.comment

            return subcomp_json

        def dump_connections(
            connections: List[ConnectionDefinition]
        ) -> List[OrderedDict]:
            """Dump connections."""
            connections_json = []

            for conn in connections:
                conn_json: OrderedDict = OrderedDict()
                conn_json["src"] = conn.src
                conn_json["dests"] = conn.dests
                if conn.params is not None and len(conn.params) > 0:
                    conn_json["params"] = conn.params
                connections_json.append(conn_json)
            return connections_json

        graphlet_json["subcomponents"] = dump_subcomponents(self.subcomponents)
        graphlet_json["connections"] = dump_connections(self.connections)
        if self.devviz_components is not None:
            graphlet_json["devvizComponents"] = self.devviz_components
        if self.generated_from_gdl:
            graphlet_json["generatedFromGDL"] = self.generated_from_gdl

        return graphlet_json
