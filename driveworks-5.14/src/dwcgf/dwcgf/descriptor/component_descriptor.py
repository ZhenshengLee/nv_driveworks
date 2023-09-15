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
"""Base class and data structures for Component descriptor."""
from collections import OrderedDict
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from .descriptor import Descriptor


class ParameterDefinition:
    """Class for parameters of component descriptor and application descriptor."""

    DefaultValueType = Union[
        bool, int, float, str, List[bool], List[int], List[float], List[str]
    ]

    def __init__(
        self,
        *,
        name: str,
        parameter_type: str,
        array_size: Optional[int] = None,
        default_value: Optional[DefaultValueType] = None,
    ):
        """Create a ParameterDefinition instance for an entry in parameters section.

        @param name           name of the parameter
        @param parameter_type type of the parameter
        @param array_size     indicates if the parameter is array parameter and the array size
                              if array_size is None, then it's not an array parameter, array_size
                              bigger than 0 indicates the array size.
        @param default_value  the default value of the parameter
        """
        self._name = name
        self._type = parameter_type
        if array_size is not None and array_size < 1:
            raise ValueError(f"parameter array size cannot be less than 1: {name}")
        self._array_size = array_size
        if default_value is not None:  # defalut is provided, check the sizes
            if self._array_size is None:
                if isinstance(default_value, list):
                    raise ValueError(
                        f"non array parameter has array default value: {name}"
                    )
            else:
                if not isinstance(default_value, list):
                    raise ValueError(f"array parameter has non default value: {name}")
                if self._array_size != len(default_value):
                    raise ValueError(
                        f"array parameter size is mismatch with default value size"
                    )

        self._default = default_value

    @property
    def name(self) -> str:
        """Return the parameter name."""
        return self._name

    @property
    def is_array_parameter(self) -> bool:
        """Indicates if this parameter is parameter array."""
        return self._array_size is not None

    @property
    def type(self) -> str:
        """Return parameter type, or element type."""
        return self._type

    @property
    def array_size(self) -> Optional[int]:
        """Return array size of array parameter.

        If the pararmeter is not an array parameter, None will be returned.
        """
        return self._array_size

    @property
    def default(self) -> Optional["DefaultValueType"]:
        """Return default value of the parameter."""
        return self._default


class PortDefinition:
    """Class for port of component descriptor and application descriptor."""

    def __init__(
        self,
        *,
        name: str,
        data_type: str,
        array_size: Optional[int],
        binding_required: bool,
        comment: Optional[str] = None,
    ):
        """Create a PortDefinition instance for an entry in inputPorts / outputPorts section.

        @param name             name of the port
        @param data_type        data type of the port
        @param array_size       indicates if the port is array port and the size of the array port
                                if array_size is None, then it's not an array port.
        @param binding_required indicates the port if is required to be bound
        @param comment          comment for this port
        """
        self._name = name
        self._comment = comment
        self._type = data_type
        if array_size is not None and array_size < 1:
            raise ValueError(f"port array size cannot be less than 1: {name}")
        self._array_size = array_size
        self._binding_required = binding_required

    @property
    def name(self) -> str:
        """Return name of this port."""
        return self._name

    @property
    def comment(self) -> Optional[str]:
        """Return comment of this port."""
        return self._comment

    @property
    def data_type(self) -> str:
        """Return data type of the port."""
        return self._type

    @property
    def is_array_port(self) -> bool:
        """Indicates if the port is array port."""
        return self._array_size is not None

    @property
    def array_size(self) -> Optional[int]:
        """Return the array size.

        If the port is not an array port, None will be returned.
        """
        return self._array_size

    @property
    def binding_required(self) -> bool:
        """Indicates if this port has to be bound or not."""
        return self._binding_required


class ComponentDescriptor(Descriptor):
    """Base class for component descriptors."""

    def __init__(
        self,
        file_path: Path,
        *,
        name: str,
        parameters: Optional[Dict[str, ParameterDefinition]] = None,
        input_ports: Optional[Dict[str, PortDefinition]] = None,
        output_ports: Optional[Dict[str, PortDefinition]] = None,
        comment: Optional[str] = None,
    ):
        """Base for NodeDescriptor and GraphletDescriptor.

        @param file_path    path of the component descriptor file
        @param name         name of the descriptor
        @param              parameters of the component descriptor
        @param input_ports  input ports of the component descriptor
        @param output_ports output ports of the component descriptor
        @param comment      comment for the descriptor
        """
        super().__init__(file_path)
        self._name = name
        self._parameters = parameters if parameters is not None else {}
        self._input_ports = input_ports if input_ports is not None else {}
        self._output_ports = output_ports if output_ports is not None else {}
        self._comment = comment

    @property
    def comment(self) -> Optional[str]:
        """Return comment of this descriptor."""
        return self._comment

    @property
    def name(self) -> str:
        """Return name of this descriptor."""
        return self._name

    @name.setter
    def name(self, new_name: str) -> None:
        """Set new name."""
        self._name = new_name

    @property
    def parameters(self) -> Dict[str, ParameterDefinition]:
        """Return the parameter definition of this component.

        The keys are parameter names, values are ParameterDefinition instances.
        """
        return self._parameters

    @property
    def input_ports(self) -> Dict[str, PortDefinition]:
        """Return the input port definition of this component.

        The keys are input port names.
        """
        return self._input_ports

    @property
    def output_ports(self) -> Dict[str, PortDefinition]:
        """Return the output port definition of this component.

        The keys are output port names.
        """
        return self._output_ports

    @classmethod
    def from_json_data(
        cls, content: Dict, path: Union[str, Path]
    ) -> "ComponentDescriptor":
        """Create ComponentDescriptor from JSON data.

        ComponentDescriptor instance can be used for constructing
        GraphletDescriptor, NodeDescriptor and ApplicationDescriptor

        Example:
            comp = ComponentDescriptor.from_json_data(json, path)
            node = NodeDescriptor(
                path,
                name=comp.name,
                parameters=comp.parameters,
                input_ports=comp.input_ports,
                ...
            )
        """
        path = Path(path)

        def create_ports(ports_raw: Dict) -> Dict[str, PortDefinition]:
            """Helper for create input / ouptut ports."""
            ports = {
                key: PortDefinition(
                    name=key,
                    data_type=value["type"],
                    array_size=value.get("array", None),
                    binding_required=value.get("bindingRequired", False),
                    comment=value.get("comment", None),
                )
                for key, value in ports_raw.items()
            }
            return ports

        parameters = {
            key: ParameterDefinition(
                name=key,
                parameter_type=value["type"],
                array_size=value.get("array", None),
                default_value=value.get("default", None),
            )
            for key, value in content.get("parameters", {}).items()
        }

        input_ports = create_ports(content.get("inputPorts", {}))
        output_ports = create_ports(content.get("outputPorts", {}))

        return ComponentDescriptor(
            path,
            name=content["name"],
            parameters=parameters,
            input_ports=input_ports,
            output_ports=output_ports,
            comment=content.get("comment", None),
        )

    def to_json_data(self) -> OrderedDict:
        """Convert ComponentDescriptor to JSON data.

        Should only be used by GraphletDescriptor, NodeDescriptor and ApplicationDescriptor.
        """

        def dump_ports(ports: Dict[str, PortDefinition]) -> OrderedDict:
            """Dump input and output ports."""
            ports_json: OrderedDict = OrderedDict()
            for k, v in ports.items():
                ports_json[k] = OrderedDict(type=v.data_type)
                if v.is_array_port:
                    ports_json[k]["array"] = v.array_size
                if v.binding_required:
                    ports_json[k]["bindingRequired"] = v.binding_required
                if v.comment:
                    ports_json[k]["comment"] = v.comment
            return ports_json

        def dump_parameters(parameters: Dict[str, ParameterDefinition]) -> OrderedDict:
            """Dump parameters."""
            parameters_json: OrderedDict = OrderedDict()
            for k, v in parameters.items():
                parameters_json[k] = OrderedDict(type=v.type)
                if v.is_array_parameter:
                    parameters_json[k]["array"] = v.array_size
                if v.default is not None:
                    parameters_json[k]["default"] = v.default
            return parameters_json

        json_data: OrderedDict = OrderedDict()
        if self.comment:
            json_data["comment"] = self.comment

        json_data["name"] = self.name
        json_data["inputPorts"] = dump_ports(self.input_ports)
        json_data["outputPorts"] = dump_ports(self.output_ports)
        json_data["parameters"] = dump_parameters(self.parameters)

        return json_data
