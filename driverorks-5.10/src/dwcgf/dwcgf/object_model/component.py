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
"""Data structures for component base."""
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from dwcgf.descriptor import ParameterDefinition
from dwcgf.transaction import UndoContext

from .object_model_channel import object_model_channel
from .port import Port
from .port import PortArray
from .port import PortArrayItem

if TYPE_CHECKING:
    from .graphlet import Graphlet  # noqa: F401


class Component:
    """Base class for component (Node & Graphlet)."""

    @object_model_channel.ignore
    def __init__(
        self,
        *,
        name: str,
        comment: Optional[str] = None,
        parameters: Optional[Dict[str, ParameterDefinition]] = None,
        input_ports: Optional[Dict[str, Port]] = None,
        output_ports: Optional[Dict[str, Port]] = None,
    ):
        """Create Component instance."""
        self._name = name
        self._comment = comment
        self._parameters = parameters if parameters is not None else {}
        self._input_ports = input_ports if input_ports is not None else {}
        self._output_ports = output_ports if output_ports is not None else {}

        for p in self._input_ports.values():
            p.parent = self
        for p in self._output_ports.values():
            p.parent = self

        self._parent: Optional["Graphlet"] = None

        self._descriptor_path: Optional[Path] = None
        # '_self_is_modified' marks if the Component itself is modified by transform actions
        self._self_is_modified = False

    @property
    def is_modified(self) -> bool:
        """Return if component is modified."""
        return self._self_is_modified

    @property
    def id(self) -> str:
        """Return the unique id in the DAG."""
        if self.parent is not None and self.parent.id is not None:
            return f"{self.parent.id}.{self.name}"
        return self.name

    @property
    def name(self) -> str:
        """Return this component name."""
        return self._name

    @name.setter
    def name(self, new_name: str) -> None:
        """Set component name."""
        self._set_name(new_name)

    @object_model_channel.pair_self
    def _set_name(self, new_name: str) -> UndoContext:
        """Set component name."""
        ret = UndoContext(self, self._name)
        self._name = new_name
        self._self_is_modified = True
        return ret

    @property
    def comment(self) -> Optional[str]:
        """Return this comment of this component."""
        return self._comment

    @property
    def descriptor_path(self) -> Optional[Path]:
        """Return the descriptor path of this component.

        The path is used as the key to find the actual descriptor.
        """
        return self._descriptor_path

    @descriptor_path.setter
    def descriptor_path(self, new_path: Path) -> None:
        """Update the path of descriptor of this component."""
        self._set_descriptor_path(new_path)

    @object_model_channel.pair_self
    def _set_descriptor_path(self, new_path: Path) -> UndoContext:
        """Update the path of the descriptor of this component."""
        ret = UndoContext(self, self._descriptor_path)
        self._descriptor_path = new_path
        # when a new descriptor path is given, the is_modified flag should be clear
        self._self_is_modified = False
        return ret

    @property
    def parameters(self) -> Dict[str, ParameterDefinition]:
        """Return default parameters."""
        return self._parameters

    @object_model_channel.pair
    def insert_parameter(self, definition: ParameterDefinition) -> UndoContext:
        """Insert a parameter for this component (undo-able)."""
        if definition.name in self._parameters:
            raise ValueError(f"Parameter '{definition.name}' already exists.")
        self._parameters[definition.name] = definition
        self._self_is_modified = True
        return UndoContext(self, definition.name)

    @insert_parameter.pair
    def remove_parameter(self, name: str) -> UndoContext:
        """Remove a parameter from this component (undo-able)."""
        if name not in self._parameters:
            raise ValueError(f"Cannot find parameter '{name}'")
        definition = self._parameters.pop(name)
        self._self_is_modified = True
        return UndoContext(self, definition)

    @property
    def input_ports(self) -> Dict[str, Port]:
        """Return input ports."""
        return self._input_ports

    @staticmethod
    def _get_port(port_dict: Dict[str, Port], name: str) -> Optional[Port]:
        """Get a port by name.

        if name contains index, the PortArrayItem will be returned.
        """
        ret: Optional[Port] = None
        if "[" in name:
            # port array
            port_array_name = name.split("[")[0]
            idx = int(name.split("[")[1].split("]")[0])
            if port_array_name in port_dict:
                p = port_dict[port_array_name]
                if isinstance(p, PortArray) and idx < len(p.ports):
                    ret = p.ports[idx]
        else:
            if name in port_dict:
                ret = port_dict[name]
        return ret

    def get_input_port(self, name: str) -> Port:
        """Get input port of this component.

        If there is a PortArray named FOO_BAR, name='FOO_BAR' returns the PortArray,
        name='FOO_BAR[idx]' gives PortWithIndex of PortArray.
        """
        found = Component._get_port(self.input_ports, name)
        if found is None:
            raise ValueError(
                f"Cannot find input port with name: '{name}' in component '{self._name}'"
            )
        return found

    def get_output_port(self, name: str) -> Port:
        """Get output port of this component.

        If there is a PortArray named FOO_BAR, name='FOO_BAR' returns the PortArray,
        name='FOO_BAR[idx]' gives PortWithIndex of PortArray.
        """
        found = Component._get_port(self.output_ports, name)
        if found is None:
            raise ValueError(
                f"Cannot find output port with name '{name}' in component '{self._name}'"
            )
        return found

    @property
    def output_ports(self) -> Dict[str, Port]:
        """Return output ports."""
        return self._output_ports

    @property
    def parent(self) -> Optional["Graphlet"]:
        """Return the component owning this component.

        Maybe None if this component is the root component
        """
        return self._parent

    @parent.setter
    def parent(self, new_parent: Optional["Graphlet"]) -> None:
        """Set new parent."""
        self._set_parent(new_parent)

    @object_model_channel.pair_self
    def _set_parent(self, new_parent: Optional["Graphlet"]) -> UndoContext:
        """Set new parent."""
        ret = UndoContext(self, self._parent)
        self._parent = new_parent
        return ret

    def insert_port(self, is_input: bool, port: Port) -> None:
        """Insert a port to either input port dict or output port dict.

        @param is_input if True, the port will be added to input_ports,
                         otherwise will be added to output_ports.
        @param port     the port to be added.

        Note: this API is undo-able.
        """
        self._insert_port_with_connection(is_input, port)

    @object_model_channel.pair
    def remove_port(self, is_input: bool, name: str) -> UndoContext:
        """Remove a port from either input port dict or output port dict.

        Note: this API is undo-able.
        """
        ports = self._input_ports if is_input else self._output_ports
        if name not in ports:
            raise ValueError(f"Cannot find port '{name}'")
        port = ports.pop(name)
        ret = UndoContext(self, is_input, port, port.upstream, port.downstreams)
        port.upstream = None
        port.downstreams.clear()
        self._self_is_modified = True
        return ret

    @object_model_channel.pair_self
    def update_port_name(self, is_input: bool, name: str, new_name: str) -> UndoContext:
        """Update the port name."""
        if name == new_name:
            return UndoContext.NO_CHANGE
        ports = self._input_ports if is_input else self._output_ports
        if name not in ports:
            raise ValueError(f"Cannot find port '{name}'")
        if new_name in ports:
            raise ValueError(f"Port name '{new_name}' already exists")
        # record the key order into an array
        ports_array = []
        for n, p in ports.items():
            if n == name:  # replace the name
                n = new_name
            ports_array.append((n, p))
        ports.clear()
        # insert the keys in original order
        for item in ports_array:
            n, p = item
            ports[n] = p
        # update the name in Port instance
        ports[new_name].name = new_name
        self._self_is_modified = True
        return UndoContext(self, is_input, new_name, name)

    @remove_port.pair
    def _insert_port_with_connection(
        self,
        is_input: bool,
        port: Port,
        upstream: Optional[Port] = None,
        downstreams: Optional[List[Port]] = None,
    ) -> UndoContext:
        """Undo of remove_port."""
        ports = self._input_ports if is_input else self._output_ports
        if port.name in ports:
            raise ValueError(f"Port '{port.name}' already exists.")
        if port.upstream is not None:
            raise ValueError(f"new port already has upstream")
        ports[port.name] = port
        if upstream is not None:
            port.upstream = upstream
        if downstreams is not None:
            for downstream in downstreams:
                port.insert_downstream(downstream)
        self._self_is_modified = True
        return UndoContext(self, is_input, port.name)

    def resize_port_array(self, is_input: bool, name: str, new_size: int) -> None:
        """Resize a port array."""
        found = Component._get_port(
            self.input_ports if is_input else self.output_ports, name
        )
        if found is not None and isinstance(found, PortArray):
            array_size = len(found.ports)
            if new_size == array_size:
                return
            elif new_size > array_size:
                for idx in range(array_size, new_size):
                    found.insert_port(PortArrayItem(found), idx)
            else:
                for idx in range(array_size, new_size, -1):
                    found.remove_port_at(idx)
            self._self_is_modified = True
        else:
            raise ValueError(f"Array port '{name}' cannot be found.")
