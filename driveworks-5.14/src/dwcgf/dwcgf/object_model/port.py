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
"""Data structures for port."""
import enum
from typing import Dict
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union
from dwcgf.descriptor import PortDefinition
from dwcgf.transaction import UndoContext

from .object_model_channel import object_model_channel

if TYPE_CHECKING:
    from .application import Application  # noqa: F401
    from .component import Component  # noqa: F401


@enum.unique
class PortType(enum.Enum):
    """Indicates if the port is as input/output/free port."""

    # the port is a input port
    INPUT = 0
    # the port is a output port
    OUTPUT = 1
    # inbound port is used to connect entities outside to the DAG.
    INBOUND = 2
    # outbound port is used to connect the DAG to entities outside of the DAG.
    OUTBOUND = 3
    # the port don't have parent associated
    FREE = 4


class Port:
    """class for input/output port."""

    ParentType = Union["PortArray", "Component"]

    @object_model_channel.ignore
    def __init__(
        self, *, name: str, data_type: str, binding_required: Optional[bool] = False
    ):
        """Create Port instance.

        @param name             name of this port
        @param data_type        port data type
        @param binding_required if the port is required to be bound
        """
        self._name = name
        self._binding_required = (
            binding_required if binding_required is not None else False
        )
        self._data_type = data_type
        self._parent: Optional["Port.ParentType"] = None
        self._upstream: Optional["Port"] = None
        self._downstreams: List["Port"] = []
        self._parameters: Dict[str, Union[str, int, bool]] = {}

    @property
    def name(self) -> str:
        """Return the name of this port."""
        return self._name

    @name.setter
    def name(self, new_name: str) -> None:
        """Set name for this port."""
        self._set_name(new_name)

    @object_model_channel.pair_self
    def _set_name(self, new_name: str) -> UndoContext:
        """Set new name for this port."""
        if self._name == new_name:
            return UndoContext.NO_CHANGE
        port_type = self.port_type
        if port_type in (PortType.OUTBOUND, PortType.INBOUND):
            raise ValueError("INBOUND or OUTBOUND port name cannot be changed")
        ret = UndoContext(self, self._name)
        self._name = new_name
        # if parent is None, no parent is needed to sync the name
        # if parent is PortArray, _set_name should never be called
        # we cannot directly check isinstance(self.parent, Component)
        # because import .component is a circular dependency.
        if self.parent is None or isinstance(self.parent, PortArray):
            pass
        else:  # isinstance(self.parent, Component)
            # if parent is Component, call parent API to change the name
            # so that the name is in sync

            # mypy is not very good at reasoning about the type of
            # self.parent inside this else block.
            # following line is to make mypy happy
            parent: "Component" = self.parent  # type: ignore
            assert port_type in (
                PortType.INPUT,
                PortType.OUTPUT,
            ), f"Unexpected port type: {port_type}"
            is_input = port_type == PortType.INPUT
            ports = parent.input_ports if is_input else parent.output_ports

            for name, port in ports.items():
                if self is port:
                    parent.update_port_name(is_input, name, new_name)
                    break
        return ret

    @property
    def port_type(self) -> PortType:
        """Indicate if the port is input/output or free."""
        if self.parent is None:
            return PortType.FREE
        if isinstance(self.parent, PortArray):
            return self.parent.port_type
        # mypy is not very good at reasoning about the type of
        # self.parent after above two if-returns
        # following line is to make mypy happy
        parent: "Component" = self.parent  # type: ignore
        from .graphlet import Graphlet

        if self in parent.input_ports.values():
            return PortType.INPUT
        elif self in parent.output_ports.values():
            return PortType.OUTPUT
        elif isinstance(parent, Graphlet) and self in parent.inbound_ports.values():
            return PortType.INBOUND
        elif isinstance(parent, Graphlet) and self in parent.outbound_ports.values():
            return PortType.OUTBOUND
        else:
            raise ValueError("Port has wrong parent.")

    @property
    def binding_required(self) -> bool:
        """Return if binding required for this port."""
        return self._binding_required

    @binding_required.setter
    def binding_required(self, new_value: bool) -> None:
        """Set new binding required value."""
        self._set_binding_required(new_value)

    @object_model_channel.pair_self
    def _set_binding_required(self, new_value: bool) -> UndoContext:
        """Set new binding required value."""
        ret = UndoContext(self, self._binding_required)
        self._binding_required = new_value
        return ret

    @property
    def id(self) -> str:
        """Return port id."""
        if self.parent is not None and self.parent.id is not None:
            return f"{self.parent.id}.{self.name}"
        else:
            return self.name

    @property
    def data_type(self) -> str:
        """Return the data type of the port."""
        return self._data_type

    @data_type.setter
    def data_type(self, new_type: str) -> None:
        """Set port data type."""
        self._set_data_type(new_type)

    @object_model_channel.pair_self
    def _set_data_type(self, new_type: str) -> UndoContext:
        """Set port data type."""
        ret = UndoContext(self, self._data_type)
        self._data_type = new_type
        return ret

    def component(self) -> Optional["Component"]:
        """Return the component who owns this Port."""
        if self.parent is not None:
            if not isinstance(self.parent, PortArray):
                parent: Optional["Component"] = self.parent  # type: ignore
                return parent
            else:
                ret: Optional["Component"] = self.parent.parent
                return ret
        return None

    @property
    def parent(self) -> Optional["ParentType"]:
        """Return the port array or component or application who owns this port."""
        return self._parent

    @parent.setter
    def parent(self, new_parent: Optional[ParentType]) -> None:
        """Set parent for this port."""
        self._set_parent(new_parent)

    @object_model_channel.pair_self
    def _set_parent(self, new_parent: Optional[ParentType]) -> UndoContext:
        """Set new parent."""
        ret = UndoContext(self, self._parent)
        self._parent = new_parent
        return ret

    @property
    def upstream(self) -> Optional["Port"]:
        """Return upstream of this port, maybe None."""
        return self._upstream

    @upstream.setter
    def upstream(self, new_upstream: Optional["Port"]) -> None:
        """Set upstream for this port.

        @param new_upstream new upstream can be either a Port or a PortArrayItem.
        """
        self._set_upstream(new_upstream)

    @object_model_channel.pair_self
    def _set_upstream(self, new_upstream: Optional["Port"]) -> UndoContext:
        """Set upstream for this port."""
        ret = UndoContext(self, self._upstream)
        self._upstream = new_upstream
        return ret

    @property
    def downstreams(self) -> List["Port"]:
        """Return all downstreams of this port."""
        return self._downstreams

    @object_model_channel.pair
    def insert_downstream(
        self, dest_port: "Port", index: Optional[int] = None
    ) -> UndoContext:
        """Add a port to be this port's downstream.

        Connect this port with another port
        """
        if dest_port in self._downstreams:
            raise ValueError(f"new port '{dest_port.name}' is the downstream already.")
        if index is not None:
            if index > len(self._downstreams):
                index = len(self._downstreams)
            self._downstreams.insert(index, dest_port)
        else:
            index = len(self._downstreams)
            self._downstreams.append(dest_port)

        return UndoContext(self, index)

    @insert_downstream.pair
    def _remove_downstream_by_index(self, index: int) -> UndoContext:
        """Remove the downstream by index."""
        if index >= len(self._downstreams):
            raise ValueError("Downstream index '{index}' out of bound")
        downstream = self._downstreams.pop(index)
        return UndoContext(self, downstream, index)

    def remove_downstream(self, dest_port: "Port") -> None:
        """Remove a port from this port's downstreams.

        if the dest_port is not one of the downstream, it will throw.
        """
        index = self._downstreams.index(dest_port)
        self._remove_downstream_by_index(index)

    @property
    def parameters(self) -> Dict[str, Union[str, int, bool]]:
        """Return the connection parameters associated with the port."""
        return self._parameters

    @parameters.setter
    def parameters(self, new_parameters: Dict[str, Union[str, int, bool]]) -> None:
        """Associate parameters with the port."""
        self._set_parameters(new_parameters)

    @object_model_channel.pair_self
    def _set_parameters(
        self, new_parameters: Dict[str, Union[str, int, bool]]
    ) -> UndoContext:
        """Associate new parameters with the port."""
        ret = UndoContext(self, self._parameters)
        self._parameters = new_parameters
        return ret

    @staticmethod
    def from_descriptor(
        desc: Dict[str, PortDefinition], is_input: bool
    ) -> Dict[str, "Port"]:
        """Init an dict of inputPorts / outputPorts from descriptor."""

        ret: Dict[str, "Port"] = {}
        for definition in desc.values():
            if definition.is_array_port:
                port_array = PortArray(
                    name=definition.name,
                    data_type=definition.data_type,
                    array_size=definition.array_size,
                    binding_required=definition.binding_required,
                )
                ret[definition.name] = port_array
            else:
                port = Port(
                    name=definition.name,
                    data_type=definition.data_type,
                    binding_required=definition.binding_required,
                )
                ret[definition.name] = port
        return ret


class PortArrayItem(Port):
    """class for items inside PortArray."""

    @object_model_channel.ignore
    def __init__(self, parent: "PortArray"):
        """Create an item of PortArray."""
        super().__init__(name="", data_type="")
        self._parent = parent

    @property
    def name(self) -> str:
        """Return the name of this port."""
        # self._name is ignored
        if isinstance(self.parent, PortArray):
            return f"{self.parent.name}[{self.index}]"
        else:
            return ""

    @name.setter
    def name(self, new_name: str) -> None:
        """Disallow setting name."""
        raise ValueError(
            "DO NOT set name for PortArrayItem, change parent name instead."
        )

    @property
    def index(self) -> int:
        """The index of this PortArrayItem inside the parent PortArray."""
        if isinstance(self.parent, PortArray):
            return self.parent.ports.index(self)
        else:
            raise ValueError("PortArrayItem doesn't have parent.")

    @property
    def id(self) -> str:
        """Return id of this port."""
        if isinstance(self.parent, PortArray):
            return f"{self.parent.id}[{self.index}]"
        else:
            return ""

    @property
    def binding_required(self) -> bool:
        """Return if binding required for this port."""
        if isinstance(self.parent, PortArray):
            return self.parent._binding_required
        else:
            return False

    @binding_required.setter
    def binding_required(self) -> None:
        """Set binding required is not allowed."""
        raise ValueError("Cannot set binding required for PortArrayItem")

    @property
    def data_type(self) -> str:
        """Return the data type of the port."""
        if isinstance(self.parent, PortArray):
            return self.parent.data_type
        else:
            return ""

    @data_type.setter
    def data_type(self, new_data_type: str) -> None:
        """Set port data type is not allowed."""
        raise ValueError("Cannot set data type for PortArrayItem")


class PortArray(Port):
    """class for port array."""

    PortArrayParentType = Union["Component", "Application"]

    @object_model_channel.ignore
    def __init__(
        self,
        *,
        name: str,
        data_type: str,
        array_size: int,
        binding_required: Optional[bool] = None,
    ):
        """Create a port array instance.

        @param name             name of the port array
        @param data_type        port data type
        @param array_size       port array size
        @param binding_required if the port is required to be bound
        """
        super().__init__(
            name=name, data_type=data_type, binding_required=binding_required
        )

        if array_size < 1:
            raise ValueError(f"Port array size has to be >=1, port: '{name}'")

        self._ports: List[PortArrayItem] = []

        for _ in range(array_size):
            port = PortArrayItem(self)
            self._ports.append(port)

    @property
    def ports(self) -> List[PortArrayItem]:
        """Return port array."""
        return self._ports

    def remove_port(self, port: PortArrayItem) -> None:
        """Remove a port from port array.

        If the port is not a part of the port array, it will throw.
        """

        index = self._ports.index(port)
        self.remove_port_at(index)

    @object_model_channel.pair
    def remove_port_at(self, idx: int) -> UndoContext:
        """Remove port at given index."""

        if idx >= len(self._ports):
            raise ValueError(f"Port index out of bound, port: '{self.name}'")

        port = self._ports.pop(idx)
        port.parent = None
        ret = UndoContext(self, port, idx, port.upstream, port.downstreams)
        port.upstream = None
        port.downstreams.clear()
        return ret

    @remove_port_at.pair
    def _insert_port_with_connection(
        self,
        port: PortArrayItem,
        index: Optional[int] = None,
        upstream: Optional[Port] = None,
        downstreams: Optional[List[Port]] = None,
    ) -> UndoContext:
        """Insert port to port array with connection."""
        if index is not None:
            if index > len(self._ports):
                index = len(self._ports)
            self._ports.insert(index, port)
        else:
            index = len(self._ports)
            self._ports.append(port)
        port.parent = self
        if upstream is not None:
            port.upstream = upstream
        if downstreams is not None:
            for downstream in downstreams:
                port.insert_downstream(downstream)
        return UndoContext(self, index)

    def insert_port(self, port: PortArrayItem, index: Optional[int] = None) -> None:
        """Insert port to port array."""
        self._insert_port_with_connection(port, index)
