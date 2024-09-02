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
# Copyright (c) 2022-2024 NVIDIA Corporation. All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property and proprietary
# rights in and to this software and related documentation and any modifications thereto.
# Any use, reproduction, disclosure or distribution of this software and related
# documentation without an express license agreement from NVIDIA Corporation is
# strictly prohibited.
#
#########################################################################################
"""Data structures for Graphlet."""
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

from dwcgf.transformation.descriptor import (
    ApplicationDescriptor,
    ConnectionDefinition,
    DescriptorLoader,
    DescriptorType,
    GraphletDescriptor,
    ParameterDefinition,
    SubcomponentDefinition,
)
from dwcgf.transformation.json_merge_patch import merge
from dwcgf.transformation.transaction import UndoContext

from .component import Component
from .node import Node
from .object_model_channel import object_model_channel
from .port import Port, PortArray, PortType


class Graphlet(Component):
    """class for Graphlet instance in DAG."""

    @staticmethod
    def is_external(port_name: str) -> bool:
        """Check if the port name stands for external port.

        throw if the port name is empty string.
        """
        if port_name == "":
            raise ValueError("Port name cannot be empty string.")
        return port_name.startswith("EXTERNAL:")

    @staticmethod
    def create_subcomponents(
        loader: DescriptorLoader, desc: Union[GraphletDescriptor, ApplicationDescriptor]
    ) -> Dict[str, Component]:
        """Create all subcomponents of a Graphlet or Application.

        @param loader loader loads the desc.
        @param desc   Descriptor for the Graphlet or Application.
        """
        subcomps: Dict[str, Component] = {}
        subcomp_descs = loader.get_subcomponent_descriptors(desc)
        for k, v in subcomp_descs.items():
            if v.desc_type is DescriptorType.NODE:
                node = Node.from_descriptor(k, v)
                subcomps[k] = node
            elif v.desc_type is DescriptorType.GRAPHLET:
                graphlet = Graphlet.from_descriptor(k, loader, v)
                subcomps[k] = graphlet
            else:
                # should not occur
                raise ValueError("Messed up")
        return subcomps

    @staticmethod
    def resolve_id(full_id: str) -> str:
        """Remove the __self__ in the full_id. __self__ means this component.

        Examples:
        - __self__                  =>  __self__
        - __self__.foo.bar          => foo.bar
        - __self__.__self__         => __self__
        - foo.__self__.__self__.bar => foo.bar
        - foo.__self__.bar.__self__ => foo.bar
        - foo.bar                   => foo.bar
        - foo                       => foo
        - <empty>                   => <empty>
        - .foo                      => <exception>
        - foo..bar                  => <exception>
        - foo.bar.                  => <exception>
        """
        ids = full_id.split(".")
        if len([name for name in ids if name == ""]) > 0:
            raise ValueError(f"Invalid component ID: '{full_id}'")
        if len(ids) <= 1:
            return full_id
        else:
            # take first name no matter what
            new_ids = [ids[0]]

            # remove all "__self__" begin from second name
            for name in ids[1:]:
                if name != "__self__":
                    new_ids.append(name)

            # remove first "__self__" if there is other name
            if len(new_ids) >= 2 and new_ids[0] == "__self__":
                new_ids = new_ids[1:]
            return ".".join(new_ids)

    @staticmethod
    def get_component_impl(comp: "Graphlet", relative_name: str) -> Optional[Component]:
        """Get a component from Graphlet using relative name."""
        if relative_name == "":
            raise ValueError("get_component: relative name cannot be empty")
        names = relative_name.split(".")
        if names[0] == "__self__":
            if len(names) == 1:
                return comp
            else:
                return Graphlet.get_component_impl(comp, ".".join(names[1:]))

        if names[0] in comp.subcomponents:
            subcomp = comp.subcomponents[names[0]]
            if len(names) == 1:
                return subcomp
            elif isinstance(subcomp, Graphlet):
                return Graphlet.get_component_impl(subcomp, ".".join(names[1:]))
            else:
                # subcomp is Node has no subcomponents
                return None
        else:
            return None

    @object_model_channel.ignore
    def __init__(
        self,
        *,
        name: str,
        comment: Optional[str],
        parameters: Optional[Dict[str, ParameterDefinition]] = None,
        input_ports: Optional[Dict[str, Port]] = None,
        output_ports: Optional[Dict[str, Port]] = None,
        subcomponents: Optional[Dict[str, Component]] = None,
        parameter_mappings: Optional[
            Dict[str, Dict[str, SubcomponentDefinition.ParameterValueType]]
        ] = None,
        connection_definitions: Optional[List[ConnectionDefinition]] = None,
    ):
        """Create Graphlet instance."""
        super().__init__(
            name=name,
            comment=comment,
            parameters=parameters,
            input_ports=input_ports,
            output_ports=output_ports,
        )
        self._subcomponents = subcomponents if subcomponents is not None else {}
        self._parameter_mappings = (
            parameter_mappings if parameter_mappings is not None else {}
        )

        self._connection_definitions = (
            connection_definitions if connection_definitions is not None else []
        )

        self._inbound_ports: Dict[str, Port] = {}
        self._outbound_ports: Dict[str, Port] = {}

        for comp in self._subcomponents.values():
            comp.parent = self

        self._init_connections(self._connection_definitions)

        self._check_connections(self._connection_definitions)

    @staticmethod
    def from_descriptor(
        name: str, loader: DescriptorLoader, desc: GraphletDescriptor
    ) -> "Graphlet":
        """Create a Graphlet instance using GraphletDescriptor.

        @param name   name of the instance.
        @param loader the loader which loads the desc.
        @param desc   the descriptor.
        """
        try:
            graphlet = Graphlet(
                name=name,
                comment=desc.comment,
                parameters=deepcopy(desc.parameters),
                input_ports=Port.from_descriptor(desc.input_ports, True),
                output_ports=Port.from_descriptor(desc.output_ports, False),
                subcomponents=Graphlet.create_subcomponents(loader, desc),
                parameter_mappings={
                    subcomp_name: deepcopy(subcomp.parameters)
                    for subcomp_name, subcomp in desc.subcomponents.items()
                },
                connection_definitions=deepcopy(desc.connections),
            )
        except Exception as e:
            raise Exception(f"Failed to load graphlet '{desc.file_path}'") from e

        graphlet.descriptor_path = desc.file_path

        return graphlet

    @property
    def is_modified(self) -> bool:
        """Return if either self or one of subcomponents is modified."""
        for subcomp in self.subcomponents.values():
            if subcomp.is_modified:
                return True
        return super().is_modified

    def _init_connections(self, connections: List[ConnectionDefinition]) -> None:
        """Init the Port instances and connect them accordingly."""
        for conn in connections:
            if Graphlet.is_external(conn.src):
                if conn.src not in self._inbound_ports:
                    src_port = Port(name=conn.src, data_type="", binding_required=False)
                    src_port.parent = self
                    self._inbound_ports[conn.src] = src_port
                else:
                    raise ValueError(
                        f"Graphlet._init_connection: Duplicate inbound port '{conn.src}'."
                    )
            else:
                src_port = self.get_port_by_name(conn.src, True)
            if isinstance(src_port, PortArray):
                raise ValueError(
                    f"Graphlet._init_connection: cannot connect port array directly: "
                    f"src '{conn.src}'."
                )
            for dest, _ in conn.dests.items():
                if Graphlet.is_external(dest):
                    if dest not in self._outbound_ports:
                        dest_port = Port(
                            name=dest, data_type="", binding_required=False
                        )
                        dest_port.parent = self
                        self._outbound_ports[dest] = dest_port
                    else:
                        raise ValueError(
                            f"Graphlet._init_connection: Duplicate outbound port '{dest}'."
                        )
                else:
                    dest_port = self.get_port_by_name(dest, False)
                if isinstance(dest_port, PortArray):
                    raise ValueError(
                        f"Graphlet._init_connection: cannot connect port array directly: "
                        f"dest '{dest}'."
                    )
                if dest_port.upstream:
                    raise ValueError(
                        f"Graphlet._init_connection: Illegal connection from "
                        f"src:'{conn.src}' to dest:'{dest}', "
                        f"dest port:'{dest_port.id}' is already in the connection "
                        f"with upstream:'{dest_port.upstream.id}'"
                    )
                src_port.insert_downstream(dest_port)
                dest_port.upstream = src_port

    def try_get_port_by_name(self, relative_name: str, is_src: bool) -> Optional[Port]:
        """Get a Port instance by relative name. Including ports of direct children."""
        try:
            p = self.get_port_by_name(relative_name, is_src)
        except Exception:
            p = None
        return p

    def get_port_by_name(self, relative_name: str, is_src: bool) -> Port:
        """Get a Port instance by relative name.

        Including external ports and ports of direct children.
        """
        if Graphlet.is_external(relative_name):
            return (
                # if external port not found, throw
                self._inbound_ports[relative_name]
                if is_src
                else self._outbound_ports[relative_name]
            )
        if "." in relative_name:
            # if len != 2, throw on purpose
            subcomp_name, port_name = relative_name.split(".")
            if subcomp_name not in self.subcomponents:
                raise ValueError(
                    f"Cannot find component name {subcomp_name} in {relative_name}"
                )
            sub_comp = self.subcomponents[subcomp_name]
            return (
                sub_comp.get_output_port(port_name)
                if is_src
                else sub_comp.get_input_port(port_name)
            )
        else:
            return (
                self.get_input_port(relative_name)
                if is_src
                else self.get_output_port(relative_name)
            )

    def get_component(self, relative_name: str) -> Optional[Component]:
        """Get component from Graphlet using relative name."""
        return Graphlet.get_component_impl(self, relative_name)

    def get_component_by_id(self, full_id: str) -> Optional[Component]:
        """Get component from Graphlet by full ID."""
        resolved_id = Graphlet.resolve_id(full_id)
        if resolved_id == "__self__":
            return self
        if resolved_id == self.id:
            return self
        if resolved_id.startswith(self.id + "."):
            return self.get_component(resolved_id[len(self.id) + 1 :])
        return None

    def get_all_connections_for_subcomponent(
        self, comp_name: str
    ) -> List[Tuple[str, List[str]]]:
        """Get all connections for subcompoent in this graphlet."""
        if comp_name not in self.subcomponents:
            raise ValueError(
                f"Subcomponent name '{comp_name}' not found in component {self.name}."
            )
        comp = self.subcomponents[comp_name]
        connections = []
        for _, port in comp.input_ports.items():
            ports = []
            if isinstance(port, PortArray):
                ports = port.ports
            else:
                ports = [port]
            for port in ports:
                if port.upstream is not None:
                    if port.upstream.component() is None:
                        raise ValueError(
                            f"port {port.name}'s upstream '{port.upstream.name}' not found."
                        )
                    upstream_comp = port.upstream.component()
                    upstream_port_name = (
                        upstream_comp.name + "." + port.upstream.name  # type: ignore
                    )
                    if upstream_comp is self:
                        upstream_port_name = port.upstream.name
                    connections.append(
                        (upstream_port_name, [comp_name + "." + port.name])
                    )

        for _, port in comp.output_ports.items():
            ports = []
            if isinstance(port, PortArray):
                ports = port.ports
            else:
                ports = [port]
            for port in ports:
                if port.downstreams is not None and len(port.downstreams) > 0:
                    dests = []
                    for downstream_port in port.downstreams:
                        if downstream_port.component() is None:
                            raise ValueError(
                                f"port {port.name}'s downstream '{downstream_port.name}' not found."
                            )
                        downstream_comp = downstream_port.component()
                        downstream_port_name = (
                            downstream_comp.name  # type: ignore
                            + "."
                            + downstream_port.name
                        )
                        if downstream_comp is self:
                            downstream_port_name = downstream_port.name
                        dests.append(downstream_port_name)
                    connections.append((comp_name + "." + port.name, dests))
        return connections

    @property
    def subcomponents(self) -> Dict[str, Component]:
        """Get all subcomponents."""
        return self._subcomponents

    @property
    def children(self) -> List[Component]:
        """Return subcomponents as list."""
        return [value for key, value in self._subcomponents.items()]

    @property
    def parameter_mappings(
        self,
    ) -> Dict[str, Dict[str, SubcomponentDefinition.ParameterValueType]]:
        """Get all subcomponents."""
        return self._parameter_mappings

    @object_model_channel.pair
    def insert_parameter_mappings(
        self,
        subcomp_name: str,
        parameter_mappings: Dict[str, SubcomponentDefinition.ParameterValueType],
    ) -> UndoContext:
        """Insert parameter mappings for given subcomponent (undo-able).

        @param subcomp_name       the subcomponent name
        @param parameter_mappings the new parameter mappings
        """
        if subcomp_name not in self.subcomponents.keys():
            raise ValueError(f"No such subcomponent: '{subcomp_name}'")
        if subcomp_name not in self._parameter_mappings:
            self._parameter_mappings[subcomp_name] = parameter_mappings
            ret = UndoContext(self, subcomp_name, parameter_mappings.keys())
        else:
            for param in parameter_mappings.keys():
                if param in self._parameter_mappings[subcomp_name]:
                    raise ValueError(
                        f"Parameter '{param}' already exists in the mapping."
                    )
            self._parameter_mappings[subcomp_name].update(parameter_mappings)
            ret = UndoContext(self, subcomp_name, parameter_mappings.keys())
        self._self_is_modified = True
        return ret

    @insert_parameter_mappings.pair
    def remove_parameter_mappings(
        self, subcomp_name: str, parameter_names: List[str]
    ) -> UndoContext:
        """Remove parameter mappings from given subcomponent (undo-able).

        @param subcomp_name    the subcomponent name
        @param parameter_names the parameter names to be deleted.
        """
        if subcomp_name not in self._parameter_mappings.keys():
            raise ValueError(f"No such subcomponent: '{subcomp_name}'")
        for param in parameter_names:
            if param not in self._parameter_mappings[subcomp_name]:
                raise ValueError(f"Parameter '{param}' cannot found in the mapping.")
        removed_mappings = {
            param: self._parameter_mappings[subcomp_name].pop(param)
            for param in parameter_names
        }
        self._self_is_modified = True
        return UndoContext(self, subcomp_name, removed_mappings)

    @property
    def inbound_ports(self) -> Dict[str, Port]:
        """Return the inbound ports of this graphlet.

        Inbound port is used to connect entities outside to the DAG.
        If there is no inbound port defined, empty list will be returned.
        """
        return self._inbound_ports

    @property
    def outbound_ports(self) -> Dict[str, Port]:
        """Return the outbound ports of this graphlet.

        Outbound port is used to connect the DAG to entities outside the DAG.
        If there is no outbound port defined, empty list will be returned.
        """
        return self._outbound_ports

    @property
    def connection_definitions(self) -> List[ConnectionDefinition]:
        """Return newest connection definition of this component."""
        return self._connection_definitions

    @object_model_channel.pair
    def insert_subcomponent(
        self,
        comp: Component,
        parameter_mappings: Optional[
            Dict[str, SubcomponentDefinition.ParameterValueType]
        ] = None,
    ) -> UndoContext:
        """Insert a subcomponent with given parameter mappings (undo-able)."""
        if comp.name in self._subcomponents:
            raise ValueError(
                f"Newly added component '{comp.name}' is already a child of '{self.id}'"
            )
        if comp.parent is not None:
            raise ValueError(
                f"Newly added component '{comp.name}' is already a child of '{comp.parent.id}'"
            )
        self._subcomponents[comp.name] = comp
        if parameter_mappings is not None:
            self._parameter_mappings[comp.name] = parameter_mappings
        self._self_is_modified = True
        comp.parent = self
        return UndoContext(self, comp.name)

    @object_model_channel.pair_self
    def replace_subcomponent(
        self,
        comp: Component,
        parameter_mappings: Optional[
            Dict[str, SubcomponentDefinition.ParameterValueType]
        ] = None,
    ) -> UndoContext:
        """Replace an existed subcomponent with a new subcomponent type."""
        if comp.name not in self._subcomponents:
            raise ValueError(
                f"Subcomponent '{comp.name}' not exists, \
                could not replace with new subcomponent type"
            )

        # checking connections
        old_comp = self._subcomponents[comp.name]
        # checking connected input ports
        conn_errors = []
        for _, port in old_comp.input_ports.items():
            ports = []
            if isinstance(port, PortArray):
                ports = port.ports
            else:
                ports = [port]
            for p in ports:
                if p.upstream is not None and comp.try_get_input_port(p.name) is None:
                    conn_errors.append(
                        f"Mismatch input port: original component has input '{p.name}' connected,"
                        f" but new component has no input port named '{p.name}'"
                    )
        # checking connected output ports
        for _, port in old_comp.output_ports.items():
            ports = []
            if isinstance(port, PortArray):
                ports = port.ports
            else:
                ports = [port]
            for p in ports:
                if len(p.downstreams) > 0 and comp.try_get_output_port(p.name) is None:
                    conn_errors.append(
                        f"Mismatch output port: original component has output '{p.name}' connected,"
                        f" but new component has no output port named '{p.name}'"
                    )

        if len(conn_errors) > 0:
            error_msg = "\n".join(conn_errors)
            raise ValueError(f"Replace subcomponent error:\n{error_msg}")

        # replace subcomponent
        old_comp = self._subcomponents.pop(comp.name)
        # replace subcomponent parameter mappings
        old_parameter_mappings = None
        if comp.name in self._parameter_mappings:
            old_parameter_mappings = self._parameter_mappings.pop(comp.name)

        self._subcomponents[comp.name] = comp
        if parameter_mappings is not None:
            self._parameter_mappings[comp.name] = parameter_mappings

        # replace subcomponent parent
        comp.parent = self
        # TODO(shuail) cannot set old_comp.parent to None because the old_comp
        # is referenced by Application processes
        # old_comp.parent = None

        # replace subcomponent input connections
        # no need to modify self._connection_definitions because already checked
        for _, port in old_comp.input_ports.items():
            ports = []
            if isinstance(port, PortArray):
                ports = port.ports
            else:
                ports = [port]
            for p in ports:
                if p.upstream is not None:
                    upstream = p.upstream
                    # break p with upstream
                    p.upstream = None
                    upstream.remove_downstream(p)
                    # connect new p with upstream
                    new_p = comp.get_input_port(p.name)
                    new_p.upstream = upstream
                    upstream.insert_downstream(new_p)
        # replace subcomponent output connections
        for _, port in old_comp.output_ports.items():
            ports = []
            if isinstance(port, PortArray):
                ports = port.ports
            else:
                ports = [port]
            for p in ports:
                if len(p.downstreams) > 0:
                    downstreams = p.downstreams
                    new_p = comp.get_output_port(p.name)
                    for d in downstreams:
                        assert d.upstream is p
                        # break p with downstream
                        d.upstream = None
                        p.remove_downstream(d)
                        # connect new p with downstream
                        d.upstream = new_p
                        new_p.insert_downstream(d)

        self._self_is_modified = True

        return UndoContext(self, old_comp, old_parameter_mappings)

    @insert_subcomponent.pair
    def remove_subcomponent(self, comp_name: str) -> UndoContext:
        """Remove a subcomponent with given name (undo-able)."""
        if comp_name not in self._subcomponents:
            raise ValueError(f"Subcomponent name '{comp_name}' not found.")
        comp = self._subcomponents.pop(comp_name)
        comp.parent = None

        parameter_mappings = None
        if comp_name in self._parameter_mappings:
            parameter_mappings = self._parameter_mappings.pop(comp_name)

        self._self_is_modified = True
        return UndoContext(self, comp, parameter_mappings)

    def _check_connections(
        self, connection_definitions: List[ConnectionDefinition]
    ) -> None:
        """Build a internal connection map for quick look-up."""

        for conn in connection_definitions:
            src_port = self.get_port_by_name(conn.src, True)
            if src_port.component() is None:
                raise ValueError(
                    f"Inside graphlet '{self.id}' a src port '{src_port.id}' has no parent."
                )
            if len(conn.dests) == 0:
                raise ValueError(
                    f"Inside graphlet '{self.id}' a src port '{src_port.id}' has no dest."
                )
            dests_to_this_graphlet: List[Port] = []
            for dest in conn.dests.keys():
                dest_port = self.get_port_by_name(dest, False)
                if dest_port.component() is None:
                    raise ValueError(
                        f"Inside graphlet '{self.id}' a dest port "
                        f"'{dest_port.id}'('src = {conn.src}') has no parent."
                    )
                if (
                    # 1) Inbound -> Outbound
                    # 2) Inbound -> self output
                    # 3) self input -> Outbound
                    # 4) self input -> self output
                    src_port.component()
                    is dest_port.component()
                ):
                    raise ValueError(
                        f"Illegal connection from src: '{conn.src}' to dest: '{dest}' "
                        f"in graphlet '{self.id}'"
                    )

                if (
                    dest_port.component() is self
                    and dest_port.port_type is not PortType.OUTBOUND
                ):
                    dests_to_this_graphlet.append(dest_port)
            if len(dests_to_this_graphlet) > 1:
                names = [p.name for p in dests_to_this_graphlet]
                raise ValueError(
                    f"The subcomponent output port ('{src_port.id}') cannot connects "
                    f"to more than 1 graphlet output: '{names}'"
                )

    def remove_connections(
        self, src: str, dests: List[str], allow_connection_not_found: bool = False
    ) -> None:
        """Remove connections by the src name and the dests names (undo-able)."""
        if src == "" or "" in dests:
            raise ValueError("Port name cannot be empty string.")
        conn_def = None
        for conn in self._connection_definitions:
            if conn.src == src:
                conn_def = conn
                break

        if conn_def is None:
            if not allow_connection_not_found:
                raise ValueError(
                    f"Cannot find connection from src: '{src} "
                    f"in {type(self).__name__} '{self.name}'"
                )
            else:
                # If the connection is missing and 'allow_connection_not_found' is set to True,
                # then return early.
                return

        not_found_dests = set(dests) - set(conn_def.dests)
        if len(not_found_dests) > 0:
            raise ValueError(
                f"Cannot find connection from src: '{src} -> {not_found_dests}' "
                f"in {type(self).__name__} '{self.name}'"
            )

        src_port = self.get_port_by_name(src, True)
        # remove the dests
        for dest in dests:
            dest_port = self.get_port_by_name(dest, False)
            assert dest_port.upstream is src_port
            src_port.remove_downstream(dest_port)
            dest_port.upstream = None
            object_model_channel.dict_remove(conn_def.dests, dest)
            if dest_port.port_type is PortType.OUTBOUND:
                object_model_channel.dict_remove(self._outbound_ports, dest)
                dest_port.parent = None

        if len(conn_def.dests) == 0:
            # remove the connection_definition
            object_model_channel.list_remove_by_value(
                self._connection_definitions, conn_def
            )
            # remove INBOUND if no downstreams
            if src_port.port_type is PortType.INBOUND:
                object_model_channel.dict_remove(self._inbound_ports, src)
                src_port.parent = None

        self._check_connections(self._connection_definitions)
        self._self_is_modified = True
        object_model_channel.attr_update(self, "_self_is_modified", True)

    def remove_connection_to_dest(self, dest: str) -> None:
        """Remove connections to the dest name (undo-able)."""
        if dest == "":
            raise ValueError("Dest name cannot be empty string.")
        for conn in self._connection_definitions[:]:
            if dest in conn.dests:
                self.remove_connections(conn.src, [dest])

    def verify_new_connections(self, new_connection: ConnectionDefinition) -> None:
        """Verify if all internal connections are valid with a new connection."""

        src_port = self.try_get_port_by_name(new_connection.src, True)
        is_external_src = Graphlet.is_external(new_connection.src)
        if src_port is None:
            if not is_external_src:
                raise ValueError(
                    f"Cannot find src port for the new connection: '{new_connection.src}'"
                )

        dests_to_this_graphlet: List[Port] = []
        if src_port is not None:
            if isinstance(src_port, PortArray):
                raise ValueError(
                    f"Cannot connect port array directly: src '{new_connection.src}'"
                )
            for d in src_port.downstreams:
                if d.component() is self and d.port_type is not PortType.OUTBOUND:
                    dests_to_this_graphlet.append(d)

        if len(new_connection.dests) == 0:
            raise ValueError(
                f"New connection contains no dest: src: {new_connection.src}"
            )
        for dest in new_connection.dests:
            is_external_dest = Graphlet.is_external(dest)
            dest_port = self.try_get_port_by_name(dest, False)
            if dest_port is None and not is_external_dest:
                raise ValueError(
                    f"Cannot find dest port for the new connection: 'new_connection.src >> {dest}'"
                )
            if dest_port is not None:
                if isinstance(dest_port, PortArray):
                    raise ValueError(
                        f"Cannot connect port array directly: dest '{dest}'"
                    )

                if dest_port.upstream is not None:
                    raise ValueError(
                        f"Dest port '{dest}' has already connected to src '{dest_port.upstream.id}'"
                    )
                if src_port and src_port.component() is dest_port.component():
                    raise ValueError(
                        f"Cannot connect the output & input of the same component: "
                        f"'{new_connection.src} >> {dest}'"
                    )
                if (
                    dest_port.component() is self
                    and dest_port.port_type is not PortType.OUTBOUND
                ):
                    dests_to_this_graphlet.append(dest_port)
            # else: means dest_port is an outbound and will be created when inserting
        if len(dests_to_this_graphlet) > 1:
            names = [p.name for p in dests_to_this_graphlet]
            raise ValueError(
                f"The subcomponent output port ('{new_connection.src}') cannot connects "
                f"to more than 1 graphlet output: '{names}'"
            )

    def insert_connections(
        self,
        connection_definition: ConnectionDefinition,
        *,
        index: Optional[int] = None,
    ) -> None:
        """Insert connections to the given index (if provided) by a ConnectionDefinition.

        Note: this API is undo-able.
        """
        self.verify_new_connections(connection_definition)
        src_port = self.try_get_port_by_name(connection_definition.src, True)
        if src_port is None:
            # already verified the new connection
            assert Graphlet.is_external(connection_definition.src)
            # new external connection, create new inbound port
            src_port = Port(
                name=connection_definition.src, data_type="", binding_required=False
            )
            src_port.parent = self
            object_model_channel.dict_insert(
                self._inbound_ports, connection_definition.src, src_port
            )

        for dest in connection_definition.dests:
            dest_port = self.try_get_port_by_name(dest, False)
            if dest_port is None:
                assert Graphlet.is_external(dest)
                dest_port = Port(name=dest, data_type="", binding_required=False)
                dest_port.parent = self
                object_model_channel.dict_insert(self._outbound_ports, dest, dest_port)
            src_port.insert_downstream(dest_port)
            dest_port.upstream = src_port

        conn_definition_copy = deepcopy(connection_definition)

        conn_def: ConnectionDefinition = None
        for conn in self._connection_definitions:
            if conn.src == conn_definition_copy.src:
                conn_def = conn
                break
        if conn_def is not None:
            for dest, params in conn_definition_copy.dests.items():
                object_model_channel.dict_insert(conn_def.dests, dest, params)
        else:
            object_model_channel.list_insert(
                self._connection_definitions, conn_definition_copy, index
            )

        self._check_connections(self._connection_definitions)
        self._self_is_modified = True
        object_model_channel.attr_update(self, "_self_is_modified", True)

    def update_connection_params(
        self,
        src: str,
        new_params: Dict[str, Union[bool, int, str]],
    ) -> None:
        """Update connection parameters by the src name (undo-able)."""

        conn_def = None
        for conn in self._connection_definitions:
            if conn.src == src:
                conn_def = conn
                break

        if conn_def is None:
            raise ValueError(f"Cannot find connection from src: {src}.")

        merged_params = deepcopy(conn_def.params)
        merge(merged_params, new_params)
        self.replace_connection_params(src=src, new_params=merged_params)

    def update_connection_consumer_only_params(
        self, src: str, dest: str, new_params: Dict[str, Union[bool, int, str]]
    ) -> None:
        """Update connection consumer-only parameters by src and dest name (undo-able)."""
        conn_def = None
        for conn in self._connection_definitions:
            if conn.src == src:
                conn_def = conn
                break

        if conn_def is None:
            raise ValueError(f"Cannot find connection from src: {src} to {dest}.")

        if dest not in conn_def.dests:
            raise ValueError(f"Cannot find connection from src: {src} to {dest}.")

        merged_params = deepcopy(conn_def.dests[dest])
        merge(merged_params, new_params)
        self.replace_connection_consumer_only_params(
            src=src, dest=dest, new_params=merged_params
        )

    @object_model_channel.pair_self
    def replace_connection_params(
        self,
        src: str,
        new_params: Dict[str, Union[bool, int, str]],
    ) -> UndoContext:
        """Replace connection parameters by the src name (undo-able)."""

        conn_def = None
        for conn in self._connection_definitions:
            if conn.src == src:
                conn_def = conn
                break

        if conn_def is None:
            raise ValueError(f"Cannot find connection from src: {src}.")

        old_params = conn_def.params
        conn_def.params = deepcopy(new_params)

        self._self_is_modified = True

        return UndoContext(self, src, old_params)

    @object_model_channel.pair_self
    def replace_connection_consumer_only_params(
        self, src: str, dest: str, new_params: Dict[str, Union[bool, int, str]]
    ) -> UndoContext:
        """Replace connection consumer-only parameters by src and dest name (undo-able)."""
        conn_def = None
        for conn in self._connection_definitions:
            if conn.src == src:
                conn_def = conn
                break

        if conn_def is None:
            raise ValueError(f"Cannot find connection from src: {src} to {dest}.")

        if dest not in conn_def.dests:
            raise ValueError(f"Cannot find connection from src: {src} to {dest}.")

        old_params = conn_def.dests[dest]
        conn_def.dests[dest] = deepcopy(new_params)

        self._self_is_modified = True

        return UndoContext(self, src, dest, old_params)
