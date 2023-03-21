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
"""Data structures for Graphlet."""
from copy import deepcopy
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from dwcgf.descriptor import ApplicationDescriptor
from dwcgf.descriptor import ConnectionDefinition
from dwcgf.descriptor import DescriptorLoader
from dwcgf.descriptor import DescriptorType
from dwcgf.descriptor import GraphletDescriptor
from dwcgf.descriptor import ParameterDefinition
from dwcgf.descriptor import SubcomponentDefinition
from dwcgf.json_merge_patch import merge
from dwcgf.transaction import UndoContext

from .component import Component
from .node import Node
from .object_model_channel import object_model_channel
from .port import Port
from .port import PortArray

if TYPE_CHECKING:
    from .application import Application  # noqa: F401


class Graphlet(Component):
    """class for Graphlet instance in DAG."""

    # empty string in src or dests means inbound and outbound respectively
    _INBOUND_PORT_NAME: str = ""
    _OUTBOUND_PORT_NAME: str = ""

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
        devviz_components: Optional[Dict] = None,
        generated_from_gdl: bool = False,
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

        self._inbound_ports: List[Port] = []
        self._outbound_ports: List[Port] = []

        for comp in self._subcomponents.values():
            comp.parent = self

        self._init_connections(self._connection_definitions)
        self._devviz_components = devviz_components
        self._generated_from_gdl = generated_from_gdl

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
                devviz_components=deepcopy(desc.devviz_components),
                generated_from_gdl=desc.generated_from_gdl,
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

    @property
    def devviz_components(self) -> Optional[Dict]:
        """For migration and should be deprecated after that."""
        return self._devviz_components

    @object_model_channel.pair_self
    def update_devviz_components(self, devviz_components_new: dict) -> UndoContext:
        """User defined Actions."""
        if devviz_components_new is self._devviz_components:
            return UndoContext.NO_CHANGE
        ret = UndoContext(self, self._devviz_components)
        self._devviz_components = devviz_components_new
        self._self_is_modified = True
        return ret

    @property
    def generated_from_gdl(self) -> bool:
        """For migration. Should be deprecated after migration."""
        return self._generated_from_gdl

    def _init_connections(self, connections: List[ConnectionDefinition]) -> None:
        """Init the Port instances and connect them accordingly."""
        for conn in connections:
            if conn.src == Graphlet._INBOUND_PORT_NAME:
                src_port = Port(name="", data_type="", binding_required=False)
                src_port.parent = self
                self._inbound_ports.append(src_port)
            else:
                src_port = self.get_port_by_name(conn.src, True)
            for dest, _ in conn.dests.items():
                if dest == Graphlet._OUTBOUND_PORT_NAME:
                    dest_port = Port(name="", data_type="", binding_required=False)
                    dest_port.parent = self
                    self._outbound_ports.append(dest_port)
                else:
                    dest_port = self.get_port_by_name(dest, False)
                if dest_port.upstream:
                    raise ValueError(
                        f"Graphlet._init_connection: Illegal connection from src:'{conn.src}' to dest:'{dest}', \
                        dest port:'{dest_port.id}' is already in the connection \
                        with upstream:'{dest_port.upstream.id}'"
                    )
                src_port.insert_downstream(dest_port)
                dest_port.upstream = src_port

    def get_port_by_name(self, relative_name: str, is_src: bool) -> Port:
        """Get a Port instance by relative name. Including ports of direct children."""
        if "." in relative_name:
            # if len != 2, throw on purpose
            subcomp_name, port_name = relative_name.split(".")
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
            raise ValueError(f"Subcomponent name '{comp_name}' not found.")
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
                        if downstream_port.component() is not None:
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
    def inbound_ports(self) -> List[Port]:
        """Return the inbound ports of this graphlet.

        Inbound port is used to connect entities outside to the DAG.
        If there is no inbound port defined, empty list will be returned.
        """
        return self._inbound_ports

    @property
    def outbound_ports(self) -> List[Port]:
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

        old_comp = self._subcomponents.pop(comp.name)
        old_parameter_mappings = None
        if comp.name in self._parameter_mappings:
            old_parameter_mappings = self._parameter_mappings.pop(comp.name)

        self._subcomponents[comp.name] = comp
        if parameter_mappings is not None:
            self._parameter_mappings[comp.name] = parameter_mappings

        if comp.parent is None:
            comp.parent = self

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

    @object_model_channel.pair
    def remove_connections(self, src: str, dests: List[str]) -> UndoContext:
        """Remove connections by the src name and the dests names (undo-able)."""
        conn_def = None
        for conn in self._connection_definitions:
            if conn.src == src:
                valid = True
                for dest in dests:
                    if dest not in conn.dests:
                        valid = False
                        break
                if valid:
                    conn_def = conn
                    break

        if conn_def is None:
            for conn in self._connection_definitions:
                if conn.src == src:
                    for dest in dests:
                        if dest not in conn.dests:
                            print(dest)
            raise ValueError(f"Cannot find connection from src: '{src}'")

        for dest in dests:
            if dest not in conn_def.dests:
                raise ValueError(
                    f"One of dest port '{dest}' is not in _connection_definition"
                )

        index = self._connection_definitions.index(conn_def)

        inbound_index = None
        if src == Graphlet._INBOUND_PORT_NAME:
            target_port = None
            for port in self._inbound_ports:
                dests_candidate = set(
                    map(
                        lambda x: f"{x.component().name}.{x.name}",  # type: ignore
                        port.downstreams,
                    )
                )
                if (
                    set(dests) == dests_candidate
                ):  # only to delete the inbound port if all downstreams are deleted
                    target_port = port
                    break
            if target_port is not None:
                inbound_index = self._inbound_ports.index(target_port)
                self._inbound_ports.pop(inbound_index)

        outbound_index = None

        src_port = None
        for dest in dests:
            dest_port = None
            if dest != Graphlet._OUTBOUND_PORT_NAME:
                dest_port = self.get_port_by_name(dest, False)
                src_port = dest_port.upstream
            else:  # dest == Graphlet._OUTBOUND_PORT_NAME
                # now src will not be INBOUD_PORT, call get_port_by_name directly
                src_port = self.get_port_by_name(src, True)
                for port in src_port.downstreams:
                    if port.name == Graphlet._OUTBOUND_PORT_NAME:
                        dest_port = port
                        # remove the outbound port
                        outbound_index = self._outbound_ports.index(dest_port)
                        self._outbound_ports.pop(outbound_index)
                        break
            dest_port.upstream = None  # type: ignore
            src_port.remove_downstream(dest_port)  # type: ignore

        if len(conn_def.dests) == len(dests):
            self._connection_definitions.pop(index)
            ret = UndoContext(
                self,
                conn_def,
                index=index,
                inbound_index=inbound_index,
                outbound_index=outbound_index,
            )
        else:
            removed_dests = {}
            for dest in dests:
                removed_dests[dest] = conn_def.dests.pop(dest)
            removed_connection = ConnectionDefinition(
                src=src, dests=removed_dests, params={}
            )
            ret = UndoContext(
                self,
                removed_connection,
                dest_hint=list(conn_def.dests.keys())[0],
                inbound_index=inbound_index,
                outbound_index=outbound_index,
            )  # at least one dest port exists

        self._self_is_modified = True
        return ret

    @remove_connections.pair
    def insert_connections(
        self,
        connection_definition: ConnectionDefinition,
        *,
        index: Optional[int] = None,
        dest_hint: Optional[str] = None,
        inbound_index: Optional[int] = None,
        outbound_index: Optional[int] = None,
    ) -> UndoContext:
        """Insert connections to the given index (if provided) by a ConnectionDefinition.

        Note: this API is undo-able.
        """
        for dest in connection_definition.dests.keys():
            if dest != Graphlet._OUTBOUND_PORT_NAME:
                dest_port = self.get_port_by_name(dest, False)
                if dest_port.upstream:
                    raise ValueError(
                        f"Insert connections: Illegal connection from src:'{connection_definition.src}' to dest:'{dest}', \
                        dest port:'{dest_port.id}' is already in the connection \
                        with upstream:'{dest_port.upstream.id}'"
                    )

        conn_def = None
        for conn in self._connection_definitions:
            if conn.src == connection_definition.src:
                if conn.src != Graphlet._INBOUND_PORT_NAME:
                    conn_def = conn
                    break
                else:
                    # find conn_def by dest_hint
                    if dest_hint in conn.dests:
                        conn_def = conn
                        break

        if conn_def is not None:
            for dest in connection_definition.dests.keys():
                if dest in conn_def.dests:
                    raise ValueError(f"'{dest}' is already in the connection")

            conn_def.dests.update(connection_definition.dests)
        else:
            if index is not None:
                self._connection_definitions.insert(index, connection_definition)
            else:
                self._connection_definitions.append(connection_definition)

        if connection_definition.src == Graphlet._INBOUND_PORT_NAME:
            if dest_hint is not None:
                src_port = self._get_inbound_port(dest_hint)
            else:
                src_port = self._create_in_out_bound_port(
                    inbound_index=inbound_index, outbound_index=outbound_index
                )
        else:
            src_port = self.get_port_by_name(connection_definition.src, True)

        dest_ports = list()
        for dest in connection_definition.dests.keys():
            if dest == Graphlet._OUTBOUND_PORT_NAME:
                dest_port = self._create_in_out_bound_port(
                    inbound=False,
                    inbound_index=inbound_index,
                    outbound_index=outbound_index,
                )
            else:
                dest_port = self.get_port_by_name(dest, False)
            dest_ports.append(dest_port)

        for dest_port in dest_ports:
            dest_port.upstream = src_port
            src_port.insert_downstream(dest_port)

        self._self_is_modified = True
        return UndoContext(
            self, connection_definition.src, list(connection_definition.dests.keys())
        )

    def update_connection_params(
        self,
        src: str,
        new_params: Dict[str, Union[bool, int, str]],
        dest_hint: Optional[str],
    ) -> None:
        """Update connection parameters by the src name (undo-able)."""
        if src == Graphlet._INBOUND_PORT_NAME and dest_hint is None:
            raise ValueError(f"Must provide dest hint for inbound src: {src}.")

        conn_def = None
        for conn in self._connection_definitions:
            if conn.src == src and src != "":
                conn_def = conn
                break
            elif conn.src == src and src == "":
                assert dest_hint is not None
                if dest_hint in conn.dests:
                    conn_def = conn
                    break

        if conn_def is None:
            raise ValueError(f"Cannot find connection from src: {src}.")

        merged_params = deepcopy(conn_def.params)
        merge(merged_params, new_params)
        self.replace_connection_params(
            src=src, dest_hint=dest_hint, new_params=merged_params
        )

    def update_connection_consumer_only_params(
        self, src: str, dest: str, new_params: Dict[str, Union[bool, int, str]]
    ) -> None:
        """Update connection consumer-only parameters by src and dest name (undo-able)."""
        conn_def = None
        for conn in self._connection_definitions:
            if conn.src == src:
                valid = True
                if dest not in conn.dests:
                    valid = False
                if valid:
                    conn_def = conn
                    break

        if conn_def is None:
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
        dest_hint: Optional[str],
        new_params: Dict[str, Union[bool, int, str]],
    ) -> UndoContext:
        """Replace connection parameters by the src name (undo-able)."""
        if src == Graphlet._INBOUND_PORT_NAME and dest_hint is None:
            raise ValueError(f"Must provide dest hint for inbound src: {src}.")

        conn_def = None
        for conn in self._connection_definitions:
            if conn.src == src and src != "":
                conn_def = conn
                break
            elif conn.src == src and src == "":
                assert dest_hint is not None
                if dest_hint in conn.dests:
                    conn_def = conn
                    break

        if conn_def is None:
            raise ValueError(f"Cannot find connection from src: {src}.")

        old_params = deepcopy(conn_def.params)
        conn_def.params = deepcopy(new_params)

        self._self_is_modified = True

        return UndoContext(self, src, dest_hint, old_params)

    @object_model_channel.pair_self
    def replace_connection_consumer_only_params(
        self, src: str, dest: str, new_params: Dict[str, Union[bool, int, str]]
    ) -> UndoContext:
        """Replace connection consumer-only parameters by src and dest name (undo-able)."""
        conn_def = None
        for conn in self._connection_definitions:
            if conn.src == src:
                valid = True
                if dest not in conn.dests:
                    valid = False
                if valid:
                    conn_def = conn
                    break

        if conn_def is None:
            raise ValueError(f"Cannot find connection from src: {src} to {dest}.")

        old_params = deepcopy(conn_def.dests[dest])
        conn_def.dests[dest] = deepcopy(new_params)

        self._self_is_modified = True

        return UndoContext(self, src, dest, old_params)

    def _get_inbound_port(self, dest_hint: str) -> Port:
        dest_hint_parent, dest_hint_node = dest_hint.split(".")
        for port in self._inbound_ports:
            for downstream in port.downstreams:
                if (
                    downstream.component().name == dest_hint_parent  # type: ignore
                    and downstream.name == dest_hint_node
                ):
                    return port

        # cannot find the existing inbound port, raise Error
        raise ValueError(f"Cannot find inbound port with dest_hint: {dest_hint}")

    def _create_in_out_bound_port(
        self,
        *,
        inbound: bool = True,
        inbound_index: Optional[int] = None,
        outbound_index: Optional[int] = None,
    ) -> Port:
        port = Port(name="", data_type="", binding_required=False)
        port.parent = self
        if inbound:
            if inbound_index is None:
                self._inbound_ports.append(port)
            else:
                self._inbound_ports.insert(inbound_index, port)
        else:
            if outbound_index is None:
                self._outbound_ports.append(port)
            else:
                self._outbound_ports.insert(outbound_index, port)
        return port
