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
"""Data structures for Action."""
from copy import deepcopy
from pathlib import Path
from typing import Dict
from typing import Tuple

from dwcgf.action import Action
from dwcgf.action import ActionAttribute
from dwcgf.action.action_factory import ActionFactory
from dwcgf.descriptor import ConnectionDefinition
from dwcgf.descriptor import DescriptorType
from dwcgf.descriptor import ParameterDefinition
from dwcgf.descriptor import PortDefinition
from dwcgf.object_model import Graphlet
from dwcgf.object_model import Node
from dwcgf.object_model import Port


####################################################################################################
# Utils
####################################################################################################
@ActionFactory.register("touch")
class Touch(Action):
    """class for touch action."""

    extra_attributes: Tuple = tuple()

    def transform_impl(self, target: Graphlet) -> None:
        """action implementation."""
        # directly modify the internal (cannot undo)
        target._self_is_modified = True


####################################################################################################
# Subcomponents
####################################################################################################
@ActionFactory.register("insert-subcomponent")
class InsertSubcomponent(Action):
    """class for insert-subcomponent action."""

    extra_attributes = (
        ActionAttribute(
            name="componentName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of new subcomponent.",
        ),
        ActionAttribute(
            name="componentType",
            is_required=True,
            attr_type=str,
            default="",
            description="Relative descriptor file path of new subcomponent.",
        ),
        ActionAttribute(
            name="parameterMappings",
            is_required=False,
            attr_type=dict,
            default=None,
            description="Parameter mappings for the new subcomponent.",
        ),
    )

    def transform_impl(self, target: Graphlet) -> None:
        """action implementation."""
        name = self.get_attribute("componentName")
        desc_path_str: str = self.get_attribute("componentType")
        desc_path = Path(desc_path_str)
        parameter_mappings = self.get_attribute("parameterMappings")

        # desc_path is specified in absolute path
        # or there is no ActionQueue owning this Action
        # then desc_path should be directly point to the file
        if not desc_path.is_absolute():
            if self.scope is not None:
                if self.scope.descriptor_path is None:
                    raise ValueError(
                        "Containing ActionQueue don't have descriptor file path associate with it."
                    )
                desc_path = self.scope.descriptor_path.parent / desc_path

        desc = self._loader.get_descriptor_by_path(desc_path)

        if desc.desc_type == DescriptorType.GRAPHLET:
            comp = Graphlet.from_descriptor(name, self._loader, desc)
        elif desc.desc_type == DescriptorType.NODE:
            comp = Node.from_descriptor(name, desc)
        else:
            raise ValueError(
                f"Wrong componentType for insert-subcomponent action: '{desc_path_str}'"
            )

        target.insert_subcomponent(comp, parameter_mappings)


@ActionFactory.register("replace-subcomponent")
class ReplaceSubcomponent(Action):
    """class for replace-subcomponent action."""

    extra_attributes = (
        ActionAttribute(
            name="componentName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of replaced subcomponent.",
        ),
        ActionAttribute(
            name="componentType",
            is_required=True,
            attr_type=str,
            default="",
            description="Relative descriptor file path of new subcomponent.",
        ),
        ActionAttribute(
            name="parameterMappings",
            is_required=False,
            attr_type=dict,
            default=None,
            description="Parameter mappings for the new subcomponent.",
        ),
    )

    def transform_impl(self, target: Graphlet) -> None:
        """action implementation."""
        name = self.get_attribute("componentName")
        desc_path_str: str = self.get_attribute("componentType")
        desc_path = Path(desc_path_str)
        parameter_mappings = self.get_attribute("parameterMappings")

        # desc_path is specified in absolute path
        # or there is no ActionQueue owning this Action
        # then desc_path should be directly point to the file
        if not desc_path.is_absolute():
            if self.scope is not None:
                if self.scope.descriptor_path is None:
                    raise ValueError(
                        "Containing ActionQueue don't have descriptor file path associate with it."
                    )
                desc_path = self.scope.descriptor_path.parent / desc_path

        desc = self._loader.get_descriptor_by_path(desc_path)
        if desc.desc_type == DescriptorType.GRAPHLET:
            comp = Graphlet.from_descriptor(name, self._loader, desc)
        elif desc.desc_type == DescriptorType.NODE:
            comp = Node.from_descriptor(name, desc)
        else:
            raise ValueError(
                f"Wrong componentType for replace-subcomponent action: '{desc_path_str}'"
            )
        target.replace_subcomponent(comp, parameter_mappings)


@ActionFactory.register("remove-subcomponent")
class RemoveSubcomponent(Action):
    """class for remove-subcomponent action."""

    extra_attributes = (
        ActionAttribute(
            name="componentName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of removed subcomponent.",
        ),
        ActionAttribute(
            name="expectExists",
            is_required=False,
            attr_type=bool,
            default=True,
            description="Action succeeds only if subcomponent exists",
        ),
    )

    def transform_impl(self, target: Graphlet) -> None:
        """action implementation."""
        comp_name = self.get_attribute("componentName")
        if "." in comp_name:
            comp_names = comp_name.split(".")
            direct_target_name = ".".join(comp_names[:-1])
            comp_name = comp_names[-1]
            target = target.get_component(direct_target_name)
            if target is None and not self.get_attribute("expectExists"):
                return
        else:
            if comp_name not in target.subcomponents and not self.get_attribute(
                "expectExists"
            ):
                return
        for src, dests in target.get_all_connections_for_subcomponent(comp_name):
            target.remove_connections(src, dests)
        target.remove_subcomponent(comp_name)


@ActionFactory.register("replace-switchboard")
class ReplaceSwitchboard(Action):
    """class for switchboard-type replacement."""

    extra_attributes = (
        ActionAttribute(
            name="switchboardOutput",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of output being replaced.",
        ),
        ActionAttribute(
            name="source",
            is_required=True,
            attr_type=str,
            default=True,
            description="Name of output used for replacement",
        ),
        ActionAttribute(
            name="params",
            is_required=True,
            attr_type=dict,
            default={},
            description="Producer side channel parameters.",
        ),
    )

    def _replace_output(
        self,
        target: Graphlet,
        original_src: str,
        new_src: str,
        is_internal_output: bool,
    ) -> None:
        connection_definitions = deepcopy(target.connection_definitions)
        replacement_made = False
        for connection_definition in connection_definitions:
            src = connection_definition.src
            if src == original_src:
                dest_names = list(connection_definition.dests.keys())
                target.remove_connections(src, dest_names)
                new_dests = {}
                for key in connection_definition.dests.keys():
                    # skip the outbound dest when performing replacement
                    # graphlet input -> graphlet output is not allowed
                    if "." in key:
                        new_dests[key] = connection_definition.dests[key]
                new_connection_definition = ConnectionDefinition(
                    src=new_src, dests=new_dests, params=connection_definition.params,
                )
                if len(new_dests) > 0:
                    target.insert_connections(new_connection_definition)
                replacement_made = True
        if not replacement_made and is_internal_output:
            raise ValueError(f"internal connections with {original_src} not found.")

    def transform_impl(self, target: Graphlet) -> None:
        """action implementation."""
        switchboard_output = self.get_attribute("switchboardOutput")
        new_src = self.get_attribute("source")
        params = self.get_attribute("params")
        mailbox_auto = {}
        if "mailbox" in params.keys() and params["mailbox"] == "auto":
            mailbox_auto["mailbox"] = "auto"

        # 1. go into subcomponent (and subsubcomponent) and create new input port
        # 1.1 determine the port type
        switchboard_tokens = switchboard_output.split(".")
        depth = len(switchboard_tokens)
        if depth not in [3, 4, 5]:
            raise ValueError(
                f"Expected <subcomp>.<subsubcomp>.<subsubsubcomp>.<subsubsubsubcomp>.<output>; "
                + f"received {switchboard_output}"
            )

        subcomponent_name = switchboard_tokens[0]
        subcomponent = target.subcomponents[subcomponent_name]
        original_port_name = ".".join(switchboard_tokens[-2:])
        if depth == 3:
            data_type = subcomponent.get_port_by_name(
                original_port_name, True
            ).data_type
        elif depth == 4:
            subsubcomponent_name = switchboard_tokens[1]
            subsubcomponent = subcomponent.subcomponents[subsubcomponent_name]
            data_type = subsubcomponent.get_port_by_name(
                original_port_name, True
            ).data_type
        elif depth == 5:
            subsubcomponent_name = switchboard_tokens[1]
            subsubcomponent = subcomponent.subcomponents[subsubcomponent_name]

            subsubsubcomponent_name = switchboard_tokens[2]
            subsubsubcomponent = subcomponent.subcomponents[
                subsubcomponent_name
            ].subcomponents[subsubsubcomponent_name]
            data_type = subsubsubcomponent.get_port_by_name(
                original_port_name, True
            ).data_type

        # 1.2 create the new input port
        new_port_name = "SUB_" + original_port_name.replace(".", "_")
        # Handle indexed ports, e.g. PORT[#]
        new_port_name = new_port_name.replace("[", "_").replace("]", "")
        new_port = Port(name=new_port_name, data_type=data_type, binding_required=False)
        is_input = True
        subcomponent.insert_port(is_input, new_port)
        if depth == 4:
            new_subport_name = "SUB_" + new_port_name
            new_subport = Port(
                name=new_subport_name, data_type=data_type, binding_required=False
            )
            subsubcomponent.insert_port(is_input, new_subport)

        if depth == 5:
            new_subport_name = "SUB_" + new_port_name
            new_subport = Port(
                name=new_subport_name, data_type=data_type, binding_required=False
            )
            subsubcomponent.insert_port(is_input, new_subport)

            new_subsubport_name = "SUB_" + new_subport_name
            new_subsubport = Port(
                name=new_subsubport_name, data_type=data_type, binding_required=False
            )
            subsubsubcomponent.insert_port(is_input, new_subsubport)

        # 2. hook up the new source to the newly created port
        new_connection_definition = ConnectionDefinition(
            src=new_src,
            dests={(subcomponent_name + "." + new_port_name): mailbox_auto},
            params=params,
        )
        target.insert_connections(new_connection_definition)
        if depth == 4:
            new_subconnection_definition = ConnectionDefinition(
                src=new_port_name,
                dests={(subsubcomponent_name + "." + new_subport_name): mailbox_auto},
                params=params,
            )
            subcomponent.insert_connections(new_subconnection_definition)

        if depth == 5:
            new_subconnection_definition = ConnectionDefinition(
                src=new_port_name,
                dests={(subsubcomponent_name + "." + new_subport_name): mailbox_auto},
                params=params,
            )
            subcomponent.insert_connections(new_subconnection_definition)

            new_subsubconnection_definition = ConnectionDefinition(
                src=new_subport_name,
                dests={
                    (subsubsubcomponent_name + "." + new_subsubport_name): mailbox_auto
                },
                params=params,
            )
            subsubcomponent.insert_connections(new_subsubconnection_definition)

        # 3. reroute what originally came from switchboard to come from the newly created input
        # 3.1 first, check if switchboard output is directly routed to an output of the subcomponent
        direct_outputs = []
        for connection_definition in (
            subcomponent
            if depth == 3
            else subsubcomponent
            if depth == 4
            else subsubsubcomponent
        ).connection_definitions:
            if connection_definition.src == original_port_name:
                for dest in connection_definition.dests.keys():
                    if "." not in dest:
                        direct_outputs.append((dest, connection_definition.dests[dest]))
        if depth == 4:
            # look for outputs that essentially flow all the way back to the top
            redir_outputs = []
            for direct_output in direct_outputs:
                looking_for = subcomponent_name + "." + direct_output[0]
                for target_connection_definition in target.connection_definitions:
                    if looking_for == target_connection_definition.src:
                        for dest in target_connection_definition.dests.keys():
                            if "." not in dest:
                                redir_outputs.append(
                                    (dest, target_connection_definition.dests[dest])
                                )
        if depth == 5:
            # look for outputs that essentially flow all the way back to the top
            up_outputs = []
            for direct_output in direct_outputs:
                up_looking_for = subsubsubcomponent_name + "." + direct_output[0]
                for up_connection_definition in subsubcomponent.connection_definitions:
                    if up_looking_for == up_connection_definition.src:
                        for dest in up_connection_definition.dests.keys():
                            if "." not in dest:
                                up_outputs.append(
                                    (dest, up_connection_definition.dests[dest])
                                )
            upup_outputs = []
            for up_output in up_outputs:
                upup_looking_for = subsubcomponent_name + "." + up_output[0]
                for upup_connection_definition in subcomponent.connection_definitions:
                    if upup_looking_for == upup_connection_definition.src:
                        for dest in upup_connection_definition.dests.keys():
                            if "." not in dest:
                                upup_outputs.append(
                                    (dest, upup_connection_definition.dests[dest])
                                )
        # 3.2 replace the direct outputs
        if depth == 3:
            for direct_output in direct_outputs:
                self._replace_output(
                    target, subcomponent_name + "." + direct_output[0], new_src, False
                )
        elif depth == 5:
            for up_direct_output in direct_outputs:
                self._replace_output(
                    subsubcomponent,
                    subsubsubcomponent_name + "." + up_direct_output[0],
                    new_subport_name,
                    False,
                )

            for upup_direct_output in up_outputs:
                self._replace_output(
                    subcomponent,
                    subsubcomponent_name + "." + upup_direct_output[0],
                    new_port_name,
                    False,
                )

            for upupup_output in upup_outputs:
                self._replace_output(
                    target, subcomponent_name + "." + upupup_output[0], new_src, False
                )
        else:
            for direct_output in direct_outputs:
                self._replace_output(
                    subcomponent,
                    subsubcomponent_name + "." + direct_output[0],
                    new_port_name,
                    False,
                )
            for redir_output in direct_outputs:
                self._replace_output(
                    target, subcomponent_name + "." + redir_output[0], new_src, False
                )

        # 3.3 replace the internal outputs
        if depth == 3:
            self._replace_output(subcomponent, original_port_name, new_port_name, True)
        elif depth == 5:
            self._replace_output(
                subsubsubcomponent, original_port_name, new_subsubport_name, True
            )
        else:
            self._replace_output(
                subsubcomponent, original_port_name, new_subport_name, True
            )


@ActionFactory.register("remove-all-unspecified-subcomponents")
class RemoveUnspecifiedSubcomponents(Action):
    """class for remove-all-unspecified-subcomponents action."""

    extra_attributes = (
        ActionAttribute(
            name="componentsToKeep",
            is_required=True,
            attr_type=list,
            default=[],
            description="Subcomponents to keep.",
        ),
    )

    def should_keep(self, component_name: str) -> bool:
        """determine if a component name should be kept.

        some examples:
        let components_to_keep = ['top.planningAndControl.avPlanner']
        then the following need to be "kept":
          top
          top.planningAndControl.avPlanner
          top.planningAndControl.avPlanner.behaviorPlannerNode
          ...
        """

        keeps = self.get_attribute("componentsToKeep")
        for keep in keeps:
            if component_name[0 : len(keep)] in keep:
                return True
        return False

    def _find_components_to_remove(self, target: Graphlet, prefix: str = "") -> list:
        if getattr(target, "subcomponents", None) is None:
            return []

        removes = []

        for component in target.subcomponents:
            component_name = prefix + component
            should_keep = self.should_keep(component_name)
            if not should_keep:
                removes.append(component_name)
            removes.extend(
                self._find_components_to_remove(
                    target.subcomponents[component], prefix + component + "."
                )
            )
        return removes

    def transform_impl(self, target: Graphlet) -> None:
        """action implementation."""
        comp_names = self.get_attribute("componentsToKeep")
        comp_name = comp_names[0]
        removes = self._find_components_to_remove(target)

        for comp_name in removes:
            # TODO(mwatson, xda): it would be nice to reuse RemoveSubcomponent here
            # check that the component wasn't already removed
            if comp_name not in target.subcomponents:
                # already deleted
                continue

            for src, dests in target.get_all_connections_for_subcomponent(comp_name):
                target.remove_connections(src, dests)
            target.remove_subcomponent(comp_name)


@ActionFactory.register("move-subcomponent")
class MoveSubcomponentAction(Action):
    """class for move-subcomponent action."""

    extra_attributes = (
        ActionAttribute(
            name="componentName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of new subcomponent.",
        ),
        ActionAttribute(
            name="newComponentName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of new subcomponent.",
        ),
    )

    def transform_impl(self, target: Graphlet) -> None:
        """action implementation."""

        def get_comp_parent(
            component_name: str, target: Graphlet
        ) -> Tuple[Graphlet, str]:
            if "." in component_name:
                component_names = component_name.split(".")
                new_direct_target_name = ".".join(component_names[:-1])
                component_name = component_names[-1]
                return target.get_component(new_direct_target_name), component_name
            else:
                return target, component_name

        new_comp_name = deepcopy(self.get_attribute("newComponentName"))
        assert isinstance(new_comp_name, str)
        dst_target, new_comp_name = get_comp_parent(new_comp_name, target)

        if new_comp_name in dst_target.subcomponents:
            raise ValueError(f"{new_comp_name} already exists in dst graphlet.")

        comp_name = self.get_attribute("componentName")
        assert isinstance(comp_name, str)
        src_target, comp_name = get_comp_parent(comp_name, target)

        if comp_name not in src_target.subcomponents:
            raise ValueError(f"{comp_name} does not exist in src graphlet.")

        for src, dests in src_target.get_all_connections_for_subcomponent(comp_name):
            src_target.remove_connections(src, dests)

        context = src_target.remove_subcomponent(comp_name)
        comp = context.args[1]
        parameter_mappings = context.args[2]

        comp.name = new_comp_name  # call of undo-able api Component._set_name()
        dst_target.insert_subcomponent(comp, parameter_mappings)


####################################################################################################
# Parameters
####################################################################################################
@ActionFactory.register("insert-parameter")
class InsertParameter(Action):
    """class for insert-parameter action."""

    extra_attributes = (
        ActionAttribute(
            name="parameterName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of new parameter.",
        ),
        ActionAttribute(
            name="parameterType",
            is_required=True,
            attr_type=str,
            default="",
            description="Type of new parameter.",
        ),
        ActionAttribute(
            name="array",
            is_required=False,
            attr_type=int,
            default=None,
            description="Parameter array size.",
        ),
        ActionAttribute(
            name="parameterDefault",
            is_required=False,
            attr_type=(str, int, float, bool, list),
            default=None,
            description="Type of new parameter.",
        ),
    )

    def transform_impl(self, target: Graphlet) -> None:
        """action implementation."""
        definition = ParameterDefinition(
            name=self.get_attribute("parameterName"),
            parameter_type=self.get_attribute("parameterType"),
            array_size=self.get_attribute("array"),  # may return None
            default_value=self.get_attribute("parameterDefault"),
        )
        target.insert_parameter(definition)


@ActionFactory.register("remove-parameter")
class RemoveParameter(Action):
    """class for remove-parameter action."""

    extra_attributes = (
        ActionAttribute(
            name="parameterName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of new parameter.",
        ),
    )

    def transform_impl(self, target: Graphlet) -> None:
        """action implementation."""
        target.remove_parameter(self.get_attribute("parameterName"))


@ActionFactory.register("insert-parameter-mappings")
class InsertParameterMappings(Action):
    """class for insert-parameter-mappings action."""

    extra_attributes = (
        ActionAttribute(
            name="subcomponentName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of target subcomponent.",
        ),
        ActionAttribute(
            name="parameterMappings",
            is_required=True,
            attr_type=dict,
            default={},
            description="New parameter mappings.",
        ),
    )

    def transform_impl(self, target: Graphlet) -> None:
        """action implementation."""
        target.insert_parameter_mappings(
            self.get_attribute("subcomponentName"),
            self.get_attribute("parameterMappings"),
        )


@ActionFactory.register("remove-parameter-mappings")
class RemoveParameterMappings(Action):
    """class for remove-parameter-mappings action."""

    extra_attributes = (
        ActionAttribute(
            name="subcomponentName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of target subcomponent.",
        ),
        ActionAttribute(
            name="parameterNames",
            is_required=True,
            attr_type=list,
            default=[],
            description="Parameter names to delete.",
        ),
    )

    def transform_impl(self, target: Graphlet) -> None:
        """action implementation."""
        target.remove_parameter_mappings(
            self.get_attribute("subcomponentName"), self.get_attribute("parameterNames")
        )


@ActionFactory.register("update-parameter-mappings")
class UpdateParameterMppings(Action):
    """class for update-parameter-mappings."""

    extra_attributes = (
        ActionAttribute(
            name="subcomponentName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of target subcomponent.",
        ),
        ActionAttribute(
            name="parameterMappings",
            is_required=True,
            attr_type=dict,
            default={},
            description="New parameter mappings.",
        ),
    )

    def transform_impl(self, target: Graphlet) -> None:
        """action implementation."""
        parameter_mappings: Dict = self.get_attribute("parameterMappings")
        subcomp_name: str = self.get_attribute("subcomponentName")
        if subcomp_name not in target.parameter_mappings:
            raise ValueError(
                f"Subcomponent '{subcomp_name}' has empty parameter mappings"
                " or the subcomponent is not there."
            )
        for param_name in parameter_mappings.keys():
            if param_name not in target.parameter_mappings[subcomp_name]:
                raise ValueError(f"Parameter '{param_name}' cannot be found.")

        target.remove_parameter_mappings(subcomp_name, parameter_mappings.keys())
        target.insert_parameter_mappings(subcomp_name, parameter_mappings)


####################################################################################################
# Ports
####################################################################################################
@ActionFactory.register("insert-port")
class InsertPort(Action):
    """class for insert-port action."""

    extra_attributes = (
        ActionAttribute(
            name="isInput",
            is_required=True,
            attr_type=bool,
            default=True,
            description="Is input port or not.",
        ),
        ActionAttribute(
            name="portName",
            is_required=True,
            attr_type=str,
            default="",
            description="New port name.",
        ),
        ActionAttribute(
            name="portType",
            is_required=True,
            attr_type=str,
            default="",
            description="New port data type.",
        ),
        ActionAttribute(
            name="array",
            is_required=False,
            attr_type=int,
            default=None,
            description="Port array size.",
        ),
        ActionAttribute(
            name="bindingRequired",
            is_required=False,
            attr_type=bool,
            default=False,
            description="If the port is required to be bound.",
        ),
    )

    def transform_impl(self, target: Graphlet) -> None:
        """action implementation."""
        name = self.get_attribute("portName")
        is_input = self.get_attribute("isInput")
        port_dict = Port.from_descriptor(
            {
                name: PortDefinition(
                    name=name,
                    data_type=self.get_attribute("portType"),
                    array_size=self.get_attribute("array"),
                    binding_required=self.get_attribute("bindingRequired"),
                )
            },
            is_input,
        )

        target.insert_port(is_input, port_dict[name])


@ActionFactory.register("remove-port")
class RemovePort(Action):
    """class for remove-port action."""

    extra_attributes = (
        ActionAttribute(
            name="isInput",
            is_required=True,
            attr_type=bool,
            default=True,
            description="Is input port or not.",
        ),
        ActionAttribute(
            name="portName",
            is_required=True,
            attr_type=str,
            default="",
            description="New port name.",
        ),
    )

    def transform_impl(self, target: Graphlet) -> None:
        """action implementation."""
        name = self.get_attribute("portName")
        is_input = self.get_attribute("isInput")
        target.remove_port(is_input, name)


####################################################################################################
# Connections
####################################################################################################
@ActionFactory.register("remove-connection")
class RemoveConnection(Action):
    """class for remove-connection action."""

    extra_attributes = (
        ActionAttribute(
            name="src",
            is_required=True,
            attr_type=str,
            default="",
            description="Source port name.",
        ),
        ActionAttribute(
            name="dests",
            is_required=True,
            attr_type=list,
            default=[],
            description="Destination port name.",
        ),
    )

    def transform_impl(self, target: Graphlet) -> None:
        """action implementation."""
        src = self.get_attribute("src")
        dests: Dict = self.get_attribute("dests")
        if len(dests) == 0:
            raise ValueError("dests cannot be empty")
        target.remove_connections(src, dests)


@ActionFactory.register("insert-connection")
class InsertConnection(Action):
    """class for insert-connection action."""

    extra_attributes = (
        ActionAttribute(
            name="src",
            is_required=True,
            attr_type=str,
            default="",
            description="Source port name.",
        ),
        ActionAttribute(
            name="dests",
            is_required=True,
            attr_type=dict,
            default={},
            description="Destination port name + consumer side channel parameters.",
        ),
        ActionAttribute(
            name="params",
            is_required=False,
            attr_type=dict,
            default={},
            description="Producer side channel parameters.",
        ),
    )

    def transform_impl(self, target: Graphlet) -> None:
        """action implementation.

        Example:
        {
            "action": "insert-connection",
            "src": "nodeA.portA",
            "dests": {
                "nodeB.portB": {"paramB": "valueB"},
                "nodeC.portC": {}
            },
            "params": {
                "paramsD": "valueD"
            }
        }
        """
        src: str = self.get_attribute("src")
        dests: Dict = self.get_attribute("dests")
        params: Dict = self.get_attribute("params")

        if len(params) != 0:
            for i in target.connection_definitions:
                if i.src == src:
                    raise ValueError(f"ERROR! src: {src} exists. Cannot update params!")

        for dest, param in dests.items():
            connection_definition = ConnectionDefinition(
                src=src, dests={dest: param}, params=params
            )
            target.insert_connections(connection_definition)


@ActionFactory.register("insert-connection-nested")
class InsertConnectionNested(Action):
    """
    class for insert-connection-nested action.

    note: this action only works when the destination is nested. if necessary,
          additional code is required to support nested sources
    """

    extra_attributes = (
        ActionAttribute(
            name="src",
            is_required=True,
            attr_type=str,
            default="",
            description="Source port name.",
        ),
        ActionAttribute(
            name="dests",
            is_required=True,
            attr_type=dict,
            default={},
            description="Destination port name + consumer side channel parameters.",
        ),
        ActionAttribute(
            name="params",
            is_required=False,
            attr_type=dict,
            default={},
            description="Producer side channel parameters.",
        ),
    )

    def _get_dest_port(self, target: Graphlet, dest: str) -> Port:
        """get the port of a (possibly) nested destination."""
        dest_path = dest.split(".")
        subcomponent = target
        while len(dest_path) > 2:
            subcomponent = subcomponent.subcomponents[dest_path[0]]
            dest_path = dest_path[1:]

        dest_path = dest.split(".")
        subcomponent_port_name = ".".join(dest_path[-2:])
        is_src = False
        dest_port = subcomponent.get_port_by_name(subcomponent_port_name, is_src)
        return dest_port

    def transform_impl(self, target: Graphlet) -> None:
        """action implementation."""

        src: str = self.get_attribute("src")
        dests: Dict = self.get_attribute("dests")
        params: Dict = self.get_attribute("params")

        # sanity check for limitations
        if len(src.split(".")) > 2:
            raise ValueError(f"ERROR! nested src: {src} is not supported")

        if len(params) != 0:
            for i in target.connection_definitions:
                if i.src == src:
                    raise ValueError(f"ERROR! src: {src} exists. Cannot update params!")

        for dest, param in dests.items():
            # sanity check for limitations
            depth = len(dest.split("."))

            # sanity check for limitations
            if depth < 3:
                raise ValueError(
                    f"ERROR! only support nesting more than three level "
                    "currently dest is not support (incompatible dest: {dest})"
                )

            # get the type of the port
            dest_port = self._get_dest_port(target, dest)

            # create the proxy ports and port
            dest_path = dest.split(".")
            subcomponent = target
            # TODO(mwatson, xda): a future enhancement might be to add a while
            #                     loop here and iterate through the nestings. It
            #                     is not currently needed and difficult to test,
            #                     so it is not implemented here.

            count = 0
            loop = depth - 2
            while count < loop:
                # create an inbound/proxy-accessible port into the destination
                child_name = dest_path[count]
                child = subcomponent.subcomponents[child_name]
                port_name = "PROXY_VIEW_" + "_".join(dest_path[count:])
                new_port = Port(
                    name=port_name,
                    data_type=dest_port.data_type,
                    binding_required=False,
                )
                is_input = True
                child.insert_port(is_input, new_port)

                # add an "external" connection between the newly created port and the destination
                connection_definition = ConnectionDefinition(
                    src=src,
                    dests={(child_name + "." + port_name): param},
                    params=params,
                )
                subcomponent.insert_connections(connection_definition)

                subcomponent = child
                src = port_name
                count = count + 1

            # add a connection internal to the component
            connection_src = port_name
            connection_dest = ".".join(dest_path[-2:])
            connection_definition = ConnectionDefinition(
                src=connection_src, dests={connection_dest: param}, params=params
            )
            child.insert_connections(connection_definition)


@ActionFactory.register("update-connection-params")
class UpdateConnectionParams(Action):
    """class for update-connection-params action."""

    extra_attributes = (
        ActionAttribute(
            name="src",
            is_required=True,
            attr_type=str,
            default="",
            description="Source port name.",
        ),
        ActionAttribute(
            name="newParams",
            is_required=True,
            attr_type=dict,
            default={},
            description="New params to be updated.",
        ),
    )

    def transform_impl(self, target: Graphlet) -> None:
        """action implementation."""
        target.update_connection_params(
            src=self.get_attribute("src"), new_params=self.get_attribute("newParams"),
        )


@ActionFactory.register("update-connection-params-multiple")
class UpdateConnectionParamsMultiple(Action):
    """class for update-connection-params-multiple action."""

    extra_attributes = (
        ActionAttribute(
            name="srcs",
            is_required=True,
            attr_type=list,
            default="",
            description="Source port names.",
        ),
        ActionAttribute(
            name="newParams",
            is_required=True,
            attr_type=dict,
            default={},
            description="New params to be updated.",
        ),
    )

    def transform_impl(self, target: Graphlet) -> None:
        """action implementation."""
        for src in self.get_attribute("srcs"):
            target.update_connection_params(
                src=src, new_params=self.get_attribute("newParams"),
            )


@ActionFactory.register("update-connection-consumer-only-params")
class UpdateConnectionConsumerOnlyParams(Action):
    """class for update-connection-consumer-only-params action."""

    extra_attributes = (
        ActionAttribute(
            name="src",
            is_required=True,
            attr_type=str,
            default="",
            description="Source port name.",
        ),
        ActionAttribute(
            name="dest",
            is_required=True,
            attr_type=str,
            default="",
            description="Destination port name.",
        ),
        ActionAttribute(
            name="newParams",
            is_required=True,
            attr_type=dict,
            default={},
            description="New params to be updated.",
        ),
    )

    def transform_impl(self, target: Graphlet) -> None:
        """action implementation."""
        target.update_connection_consumer_only_params(
            src=self.get_attribute("src"),
            dest=self.get_attribute("dest"),
            new_params=self.get_attribute("newParams"),
        )


@ActionFactory.register("replace-connection-params")
class ReplaceConnectionParams(Action):
    """class for replace-connection-params action."""

    extra_attributes = (
        ActionAttribute(
            name="src",
            is_required=True,
            attr_type=str,
            default="",
            description="Source port name.",
        ),
        ActionAttribute(
            name="newParams",
            is_required=True,
            attr_type=dict,
            default={},
            description="New params to be updated.",
        ),
    )

    def transform_impl(self, target: Graphlet) -> None:
        """action implementation."""
        target.replace_connection_params(
            src=self.get_attribute("src"), new_params=self.get_attribute("newParams"),
        )


@ActionFactory.register("replace-connection-consumer-only-params")
class ReplaceConnectionConsumerOnlyParams(Action):
    """class for replace-connection-consumer-only-params action."""

    extra_attributes = (
        ActionAttribute(
            name="src",
            is_required=True,
            attr_type=str,
            default="",
            description="Source port name.",
        ),
        ActionAttribute(
            name="dest",
            is_required=True,
            attr_type=str,
            default="",
            description="Destination port name.",
        ),
        ActionAttribute(
            name="newParams",
            is_required=True,
            attr_type=dict,
            default={},
            description="New params to be updated.",
        ),
    )

    def transform_impl(self, target: Graphlet) -> None:
        """action implementation."""
        target.replace_connection_consumer_only_params(
            src=self.get_attribute("src"),
            dest=self.get_attribute("dest"),
            new_params=self.get_attribute("newParams"),
        )


@ActionFactory.register("check-connection")
class CheckConnection(Action):
    """class for check-connection action."""

    extra_attributes = (
        ActionAttribute(
            name="src",
            is_required=True,
            attr_type=str,
            default="",
            description="Source port name.",
        ),
        ActionAttribute(
            name="dests",
            is_required=True,
            attr_type=list,
            default=[],
            description="Destination port name.",
        ),
    )

    def transform_impl(self, target: Graphlet) -> None:
        """action implementation.

        Example:
        {
            "action": "check-connection",
            "src": "nodeA.portA",
            "dests": [
                "nodeB.portB",
                "nodeC.portC",
            ]
        }
        """
        src: str = self.get_attribute("src")
        dests: list = self.get_attribute("dests")

        conn_defs = target.connection_definitions
        for i in conn_defs:
            if i.src == src:
                target_dests = list(i.dests.keys())
                target_dests.sort()
                dests.sort()
                if target_dests != dests:
                    new_dests = dests[:]
                    new_target_dests = target_dests[:]
                    for d in dests:
                        if d in target_dests:
                            new_dests.remove(d)
                            new_target_dests.remove(d)

                    raise ValueError(
                        f"Connection Check fails: src: {src}\n should connect to dests:\n\n"
                        f"{dests}\n \ninstead of\n\n {target_dests} \n\n"
                        f"should connect to dests:\n\n {new_dests} \n\n"
                        f"instead of \n\n{new_target_dests}\n\n"
                    )
