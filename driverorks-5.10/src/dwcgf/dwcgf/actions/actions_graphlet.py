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
"""Data structures for Action."""
from copy import deepcopy
from pathlib import Path
from typing import Dict
from typing import Tuple

from dwcgf.action import Action
from dwcgf.action import ActionAttribute
from dwcgf.action.action_factory import ActionFactory
from dwcgf.descriptor import ConnectionDefinition
from dwcgf.descriptor import DescriptorFactory
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

        if desc_path.is_file():
            desc = self._loader.get_descriptor_by_path(desc_path)
        else:
            Warning(f"{str(desc_path)} not exist, creating empty descriptor.")
            desc_type = DescriptorFactory.determine_descriptor_type(desc_path)
            if desc_type not in [DescriptorType.GRAPHLET, DescriptorType.NODE]:
                raise ValueError(
                    f"Unsupported desc type for subcomponent insertion: {desc_path_str}"
                )
            desc = DescriptorFactory._factory[desc_type].empty_desc(name, desc_path)
            self._loader.add_descriptor(desc)

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
    )

    def transform_impl(self, target: Graphlet) -> None:
        """action implementation."""
        comp_name = self.get_attribute("componentName")
        if "." in comp_name:
            comp_names = comp_name.split(".")
            direct_target_name = ".".join(comp_names[:-1])
            comp_name = comp_names[-1]
            target = target.get_component(direct_target_name)
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
        ActionAttribute(
            name="destHint",
            is_required=False,
            attr_type=str,
            default=None,
            description="Used for inserting inbound port only",
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
        dest_hint: str = self.get_attribute("destHint")

        if len(params) != 0:
            for i in target.connection_definitions:
                if i.src == src:
                    raise ValueError(f"ERROR! src: {src} exists. Cannot update params!")

        for dest, param in dests.items():
            connection_definition = ConnectionDefinition(
                src=src, dests={dest: param}, params=params
            )
            target.insert_connections(connection_definition, dest_hint=dest_hint)


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
            name="destHint",
            is_required=False,
            attr_type=str,
            default=None,
            description="Dest port names if src is a inbound port.",
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
            src=self.get_attribute("src"),
            new_params=self.get_attribute("newParams"),
            dest_hint=self.get_attribute("destHint"),
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
            name="destHint",
            is_required=False,
            attr_type=str,
            default=None,
            description="One of the dest port name if src is a inbound port.",
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
            src=self.get_attribute("src"),
            new_params=self.get_attribute("newParams"),
            dest_hint=self.get_attribute("destHint"),
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
                    raise ValueError(
                        f"Connection Check fails: src: {src}\n should connect to dests:\n\n"
                        f"{dests}\n \ninstead of\n\n {target_dests}"
                    )
