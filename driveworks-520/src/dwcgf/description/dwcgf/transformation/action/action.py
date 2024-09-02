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
"""Data structure for Action."""
from copy import deepcopy
import json
from typing import Dict, List, Optional, Tuple, Type, TYPE_CHECKING, Union

from dwcgf.transformation.descriptor import ActionDefinition, DescriptorLoader
from dwcgf.transformation.object_model import (
    Application,
    Graphlet,
    object_model_channel,
)
from dwcgf.transformation.transaction import Edit, UndoContext

from .action_channel import action_channel

if TYPE_CHECKING:
    from .action_queue import ActionQueue


class ActionAttribute:
    """Each action can have extra attributes."""

    allowed_types = (str, int, float, bool, dict, list)
    AttributeType = Union[str, int, float, bool, Dict, List]

    def __init__(
        self,
        *,
        name: str,
        is_required: bool,
        attr_type: Union[Type, Tuple[Type, ...]],
        default: Optional["AttributeType"],
        description: Optional[str] = None,
    ):
        """Represent an extra attribute.

        @param name        attribute name.
        @param is_required if the attribute is required to be provided.
        @param attr_type   type of the attribute, supported types are
                           listed in ActionAttribute.allowed_types (or tuple of them).
        @param default     default value is the attribute is not provided.
        @param description the description about this attribute.
        """
        if not isinstance(attr_type, tuple):
            attr_type = (attr_type,)
        for t in attr_type:
            if t not in type(self).allowed_types:
                raise ValueError(
                    "Action extra attribute only allow to be following types:"
                    f" '{type(self).allowed_types}', given: '{t}', name: '{name}'"
                )
        if default is not None and not isinstance(default, type(self).allowed_types):
            raise ValueError(f"wrong default value type, name: '{name}', '{default}'")
        self._name = name
        self._is_required = is_required
        self._attr_type = attr_type
        self._default = default
        self._value = self._default
        self._description = description if description is not None else ""

    @property
    def name(self) -> str:
        """Return the attribute name."""
        return self._name

    @property
    def is_required(self) -> bool:
        """Return if the attribute is required."""
        return self._is_required

    @property
    def default(self) -> Optional["AttributeType"]:
        """Return default value of this attribute."""
        return self._default

    @property
    def description(self) -> str:
        """Return the description of the attribute."""
        return self._description

    @property
    def value(self) -> Optional["AttributeType"]:
        """Return the value."""
        return self._value

    @value.setter
    def value(self, new_val: "AttributeType") -> None:
        if not isinstance(new_val, self._attr_type):
            raise ValueError(
                f"new attribute value has wrong type, expect:"
                f" '{self._attr_type}', name: {self._name}"
            )
        self._value = new_val


class Action:
    """Base class for all transformation actions."""

    action_type: str = "ActionBase"  # for type checking, should not be used

    extra_attributes: Tuple[ActionAttribute, ...] = ()

    def __init__(self, action_definition: ActionDefinition, loader: DescriptorLoader):
        """Create an Action instance.

        @param scope             ActionQueue who owns this action
        @param action_definition definition of the action
        @param loader            loader for object model descriptors
        """
        if self.action_type != action_definition.type:
            raise ValueError(
                f"Wrong action definition type: '{action_definition.type}',"
                f" expect: {self.action_type}"
            )
        self._scope: Optional["ActionQueue"] = None
        self._action_definition = action_definition
        self._loader = loader
        self._extra_attributes = {
            attr.name: deepcopy(attr) for attr in self.extra_attributes
        }

        provided_attributes = set(action_definition.attrs.keys())
        extra_keys = set(self._extra_attributes.keys())
        redundent_attributes = provided_attributes - extra_keys
        if len(redundent_attributes) > 0:
            raise ValueError(
                f"Unrecognized action parameters '{redundent_attributes}' for"
                f" action {self.action_type}."
            )

        # init extra attributes
        for attr in self._extra_attributes.values():
            if attr.name in action_definition.attrs:
                attr.value = action_definition.attrs.get(attr.name)
            elif attr.is_required:
                raise ValueError(
                    f"{self.action_type} requires attribute '{attr.name}', but not provided.\n"
                    + json.dumps(action_definition.to_json_data(), indent=2)
                )

        # clear attrs in action_definition
        self._action_definition.attrs.clear()

        self._edit: Edit = None

    def get_attribute(self, attr_name: str) -> Optional[ActionAttribute.AttributeType]:
        """Get extra attribute value."""
        if attr_name not in self._extra_attributes:
            raise ValueError(f"Attribute name '{attr_name}' is not defined.")
        return self._extra_attributes[attr_name].value

    @action_channel.pair_self
    def set_attribute(
        self, attr_name: str, new_val: ActionAttribute.AttributeType
    ) -> UndoContext:
        """Set extra attribute value."""
        if attr_name not in self._extra_attributes:
            raise ValueError(f"Attribute name '{attr_name}' is not defined.")
        ret = UndoContext(self, attr_name, self.get_attribute(attr_name))
        self._extra_attributes[attr_name].value = new_val
        return ret

    @property
    def scope(self) -> Optional["ActionQueue"]:
        """Return the scope owns this action."""
        return self._scope

    @scope.setter
    def scope(self, new_scope: "ActionQueue") -> None:
        """Set new scope."""
        self._set_scope(new_scope)

    @action_channel.pair_self
    def _set_scope(self, new_scope: "ActionQueue") -> UndoContext:
        """Undo/redo-able API."""
        ret = UndoContext(self, self._scope)
        self._scope = new_scope
        return ret

    @property
    def comment(self) -> Optional[str]:
        """Return comment of this action."""
        return self._action_definition.comment

    @comment.setter
    def comment(self, new_comment: str) -> None:
        """Set new comment."""
        self._set_comment(new_comment)

    @action_channel.pair_self
    def _set_comment(self, new_comment: str) -> UndoContext:
        """Undo/redo-able API."""
        ret = UndoContext(self, self._action_definition.comment)
        self._action_definition.comment = new_comment
        return ret

    @property
    def instances(self) -> List[str]:
        """Return action target IDs."""
        return self.definition.instances

    def remove_instance(self, instance: str) -> None:
        """Remove an action instance.

        @param instance the instance name.
        """
        index = self._action_definition.instances.index(instance)
        self._remove_instance_by_index(index)

    @action_channel.pair
    def _remove_instance_by_index(self, index: int) -> UndoContext:
        """Remove a instance by index."""

        if index >= len(self._action_definition.instances):
            raise ValueError("Index out of bound")
        value = self._action_definition.instances.pop(index)
        return UndoContext(self, value, index)

    @_remove_instance_by_index.pair
    def insert_instance(self, instance: str, index: Optional[int] = None) -> bool:
        """Insert a instance to the action.

        @param instance new instance name.
        @param index    if given, the instance will be added at given index(or appended),
                        otherwise, the instance will be appended at the end.
        """

        if index is not None:
            if index > len(self._action_definition.instances):
                index = len(self._action_definition.instances)
            self._action_definition.instances.insert(index, instance)
        else:
            index = len(self._action_definition.instances)
            self._action_definition.instances.append(instance)

        return UndoContext(self, index)

    @property
    def definition(self) -> ActionDefinition:
        """Return the definition of this action."""
        self._action_definition.attrs.clear()
        self._action_definition.attrs.update(
            {
                name: attr.value
                for name, attr in self._extra_attributes.items()
                if not (not attr.is_required and attr.value != attr.default)
            }
        )
        return self._action_definition

    @action_channel.pair
    def redo(self) -> UndoContext:
        """Redo the transformation.

        Throw if undo() ever be called yet.
        """
        # if redo() throw, which means code is not implemented correctly
        # in this case, so didn't consider how to restore a clean state
        # when redo() throws.
        self._edit.redo()
        return UndoContext(self)

    @redo.pair
    def undo(self) -> UndoContext:
        """Undo the transformation.

        Throw if transform() ever be called yet, because self._edit is None.
        """
        # if undo() throw, which means code is not implemented correctly
        # in this case, so didn't consider how to restore a clean state
        # when undo() throws.
        self._edit.undo()
        return UndoContext(self)

    @action_channel.pair
    def transform(self, targets: List[Union[Application, Graphlet]]) -> UndoContext:
        """Common entry point for all actions."""

        if self._edit is not None:
            self.redo()
        else:

            def exception_handler(e: Exception) -> bool:
                self._edit.undo()
                return True  # rethrow the exception

            object_model_channel.capture_exception(Exception, exception_handler)
            with object_model_channel.recording() as self._edit:
                for target in targets:
                    try:
                        self.transform_impl(target)
                    except Exception as e:
                        # re-raise with action info
                        target_name: str = ""
                        target_type: str = ""
                        if isinstance(target, Application):
                            target_name = target.name
                            target_type = "application type"
                        elif isinstance(target, Graphlet):
                            target_name = target.id
                            target_type = "graphlet id"

                        action_content: str = json.dumps(
                            self._action_definition.to_json_data(), indent=2
                        )

                        action_file = None
                        if (
                            self.scope is not None
                            and self.scope.descriptor_path is not None
                        ):
                            action_file = self.scope.descriptor_path

                        original_msg: str = str(e)

                        detailed_exception = type(e)(
                            "\n"
                            f"{target_type}: '{target_name}'\n"
                            f"action name: '{self.__class__.__name__}'\n"
                            f"action content: '{action_content}'\n"
                            f"action file: '{action_file}'\n"
                            "\n"
                            f"{original_msg}\n"
                        )
                        print(
                            "Exception captured when execute transform_impl: ",
                            detailed_exception,
                        )
                        raise detailed_exception

        return UndoContext(self, targets)

    @transform.pair
    def _revert_transform(
        self, targets: List[Union[Application, Graphlet]]
    ) -> UndoContext:
        """Used to revert the transform() call.

        NOTE: this API should never be called directly.
        """
        if self._edit is not None:
            self._edit.undo()
            self._edit = None

        return UndoContext(self, targets)

    def transform_impl(self, targets: Union[Application, Graphlet]) -> None:
        """API to transform the target instance and underlying descriptor."""
        raise NotImplementedError(
            f"Action subtype({self.__class__.__name__}) should"
            " have transform_impl() implemented"
        )
