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
"""For transformation descriptor."""
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from .descriptor import Descriptor
from .descriptor_factory import DescriptorFactory
from .descriptor_factory import DescriptorType


class MergePatchBehaviorDefinition:
    """Wrapper for JSON merge patch behaviors."""

    def __init__(
        self, *, allow_adding: bool = True, allow_deleting_nonexistent_item: bool = True
    ):
        """By default allow adding and allow deleting nonexistent item."""
        self._allow_adding = allow_adding
        self._allow_deleting_nonexistent_item = allow_deleting_nonexistent_item

    @property
    def allow_adding(self) -> bool:
        """Return if allow adding during merge patch operation."""
        return self._allow_adding

    @property
    def allow_deleting_nonexistent_item(self) -> bool:
        """Return if allow deleting non-existent items."""
        return self._allow_deleting_nonexistent_item


class ActionDefinition:
    """Action definitions inside transformation descriptor."""

    def __init__(
        self,
        *,
        action_type: str,
        instances: Optional[List[str]] = None,
        attrs: Optional[Dict] = None,
        comment: Optional[str] = None,
    ):
        """ActionDefinition represents an item in actions section."""
        self._type = action_type
        self._instances = instances if instances is not None else ["__self__"]
        self._attrs = attrs if attrs is not None else {}
        self._comment = comment

    @property
    def type(self) -> str:
        """Return the action type."""
        return self._type

    @property
    def instances(self) -> List[str]:
        """Return the target instances name."""
        return self._instances

    @property
    def attrs(self) -> Dict:
        """Return the action attributes."""
        return self._attrs

    @property
    def comment(self) -> Optional[str]:
        """Return the comment."""
        return self._comment

    @classmethod
    def from_json_data(cls, content: Dict) -> "ActionDefinition":
        """Create an ActionDefinition from JSON data."""

        content = deepcopy(content)
        action_type = cast(str, content.get("action"))
        instances = content.get("instances", None)
        comment = content.get("comment", None)
        attrs = content.get("attributes", {})

        if instances is None:
            instances = ["__self__"]
        elif len(instances) == 0:
            raise ValueError("instances cannot be empty")

        return ActionDefinition(
            action_type=action_type, instances=instances, attrs=attrs, comment=comment
        )

    def to_json_data(self) -> OrderedDict:
        """Dump the ActionDefinition to JSON data."""

        action_json: OrderedDict = OrderedDict()
        if self.comment is not None:
            action_json["comment"] = self.comment

        action_json["action"] = self.type
        if len(self.instances) == 1 and self.instances[0] == "__self__":
            pass
        else:
            action_json["instances"] = self.instances

        if self.attrs:
            action_json["attributes"] = self.attrs

        return action_json


@DescriptorFactory.register(DescriptorType.TRANSFORMATION)
class TransformationDescriptor(Descriptor):
    """class for transformation file."""

    def __init__(
        self,
        file_path: Path,
        *,
        target: str,
        target_type: str,
        actions: Optional[List[ActionDefinition]] = None,
        comment: Optional[str] = None,
    ):
        """Create TransformationDescriptor instance.

        @param file_path path of this descriptor file
        """
        super().__init__(file_path)
        self._target = target
        self._target_type = target_type
        self._actions = actions if actions is not None else []
        self._comment = comment

    @property
    def comment(self) -> Optional[str]:
        """Return comment of this descriptor."""
        return self._comment

    @property
    def target(self) -> str:
        """Return the target descriptor name."""
        return self._target

    @property
    def target_type(self) -> str:  # Literal["app", "graphlet"], supported in python 3.8
        """Return the target descriptor type."""
        return self._target_type

    @property
    def actions(self) -> List[ActionDefinition]:
        """Return the ActionDefinitions of the transformation file."""
        return self._actions

    @property
    def referenced_descriptors(self) -> List[Path]:
        """Return the referenced transformation file."""
        ret = []
        for action in self.actions:
            if action.type == "perform-transformation":
                ret.append(
                    self.dirname  # type: ignore
                    / action.attrs.get("transformationDesc")
                )
        return ret

    @classmethod
    def from_json_data(
        cls, content: Dict, path: Union[str, Path]
    ) -> "TransformationDescriptor":
        """Create a TransformationDescriptor instance from JSON data."""

        path = Path(path)

        actions = [
            ActionDefinition.from_json_data(action)
            for action in content.get("actions", [])
        ]

        return TransformationDescriptor(
            path,
            target=content.get("target"),  # type: ignore
            target_type=content.get("targetType"),  # type: ignore
            actions=actions,
            comment=content.get("comment", None),
        )

    def to_json_data(self) -> OrderedDict:
        """Dump the transformation descriptor to JSON data."""

        trans_json: OrderedDict = OrderedDict()

        if self.comment is not None:
            trans_json["comment"] = self.comment

        trans_json["target"] = self.target
        trans_json["targetType"] = self.target_type

        trans_json["actions"] = [action.to_json_data() for action in self.actions]

        return trans_json
