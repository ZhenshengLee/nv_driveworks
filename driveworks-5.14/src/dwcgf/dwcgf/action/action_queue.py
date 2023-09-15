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
"""ActionQueue."""
from copy import deepcopy
from pathlib import Path
from typing import List
from typing import Optional
from typing import Union
from dwcgf.descriptor import ActionDefinition
from dwcgf.descriptor import DescriptorLoader
from dwcgf.descriptor import TransformationDescriptor
from dwcgf.object_model import Application
from dwcgf.object_model import Graphlet

from .action import Action
from .action import ActionAttribute
from .action_factory import ActionFactory


@ActionFactory.register("perform-transformation")
class ActionQueue(Action):
    """class for construct action queue based on transformation file.

    ActionQueue implements perform-transformation action
    ActionQueue corresponding to a transformation file (transformation scope)
    """

    extra_attributes = (
        ActionAttribute(
            name="transformationDesc",
            is_required=True,
            attr_type=str,
            default="",
            description="Relative path to the transformation descriptor file.",
        ),
    )

    @staticmethod
    def create(
        action_definition: ActionDefinition,
        om_loader: DescriptorLoader,
        loader: Optional[DescriptorLoader],
    ) -> Union[Action, List[Action]]:
        """ActionQueue version of create.

        Used to create ActionQueues out of perform-transformation action definition.
        @param action_definition action definition of this ActionQueue.
        @param loader            the loader loads the application and component descriptors.
        @param loader            the loader loads the transformation files, this is needed
                                 for action need to load another transformation descriptor file.
        """
        if loader is None:
            raise ValueError("cannot create ActionQueue with loader is None.")
        desc = loader.get_descriptor_by_path(
            action_definition.get("transformationDesc")
        )
        return ActionQueue.from_descriptor(om_loader, loader, desc, action_definition)

    @staticmethod
    def from_descriptor(
        om_loader: DescriptorLoader,
        loader: DescriptorLoader,
        desc: TransformationDescriptor,
        action_definition: Optional[ActionDefinition] = None,
    ) -> "ActionQueue":
        """Construct an ActionQueue from transformation descriptor.

        @param om_loader         loader which loads object model descriptors.
        @param loader            loader which loads transformation descriptors.
        @param desc              the transformation descriptor to be used to
                                 construct the ActionQueue.
        @param action_definition the action definition represent this ActionQueue action.
        """

        actions: List[Action] = []
        for action_def in desc.actions:
            if action_def.type == "perform-transformation":
                action = ActionQueue.from_descriptor(
                    om_loader,
                    loader,
                    loader.get_descriptor_by_path(
                        desc.dirname / action_def.attrs.get("transformationDesc")
                    ),
                    action_def,
                )
            else:
                action = ActionFactory.create(deepcopy(action_def), om_loader, loader)
            actions.append(action)

        if action_definition is None:
            action_definition = ActionDefinition.from_json_data(
                {
                    "action": "perform-transformation",
                    "instances": ["__self__"],
                    "attributes": {"transformationDesc": str(desc.file_path)},
                }
            )
        else:
            action_definition = deepcopy(action_definition)

        aq = ActionQueue(
            loader=om_loader,
            action_definition=action_definition,
            actions=actions,
            target=desc.target,
            targetType=desc.target_type,
            comment=desc.comment,
        )

        aq.descriptor_path = desc.file_path

        return aq

    def __init__(
        self,
        *,
        loader: DescriptorLoader,
        action_definition: Optional[ActionDefinition] = None,
        actions: Optional[List[Action]] = None,
        target: Optional[str] = None,
        targetType: Optional[str] = None,
        comment: Optional[str] = None,
    ):
        """Create ActionQueue instance.

        @param parent_scope scope owns this ActionQueue, maybe None when this ActionQueue is root
        @param action_definition definition for this ActionQueue
        @param transformation transformation file for the input instance
        """
        super().__init__(action_definition, loader)
        self._actions: List[Action] = actions if actions is not None else []
        self._target = target
        self._targetType = targetType

        for sub_action in self._actions:
            sub_action.scope = self

        self._descriptor_path: Optional[Path] = None

    @property
    def descriptor_path(self) -> Optional[Path]:
        """Return the descriptor path of this action queue.

        The path is used as the key to find the actual descriptor.
        """
        return self._descriptor_path

    @descriptor_path.setter
    def descriptor_path(self, new_path: Path) -> None:
        """Update the path of descriptor of this action queue."""
        self._descriptor_path = new_path

    def transform(self, targets: List[Union[Application, Graphlet]]) -> None:
        """API to perform transformation."""
        for scope_target in targets:
            # application id is none
            for action in self._actions:
                action_targets = []
                for action_instance in action.instances:
                    action_target = scope_target.get_component(action_instance)
                    if action_target is None:
                        raise ValueError(
                            f"Target cannot be found for action:"
                            f" '{action.action_type}: {action_instance}'"
                        )
                    action_targets.append(action_target)
                action.transform(action_targets)
