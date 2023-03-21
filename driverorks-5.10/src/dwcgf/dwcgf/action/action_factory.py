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
"""ActionFactory."""
from typing import Dict
from typing import Optional
from typing import Type
from typing import TYPE_CHECKING

from dwcgf.descriptor import ActionDefinition
from dwcgf.descriptor import DescriptorLoader


if TYPE_CHECKING:
    from .action_base import Action


class ActionFactory:
    """class for registering all action implementations.

    Also provides unified entrance for construct the actions.
    """

    _factory: Dict[str, Type["Action"]] = {}

    class register:
        """New action class need to be registered into this factory."""

        def __init__(self, action: str):
            """Remembers action type name.

            @param action action type name.
            """
            self._type = action

        def __call__(self, action_class: Type["Action"]) -> Type["Action"]:
            """Register an Action class to the factory."""
            if self._type in ActionFactory._factory:
                raise ValueError(
                    f"Action type '{action_class.action_type}' has already been registered"
                )
            ActionFactory._factory[self._type] = action_class
            # add a static variable action_type to the registered class
            action_class.action_type = self._type
            return action_class

    @staticmethod
    def create(
        action_definition: ActionDefinition,
        om_loader: DescriptorLoader,
        loader: Optional[DescriptorLoader],
    ) -> "Action":
        """Create an action instance of given type. Factory for all action types.

        @param action_definition action definition from transformation file
        @param om_loader         the loader loads the application and component descriptors.
        @param loader            the loader loads the transformation files, this is needed
                                 for action need to load another transformation descriptor file.
        """
        if action_definition.type not in ActionFactory._factory:
            raise NotImplementedError(
                f"Action {action_definition.type} is not registered"
            )

        action_class = ActionFactory._factory[action_definition.type]
        # if user define create static method for the action, this class will hijack action
        # __init__, if create is not defined, action constructor is used directly
        if "create" in action_class.__dict__:
            return action_class.create(action_definition, om_loader, loader)
        else:
            return action_class(action_definition, om_loader)
