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
from collections import OrderedDict
from copy import deepcopy
from typing import cast
from typing import Dict
from typing import List

from dwcgf.action import Action
from dwcgf.action import ActionAttribute
from dwcgf.action.action_factory import ActionFactory
from dwcgf.descriptor import EpochDefinition
from dwcgf.descriptor import HyperepochDefinition
from dwcgf.descriptor import ResourceDefinition
from dwcgf.descriptor import ScheduleDefinition
from dwcgf.descriptor import StateDefinition
from dwcgf.descriptor import STMExternalRunnableDefinition
from dwcgf.object_model import Application
from dwcgf.object_model import PassDependencyDefinition
from dwcgf.object_model import ProcessDefinition
from dwcgf.object_model import ProcessServiceDefinition
from dwcgf.object_model import STMSchedule


####################################################################################################
# Epoch
####################################################################################################
@ActionFactory.register("insert-epoch")
class InsertEpoch(Action):
    """class for insert-epoch action."""

    extra_attributes = (
        ActionAttribute(
            name="scheduleNames",
            is_required=True,
            attr_type=list,
            default=[],
            description="Name of the target schedules.",
        ),
        ActionAttribute(
            name="hyperepochName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the target hyperepoch.",
        ),
        ActionAttribute(
            name="epochName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the epoch to be inserted.",
        ),
        ActionAttribute(
            name="period",
            is_required=True,
            attr_type=int,
            default=0,
            description="epoch period.",
        ),
        ActionAttribute(
            name="frames",
            is_required=False,
            attr_type=int,
            default=1,
            description="epoch frames.",
        ),
        ActionAttribute(
            name="passes",
            is_required=True,
            attr_type=list,
            default=[[]],
            description="epoch passes.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        for schedule_name in cast(list, self.get_attribute("scheduleNames")):
            target.insert_epoch(
                schedule=schedule_name,
                hyperepoch=self.get_attribute("hyperepochName"),
                epoch=EpochDefinition(
                    name=self.get_attribute("epochName"),
                    period=self.get_attribute("period"),
                    frames=self.get_attribute("frames"),
                    passes=self.get_attribute("passes"),
                ),
            )


@ActionFactory.register("remove-epoch")
class RemoveEpoch(Action):
    """class for remove-epoch action."""

    extra_attributes = (
        ActionAttribute(
            name="scheduleNames",
            is_required=False,
            attr_type=list,
            default=[],
            description="Name of the target schedules.",
        ),
        ActionAttribute(
            name="hyperepochName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the target hyperepoch.",
        ),
        ActionAttribute(
            name="epochName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the epoch to be removed.",
        ),
        ActionAttribute(
            name="allSchedules",
            is_required=False,
            attr_type=bool,
            default=False,
            description="The bool indicating whether apply to all schedules.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        if self.get_attribute("scheduleNames") and self.get_attribute("allSchedules"):
            raise ValueError(
                f"Please do not specify schedule names with 'allSchedules' on."
            )
        elif not self.get_attribute("scheduleNames") and not self.get_attribute(
            "allSchedules"
        ):
            raise ValueError(f"Please specify schedule names with 'allSchedules' off.")

        for schedule_name in (
            cast(list, self.get_attribute("scheduleNames"))
            if not self.get_attribute("allSchedules")
            else target.schedules.keys()
        ):
            target.remove_epoch(
                schedule=schedule_name,
                hyperepoch=self.get_attribute("hyperepochName"),
                epoch_name=self.get_attribute("epochName"),
            )


@ActionFactory.register("update-epoch-name")
class UpdateEpochName(Action):
    """class for update-epoch-name action."""

    extra_attributes = (
        ActionAttribute(
            name="scheduleNames",
            is_required=True,
            attr_type=list,
            default=[],
            description="Name of the target schedules.",
        ),
        ActionAttribute(
            name="hyperepochName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the target hyperepoch.",
        ),
        ActionAttribute(
            name="epochName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the target epoch.",
        ),
        ActionAttribute(
            name="newName",
            is_required=True,
            attr_type=str,
            default="",
            description="New name for the target epoch.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        for schedule_name in cast(list, self.get_attribute("scheduleNames")):
            target.update_epoch_name(
                schedule=schedule_name,
                hyperepoch=self.get_attribute("hyperepochName"),
                epoch_name=self.get_attribute("epochName"),
                new_name=self.get_attribute("newName"),
            )


@ActionFactory.register("update-epoch-period")
class UpdateEpochPeriod(Action):
    """class for update-epoch-period action."""

    extra_attributes = (
        ActionAttribute(
            name="scheduleNames",
            is_required=True,
            attr_type=list,
            default=[],
            description="Name of the target schedules.",
        ),
        ActionAttribute(
            name="hyperepochName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the target hyperepoch.",
        ),
        ActionAttribute(
            name="epochName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the target epoch.",
        ),
        ActionAttribute(
            name="newPeriod",
            is_required=True,
            attr_type=int,
            default=10_000_000,
            description="New period for the target epoch.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        for schedule_name in cast(list, self.get_attribute("scheduleNames")):
            target.update_epoch_period(
                schedule=schedule_name,
                hyperepoch=self.get_attribute("hyperepochName"),
                epoch_name=self.get_attribute("epochName"),
                new_period=self.get_attribute("newPeriod"),
            )


@ActionFactory.register("update-epoch-frame")
class UpdateEpochFrame(Action):
    """class for update-epoch-frame action."""

    extra_attributes = (
        ActionAttribute(
            name="scheduleNames",
            is_required=True,
            attr_type=list,
            default=[],
            description="Name of the target schedules.",
        ),
        ActionAttribute(
            name="hyperepochName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the target hyperepoch.",
        ),
        ActionAttribute(
            name="epochName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the target epoch.",
        ),
        ActionAttribute(
            name="newFrames",
            is_required=True,
            attr_type=int,
            default=1,
            description="New frames for the target epoch.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        for schedule_name in cast(list, self.get_attribute("scheduleNames")):
            target.update_epoch_frame(
                schedule=schedule_name,
                hyperepoch=self.get_attribute("hyperepochName"),
                epoch_name=self.get_attribute("epochName"),
                new_frames=self.get_attribute("newFrames"),
            )


@ActionFactory.register("insert-epoch-passes")
class InsertEpochPasses(Action):
    """class for insert-epoch-passes action."""

    extra_attributes = (
        ActionAttribute(
            name="scheduleNames",
            is_required=False,
            attr_type=list,
            default=[],
            description="Name of the target schedules.",
        ),
        ActionAttribute(
            name="hyperepochName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the target hyperepoch.",
        ),
        ActionAttribute(
            name="epochName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the target epoch.",
        ),
        ActionAttribute(
            name="passes",
            is_required=True,
            attr_type=list,
            default=None,
            description="Passes to be inserted",
        ),
        ActionAttribute(
            name="pipelinePhase",
            is_required=False,
            attr_type=int,
            default=0,
            description="Pipeline Phase in the passes list where the insertion takes place.",
        ),
        ActionAttribute(
            name="index",
            is_required=False,
            attr_type=int,
            default=None,
            description="Index in the passes list where the insertion takes place.",
        ),
        ActionAttribute(
            name="allSchedules",
            is_required=False,
            attr_type=bool,
            default=False,
            description="The bool indicating whether apply to all schedules.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        if self.get_attribute("scheduleNames") and self.get_attribute("allSchedules"):
            raise ValueError(
                f"Please do not specify schedule names with 'allSchedules' on."
            )
        elif not self.get_attribute("scheduleNames") and not self.get_attribute(
            "allSchedules"
        ):
            raise ValueError(f"Please specify schedule names with 'allSchedules' off.")

        for schedule_name in (
            cast(list, self.get_attribute("scheduleNames"))
            if not self.get_attribute("allSchedules")
            else target.schedules.keys()
        ):
            target.insert_epoch_passes(
                schedule=schedule_name,
                hyperepoch=self.get_attribute("hyperepochName"),
                epoch_name=self.get_attribute("epochName"),
                content=self.get_attribute("passes"),
                pipeline_phase=self.get_attribute("pipelinePhase"),
                insert_at=self.get_attribute("index"),
            )


@ActionFactory.register("remove-epoch-passes")
class RemoveEpochPasses(Action):
    """class for remove-epoch-passes action."""

    extra_attributes = (
        ActionAttribute(
            name="scheduleNames",
            is_required=False,
            attr_type=list,
            default=[],
            description="Name of the target schedules.",
        ),
        ActionAttribute(
            name="hyperepochName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the target hyperepoch.",
        ),
        ActionAttribute(
            name="epochName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the target epoch.",
        ),
        ActionAttribute(
            name="passes",
            is_required=True,
            attr_type=list,
            default=None,
            description="Passes to be removed",
        ),
        ActionAttribute(
            name="allSchedules",
            is_required=False,
            attr_type=bool,
            default=False,
            description="The bool indicating whether apply to all schedules.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        if self.get_attribute("scheduleNames") and self.get_attribute("allSchedules"):
            raise ValueError(
                f"Please do not specify schedule names with 'allSchedules' on."
            )
        elif not self.get_attribute("scheduleNames") and not self.get_attribute(
            "allSchedules"
        ):
            raise ValueError(f"Please specify schedule names with 'allSchedules' off.")

        for schedule_name in (
            cast(list, self.get_attribute("scheduleNames"))
            if not self.get_attribute("allSchedules")
            else target.schedules.keys()
        ):
            target.remove_epoch_passes(
                schedule=schedule_name,
                hyperepoch=self.get_attribute("hyperepochName"),
                epoch_name=self.get_attribute("epochName"),
                content=self.get_attribute("passes"),
            )


@ActionFactory.register("insert-epoch-alias-groups")
class InsertEpochAliasGroups(Action):
    """class for insert-epoch-alias-groups."""

    extra_attributes = (
        ActionAttribute(
            name="scheduleNames",
            is_required=True,
            attr_type=list,
            default=[],
            description="Schedule names of the target hyperepoch",
        ),
        ActionAttribute(
            name="hyperepochName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the target hyperepoch.",
        ),
        ActionAttribute(
            name="epochName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the epoch to be inserted.",
        ),
        ActionAttribute(
            name="aliasGroups",
            is_required=True,
            attr_type=dict,
            default={},
            description="Alias of the epoch to be inserted, \
                         the key is the group name and \
                         the value contains an array of \
                         fully qualified components \
                         which should have affinity to this aliasGroup",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        for schedule_name in cast(list, self.get_attribute("scheduleNames")):
            target.insert_epoch_alias_groups(
                schedule_name=schedule_name,
                hyperepoch_name=self.get_attribute("hyperepochName"),
                epoch_name=self.get_attribute("epochName"),
                alias_groups=self.get_attribute("aliasGroups"),
            )


@ActionFactory.register("remove-epoch-alias-groups")
class RemoveEpochAliasGroups(Action):
    """class for remove-epoch-alias-groups."""

    extra_attributes = (
        ActionAttribute(
            name="scheduleNames",
            is_required=True,
            attr_type=list,
            default=[],
            description="Schedule names of the target hyperepoch",
        ),
        ActionAttribute(
            name="hyperepochName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the target hyperepoch.",
        ),
        ActionAttribute(
            name="epochName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the epoch to be removed.",
        ),
        ActionAttribute(
            name="aliasGroups",
            is_required=True,
            attr_type=dict,
            default={},
            description="Alias of the epoch to be removed, \
                         the key is the group name and \
                         the value contains an array of \
                         fully qualified components \
                         which should have affinity to this aliasGroup",
        ),
    )

    def transform_impl(self, targets: Application) -> None:
        """action implementation."""
        for schedule_name in cast(list, self.get_attribute("scheduleNames")):
            targets.remove_epoch_alias_groups(
                schedule_name=schedule_name,
                hyperepoch_name=self.get_attribute("hyperepochName"),
                epoch_name=self.get_attribute("epochName"),
                alias_groups=self.get_attribute("aliasGroups"),
            )


@ActionFactory.register("update-epoch-alias-groups")
class UpdateEpochAliasGroups(Action):
    """class for update-epoch-alias-groups."""

    extra_attributes = (
        ActionAttribute(
            name="scheduleNames",
            is_required=True,
            attr_type=list,
            default=[],
            description="Schedule names of the target hyperepoch",
        ),
        ActionAttribute(
            name="hyperepochName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the target hyperepoch.",
        ),
        ActionAttribute(
            name="epochName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the epoch to be updated.",
        ),
        ActionAttribute(
            name="aliasGroups",
            is_required=True,
            attr_type=dict,
            default={},
            description="Alias of the epoch to be updated, \
                         the key is the group name and \
                         the value contains an array of \
                         fully qualified components \
                         which should have affinity to this aliasGroup",
        ),
    )

    def transform_impl(self, targets: Application) -> None:
        """action implementation."""
        for schedule_name in cast(list, self.get_attribute("scheduleNames")):
            targets.update_epoch_alias_groups(
                schedule_name=schedule_name,
                hyperepoch_name=self.get_attribute("hyperepochName"),
                epoch_name=self.get_attribute("epochName"),
                alias_groups=self.get_attribute("aliasGroups"),
            )


####################################################################################################
# Hyperepoch
####################################################################################################
@ActionFactory.register("insert-hyperepoch")
class InsertHyperepoch(Action):
    """class for insert-hyperepoch action."""

    extra_attributes = (
        ActionAttribute(
            name="scheduleNames",
            is_required=True,
            attr_type=list,
            default=[],
            description="Name of the target schedules.",
        ),
        ActionAttribute(
            name="hyperepochName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the hyperepoch to be inserted.",
        ),
        ActionAttribute(
            name="period",
            is_required=True,
            attr_type=int,
            default=0,
            description="Hyperepoch period.",
        ),
        ActionAttribute(
            name="epochs",
            is_required=True,
            attr_type=dict,
            default=None,
            description="epochs of the hyperepoch to be added.",
        ),
        ActionAttribute(
            name="resources",
            is_required=True,
            attr_type=dict,
            default=None,
            description="epochs of the hyperepoch to be added.",
        ),
        ActionAttribute(
            name="monitoringPeriod",
            is_required=False,
            attr_type=int,
            default=None,
            description="Hyperepoch monitoring threshold.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        epoch_json = self.get_attribute("epochs")
        if isinstance(epoch_json, dict):
            epochs = {
                epoch_name: EpochDefinition(
                    name=epoch_name,
                    period=ScheduleDefinition.convert_period(epoch_raw.get("period")),
                    frames=epoch_raw.get("frames", 1),
                    passes=epoch_raw.get("passes", [[]]),
                )
                for epoch_name, epoch_raw in epoch_json.items()
            }

        resource_json = self.get_attribute("resources")
        if isinstance(resource_json, dict):
            resources = {
                specifier: ResourceDefinition(specifier=specifier, passes=passes)
                for specifier, passes in resource_json.items()
            }

        for schedule_name in cast(list, self.get_attribute("scheduleNames")):
            target.insert_hyperepoch(
                schedule=schedule_name,
                hyperepoch=HyperepochDefinition(
                    name=self.get_attribute("hyperepochName"),
                    period=self.get_attribute("period"),
                    monitoringPeriod=self.get_attribute("monitoringPeriod"),
                    epochs=epochs,
                    resources=resources,
                ),
            )


@ActionFactory.register("remove-hyperepoch")
class RemoveHyperepoch(Action):
    """class for insert-hyperepoch action."""

    extra_attributes = (
        ActionAttribute(
            name="scheduleNames",
            is_required=True,
            attr_type=list,
            default=[],
            description="Name of the target schedules.",
        ),
        ActionAttribute(
            name="hyperepochName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the hyperepoch to be removed.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        for schedule_name in cast(list, self.get_attribute("scheduleNames")):
            target.remove_hyperepoch(
                schedule=schedule_name,
                hyperepoch_name=self.get_attribute("hyperepochName"),
            )


@ActionFactory.register("insert-hyperepoch-resource-assignment")
class InsertHyperepochResourceAssignment(Action):
    """class for insert-hyperepoch-resource-assignment action."""

    extra_attributes = (
        ActionAttribute(
            name="scheduleNames",  # for example: ["standardSchedule"]
            is_required=True,
            attr_type=list,
            default=[],
            description="Schedule names of the target hyperepoch",
        ),
        ActionAttribute(
            name="hyperepochName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the target hyperepoch.",
        ),
        ActionAttribute(
            name="resourceAssignment",
            is_required=True,
            attr_type=dict,
            default={},
            description="Resource assignment of the target hyperepoch.",
        ),
    )

    def transform_impl(self, targets: Application) -> None:
        """action implementation."""
        for schedule_name in cast(list, self.get_attribute("scheduleNames")):
            targets.insert_hyperepoch_resource_assignment(
                schedule_name=schedule_name,
                hyperepoch_name=self.get_attribute("hyperepochName"),
                resource_assignment=self.get_attribute("resourceAssignment"),
            )


@ActionFactory.register("remove-hyperepoch-resource-assignment")
class RemoveHyperepochResourceAssignment(Action):
    """class for remove-hyperepoch-resource-assignment action."""

    extra_attributes = (
        ActionAttribute(
            name="scheduleNames",  # for example: ["standardSchedule"]
            is_required=True,
            attr_type=list,
            default=[],
            description="Schedule names of the target hyperepoch",
        ),
        ActionAttribute(
            name="hyperepochName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the target hyperepoch.",
        ),
        ActionAttribute(
            name="resourceAssignment",
            is_required=True,
            attr_type=dict,
            default={},
            description="Resource assignment of the target hyperepoch.",
        ),
    )

    def transform_impl(self, targets: Application) -> None:
        """action implementation."""
        for schedule_name in cast(list, self.get_attribute("scheduleNames")):
            targets.remove_hyperepoch_resource_assignment(
                schedule_name=schedule_name,
                hyperepoch_name=self.get_attribute("hyperepochName"),
                resource_assignment=self.get_attribute("resourceAssignment"),
            )


@ActionFactory.register("remove-hyperepoch-resource")
class RemoveHyperepochResource(Action):
    """class for remove-hyperepoch-resource action."""

    extra_attributes = (
        ActionAttribute(
            name="scheduleNames",  # for example: ["standardSchedule"]
            is_required=True,
            attr_type=list,
            default=[],
            description="Schedule names of the target hyperepoch",
        ),
        ActionAttribute(
            name="hyperepochName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the target hyperepoch.",
        ),
        ActionAttribute(
            name="resources",
            is_required=True,
            attr_type=list,
            default=[],
            description="Resources of the target hyperepoch.",
        ),
    )

    def transform_impl(self, targets: Application) -> None:
        """action implementation."""
        for schedule_name in cast(list, self.get_attribute("scheduleNames")):
            targets.remove_hyperepoch_resource(
                schedule_name=schedule_name,
                hyperepoch_name=self.get_attribute("hyperepochName"),
                resources=self.get_attribute("resources"),
            )


@ActionFactory.register("insert-hyperepoch-resource")
class InsertHyperepochResource(Action):
    """class for insert-hyperepoch-resource action."""

    extra_attributes = (
        ActionAttribute(
            name="scheduleNames",  # for example: ["standardSchedule"]
            is_required=True,
            attr_type=list,
            default=[],
            description="Schedule names of the target hyperepoch",
        ),
        ActionAttribute(
            name="hyperepochName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the target hyperepoch.",
        ),
        ActionAttribute(
            name="newResources",
            is_required=True,
            attr_type=dict,
            default={},
            description="New resources of the target hyperepoch.",
        ),
    )

    def transform_impl(self, targets: Application) -> None:
        """action implementation."""
        for schedule_name in cast(list, self.get_attribute("scheduleNames")):
            targets.insert_hyperepoch_resource(
                schedule_name=schedule_name,
                hyperepoch_name=self.get_attribute("hyperepochName"),
                new_resources=self.get_attribute("newResources"),
            )


@ActionFactory.register("remove-hyperepoch-resource-assignment-by-pass-name")
class RemoveHyperepochResourceAssignmentByPassName(Action):
    """class for remove-hyperepoch-resource-assignment-by-pass-name action."""

    extra_attributes = (
        ActionAttribute(
            name="scheduleNames",  # for example: ["standardSchedule"]
            is_required=False,
            attr_type=list,
            default=[],
            description="Schedule names of the target hyperepoch",
        ),
        ActionAttribute(
            name="hyperepochName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the target hyperepoch.",
        ),
        ActionAttribute(
            name="passes",
            is_required=True,
            attr_type=list,
            default=None,
            description="Passes to be removed",
        ),
        ActionAttribute(
            name="allSchedules",
            is_required=False,
            attr_type=bool,
            default=False,
            description="The bool indicating whether apply to all schedules.",
        ),
        ActionAttribute(
            name="allowPassNotFound",
            is_required=False,
            attr_type=bool,
            default=False,
            description="The bool indicating whether allow pass not found.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        if self.get_attribute("scheduleNames") and self.get_attribute("allSchedules"):
            raise ValueError(
                f"Please do not specify schedule names with 'allSchedules' on."
            )
        elif not self.get_attribute("scheduleNames") and not self.get_attribute(
            "allSchedules"
        ):
            raise ValueError(f"Please specify schedule names with 'allSchedules' off.")

        hyperepoch_name = self.get_attribute("hyperepochName")
        passes = self.get_attribute("passes")
        allow_pass_not_found = self.get_attribute("allowPassNotFound")
        scheduel_resourceRemove_dict: Dict[str, Dict[str, List[str]]] = {}

        for schedule_name in (
            cast(list, self.get_attribute("scheduleNames"))
            if not self.get_attribute("allSchedules")
            else target.schedules.keys()
        ):
            if schedule_name not in target._stm_schedules:
                raise ValueError(f"Cannot find {schedule_name}")

            schedule_definition: STMSchedule = target._stm_schedules[schedule_name]
            if hyperepoch_name not in schedule_definition.schedule.hyperepochs:
                raise ValueError(
                    f"Cannot find hyperepoch name '{hyperepoch_name}' in '{schedule_name}'"
                )

            target_hyperepoch: HyperepochDefinition = schedule_definition.schedule.hyperepochs[
                hyperepoch_name
            ]
            existing_resources: Dict[
                str, ResourceDefinition
            ] = target_hyperepoch.resources
            resource_remove_dict: Dict[str, List[str]] = {}
            for pass_name in passes:
                pass_found = False
                for resource_name, resource in existing_resources.items():
                    pass_list = resource.passes
                    if pass_name in pass_list:
                        pass_found = True
                        if resource_name in resource_remove_dict:
                            resource_remove_dict[resource_name].extend([pass_name])
                        else:
                            resource_remove_dict[resource_name] = [pass_name]
                        continue
                if not pass_found and not allow_pass_not_found:
                    raise ValueError(
                        f"Cannot find pass name '{pass_name}' in all resources\
in schedule '{schedule_name}'"
                    )
            scheduel_resourceRemove_dict[schedule_name] = resource_remove_dict

        for schedule_name in (
            cast(list, self.get_attribute("scheduleNames"))
            if not self.get_attribute("allSchedules")
            else target.schedules.keys()
        ):
            target.remove_hyperepoch_resource_assignment(
                schedule_name=schedule_name,
                hyperepoch_name=hyperepoch_name,
                resource_assignment=scheduel_resourceRemove_dict[schedule_name],
            )


@ActionFactory.register("update-hyperepoch-period")
class UpdateHyperepochPeriod(Action):
    """class for update-hyperepoch-period action."""

    extra_attributes = (
        ActionAttribute(
            name="scheduleNames",  # for example: ["standardSchedule"]
            is_required=True,
            attr_type=list,
            default=[],
            description="Schedule names of the target hyperepoch",
        ),
        ActionAttribute(
            name="hyperepochName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the target hyperepoch.",
        ),
        ActionAttribute(
            name="newPeriod",
            is_required=True,
            attr_type=int,
            default=None,
            description="New period of the target hyperepoch.",
        ),
    )

    def transform_impl(self, targets: Application) -> None:
        """action implementation."""
        for schedule_name in cast(list, self.get_attribute("scheduleNames")):
            targets.update_hyperepoch_period(
                schedule_name=schedule_name,
                hyperepoch_name=self.get_attribute("hyperepochName"),
                new_period=self.get_attribute("newPeriod"),
            )


@ActionFactory.register("update-hyperepoch-monitoringPeriod")
class UpdateHyperepochMonitoringPeriod(Action):
    """class for update-hyperepoch-monitoringPeriod action."""

    extra_attributes = (
        ActionAttribute(
            name="scheduleNames",  # for example: ["standardSchedule"]
            is_required=True,
            attr_type=list,
            default=[],
            description="Schedule names of the target hyperepoch",
        ),
        ActionAttribute(
            name="hyperepochName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the target hyperepoch.",
        ),
        ActionAttribute(
            name="newMonitoringPeriod",
            is_required=True,
            attr_type=int,
            default=None,
            description="New monitoring threshold of the target hyperepoch.",
        ),
    )

    def transform_impl(self, targets: Application) -> None:
        """action implementation."""
        for schedule_name in cast(list, self.get_attribute("scheduleNames")):
            targets.update_hyperepoch_monitoringPeriod(
                schedule_name=schedule_name,
                hyperepoch_name=self.get_attribute("hyperepochName"),
                new_monitoringPeriod=self.get_attribute("newMonitoringPeriod"),
            )


@ActionFactory.register("update-hyperepoch-name")
class UpdateHyperepochName(Action):
    """class for update-hyperepoch-name action."""

    extra_attributes = (
        ActionAttribute(
            name="scheduleNames",  # for example: ["standardSchedule"]
            is_required=True,
            attr_type=list,
            default=[],
            description="Schedule names of the target hyperepoch",
        ),
        ActionAttribute(
            name="hyperepochName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the target hyperepoch.",
        ),
        ActionAttribute(
            name="newHyperepochName",
            is_required=True,
            attr_type=str,
            default="",
            description="New name of the target hyperepcoh.",
        ),
    )

    def transform_impl(self, targets: Application) -> None:
        """action implementation."""
        for schedule_name in cast(list, self.get_attribute("scheduleNames")):
            targets.update_hyperepoch_name(
                schedule_name=schedule_name,
                hyperepoch_name=self.get_attribute("hyperepochName"),
                new_hyperepoch_name=self.get_attribute("newHyperepochName"),
            )


####################################################################################################
# Process
####################################################################################################
@ActionFactory.register("update-process-name")
class UpdateProcessName(Action):
    """class for update-process-name action."""

    extra_attributes = (
        ActionAttribute(
            name="processName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the target process.",
        ),
        ActionAttribute(
            name="newName",
            is_required=True,
            attr_type=str,
            default="",
            description="New name of the target process.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        target.update_process_name(
            name=self.get_attribute("processName"),
            new_name=self.get_attribute("newName"),
        )


@ActionFactory.register("insert-process")
class InsertProcess(Action):
    """class for insert-process action."""

    extra_attributes = (
        ActionAttribute(
            name="processName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the process.",
        ),
        ActionAttribute(
            name="runOn",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of target runtime hardware.",
        ),
        ActionAttribute(
            name="executable",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of process binary.",
        ),
        ActionAttribute(
            name="logSpec",
            is_required=False,
            attr_type=str,
            default=None,
            description="Name of process binary.",
        ),
        ActionAttribute(
            name="argv",
            is_required=False,
            attr_type=dict,
            default=None,
            description="Argument dict of the process to be added.",
        ),
        ActionAttribute(
            name="services",
            is_required=False,
            attr_type=dict,
            default=None,
            description="services dict of the process to be enabled.",
        ),
        ActionAttribute(
            name="subcomponents",
            is_required=False,
            attr_type=list,
            default=None,
            description="Subcomponents of the process to be added.",
        ),
        ActionAttribute(
            name="extraInfo",
            is_required=False,
            attr_type=dict,
            default=None,
            description="Extra info of the process to be added.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        name = self.get_attribute("processName")
        # Construct the ProcessServiceDefinition
        services: OrderedDict = OrderedDict()
        if self.get_attribute("services") is not None:
            for service_name, service_raw in self.get_attribute("services").items():
                services[service_name] = ProcessServiceDefinition(
                    name=service_name,
                    parameters=service_raw["parameters"],
                    passes=service_raw["passes"],
                )
        process = ProcessDefinition(
            dirname=target.descriptor_path.parent,
            name=name,
            executable=self.get_attribute("executable"),
            run_on=self.get_attribute("runOn"),
            log_spec=self.get_attribute("logSpec"),
            argv=self.get_attribute("argv"),
            services=services if services and len(services) else None,
            subcomponents=self.get_attribute("subcomponents"),
            extra_info=self.get_attribute("extraInfo"),
        )
        target.insert_process(process, self._loader)


@ActionFactory.register("remove-process")
class RemoveProcess(Action):
    """class for remove-process action."""

    extra_attributes = (
        ActionAttribute(
            name="processName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the process.",
        ),
        ActionAttribute(
            name="expectExists",
            is_required=False,
            attr_type=bool,
            default=True,
            description="Action succeeds only if process exists",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        if self.get_attribute(
            "processName"
        ) not in target.processes and not self.get_attribute("expectExists"):
            return
        target.remove_process(name=self.get_attribute("processName"))


@ActionFactory.register("insert-process-argument")
class InsertProcessArg(Action):
    """class for insert-process-argument action."""

    extra_attributes = (
        ActionAttribute(
            name="processName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the target process.",
        ),
        ActionAttribute(
            name="argument",
            is_required=True,
            attr_type=str,
            default=None,
            description="Key of the argument to be inserted.",
        ),
        ActionAttribute(
            name="argumentValue",
            is_required=True,
            attr_type=(bool, str, list),
            default=None,
            description="Value of the argument to be inserted.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        target.insert_process_arg(
            process_name=self.get_attribute("processName"),
            arg=self.get_attribute("argument"),
            arg_value=self.get_attribute("argumentValue"),
        )


@ActionFactory.register("remove-process-argument")
class RemoveProcessArg(Action):
    """class for remove-process-args action."""

    extra_attributes = (
        ActionAttribute(
            name="processName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the target process.",
        ),
        ActionAttribute(
            name="argument",
            is_required=True,
            attr_type=str,
            default=None,
            description="key of the arguments to be removed",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        target.remove_process_arg(
            process_name=self.get_attribute("processName"),
            arg=self.get_attribute("argument"),
        )


@ActionFactory.register("insert-process-subcomponent")
class InsertProcessSubcomponent(Action):
    """class for insert-process-subcomponent action."""

    extra_attributes = (
        ActionAttribute(
            name="processName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the target process.",
        ),
        ActionAttribute(
            name="subcomponentID",
            is_required=True,
            attr_type=list,
            default=None,
            description="Content of the arguments to be inserted",
        ),
        ActionAttribute(
            name="index",
            is_required=False,
            attr_type=int,
            default=None,
            description="Index in the argument list where the insertion takes place.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        target.insert_process_subcomponent(
            process_name=self.get_attribute("processName"),
            content=self.get_attribute("subcomponentID"),
            insert_at=self.get_attribute("index"),
        )


@ActionFactory.register("remove-process-subcomponent")
class RemoveProcessSubcomponents(Action):
    """class for remove-process-subcomponent action."""

    extra_attributes = (
        ActionAttribute(
            name="processName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the target process.",
        ),
        ActionAttribute(
            name="subcomponentID",
            is_required=True,
            attr_type=list,
            default=None,
            description="Content of the arguments to be removed.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        target.remove_process_subcomponent(
            process_name=self.get_attribute("processName"),
            content=self.get_attribute("subcomponentID"),
        )


####################################################################################################
# Pass Dependencies
####################################################################################################
@ActionFactory.register("insert-pass-dependencies")
class InsertPassDependencies(Action):
    """class for insert-pass-dependencies action."""

    extra_attributes = (
        ActionAttribute(
            name="scheduleNames",
            is_required=True,
            attr_type=list,
            default=[],
            description="Name of the target schedules.",
        ),
        ActionAttribute(
            name="pass",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the target pass.",
        ),
        ActionAttribute(
            name="dependencies",
            is_required=True,
            attr_type=list,
            default=[],
            description="pass dependencies.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        for schedule_name in cast(list, self.get_attribute("scheduleNames")):
            target.insert_pass_dependencies(
                schedule=schedule_name,
                pass_def=PassDependencyDefinition(
                    pass_name=self.get_attribute("pass"),
                    dependencies=self.get_attribute("dependencies"),
                ),
            )


@ActionFactory.register("remove-pass-dependencies")
class RemovePassdependencies(Action):
    """class for remove-pass-dependencies action."""

    extra_attributes = (
        ActionAttribute(
            name="scheduleNames",
            is_required=False,
            attr_type=list,
            default=[],
            description="Name of the target schedules.",
        ),
        ActionAttribute(
            name="pass",
            is_required=True,
            attr_type=list,
            default=[],
            description="Name of the target passes.",
        ),
        ActionAttribute(
            name="allSchedules",
            is_required=False,
            attr_type=bool,
            default=False,
            description="The bool indicating whether apply to all schedules.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        if self.get_attribute("scheduleNames") and self.get_attribute("allSchedules"):
            raise ValueError(
                f"Please do not specify schedule names with 'allSchedules' on."
            )
        elif not self.get_attribute("scheduleNames") and not self.get_attribute(
            "allSchedules"
        ):
            raise ValueError(f"Please specify schedule names with 'allSchedules' off.")

        pass_names = self.get_attribute("pass")
        if isinstance(pass_names, list):
            for pass_name in pass_names:
                for schedule_name in (
                    cast(list, self.get_attribute("scheduleNames"))
                    if not self.get_attribute("allSchedules")
                    else target.schedules.keys()
                ):
                    target.remove_pass_dependencies(
                        schedule=schedule_name, pass_name=pass_name
                    )
        else:
            raise TypeError("pass should be a list of str.")


@ActionFactory.register("update-pass-dependencies")
class UpdatePassDependencies(Action):
    """class for update-pass-dependencies action.

    Call insert_pass_dependencies() for new passes.
    Call update_pass_dependencies() for existing passes.
    """

    extra_attributes = (
        ActionAttribute(
            name="scheduleNames",
            is_required=False,
            attr_type=list,
            default=[],
            description="Name of the target schedules.",
        ),
        ActionAttribute(
            name="pass",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the target pass.",
        ),
        ActionAttribute(
            name="dependencies",
            is_required=True,
            attr_type=list,
            default=[],
            description="pass dependencies.",
        ),
        ActionAttribute(
            name="allSchedules",
            is_required=False,
            attr_type=bool,
            default=False,
            description="The bool indicating whether apply to all schedules.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        if self.get_attribute("scheduleNames") and self.get_attribute("allSchedules"):
            raise ValueError(
                f"Please do not specify schedule names with 'allSchedules' on."
            )
        elif not self.get_attribute("scheduleNames") and not self.get_attribute(
            "allSchedules"
        ):
            raise ValueError(f"Please specify schedule names with 'allSchedules' off.")

        for schedule in (
            cast(list, self.get_attribute("scheduleNames"))
            if not self.get_attribute("allSchedules")
            else target.schedules.keys()
        ):
            pass_def = PassDependencyDefinition(
                pass_name=self.get_attribute("pass"),
                dependencies=self.get_attribute("dependencies"),
            )

            if schedule not in target.schedules:
                raise ValueError(f"Schedule name '{schedule}' does not exist.")

            pass_dependencies = deepcopy(
                target.schedules[schedule].schedule.passDependencies
            )

            if pass_def.pass_name not in pass_dependencies:
                target.insert_pass_dependencies(schedule=schedule, pass_def=pass_def)
            else:
                for pass_dep in pass_def.dependencies:
                    pass_dependencies[pass_def.pass_name]._dependencies.append(pass_dep)
                target.update_pass_dependencies(schedule, pass_dependencies)


####################################################################################################
# STM Externel Runnable
####################################################################################################
@ActionFactory.register("insert-stm-external-runnable")
class InsertSTMExternalRunnable(Action):
    """class for insert-stm-external-runnable action."""

    extra_attributes = (
        ActionAttribute(
            name="runnableName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the runnable.",
        ),
        ActionAttribute(
            name="wcet",
            is_required=True,
            attr_type=int,
            default=0,
            description="Wcet of the runnable.",
        ),
        ActionAttribute(
            name="processorTypes",
            is_required=True,
            attr_type=list,
            default=[],
            description="Processor types of the runnable.",
        ),
        ActionAttribute(
            name="passDependencies",
            is_required=False,
            attr_type=list,
            default=[],
            description="Pass dependencies of the runnable.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        target.insert_stm_external_runnable(
            runnable_def=STMExternalRunnableDefinition(
                name=self.get_attribute("runnableName"),
                wcet=self.get_attribute("wcet"),
                processorTypes=self.get_attribute("processorTypes"),
                passDependencies=self.get_attribute("passDependencies"),
            )
        )


@ActionFactory.register("remove-stm-external-runnable")
class RemoveSTMExternalRunnable(Action):
    """class for remove-stm-external-runnable action."""

    extra_attributes = (
        ActionAttribute(
            name="runnableName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the runnable.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        target.remove_stm_external_runnable(
            runnable_name=self.get_attribute("runnableName")
        )


@ActionFactory.register("update-stm-external-runnable-name")
class UpdateSTMExternalRunnableName(Action):
    """class for update-stm-external-runnable-name action."""

    extra_attributes = (
        ActionAttribute(
            name="runnableName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the runnable.",
        ),
        ActionAttribute(
            name="newName",
            is_required=True,
            attr_type=str,
            default="",
            description="New name for the target runnable.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        target.update_stm_external_runnable_name(
            runnable_name=self.get_attribute("runnableName"),
            new_name=self.get_attribute("newName"),
        )


@ActionFactory.register("update-stm-external-runnable-wcet")
class UpdateSTMExternalRunnableWcet(Action):
    """class for update-stm-external-runnable-wcet action."""

    extra_attributes = (
        ActionAttribute(
            name="runnableName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the runnable.",
        ),
        ActionAttribute(
            name="newWcet",
            is_required=True,
            attr_type=int,
            default=0,
            description="New wcet for the target runnable.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        target.update_stm_external_runnable_wcet(
            runnable_name=self.get_attribute("runnableName"),
            new_wcet=self.get_attribute("newWcet"),
        )


@ActionFactory.register("insert-stm-external-runnable-processor-types")
class InsertSTMExternalRunnableProcessorType(Action):
    """class for insert-stm-external-runnable-process-type action."""

    extra_attributes = (
        ActionAttribute(
            name="runnableName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the runnable.",
        ),
        ActionAttribute(
            name="processorTypes",
            is_required=True,
            attr_type=list,
            default=[],
            description="Process types of the runnable to be inserted.",
        ),
        ActionAttribute(
            name="index",
            is_required=False,
            attr_type=int,
            default=None,
            description="Index in the process type list where the insertion takes place.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        target.insert_stm_ext_runnable_proc_types(
            runnable_name=self.get_attribute("runnableName"),
            content=self.get_attribute("processorTypes"),
            insert_at=self.get_attribute("index"),
        )


@ActionFactory.register("remove-stm-external-runnable-processor-types")
class RemoveSTMExternalRunnableProcessorType(Action):
    """class for remove-stm-external-runnable-process-type action."""

    extra_attributes = (
        ActionAttribute(
            name="runnableName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the runnable.",
        ),
        ActionAttribute(
            name="processorTypes",
            is_required=True,
            attr_type=list,
            default=[],
            description="Process types of the runnable to be removed.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        target.remove_stm_ext_runnable_proc_types(
            runnable_name=self.get_attribute("runnableName"),
            content=self.get_attribute("processorTypes"),
        )


@ActionFactory.register("insert-stm-external-runnable-pass-dependencies")
class InsertSTMExternalRunnablePassDependencies(Action):
    """class for insert-stm-external-runnable-pass-dependencies action."""

    extra_attributes = (
        ActionAttribute(
            name="runnableName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the runnable.",
        ),
        ActionAttribute(
            name="passDependencies",
            is_required=True,
            attr_type=list,
            default=[],
            description="Pass dependencies of the runnable to be inserted.",
        ),
        ActionAttribute(
            name="index",
            is_required=False,
            attr_type=int,
            default=None,
            description="Index in the pass dependencies list where the insertion takes place.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        target.insert_stm_ext_runnable_pass_dependencies(
            runnable_name=self.get_attribute("runnableName"),
            content=self.get_attribute("passDependencies"),
            insert_at=self.get_attribute("index"),
        )


@ActionFactory.register("remove-stm-external-runnable-pass-dependencies")
class RemoveSTMExternalRunnablePassDependencies(Action):
    """class for remove-stm-external-runnable-pass-dependencies action."""

    extra_attributes = (
        ActionAttribute(
            name="runnableName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the runnable.",
        ),
        ActionAttribute(
            name="passDependencies",
            is_required=True,
            attr_type=list,
            default=[],
            description="Pass dependencies of the runnable to be removed.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        target.remove_stm_ext_runnable_pass_dependencies(
            runnable_name=self.get_attribute("runnableName"),
            content=self.get_attribute("passDependencies"),
        )


####################################################################################################
# Schedule State
####################################################################################################
@ActionFactory.register("duplicate-schedule")
class DuplicateSchedule(Action):
    """class for duplicate-schedule action."""

    extra_attributes = (
        ActionAttribute(
            name="scheduleName",
            is_required=True,
            attr_type=str,
            default="",
            description="The existing schedule to be duplicated",
        ),
        ActionAttribute(
            name="newScheduleName",
            is_required=True,
            attr_type=str,
            default="",
            description="The name of duplicated schedule",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        schedule_name = self.get_attribute("scheduleName")
        if schedule_name not in target.schedules:
            raise ValueError(f"schedule name '{schedule_name}' cannot be found.")
        original_schedule = target.schedules[schedule_name].schedule
        duplicate = deepcopy(original_schedule)
        duplicate.name = self.get_attribute("newScheduleName")
        target.insert_schedule(duplicate)


@ActionFactory.register("insert-schedule")
class InsertSchedule(Action):
    """class for insert-schedule action."""

    extra_attributes = (
        ActionAttribute(
            name="scheduleName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the schedule to be inserted.",
        ),
        ActionAttribute(
            name="wcet",
            is_required=True,
            attr_type=str,
            default="",
            description="Wcet of the schedule to be inserted.",
        ),
        ActionAttribute(
            name="hyperepochs",
            is_required=True,
            attr_type=dict,
            default=None,
            description="Hyperepochs of the schedule to be inserted.",
        ),
        ActionAttribute(
            name="passDependencies",
            is_required=False,
            attr_type=dict,
            default=None,
            description="Pass dependencies of the schedule to be inserted.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        hyperepoch_json = self.get_attribute("hyperepochs")
        if isinstance(hyperepoch_json, dict):
            hyperepochs: dict = {}
            for specifier, value in hyperepoch_json.items():
                epoch_json = value.get("epochs")
                epochs = {
                    epoch_name: EpochDefinition(
                        name=epoch_name,
                        period=ScheduleDefinition.convert_period(
                            epoch_raw.get("period")
                        ),
                        frames=epoch_raw.get("frames", 1),
                        passes=epoch_raw.get("passes", [[]]),
                    )
                    for epoch_name, epoch_raw in epoch_json.items()
                }

                resource_json = value.get("resources")
                resources = {
                    specifier: ResourceDefinition(specifier=specifier, passes=passes)
                    for specifier, passes in resource_json.items()
                }

                hyperepochs[specifier] = HyperepochDefinition(
                    name=specifier,
                    period=value.get("period"),
                    epochs=epochs,
                    resources=resources,
                    monitoringPeriod=value.get("monitoringPeriod"),
                )

        pass_dependencies_json = self.get_attribute("passDependencies")
        if isinstance(pass_dependencies_json, dict):
            passDependencies: dict = {}
            for pass_name, dependencies in pass_dependencies_json.items():
                passDependencies[pass_name] = PassDependencyDefinition(
                    pass_name=pass_name, dependencies=dependencies
                )

        target.insert_schedule(
            schedule=ScheduleDefinition(
                name=self.get_attribute("scheduleName"),
                wcet=self.get_attribute("wcet"),
                hyperepochs=hyperepochs,
                passDependencies=passDependencies,
            )
        )


@ActionFactory.register("remove-schedule")
class RemoveSchedule(Action):
    """class for remove-schedule action."""

    extra_attributes = (
        ActionAttribute(
            name="scheduleName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the schedule to be removed.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        target.remove_schedule(schedule_name=self.get_attribute("scheduleName"))


@ActionFactory.register("update-schedule-name")
class UpdateScheduleName(Action):
    """class for update-schedule-name action."""

    extra_attributes = (
        ActionAttribute(
            name="scheduleName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the target schedule.",
        ),
        ActionAttribute(
            name="newName",
            is_required=True,
            attr_type=str,
            default="",
            description="New name of the target schedule.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        target.update_schedule_name(
            schedule_name=self.get_attribute("scheduleName"),
            new_name=self.get_attribute("newName"),
        )


@ActionFactory.register("update-schedule-wcet")
class UpdateScheduleWcet(Action):
    """class for update-schedule-wcet action."""

    extra_attributes = (
        ActionAttribute(
            name="scheduleNames",
            is_required=True,
            attr_type=list,
            default=[],
            description="Name of the target schedules.",
        ),
        ActionAttribute(
            name="newWcet",
            is_required=True,
            attr_type=str,
            default="",
            description="New wcet of the target schedule.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        for schedule_name in cast(list, self.get_attribute("scheduleNames")):
            target.update_schedule_wcet(
                schedule_name=schedule_name, new_wcet=self.get_attribute("newWcet")
            )


@ActionFactory.register("insert-state")
class InsertState(Action):
    """class for insert-state action."""

    extra_attributes = (
        ActionAttribute(
            name="stateName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the state to be inserted.",
        ),
        ActionAttribute(
            name="stmScheduleKey",
            is_required=True,
            attr_type=str,
            default="",
            description="stm schedule key of the state to be inserted.",
        ),
        ActionAttribute(
            name="default",
            is_required=False,
            attr_type=bool,
            default=False,
            description="Default state of the state to be inserted.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        target.insert_state(
            state=StateDefinition(
                name=self.get_attribute("stateName"),
                stm_scheudle_key=self.get_attribute("stmScheduleKey"),
                default=self.get_attribute("default"),
            )
        )


@ActionFactory.register("remove-state")
class RemoveState(Action):
    """class for remove-state action."""

    extra_attributes = (
        ActionAttribute(
            name="stateName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the state to be removed.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        target.remove_state(state_name=self.get_attribute("stateName"))


@ActionFactory.register("update-state-name")
class UpdateStateName(Action):
    """class for update-state-name action."""

    extra_attributes = (
        ActionAttribute(
            name="stateName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the target state.",
        ),
        ActionAttribute(
            name="newName",
            is_required=True,
            attr_type=str,
            default="",
            description="New name for the target state.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        target.update_state_name(
            state_name=self.get_attribute("stateName"),
            new_name=self.get_attribute("newName"),
        )


@ActionFactory.register("update-state-schedule")
class UpdateStateSchedule(Action):
    """class for update-state-schedule action."""

    extra_attributes = (
        ActionAttribute(
            name="stateName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the target state.",
        ),
        ActionAttribute(
            name="newSchedule",
            is_required=True,
            attr_type=str,
            default="",
            description="New schedule for the target state.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        target.update_state_schedule(
            state_name=self.get_attribute("stateName"),
            new_schedule=self.get_attribute("newSchedule"),
        )


@ActionFactory.register("select-default-state")
class SelectDefaultState(Action):
    """class for select-default-state action."""

    extra_attributes = (
        ActionAttribute(
            name="stateName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the target state.",
        ),
        ActionAttribute(
            name="newDefault",
            is_required=True,
            attr_type=bool,
            default=False,
            description="New default state for the target state.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        target.select_default_state(
            state_name=self.get_attribute("stateName"),
            new_default_state=self.get_attribute("newDefault"),
        )


####################################################################################################
# Wcet
####################################################################################################
@ActionFactory.register("update-wcet-file")
class UpdateWcetFile(Action):
    """class for update-wcet-file action."""

    extra_attributes = (
        ActionAttribute(
            name="scheduleNames",
            is_required=False,
            attr_type=list,
            default=[],
            description="The schedules needed to update wcet file",
        ),
        ActionAttribute(
            name="newWcetFileName",
            is_required=True,
            attr_type=str,
            default="",
            description="The name of new wcet file",
        ),
        ActionAttribute(
            name="allSchedules",
            is_required=False,
            attr_type=bool,
            default=False,
            description="The bool indicating whether apply to all schedules.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        if self.get_attribute("scheduleNames") and self.get_attribute("allSchedules"):
            raise ValueError(
                f"Please do not specify schedule names with 'allSchedules' on."
            )
        elif not self.get_attribute("scheduleNames") and not self.get_attribute(
            "allSchedules"
        ):
            raise ValueError(f"Please specify schedule names with 'allSchedules' off.")

        for schedule_name in (
            cast(list, self.get_attribute("scheduleNames"))
            if not self.get_attribute("allSchedules")
            else target.schedules.keys()
        ):
            target.update_wcet_file(
                schedule_name=schedule_name,
                new_wcet_file_name=self.get_attribute("newWcetFileName"),
            )


@ActionFactory.register("separate-epochs")
class SeparateEpochs(Action):
    """Create a 1-1 mapping between epochs and hyperepochs."""

    extra_attributes = (
        ActionAttribute(
            name="scheduleNames",
            is_required=True,
            attr_type=list,
            default=[],
            description="Name of the target schedules.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        for schedule_name in cast(list, self.get_attribute("scheduleNames")):
            if schedule_name not in target._stm_schedules:
                raise ValueError(f"Schedule name '{schedule_name}' does not exists.")

            cpu_index = 0
            gpu_index = 0
            dla_index = 0
            vpu_index = 0
            cuda_stream_index = 0
            cudla_stream_index = 0
            cupva_stream_index = 0
            for hyperepoch in list(
                target._stm_schedules[schedule_name].schedule.hyperepochs.values()
            ):
                epochsList = deepcopy(list(hyperepoch.epochs.values()))
                for epoch in epochsList:
                    new_resources = dict()
                    for key, resource in hyperepoch.resources.items():
                        resource_name = ""
                        if "CPU" in key:
                            resource_split = key.split(".")
                            resource_name = resource_split[0] + ".CPU" + str(cpu_index)
                            cpu_index += 1
                        elif "GPU" in key:
                            resource_name = key + str(gpu_index)
                            if "CUDA_STREAM" in key:
                                resource_split = key.split(":")
                                resource_split[0] = "".join(
                                    [i for i in resource_split[0] if not i.isdigit()]
                                )
                                resource_split[0] = resource_split[0] + str(
                                    cuda_stream_index
                                )
                                resource_name = (
                                    resource_split[0]
                                    + ":"
                                    + resource_split[1]
                                    + str(gpu_index)
                                )
                                cuda_stream_index += 1
                        elif "DLA" in key:
                            resource_name = key + str(dla_index)
                            if "CUDLA_STREAM" in key:
                                resource_split = key.split(":")
                                resource_split[0] = "".join(
                                    [i for i in resource_split[0] if not i.isdigit()]
                                )
                                resource_split[0] = resource_split[0] + str(
                                    cudla_stream_index
                                )
                                resource_name = (
                                    resource_split[0]
                                    + ":"
                                    + resource_split[1]
                                    + str(dla_index)
                                )
                                cudla_stream_index += 1
                        elif "VPU" in key:
                            resource_name = key + str(vpu_index)
                            if "PVA_STREAM" in key:
                                resource_split = key.split(":")
                                resource_split[0] = "".join(
                                    [i for i in resource_split[0] if not i.isdigit()]
                                )
                                resource_split[0] = resource_split[0] + str(
                                    cupva_stream_index
                                )
                                resource_name = (
                                    resource_split[0]
                                    + ":"
                                    + resource_split[1]
                                    + str(vpu_index)
                                )
                                cupva_stream_index += 1
                        else:
                            continue

                        new_resources[resource_name] = deepcopy(resource)
                        new_resources[resource_name].passes = []
                        for res_pass in resource.passes:
                            for pass_name_out in epoch.passes:
                                for pass_name_in in pass_name_out:
                                    if pass_name_in in res_pass:
                                        new_resources[resource_name].passes.append(
                                            res_pass
                                        )

                    gpu_index += 1
                    dla_index += 1
                    vpu_index += 1
                    hyperepoch_name = (
                        epoch.name
                        + hyperepoch.name[0].capitalize()
                        + hyperepoch.name[1:]
                    )
                    hyperepoch_monitoringPeriod = None
                    if hyperepoch.monitoringPeriod is not None:
                        hyperepoch_monitoringPeriod = (
                            hyperepoch.monitoringPeriod / epoch.frames
                        )
                    epoch.period = hyperepoch.period / epoch.frames
                    epoch.frames = 1
                    new_hyperepoch = HyperepochDefinition(
                        name=hyperepoch_name,
                        period=epoch.period,
                        epochs={epoch.name: epoch},
                        resources=new_resources,
                        monitoringPeriod=hyperepoch_monitoringPeriod,
                    )
                    target.insert_hyperepoch(
                        schedule=schedule_name, hyperepoch=new_hyperepoch
                    )
                target.remove_hyperepoch(
                    schedule=schedule_name, hyperepoch_name=hyperepoch.name
                )


@ActionFactory.register("combine-hyperepochs")
class CombineHyperepochs(Action):
    """Combine all epochs into a single hyperepoch."""

    extra_attributes = (
        ActionAttribute(
            name="scheduleNames",
            is_required=True,
            attr_type=list,
            default=[],
            description="Name of the target schedules.",
        ),
        ActionAttribute(
            name="newPeriod",
            is_required=True,
            attr_type=int,
            default=0,
            description="New hyperepoch period.",
        ),
        ActionAttribute(
            name="hyperepochName",
            is_required=False,
            attr_type=str,
            default="singleHyperepoch",
            description="New hyperepoch name.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        for schedule_name in cast(list, self.get_attribute("scheduleNames")):
            new_period = self.get_attribute("newPeriod")
            hyperepoch_name = self.get_attribute("hyperepochName")
            if schedule_name not in target._stm_schedules:
                raise ValueError(f"Schedule name '{schedule_name}' does not exists.")
            combined_resource_list = dict()
            combined_epoch_list = dict()

            for hyperepoch in list(
                target._stm_schedules[schedule_name].schedule.hyperepochs.values()
            ):
                # Iterate through all of the epochs aligns the number of frames
                # for the new period
                for epoch in deepcopy(list(hyperepoch.epochs.values())):
                    # Evenly divide the frames into the hyperepoch
                    new_num_frames = int(
                        new_period / int(hyperepoch.period / epoch.frames)
                    )
                    # If the new period isn't a multiple add one more frame
                    if (new_period % int(hyperepoch.period / epoch.frames)) > 0:
                        new_num_frames += 1
                    epoch.frames = new_num_frames
                    if epoch.period is not None:
                        epoch.period = int(new_period / epoch.frames)
                    combined_epoch_list[
                        hyperepoch.name + epoch.name.capitalize()
                    ] = epoch
                # Combine the resources
                for resource in hyperepoch.resources:
                    if resource not in combined_resource_list:
                        combined_resource_list[resource] = hyperepoch.resources[
                            resource
                        ]
                target.remove_hyperepoch(
                    schedule=schedule_name, hyperepoch_name=hyperepoch.name
                )

            combined_hyperepoch = HyperepochDefinition(
                name=hyperepoch_name,
                period=new_period,
                epochs=combined_epoch_list,
                resources=combined_resource_list,
            )
            # Insert the new one
            target.insert_hyperepoch(
                schedule=schedule_name, hyperepoch=combined_hyperepoch
            )
