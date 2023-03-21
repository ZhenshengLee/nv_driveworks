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
"""Data structures for schedule descriptor."""
from collections import OrderedDict

from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional


class ResourceDefinition:
    """class for resources definition."""

    def __init__(self, specifier: str, passes: Optional[List[str]] = None):
        """Create resource definition instance."""
        self._specifier = specifier
        self._passes = passes if passes is not None else []

    @property
    def specifier(self) -> str:
        """Return resource specifier."""
        return self._specifier

    @property
    def passes(self) -> List[str]:
        """Return passes assigned to this resource."""
        return self._passes

    @passes.setter
    def passes(self, value: List[str]) -> None:
        """Set passes assigned to this resource."""
        self._passes = value

    def to_json_data(self) -> List[str]:
        """Dump ResourceDefinition to JSON data."""
        return self._passes[:]

    def __eq__(self, other: object) -> bool:
        """Check if two ResourceDefinition objects are equal."""
        if not isinstance(other, ResourceDefinition):
            return NotImplemented

        return (
            self.to_json_data() == other.to_json_data()
            and self._specifier == other._specifier
        )


class AliasGroupDefinition:
    """class for AliasGroup descriptor."""

    def __init__(self, name: str, group_components: List[str]):
        """Create AliasGroup definition instance."""
        self._name = name
        # check duplicates
        if len(group_components) != len(set(group_components)):
            raise ValueError(f"Duplicates components in AliasGroup: {name}")
        self._group_components = (
            sorted(group_components) if group_components is not None else []
        )

    @property
    def name(self) -> str:
        """Return name of this AliasGroup."""
        return self._name

    @property
    def group_components(self) -> List[str]:
        """Return components of this AliasGroup."""
        return self._group_components

    def to_json_data(self) -> List[str]:
        """Dump AliasGroupDefinition to JSON data."""
        return self._group_components

    def check_group(self, group: "AliasGroupDefinition", should_exist: bool) -> None:
        """Sanity check if the input AliasGroup can be merged/substract.

        @param group AliasGroup to be checked
        @param should_exist bool, mark if the comp should exists in input AliasGroup
        """
        if group.name != self._name:
            raise ValueError(
                f"Group name is not matched, src: {group.name}, target: {self.name}"
            )
        for comp in group.group_components:
            if should_exist:
                if comp not in self.group_components:
                    raise ValueError(
                        f"Component to be substracted '{comp}' not in '{self.name}'"
                    )
            else:
                if comp in self._group_components:
                    raise ValueError(f"Duplicate comp found: {group.name}, {comp}")

    def merge_group(self, group: "AliasGroupDefinition") -> None:
        """Merge AliasGroup. Be sure call check_group() for sanity check.

        @param group AliasGroup to be merged
        """
        for comp in group.group_components:
            self._group_components.append(comp)
        self._group_components.sort()

    def substract_group(self, group: "AliasGroupDefinition") -> "AliasGroupDefinition":
        """Substract AliasGroup. Be sure call check_group() for sanity check.

        @param group AliasGroup to be substracted
        @return the AliasGroupDefinition actually removed.
        Note: if only one component is left, the one will also be removed.
        """
        for comp in group.group_components:
            self._group_components.remove(comp)
        if len(self._group_components) == 1:
            group.group_components.append(self._group_components.pop())
        self._group_components.sort()
        return group


class EpochDefinition:
    """class for epoch descriptor."""

    def __init__(
        self,
        *,
        name: str,
        period: int,
        frames: Optional[int] = 1,
        passes: Optional[List[List[str]]] = None,
        alias_groups: Optional[Dict[str, AliasGroupDefinition]] = None,
    ):
        """Create epoch definition instance."""
        self._name = name
        self._period = period
        self._frames = frames if frames is not None else 1
        self._passes = passes if passes is not None else [[]]
        self._alias_groups = alias_groups if alias_groups is not None else {}

    @property
    def name(self) -> str:
        """Return name of this epoch."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set name of this epoch."""
        self._name = value

    @property
    def period(self) -> int:
        """Return period of this epoch."""
        return self._period

    @period.setter
    def period(self, value: int) -> None:
        """Set period of this epoch."""
        self._period = value

    @property
    def frames(self) -> int:
        """Return frames of this epoch."""
        return self._frames

    @frames.setter
    def frames(self, value: int) -> None:
        """Set frames of this epoch."""
        self._frames = value

    @property
    def passes(self) -> List[List[str]]:
        """Return passes with pipeline info of this epoch."""
        return self._passes

    @passes.setter
    def passes(self, value: List[List[str]]) -> None:
        """Set passes with pipeline info of this epoch."""
        self._passes = value

    @property
    def alias_groups(self) -> Dict[str, AliasGroupDefinition]:
        """Return alias groups of the epoch."""
        return self._alias_groups

    @alias_groups.setter
    def alias_groups(self, new_alias_groups: Dict[str, AliasGroupDefinition]) -> None:
        """Set alias groups of this epoch."""
        self._alias_groups = new_alias_groups

    def to_json_data(self) -> OrderedDict:
        """Dump EpochDefinition to JSON data."""
        epoch_json: OrderedDict = OrderedDict()
        epoch_json["period"] = self.period
        if self.frames != 1:
            epoch_json["frames"] = self.frames
        epoch_json["passes"] = self.passes
        if self._alias_groups:
            epoch_json["aliasGroups"] = {
                name: group.to_json_data() for name, group in self._alias_groups.items()
            }
        return epoch_json

    def __eq__(self, __o: object) -> bool:
        """Compare the two instance of ProcessDefinition."""
        if not isinstance(__o, EpochDefinition):
            return NotImplemented
        return self.to_json_data() == __o.to_json_data() and self.name == __o.name


class HyperepochDefinition:
    """Class for hyperepoch descriptor."""

    def __init__(
        self,
        *,
        name: str,
        period: int,
        epochs: Dict[str, EpochDefinition] = None,
        resources: Dict[str, ResourceDefinition] = None,
    ):
        """An entry in hyperepochs section."""
        self._name = name
        self._period = period
        self._epochs = epochs if epochs is not None else {}
        self._resources = resources if resources is not None else {}

    @property
    def name(self) -> str:
        """Return name of this HyperepochDefinition."""
        return self._name

    @name.setter
    def name(self, new_name: str) -> None:
        """Set name of this HyperepochDefinition."""
        self._name = new_name

    @property
    def period(self) -> int:
        """Return period of this HyperepochDefinition."""
        return self._period

    @period.setter
    def period(self, new_period: int) -> None:
        """Set period of this HyperepochDefinition."""
        self._period = new_period

    @property
    def epochs(self) -> Dict[str, EpochDefinition]:
        """Return epochs of this hyperepoch."""
        return self._epochs

    @epochs.setter
    def epochs(self, value: Dict[str, EpochDefinition]) -> None:
        """set the epochs dict of this hyperepoch."""
        self._epochs = value

    @property
    def resources(self) -> Dict[str, ResourceDefinition]:
        """Return resources definition and assignment of this hyperepoch."""
        return self._resources

    @resources.setter
    def resources(self, value: Dict[str, ResourceDefinition]) -> None:
        """set the resource dict of this hyperepoch."""
        self._resources = value

    def to_json_data(self) -> OrderedDict:
        """Dump EpochDefinition to JSON data."""
        hyperepoch_json: OrderedDict = OrderedDict()
        hyperepoch_json["period"] = self.period

        # hyperepoch epochs
        epochs_json: OrderedDict = OrderedDict()
        for epoch_name in sorted(self.epochs.keys()):
            epoch = self.epochs[epoch_name]
            epoch_json = epoch.to_json_data()
            epochs_json[epoch_name] = epoch_json
        hyperepoch_json["epochs"] = epochs_json

        # hyperepoch resources
        resources_json: OrderedDict = OrderedDict()
        for specifier in sorted(self.resources.keys()):
            resource = self.resources[specifier]
            resources_json[specifier] = resource.passes

        hyperepoch_json["resources"] = resources_json

        return hyperepoch_json

    def __eq__(self, __o: object) -> bool:
        """Compare the two instance of HyperepochDefinition."""
        if not isinstance(__o, HyperepochDefinition):
            return NotImplemented
        return self.to_json_data() == __o.to_json_data() and self.name == __o.name


class PassDependencyDefinition:
    """Class for passDependency descriptor."""

    def __init__(self, *, pass_name: str, dependencies: List[str]):
        """An entry in passDependency section.

        @param pass_name name of this PassDependency
        @param dependencies dependencies of this PassDependency
        """
        self._pass_name = pass_name
        self._dependencies = dependencies

    @property
    def pass_name(self) -> str:
        """Return pass_name of this PassDependency."""
        return self._pass_name

    @property
    def dependencies(self) -> List[str]:
        """Return dependencies of this PassDependency."""
        return self._dependencies

    def __eq__(self, other: object) -> bool:
        """Overload the '==' operator and ignore the order of dependencies list."""
        if not isinstance(other, PassDependencyDefinition):
            raise TypeError("The __eq__ of PassDependencyDefinition type invalid")
        if other.pass_name != self.pass_name:
            return False
        if sorted(other.dependencies) != sorted(self.dependencies):
            return False
        return True

    def insert_dependency(self, dependencies: List[str]) -> None:
        """Insert dependency to this PassDependency.

        @param dependency list of dependency to be inserted
        """
        for dep in dependencies:
            if dep in self.dependencies:
                raise ValueError(
                    f"Dependency to be inserted '{dep}' already in '{self.pass_name}'"
                )
            self.dependencies.append(dep)

    def remove_dependency(self, dependencies: List[str]) -> None:
        """Remove dependency from this PassDependency.

        @param dependency list of dependency to be removed
        """
        for dep in dependencies:
            if dep not in self.dependencies:
                raise ValueError(
                    f"Dependency to be removed '{dep}' not in '{self.pass_name}'"
                )
            self.dependencies.remove(dep)


class ScheduleDefinition:
    """Class for schedule descriptor."""

    def __init__(
        self,
        *,
        name: str,
        wcet: Path,
        hyperepochs: Optional[Dict[str, HyperepochDefinition]],
        passDependencies: Optional[Dict[str, PassDependencyDefinition]],
    ):
        """An entry in stmSchedules section.

        @param name name of this schedule
        @param hyperepochs hyperepoch definitions of this schedule
        """

        self._name = name
        self._wcet = wcet
        self._hyperepochs = hyperepochs if hyperepochs is not None else {}
        self._passDependencies = (
            passDependencies if passDependencies is not None else {}
        )

    @property
    def name(self) -> str:
        """Return name of this schedule."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set the name of this schedule."""
        self._name = value

    @property
    def hyperepochs(self) -> Dict[str, HyperepochDefinition]:
        """Return all hyperepochs of this schedule."""
        return self._hyperepochs

    @hyperepochs.setter
    def hyperepochs(self, value: Dict[str, HyperepochDefinition]) -> None:
        """set the hyperepoch dict of this schedule."""
        self._hyperepochs = value

    @property
    def passDependencies(self) -> Dict[str, PassDependencyDefinition]:
        """Return passDependencies of this schedule."""
        return self._passDependencies

    @passDependencies.setter
    def passDependencies(self, value: Dict[str, PassDependencyDefinition]) -> None:
        """Set passDependencies of this schedule."""
        self._passDependencies = value

    @property
    def wcet(self) -> Path:
        """Return the wcet file path."""
        return self._wcet

    @wcet.setter
    def wcet(self, value: Path) -> None:
        """Set the wcet file path."""
        self._wcet = value

    @staticmethod
    def convert_period(period: Optional[int]) -> int:
        """Return the passed period or a default value of 10ms."""
        # TODO(hongwang): Fix the period in backend_systemdescription.py
        if period is None:
            return 10_000_000
        return period

    @classmethod
    def from_json_data(cls, name: str, content: Dict) -> "ScheduleDefinition":
        """Create ScheduleDefinition from JSON data."""

        hyperepochs = {
            name: HyperepochDefinition(
                name=name,
                period=ScheduleDefinition.convert_period(hyperepoch.get("period")),
                epochs={
                    epoch_name: EpochDefinition(
                        name=epoch_name,
                        period=ScheduleDefinition.convert_period(
                            epoch_raw.get("period")
                        ),
                        frames=epoch_raw.get("frames", 1),
                        passes=epoch_raw.get("passes", []),
                        alias_groups={
                            group_name: AliasGroupDefinition(
                                name=group_name, group_components=group_components
                            )
                            for group_name, group_components in epoch_raw.get(
                                "aliasGroups", {}
                            ).items()
                        },
                    )
                    for epoch_name, epoch_raw in hyperepoch.get("epochs", {}).items()
                },
                resources={
                    specifier: ResourceDefinition(specifier=specifier, passes=passes)
                    for specifier, passes in hyperepoch.get("resources", {}).items()
                },
            )
            for name, hyperepoch in content.get("hyperepochs", {}).items()
        }

        passDependencies = {
            name: PassDependencyDefinition(pass_name=name, dependencies=value)
            for name, value in content.get("passDependencies", {}).items()
        }

        return ScheduleDefinition(
            name=name,
            wcet=Path(content.get("wcet", "")),
            hyperepochs=hyperepochs,
            passDependencies=passDependencies,
        )

    def to_json_data(self) -> OrderedDict:
        """Dump ScheduleDefinition to JSON data."""
        schedule_json: OrderedDict = OrderedDict()

        schedule_json["wcet"] = str(self.wcet)

        hyperepochs_json: OrderedDict = OrderedDict()
        for name in sorted(self.hyperepochs.keys()):
            hyperepoch = self.hyperepochs[name]
            hyperepoch_json = hyperepoch.to_json_data()
            hyperepochs_json[name] = hyperepoch_json

        passDependencies = {}
        for key in sorted(self.passDependencies.keys()):
            item = self.passDependencies[key]
            passDependencies[key] = sorted(item.dependencies)

        if passDependencies:
            schedule_json["passDependencies"] = passDependencies

        schedule_json["hyperepochs"] = hyperepochs_json

        return schedule_json

    def insert_dependencies(self, passDependency: PassDependencyDefinition) -> None:
        """Insert dependencies to this schedule.

        @param dependencies list of dependencies to be inserted
        """
        if passDependency.pass_name not in self.passDependencies:
            self.passDependencies[passDependency.pass_name] = passDependency
        else:
            raise ValueError(f"Pass name '{passDependency.pass_name}' alread exists.")

    def remove_dependencies(self, passDependency: PassDependencyDefinition) -> None:
        """Remove dependencies from this schedule.

        @param dependencies list of dependencies to be inserted
        """
        if passDependency.pass_name in self.passDependencies:
            del (self.passDependencies[passDependency.pass_name])
        else:
            raise ValueError(f"Pass name '{passDependency.pass_name}' does not exist.")

    def __eq__(self, __o: object) -> bool:
        """Compare the two instance of ScheduleDefinition."""
        if not isinstance(__o, ScheduleDefinition):
            raise TypeError("The __eq__ of ScheduleDefinition type invalid")

        return self.to_json_data() == __o.to_json_data() and self.name == __o.name
