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
"""Data structures for Application."""
from copy import deepcopy
from pathlib import Path
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Union
import warnings
from dwcgf.descriptor import AliasGroupDefinition
from dwcgf.descriptor import ApplicationDescriptor
from dwcgf.descriptor import ConnectionDefinition
from dwcgf.descriptor import DescriptorLoader
from dwcgf.descriptor import EpochDefinition
from dwcgf.descriptor import HyperepochDefinition
from dwcgf.descriptor import ParameterDefinition
from dwcgf.descriptor import PassDependencyDefinition
from dwcgf.descriptor import ProcessDefinition
from dwcgf.descriptor import ResourceDefinition
from dwcgf.descriptor import ScheduleDefinition
from dwcgf.descriptor import SignalProducerPort
from dwcgf.descriptor import StateDefinition
from dwcgf.descriptor import STMExternalRunnableDefinition
from dwcgf.descriptor import SubcomponentDefinition
from dwcgf.transaction import UndoContext
import json_merge_patch

import yaml

from .component import Component
from .graphlet import Graphlet
from .node import Node
from .node import Pass
from .object_model_channel import object_model_channel
from .port import Port
from .process import Process


class Epoch:
    """class for epoch (TODO)."""

    def __init__(self, epoch: EpochDefinition):
        """TODO."""
        pass


class HyperEpoch:
    """class for hyperepoch (TODO)."""

    def __init__(self, hyperepoch: HyperepochDefinition):
        """TODO."""
        pass


class Resource:
    """class for resource (TODO)."""

    def __init__(self, resource: ResourceDefinition):
        """TODO."""
        pass


class State:
    """class for states (TODO)."""

    def __init__(self, state: StateDefinition):
        """TODO."""
        self.state = state


class STMSchedule:
    """class for STM schedule (TODO)."""

    def __init__(self, schedule: ScheduleDefinition):
        """TODO."""
        self.schedule = schedule


class Application(Graphlet):
    """class for application instance."""

    CMDArgumentValueType = Union[bool, str, List[str]]

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        comment: Optional[str] = None,
        log_spec: Optional[str] = None,
        version: Optional[int] = None,
        parameters: Optional[Dict[str, ParameterDefinition]] = None,
        input_ports: Optional[Dict[str, Port]] = None,
        output_ports: Optional[Dict[str, Port]] = None,
        subcomponents: Optional[Dict[str, Component]] = None,
        parameter_mappings: Optional[
            Dict[str, SubcomponentDefinition.ParameterValueType]
        ] = None,
        connection_definitions: Optional[List[ConnectionDefinition]] = None,
        states: Optional[Dict[str, State]] = None,
        stm_schedules: Optional[Dict[str, STMSchedule]] = None,
        stm_external_runnables: Optional[
            Dict[str, STMExternalRunnableDefinition]
        ] = None,
        processes: Optional[Dict[str, Process]] = None,
        required_sensors: Optional[Path] = None,
        sensor_mapping_lookups: Optional[List[Path]] = None,
        extra_info: Optional[Path] = None,
    ):
        """Create Application instance."""
        super().__init__(
            name=name,
            comment=comment,
            parameters=parameters,
            input_ports=input_ports,
            output_ports=output_ports,
            subcomponents=subcomponents,
            parameter_mappings=parameter_mappings,
            connection_definitions=connection_definitions,
        )

        if sensor_mapping_lookups is not None:
            assert isinstance(sensor_mapping_lookups, list)
            assert all(isinstance(e, Path) for e in sensor_mapping_lookups)

        self._parameters = parameters if parameters is not None else {}
        self._input_ports = input_ports if input_ports is not None else {}
        self._output_ports = output_ports if output_ports is not None else {}
        self._subcomponents = subcomponents if subcomponents is not None else {}
        self._parameter_mappings = (
            parameter_mappings if parameter_mappings is not None else {}
        )

        self._states = states if states is not None else {}
        self._stm_schedules = stm_schedules if stm_schedules is not None else {}
        self._stm_external_runnables = (
            stm_external_runnables if stm_external_runnables is not None else {}
        )

        self._processes = processes if processes is not None else {}
        for _, process in self._processes.items():
            process_subcomponents: List[Component] = []
            if process.subcomponents is None and process.desc_subcomponents is not None:
                for _name in process.desc_subcomponents:
                    subcomponent = self.get_component(_name)
                    if subcomponent is not None:
                        process_subcomponents.append(subcomponent)
                    else:
                        raise ValueError(f"Can not locate subcomponent: {_name}.")
                process.subcomponents = process_subcomponents

        self._log_spec = log_spec
        self._version = version
        self._extra_info = extra_info
        self._required_sensors = required_sensors
        self._sensor_mapping_lookups = (
            sensor_mapping_lookups if sensor_mapping_lookups is not None else []
        )

        self._constructed = False
        self._wcet_data: Dict[str, Dict] = {}

    @staticmethod
    def from_descriptor(  # type: ignore
        loader: DescriptorLoader, desc: ApplicationDescriptor
    ) -> "Application":
        """Create Application from descriptor."""
        processes: Dict[str, Process] = {}
        for process_name, process_def in desc.processes.items():
            processes[process_name] = Process.from_descriptor(
                desc=deepcopy(process_def),
                subcomponent_instances=None,  # to be set inside Application constructor
                loader=loader,
            )
        app = Application(
            name=desc.name,
            comment=desc.comment,
            log_spec=desc.log_spec,
            version=desc.version,
            parameters=desc.parameters,
            input_ports=Port.from_descriptor(desc.input_ports, True),
            output_ports=Port.from_descriptor(desc.output_ports, False),
            subcomponents=Graphlet.create_subcomponents(loader, desc),
            parameter_mappings={
                subcomp_name: deepcopy(subcomp.parameters)
                for subcomp_name, subcomp in desc.subcomponents.items()
            },
            connection_definitions=deepcopy(desc.connections),
            states={
                name: State(deepcopy(state)) for name, state in desc.states.items()
            },
            stm_schedules={
                name: STMSchedule(deepcopy(schedule))
                for name, schedule in desc.stm_schedules.items()
            },
            stm_external_runnables={
                name: deepcopy(stm_ext_runnable)
                for name, stm_ext_runnable in desc.stm_external_runnables.items()
            },
            processes=processes,
            extra_info=Path(desc.extra_info),
            required_sensors=Path(desc.required_sensors),
            sensor_mapping_lookups=deepcopy(desc.sensor_mapping_lookups),
        )

        app.descriptor_path = desc.file_path

        return app

    @property
    def log_spec(self) -> Optional[str]:
        """Return log_spec."""
        return self._log_spec

    @property
    def version(self) -> Optional[int]:
        """Return version."""
        return self._version

    @property
    def required_sensors(self) -> Optional[Path]:
        """Return required_sensors."""
        return self._required_sensors

    @property
    def sensor_mapping_lookups(self) -> List[Path]:
        """Return relative path(s) of sensor mapping lookups."""
        return self._sensor_mapping_lookups

    @property
    def extra_info(self) -> Optional[Path]:
        """Return extra_info."""
        return self._extra_info

    @extra_info.setter
    def extra_info(self, new_extra_info: Optional[Path]) -> None:
        """Set the extra info file path."""

        self._set_extra_info(new_extra_info)

    def _set_extra_info(self, new_extra_info: Optional[Path]) -> UndoContext:
        """Set a new extra info path."""
        ret = UndoContext(self, self._extra_info)
        self._extra_info = new_extra_info

        return ret

    @property
    def children(self) -> List[Component]:
        """Return subcomponents as list."""
        return [value for key, value in self._subcomponents.items()]

    @property
    def processes(self) -> Dict[str, Process]:
        """Return processes."""
        return self._processes

    @property
    def states(self) -> Dict[str, State]:
        """Return states."""
        return self._states

    @property
    def schedules(self) -> Dict[str, STMSchedule]:
        """Return schedules."""
        return self._stm_schedules

    @property
    def stm_external_runnables(self) -> Dict[str, STMExternalRunnableDefinition]:
        """Return stm external runnables."""
        return self._stm_external_runnables

    def construct(self) -> None:
        """Perform wcet distribution and pass dependency graph building."""

        # always reset pass status before construction
        self.__reset_all_passes()

        # get wcet path and check validity
        for schedule in self._stm_schedules.values():
            if schedule.schedule.wcet is None:
                raise ValueError(
                    f"WCET file unset for schedule: {schedule.schedule.name}"
                )

            wcet_path = None
            if self.descriptor_path is not None:
                wcet_path = self.descriptor_path.parent / schedule.schedule.wcet
            else:
                wcet_path = schedule.schedule.wcet

            self.__distribute_wcet(wcet_path, schedule.schedule.name)

        # in the future wcet_path will be an attribute of application
        # if self._wcet is not None:
        #     self.distribute_wcet(self._wcet)

        # build pass dependency graph
        for schedule in self._stm_schedules.values():
            self.__build_pass_dependency_graph(schedule)

        # TODO
        # epoch & client info assosiation to object model

        self._constructed = True

    def __reset_all_passes(self) -> None:
        """Reset wcet distribution and pass dependency graph status."""

        def dfs(root: Component) -> None:
            if isinstance(root, Node):
                passes = root.passes
                for pass_ in passes:
                    pass_.wcet = {}
                    pass_.dependencies = {}
                    pass_.process = None
            elif isinstance(root, (Graphlet, Application)):
                for _, component in root.subcomponents.items():
                    dfs(component)
            else:
                raise TypeError(f"Unexpected subcomponent subcomponent type: {root}")

        dfs(self)

    def __distribute_wcet(self, wcet_path: Path, schedule_name: str) -> None:
        """Distribute wcet to each pass from a file."""
        wcet_data: Dict[str, str] = {}
        with wcet_path.open() as f:
            wcet_data = yaml.safe_load(f)
        wcet_process_name: dict = {}
        for k in wcet_data.keys():
            # wcet line example: camera_master.top_arender_renderingNode_pass_0: 934294
            # key should be split into 2 part, one is the process name, the other is pass name
            k_list = k.split(".")
            if len(k_list) != 2:
                raise ValueError(f"wcet key: {k} is invalid.")
            process_name = k_list[0]
            pass_name = k_list[1]
            wcet_process_name[pass_name] = process_name

        if schedule_name in self._wcet_data.keys() and self._constructed is True:
            raise ValueError(f"{schedule_name} already loaded!")
        else:
            self._wcet_data[schedule_name] = deepcopy(wcet_data)

        pass_not_listed: list = []

        def distribute_wcet_impl(root: Component, schedule_name: str) -> None:
            if isinstance(root, Node):
                passes = root.passes
                parent_name = root.id.replace(".", "_")
                for idx, pass_ in enumerate(passes):
                    pass_name = parent_name + f"_pass_{idx}"
                    process_name = wcet_process_name.get(pass_name, None)
                    wcet_key = (
                        process_name + "." + pass_name
                        if process_name is not None
                        else pass_name
                    )
                    if wcet_key in self._wcet_data[schedule_name].keys():
                        pass_.wcet[schedule_name] = self._wcet_data[schedule_name].pop(
                            wcet_key
                        )
                        if process_name in self._processes.keys():
                            pass_.process = self._processes[process_name]
                        else:
                            raise ValueError(
                                f"Can not bind {process_name} for {wcet_key}, \
as {process_name} does not exist in the application."
                            )
                    else:
                        pass_not_listed.append(wcet_key)
            elif isinstance(root, (Graphlet, Application)):
                for component in root.subcomponents.values():
                    distribute_wcet_impl(component, schedule_name)
            else:
                raise TypeError(f"Unexpected subcomponent subcomponent type: {root}")

        distribute_wcet_impl(self, schedule_name)

        total_passes = len(self._wcet_data[schedule_name])
        print(
            f"\nAfter wcet distribution: \
wcet loaded/unused: {len(wcet_data) - total_passes}/{total_passes}, \
{len(pass_not_listed)} passes not listed in wcet file."
        )

    def get_wcet_file(self, schedule_name: str) -> Optional[str]:
        """Returns the WCET file for a given schedule."""

        if schedule_name not in self._stm_schedules:
            raise ValueError(f"Schedule {schedule_name} not present in.")

        return self._stm_schedules[schedule_name].schedule.wcet

    def collect_wcet(self, schedule_name: str) -> Dict[str, int]:
        """Collect wcet from all passes."""
        if not self._constructed:
            raise NotImplementedError(f"Can not dump wcet before construction.")

        wcet_data: dict = {}
        pass_with_empty_wcet: list = []

        def collect_wcet_impl(root: Component, schedule_name: str) -> None:
            if isinstance(root, Node):
                passes = root.passes
                parent_name = root.id.replace(".", "_")
                for idx, pass_ in enumerate(passes):
                    pass_name = parent_name + f"_pass_{idx}"
                    process_name = pass_.process_name
                    wcet_key = (
                        process_name + "." + pass_name
                        if process_name is not None
                        else pass_name
                    )
                    if pass_.wcet is not None and schedule_name in pass_.wcet:
                        wcet_data[wcet_key] = pass_.wcet[schedule_name]
                    else:
                        pass_with_empty_wcet.append(wcet_key)
            elif isinstance(root, (Graphlet, Application)):
                for component in root.subcomponents.values():
                    collect_wcet_impl(component, schedule_name)
            else:
                raise TypeError(f"Unexpected subcomponent subcomponent type: {root}")

        collect_wcet_impl(self, schedule_name)

        print(
            f"\nAfter wcet collection: \
wcet collected/unused: {len(wcet_data)}/{len(self._wcet_data[schedule_name])}, \
{len(pass_with_empty_wcet)} passes have no valid wcet."
        )

        return (
            wcet_data
            if len(self._wcet_data[schedule_name]) == 0
            else json_merge_patch.merge(wcet_data, self._wcet_data[schedule_name])
        )

    def __build_pass_dependency_graph(self, schedule: STMSchedule) -> None:
        """build the pass dependency graph from schedule and pass_dependency_predef."""
        pass_id2fullname = self.__get_pass_id_mapping()
        passDependencies = deepcopy(schedule.schedule.passDependencies)
        residual: Dict[str, List[str]] = {}

        def build_impl(root_name: str, history: List[str] = []) -> None:
            history.append(root_name)
            if root_name in passDependencies.keys():

                dependency_def = passDependencies.pop(root_name)
                pass_ = self.__get_pass_by_name(root_name)

                if pass_ is None:
                    residual[root_name] = dependency_def.dependencies
                    history.pop()
                    warnings.warn(
                        f"Can't locate pass {root_name} in subcomponents, skipped."
                    )
                    return

                res_dep: List[str] = []
                for dependency_name in dependency_def.dependencies:
                    if dependency_name not in history:

                        dependency = self.__get_pass_by_name(dependency_name)
                        if dependency is None:
                            res_dep.append(dependency_name)
                            warnings.warn(
                                f"Can't locate pass {dependency_name} in subcomponents, skipped."
                            )
                            continue

                        assert id(dependency) in pass_id2fullname
                        if schedule.schedule.name not in pass_.dependencies.keys():
                            pass_.dependencies[schedule.schedule.name] = []
                        pass_.dependencies[schedule.schedule.name].append(dependency)

                        build_impl(dependency_name, history)
                    else:
                        raise ValueError(
                            f"Loop found in pass dependency chain: {history}."
                        )

                if res_dep:
                    residual[root_name] = res_dep

            history.pop()

        while len(passDependencies) > 0:
            root_name = list(passDependencies.keys())[0]
            build_impl(root_name)

        if residual:
            schedule.schedule.residual_deps = residual

    def dump_pass_dependency_graph(self, schedule_name: str) -> dict:
        """dump the pass dependency graph to schedule passDependencies dict."""
        if not self._constructed:
            raise NotImplementedError(
                f"Can not dump pass dependency graph before construction."
            )
        if schedule_name not in self._stm_schedules:
            raise ValueError(f"Speficied schedule: {schedule_name} not found.")

        pass_id2fullname = self.__get_pass_id_mapping()
        pass_dependencies = {}

        def dump_impl(root: Component) -> None:
            if isinstance(root, Node):
                passes = root.passes
                for pass_ in passes:
                    if schedule_name in pass_.dependencies:
                        dependency_names = []
                        for dependency in pass_.dependencies[schedule_name]:
                            try:
                                dependency_name = pass_id2fullname[id(dependency)]
                            except KeyError:
                                raise KeyError(
                                    f"id {id(dependency)} not found for \
                                        {pass_.name}:{dependency.name}"
                                )
                            dependency_names.append(dependency_name)
                        pass_dependencies[
                            pass_id2fullname[id(pass_)]
                        ] = dependency_names
            elif isinstance(root, (Graphlet, Application)):
                for _, component in root.subcomponents.items():
                    dump_impl(component)
            else:
                raise TypeError(f"Unexpected subcomponent subcomponent type: {root}")

        dump_impl(self)

        if hasattr(self._stm_schedules[schedule_name].schedule, "residual_deps"):
            json_merge_patch.merge(
                pass_dependencies,
                self._stm_schedules[schedule_name].schedule.residual_deps,
            )

        for key in pass_dependencies.keys():
            pass_dependencies[key] = sorted(pass_dependencies[key])

        return pass_dependencies

    def __get_pass_by_name(self, full_pass_name: str) -> Optional[Pass]:
        pass_parents = full_pass_name.split(".")
        pass_name = pass_parents.pop()

        ret_list = []

        def dfs(root: Graphlet, parents: List[str]) -> None:
            child = parents[0]
            parents.pop(0)
            if child in root.subcomponents.keys():
                if len(parents) > 0:
                    child_component = root.subcomponents[child]
                    if isinstance(child_component, (Application, Graphlet)):
                        dfs(child_component, parents)
                else:
                    node = root.subcomponents[child]
                    assert isinstance(node, Node)
                    for pass_ in node.passes:
                        if pass_.name == pass_name:
                            ret_list.append(pass_)
                            return

        dfs(self, pass_parents)

        if ret_list:
            return ret_list[0]
        else:
            return None

    def __get_pass_id_mapping(self) -> dict:
        pass_id2fullname = {}

        def dfs_id2fullname(root: Component, parents: List[str] = []) -> None:
            if isinstance(root, Node):
                passes = root.passes
                for pass_ in passes:
                    pass_name = ".".join(parents) + f".{pass_.name}"
                    pass_id2fullname[id(pass_)] = pass_name
            elif isinstance(root, (Graphlet, Application)):
                for specifier, component in root.subcomponents.items():
                    parents.append(specifier)
                    dfs_id2fullname(component, parents)
                    parents.pop()
            else:
                raise TypeError(f"Unexpected subcomponent subcomponent type: {root}")

        dfs_id2fullname(self)

        return pass_id2fullname

    def get_component(self, relative_name: str) -> Union[Component, None]:
        """Get component from Application using relative name."""
        resolved_id = Graphlet.resolve_id(relative_name)
        if resolved_id == "__self__":
            return self
        return Graphlet.get_component_impl(self, resolved_id)

    def get_component_by_id(self, full_id: str) -> Union[Component, None]:
        """Get component from Application by full ID."""
        # bacause Application is always the root, so full_id is relative name
        return self.get_component(full_id)

    @property
    def id(self) -> None:  # type: ignore
        """Application don't have name, neither ID."""
        return None

    def __check_schedule_name(self, schedule_name: str) -> None:
        """Check if schedule_name exists."""
        if schedule_name not in self._stm_schedules:
            raise ValueError(f"Cannot find schedule name: '{schedule_name}'")

    def __check_hyperepoch_name(self, schedule_name: str, hyperepoch_name: str) -> None:
        """Check if hyperepoch_name exists."""
        self.__check_schedule_name(schedule_name)
        if (
            hyperepoch_name
            not in self._stm_schedules[schedule_name].schedule.hyperepochs
        ):
            raise ValueError(f"Cannot find hyperepoch name '{hyperepoch_name}'")

    def __check_epoch_name(
        self, schedule_name: str, hyperepoch_name: str, epoch_name: str
    ) -> None:
        """Check if epoch_name exists."""
        self.__check_hyperepoch_name(schedule_name, hyperepoch_name)
        if (
            epoch_name
            not in self._stm_schedules[schedule_name]
            .schedule.hyperepochs[hyperepoch_name]
            .epochs
        ):
            raise ValueError(f"Cannot find epoch name '{epoch_name}'")

    def __get_epoch(
        self, schedule_name: str, hyperepoch_name: str, epoch_name: str
    ) -> EpochDefinition:
        self.__check_epoch_name(schedule_name, hyperepoch_name, epoch_name)
        return (
            self._stm_schedules[schedule_name]
            .schedule.hyperepochs[hyperepoch_name]
            .epochs[epoch_name]
        )

    @object_model_channel.pair
    def insert_epoch_alias_groups(
        self,
        schedule_name: str,
        hyperepoch_name: str,
        epoch_name: str,
        alias_groups: Dict[str, List[str]],
    ) -> UndoContext:
        """Insert epoch alias groups(undo-able)."""

        assert isinstance(schedule_name, str)
        assert isinstance(hyperepoch_name, str)
        assert isinstance(epoch_name, str)
        assert isinstance(alias_groups, dict)
        assert all(isinstance(group_name, str) for group_name in alias_groups.keys())
        assert all(isinstance(group, list) for group in alias_groups.values())
        for group in alias_groups.values():
            assert all(isinstance(comp, str) for comp in group)

        alias_groups_inserted = {
            group_name: AliasGroupDefinition(
                name=group_name, group_components=group_components
            )
            for group_name, group_components in alias_groups.items()
        }
        target_epoch: EpochDefinition = self.__get_epoch(
            schedule_name, hyperepoch_name, epoch_name
        )
        target_alias_groups = target_epoch.alias_groups
        # sanity check
        for group_name, group in alias_groups_inserted.items():
            if group_name in target_alias_groups.keys():
                target_alias_groups[group_name].check_group(
                    group=group, should_exist=False
                )
        # inset epoch alias groups
        for group_name, group in alias_groups_inserted.items():
            if group_name in target_alias_groups.keys():
                target_alias_groups[group_name].merge_group(group)
            else:
                target_alias_groups[group_name] = group

        self._self_is_modified = True

        return UndoContext(
            self, schedule_name, hyperepoch_name, epoch_name, alias_groups
        )

    @insert_epoch_alias_groups.pair
    def remove_epoch_alias_groups(
        self,
        schedule_name: str,
        hyperepoch_name: str,
        epoch_name: str,
        alias_groups: Dict[str, List[str]],
    ) -> UndoContext:
        """Remove epoch alias groups."""

        assert isinstance(schedule_name, str)
        assert isinstance(hyperepoch_name, str)
        assert isinstance(epoch_name, str)
        assert isinstance(alias_groups, dict)
        assert all(isinstance(group_name, str) for group_name in alias_groups.keys())
        assert all(isinstance(group, list) for group in alias_groups.values())
        for group in alias_groups.values():
            assert all(isinstance(comp, str) for comp in group)

        alias_groups_removed = {
            group_name: AliasGroupDefinition(
                name=group_name, group_components=group_components
            )
            for group_name, group_components in alias_groups.items()
        }
        target_epoch: EpochDefinition = self.__get_epoch(
            schedule_name, hyperepoch_name, epoch_name
        )
        target_alias_groups = target_epoch.alias_groups
        # sanity check
        for group_name, group in alias_groups_removed.items():
            if group_name in target_alias_groups.keys():
                target_alias_groups[group_name].check_group(
                    group=group, should_exist=True
                )
            else:
                raise ValueError(
                    f"Cannot find group name to be removed: '{group_name}'"
                )
        # remove epoch alias groups
        for group_name, group in alias_groups_removed.items():
            removed_group = target_alias_groups[group_name].substract_group(group)
            # record the components actually be removed
            alias_groups[group_name] = removed_group.group_components
            alias_groups_removed[group_name] = removed_group
            if not target_alias_groups[group_name].group_components:
                target_alias_groups.pop(group_name)

        self._self_is_modified = True

        return UndoContext(
            self, schedule_name, hyperepoch_name, epoch_name, alias_groups
        )

    @object_model_channel.pair_self
    def update_epoch_alias_groups(
        self,
        schedule_name: str,
        hyperepoch_name: str,
        epoch_name: str,
        alias_groups: Dict[str, List[str]],
    ) -> UndoContext:
        """Update epoch alias groups.

        The alias_groups section in epoch will be fully replaced by the input one.
        """

        assert isinstance(schedule_name, str)
        assert isinstance(hyperepoch_name, str)
        assert isinstance(epoch_name, str)
        assert isinstance(alias_groups, dict)
        assert all(isinstance(group_name, str) for group_name in alias_groups.keys())
        assert all(isinstance(group, list) for group in alias_groups.values())
        for group in alias_groups.values():
            assert all(isinstance(comp, str) for comp in group)

        alias_groups_updated = {
            group_name: AliasGroupDefinition(
                name=group_name, group_components=group_components
            )
            for group_name, group_components in alias_groups.items()
        }
        target_epoch: EpochDefinition = self.__get_epoch(
            schedule_name, hyperepoch_name, epoch_name
        )
        target_alias_groups_old_dump = {
            name: group.to_json_data()
            for name, group in target_epoch.alias_groups.items()
        }
        target_epoch.alias_groups = alias_groups_updated
        self._self_is_modified = True

        return UndoContext(
            self,
            schedule_name,
            hyperepoch_name,
            epoch_name,
            target_alias_groups_old_dump,
        )

    @object_model_channel.pair
    def insert_hyperepoch_resource_assignment(
        self,
        schedule_name: str,
        hyperepoch_name: str,
        resource_assignment: Dict[str, List[str]],
    ) -> UndoContext:
        """Insert hyperepoch resource assignment(undo-able)."""

        if schedule_name not in self._stm_schedules:
            raise ValueError(f"Cannot find schedule name: '{schedule_name}'")

        schedule_definition: STMSchedule = self._stm_schedules[schedule_name]

        if hyperepoch_name not in schedule_definition.schedule.hyperepochs:
            raise ValueError(f"Cannot find hyperepoch name: '{hyperepoch_name}'")

        target_hyperepoch: HyperepochDefinition = schedule_definition.schedule.hyperepochs[
            hyperepoch_name
        ]
        existing_resources: Dict[str, ResourceDefinition] = target_hyperepoch.resources

        # This for loop is used for checking if input is valid.
        # Don't modify existing_resources in this for loop
        for specifier in resource_assignment:
            if specifier not in existing_resources:
                raise ValueError(
                    f"Cannot find resource specifier: '{specifier}'"
                    + f" in hyperepoch: '{hyperepoch_name}'"
                )
            new_resource_passes: List[str] = resource_assignment[specifier]

            # Check duplicate resource assignment
            new_resource_passes_set: Set = set(new_resource_passes)
            if len(new_resource_passes_set) < len(new_resource_passes):
                for item in new_resource_passes_set:
                    new_resource_passes.pop(item)
                duplicate_resource_passes = list(set(new_resource_passes))
                duplicate_resource_passes_str = ",".join(duplicate_resource_passes)
                raise ValueError(
                    f"Duplicate resource passes '{duplicate_resource_passes_str}'"
                    + f"found for hyperepoch '{hyperepoch_name}'"
                )

            existing_resource_passes: List[str] = existing_resources[specifier].passes
            for pass_name in new_resource_passes:
                if pass_name in existing_resource_passes:
                    raise ValueError(
                        f"'{pass_name}' already exists in hyperepoch '{hyperepoch_name}'"
                    )

        # Modify existing_resource_passes, don't raise exceptions
        for specifier in resource_assignment:
            existing_resources[specifier].passes.extend(resource_assignment[specifier])

        self._self_is_modified = True

        return UndoContext(self, schedule_name, hyperepoch_name, resource_assignment)

    @insert_hyperepoch_resource_assignment.pair
    def remove_hyperepoch_resource_assignment(
        self,
        schedule_name: str,
        hyperepoch_name: str,
        resource_assignment: Dict[str, List[str]],
    ) -> UndoContext:
        """Remove hyperepoch resource assignment(undo-able)."""
        if schedule_name not in self._stm_schedules:
            raise ValueError(f"Cannot find {schedule_name}")

        schedule_definition: STMSchedule = self._stm_schedules[schedule_name]

        if hyperepoch_name not in schedule_definition.schedule.hyperepochs:
            raise ValueError(
                f"Cannot find hyperepoch name '{hyperepoch_name}' in '{schedule_name}'"
            )

        target_hyperepoch: HyperepochDefinition = schedule_definition.schedule.hyperepochs[
            hyperepoch_name
        ]
        existing_resources: Dict[str, ResourceDefinition] = target_hyperepoch.resources

        # This for loop is used for checking if input is valid.
        # Don't modify existing_resources in this for loop
        for specifier in resource_assignment:
            if specifier not in existing_resources:
                raise ValueError(
                    f"Cannot find resource specifier '{specifier}'"
                    + f" in hyperepoch '{hyperepoch_name}'"
                )
            new_resource_passes: List[str] = resource_assignment[specifier]

            # Check duplicate resource assignment
            new_resource_passes_set: Set = set(new_resource_passes)
            if len(new_resource_passes_set) < len(new_resource_passes):
                for item in new_resource_passes_set:
                    new_resource_passes.pop(item)
                duplicate_resource_passes = list(set(new_resource_passes))
                duplicate_resource_passes_str = ",".join(duplicate_resource_passes)
                raise ValueError(
                    f"Duplicate resource passes '{duplicate_resource_passes_str}'"
                    + f"found for hyperepoch '{hyperepoch_name}'"
                )

            existing_resource_passes: List[str] = existing_resources[specifier].passes
            for pass_name in new_resource_passes:
                if pass_name not in existing_resource_passes:
                    raise ValueError(
                        f"Cannot find '{pass_name}' in hyperepoch '{hyperepoch_name}'"
                    )

        # Modify existing_resource_passes, don't raise exceptions
        for specifier in resource_assignment:
            for pass_name in resource_assignment[specifier]:
                existing_resources[specifier].passes.remove(pass_name)

        self._self_is_modified = True

        return UndoContext(self, schedule_name, hyperepoch_name, resource_assignment)

    @object_model_channel.pair_self
    def update_hyperepoch_name(
        self, schedule_name: str, hyperepoch_name: str, new_hyperepoch_name: str
    ) -> UndoContext:
        """update hyperepoch name(undo-able)."""
        if hyperepoch_name == "":
            raise ValueError(f"{hyperepoch_name} should not be empty string")

        if new_hyperepoch_name == "":
            raise ValueError(f"New hyperepoch name should not be empty string")

        if schedule_name not in self._stm_schedules:
            raise ValueError(f"Cannot find {schedule_name}")

        if hyperepoch_name == new_hyperepoch_name:
            return UndoContext.NO_CHANGE

        schedule_definition = self._stm_schedules[schedule_name]

        if hyperepoch_name not in schedule_definition.schedule.hyperepochs.keys():
            raise ValueError(
                f"Cannot find hyperepoch name '{hyperepoch_name}' in '{schedule_name}'"
            )

        if new_hyperepoch_name in schedule_definition.schedule.hyperepochs.keys():
            raise ValueError(
                f"Hyperepoch name '{new_hyperepoch_name}' already exists in '{schedule_name}'"
            )

        # record the key order into an array
        hyperepoch_array = []
        for n, hyperepoch in schedule_definition.schedule.hyperepochs.items():
            if n == hyperepoch_name:
                n = new_hyperepoch_name
            hyperepoch_array.append((n, hyperepoch))

        schedule_definition.schedule.hyperepochs.clear()
        # insert the keys in original order
        for item in hyperepoch_array:
            n, hyperepoch = item
            schedule_definition.schedule.hyperepochs[n] = hyperepoch

        # update the name in HyperepochDefinition instance
        schedule_definition.schedule.hyperepochs[
            new_hyperepoch_name
        ].name = new_hyperepoch_name

        self._self_is_modified = True

        return UndoContext(self, schedule_name, new_hyperepoch_name, hyperepoch_name)

    @object_model_channel.pair_self
    def update_hyperepoch_period(
        self, schedule_name: str, hyperepoch_name: str, new_period: int
    ) -> UndoContext:
        """Update hyperepoch period(undo-able)."""

        if schedule_name not in self._stm_schedules:
            raise ValueError(f"Cannot find {schedule_name}")

        if new_period <= 0:
            raise ValueError("New hyperepoch period should larger than 0.")

        schedule_definition = self._stm_schedules[schedule_name]

        if hyperepoch_name not in schedule_definition.schedule.hyperepochs.keys():
            raise ValueError(f"Cannot find hyperepoch name '{hyperepoch_name}'")

        old_period = schedule_definition.schedule.hyperepochs[hyperepoch_name].period
        schedule_definition.schedule.hyperepochs[hyperepoch_name].period = new_period

        self._self_is_modified = True

        return UndoContext(self, schedule_name, hyperepoch_name, old_period)

    @object_model_channel.pair_self
    def update_hyperepoch_monitoringPeriod(
        self, schedule_name: str, hyperepoch_name: str, new_monitoringPeriod: int
    ) -> UndoContext:
        """Update hyperepoch monitoring threshold(undo-able)."""

        if schedule_name not in self._stm_schedules:
            raise ValueError(f"Cannot find {schedule_name}")

        if new_monitoringPeriod <= 0:
            raise ValueError(
                "New hyperepoch monitoring threshold should larger than 0."
            )

        schedule_definition = self._stm_schedules[schedule_name]

        if hyperepoch_name not in schedule_definition.schedule.hyperepochs.keys():
            raise ValueError(f"Cannot find hyperepoch name '{hyperepoch_name}'")

        old_monitoringPeriod = schedule_definition.schedule.hyperepochs[
            hyperepoch_name
        ].monitoringPeriod
        schedule_definition.schedule.hyperepochs[
            hyperepoch_name
        ].monitoringPeriod = new_monitoringPeriod

        self._self_is_modified = True

        return UndoContext(self, schedule_name, hyperepoch_name, old_monitoringPeriod)

    @object_model_channel.pair_self
    def update_wcet_file(
        self, schedule_name: str, new_wcet_file_name: str
    ) -> UndoContext:
        """Update wcet file (undo-able)."""

        if schedule_name not in self._stm_schedules:
            raise ValueError(f"Cannot find {schedule_name}")

        schedule_definition = self._stm_schedules[schedule_name]

        old_wcet_file_name = schedule_definition.schedule.wcet
        schedule_definition.schedule.wcet = new_wcet_file_name

        self._self_is_modified = True

        return UndoContext(self, schedule_name, old_wcet_file_name)

    @object_model_channel.pair
    def insert_hyperepoch_resource(
        self, schedule_name: str, hyperepoch_name: str, new_resources: Dict
    ) -> UndoContext:
        """Insert Hyperepoch resource(undo-able)."""
        assert isinstance(new_resources, dict)

        if schedule_name not in self._stm_schedules:
            raise ValueError(f"Cannot find {schedule_name}")

        schedule_definition = self._stm_schedules[schedule_name]

        if hyperepoch_name not in schedule_definition.schedule.hyperepochs.keys():
            raise ValueError(f"Cannot find hyperepoch name '{hyperepoch_name}'")

        resources: Dict[
            str, ResourceDefinition
        ] = schedule_definition.schedule.hyperepochs[hyperepoch_name].resources
        for specifier, passes in new_resources.items():
            if specifier in resources:
                raise ValueError(
                    f"Resource specifier '{specifier}' already exists"
                    + f" in hyperepoch '{hyperepoch_name}'"
                )

        for specifier, passes in new_resources.items():
            resources[specifier] = ResourceDefinition(specifier, passes)

        self._self_is_modified = True

        return UndoContext(
            self, schedule_name, hyperepoch_name, list(new_resources.keys())
        )

    @insert_hyperepoch_resource.pair
    def remove_hyperepoch_resource(
        self, schedule_name: str, hyperepoch_name: str, resources: List[str]
    ) -> UndoContext:
        """Remove Hyperepoch resource(undo-able)."""
        assert isinstance(resources, list)
        if schedule_name not in self._stm_schedules:
            raise ValueError(f"Cannot find {schedule_name}")

        schedule_definition = self._stm_schedules[schedule_name]

        if hyperepoch_name not in schedule_definition.schedule.hyperepochs.keys():
            raise ValueError(f"Cannot find hyperepoch name '{hyperepoch_name}'")

        existing_resources: Dict[
            str, ResourceDefinition
        ] = schedule_definition.schedule.hyperepochs[hyperepoch_name].resources

        # Check if each specifier exists in hyperepoch
        for specifier in resources:
            if specifier not in existing_resources:
                raise ValueError(
                    f"Cannot fid resource specifier '{specifier}' in hyperepoch '{hyperepoch_name}'"
                )

        undo_context_resources = {}
        for specifier in resources:
            undo_context_resources[specifier] = existing_resources[specifier].passes[:]
            existing_resources.pop(specifier)

        self._self_is_modified = True

        return UndoContext(self, schedule_name, hyperepoch_name, undo_context_resources)

    @object_model_channel.pair_self
    def update_process_name(self, name: str, new_name: str) -> UndoContext:
        """Update a process name(undo-able).

        @param name        target process name.
        @param new_name    a new name to replace with.

        """
        if name not in self._processes:
            raise ValueError(f"Target process name '{name}' does not exist.")
        if new_name in self._processes:
            raise ValueError(f"New process name '{new_name}' already exists.")
        process = self._processes.pop(name)
        process.name = new_name
        self._processes[new_name] = process

        self._self_is_modified = True

        return UndoContext(self, new_name, name)

    def insert_process(
        self, process_def: ProcessDefinition, loader: Optional[DescriptorLoader] = None
    ) -> None:
        """Insert a process (undo-able)."""
        process_subcomponents: List[Component] = []
        if process_def.subcomponents is not None:
            for _name in process_def.subcomponents:
                subcomponent = self.get_component(_name)
                if subcomponent is not None:
                    process_subcomponents.append(subcomponent)
                else:
                    raise ValueError(
                        f"Can not locate subcomponent: {_name}. "
                        + f"Available subcomponents: {self.subcomponents}"
                    )

        process = Process.from_descriptor(
            desc=process_def,
            subcomponent_instances=process_subcomponents
            if process_subcomponents
            else None,
            loader=loader,
        )

        self.__insert_process_impl(process)

    def remove_process(self, name: str) -> None:
        """Remove a process with a given name (undo-able)."""
        self.__remove_process_impl(name)

    @object_model_channel.pair
    def __insert_process_impl(self, process: Process) -> UndoContext:
        """Insert a process impl (undo-able)."""
        if process.name in self._processes:
            raise ValueError(f"Process name '{process.name}' already exists.")

        self._processes[process.name] = process

        self._self_is_modified = True

        return UndoContext(self, process.name)

    @__insert_process_impl.pair
    def __remove_process_impl(self, name: str) -> UndoContext:
        """Remove a process with a given name impl (undo-able)."""
        if name not in self._processes:
            raise ValueError(f"Process name '{name}' not found.")
        process = self._processes.pop(name)

        self._self_is_modified = True

        return UndoContext(self, process)

    @object_model_channel.pair
    def insert_process_arg(
        self, process_name: str, arg: str, arg_value: CMDArgumentValueType
    ) -> UndoContext:
        """insert a process argument (undo-able)."""

        if process_name not in self._processes:
            raise ValueError(f"Process name '{process_name}' not found.")
        if self._processes[process_name].argv is not None:
            if arg in cast(dict, self._processes[process_name].argv).keys():
                raise ValueError(f"Input argument name '{arg}' already exists.")
        if len(arg) == 0:
            raise ValueError(f"Input argument name is empty.")

        dst = self._processes[process_name].argv
        if dst is None:
            dst = {arg: arg_value}
        else:
            dst[arg] = arg_value
        self._processes[process_name].argv = dst

        self._self_is_modified = True

        return UndoContext(self, process_name, arg)

    @insert_process_arg.pair
    def remove_process_arg(self, process_name: str, arg: str) -> UndoContext:
        """remove a process argument (undo-able)."""

        if process_name not in self._processes:
            raise ValueError(f"Process name '{process_name}' not found.")
        if self._processes[process_name].argv is not None:
            if arg not in cast(dict, self._processes[process_name].argv).keys():
                raise ValueError(f"Input argument name '{arg}' dose not exist.")
        if len(arg) == 0:
            raise ValueError(f"Input argument name is empty.")

        dst = self._processes[process_name].argv
        assert dst is not None
        removed_arg_value = dst.pop(arg)
        if len(dst) == 0:
            self._processes[process_name].argv = None
        else:
            self._processes[process_name].argv = dst

        self._self_is_modified = True

        return UndoContext(self, process_name, arg, removed_arg_value)

    def insert_process_subcomponent(
        self, process_name: str, content: List[str], insert_at: Optional[int] = None
    ) -> None:
        """insert some process subcomponents."""
        if process_name not in self._processes:
            raise ValueError(f"Process name '{process_name}' not found.")

        process_subcomponents: List[Component] = []
        for _name in content:
            subcomponent = self.get_component(_name)
            if subcomponent is not None:
                process_subcomponents.append(subcomponent)
            else:
                raise ValueError(f"Can not locate subcomponent: {_name}.")

        if self._processes[process_name].subcomponents is None:
            self._processes[process_name].subcomponents = []
        src = self._processes[process_name].subcomponents

        self.__insert_into_list(src, process_subcomponents, insert_at, True)

    def remove_process_subcomponent(
        self, process_name: str, content: List[str]
    ) -> None:
        """remove some process subcomponents."""
        if process_name not in self._processes:
            raise ValueError(f"Process name '{process_name}' not found.")
        if self._processes[process_name].subcomponents is None:
            raise ValueError(f"Process: '{process_name}' has no subcomponent.")

        for subcomponent_name in content:
            subcomponent = self.get_component(subcomponent_name)
            if subcomponent is None:
                raise ValueError(f"Subcomponent: '{subcomponent_name}' not found.")
            if subcomponent in cast(list, self._processes[process_name].subcomponents):
                src = self._processes[process_name].subcomponents
                ind = cast(list, src).index(subcomponent)
                self.__remove_from_list(src, ind)

        if len(cast(list, self._processes[process_name].subcomponents)) == 0:
            self._processes[process_name].subcomponents = None

    @object_model_channel.pair
    def __insert_into_list(
        self,
        src: List,
        content: List,
        insert_at: Optional[int] = None,
        remove_duplicate: bool = False,
    ) -> UndoContext:
        """insert some contents into a list (undo-able)."""
        if src is None:
            raise ValueError("src can not be None.")
        if content is None or len(content) == 0:
            return UndoContext.NO_CHANGE

        if remove_duplicate:
            identical_content = []
            for obj in content:
                if src is not None:
                    if obj not in identical_content and obj not in src:
                        identical_content.append(obj)
                else:
                    if obj not in identical_content:
                        identical_content.append(obj)
            content = identical_content
            if len(content) == 0:
                return UndoContext.NO_CHANGE

        inserted_at = 0
        if insert_at is None:
            inserted_at = len(src)
            # src = src + content
            for value in content:
                src.append(value)
        elif insert_at < 0 or insert_at >= len(src):
            warnings.warn(
                f"Index {insert_at} out of bound for src list', \
                appending to src list tail",
                RuntimeWarning,
            )
            inserted_at = len(src)
            # src = src + content
            for value in content:
                src.append(value)
        else:
            inserted_at = insert_at
            src[insert_at:insert_at] = content

        self._self_is_modified = True

        return UndoContext(self, src, inserted_at, len(content))

    @__insert_into_list.pair
    def __remove_from_list(
        self, src: List, start_at: int, length: int = 1
    ) -> UndoContext:
        """remove some contents into a list by index (undo-able)."""
        if length <= 0 or src is None:
            return UndoContext.NO_CHANGE
        if start_at < 0 or start_at >= len(src):
            warnings.warn(
                f"Index {start_at} out of bound for src list, \
                removing from src list tail",
                RuntimeWarning,
            )
            if len(src) <= length:
                start_at = 0
                length = len(src)
            else:
                start_at = len(src) - length

        if start_at + length > len(src):
            length = len(src) - start_at

        removal = []
        for _ in range(length):
            removal.append(src.pop(start_at))

        self._self_is_modified = True

        return UndoContext(self, src, removal, start_at)

    @object_model_channel.pair_self
    def update_epoch_name(
        self, schedule: str, hyperepoch: str, epoch_name: str, new_name: str
    ) -> UndoContext:
        """update a epoch's name (undo-able)."""
        if schedule not in self._stm_schedules:
            raise ValueError(f"Schedule name '{schedule}' does not exist.")
        if hyperepoch not in self._stm_schedules[schedule].schedule.hyperepochs:
            raise ValueError(f"Hyperepoch name '{hyperepoch}' does not exist.")
        if (
            epoch_name
            not in self._stm_schedules[schedule].schedule.hyperepochs[hyperepoch].epochs
        ):
            raise ValueError(f"Epoch name '{epoch_name}' does not exist.")

        dst = self._stm_schedules[schedule].schedule.hyperepochs[hyperepoch].epochs
        epoch = dst.pop(epoch_name)
        epoch.name = new_name
        dst[new_name] = epoch

        self._self_is_modified = True

        return UndoContext(self, schedule, hyperepoch, new_name, epoch_name)

    @object_model_channel.pair_self
    def update_epoch_period(
        self, schedule: str, hyperepoch: str, epoch_name: str, new_period: int
    ) -> UndoContext:
        """update a epoch's name (undo-able)."""
        if schedule not in self._stm_schedules:
            raise ValueError(f"Schedule name '{schedule}' does not exist.")
        if hyperepoch not in self._stm_schedules[schedule].schedule.hyperepochs:
            raise ValueError(f"Hyperepoch name '{hyperepoch}' does not exist.")
        if (
            epoch_name
            not in self._stm_schedules[schedule].schedule.hyperepochs[hyperepoch].epochs
        ):
            raise ValueError(f"Epoch name '{epoch_name}' does not exist.")

        dst = self._stm_schedules[schedule].schedule.hyperepochs[hyperepoch].epochs
        old_period = dst[epoch_name].period
        dst[epoch_name].period = new_period

        self._self_is_modified = True

        return UndoContext(self, schedule, hyperepoch, epoch_name, old_period)

    @object_model_channel.pair_self
    def update_epoch_frame(
        self, schedule: str, hyperepoch: str, epoch_name: str, new_frames: int
    ) -> UndoContext:
        """update a epoch's frame (undo-able)."""
        if schedule not in self._stm_schedules:
            raise ValueError(f"Schedule name '{schedule}' does not exist.")
        if hyperepoch not in self._stm_schedules[schedule].schedule.hyperepochs:
            raise ValueError(f"Hyperepoch name '{hyperepoch}' does not exist.")
        if (
            epoch_name
            not in self._stm_schedules[schedule].schedule.hyperepochs[hyperepoch].epochs
        ):
            raise ValueError(f"Epoch name '{epoch_name}' does not exist.")

        dst = self._stm_schedules[schedule].schedule.hyperepochs[hyperepoch].epochs
        old_frames = dst[epoch_name].frames
        dst[epoch_name].frames = new_frames

        self._self_is_modified = True

        return UndoContext(self, schedule, hyperepoch, epoch_name, old_frames)

    @object_model_channel.pair
    def insert_epoch(
        self, schedule: str, hyperepoch: str, epoch: EpochDefinition
    ) -> UndoContext:
        """insert a epoch (undo-able)."""
        if schedule not in self._stm_schedules:
            raise ValueError(f"Schedule name '{schedule}' does not exist.")
        if hyperepoch not in self._stm_schedules[schedule].schedule.hyperepochs:
            raise ValueError(f"Hyperepoch name '{hyperepoch}' does not exist.")
        if (
            epoch.name
            in self._stm_schedules[schedule].schedule.hyperepochs[hyperepoch].epochs
        ):
            raise ValueError(f"Epoch name '{epoch.name}' already exists.")

        dst = self._stm_schedules[schedule].schedule.hyperepochs[hyperepoch].epochs
        dst[epoch.name] = epoch

        self._self_is_modified = True

        return UndoContext(self, schedule, hyperepoch, epoch.name)

    @insert_epoch.pair
    def remove_epoch(
        self, schedule: str, hyperepoch: str, epoch_name: str
    ) -> UndoContext:
        """remove a epoch (undo-able)."""
        if schedule not in self._stm_schedules:
            raise ValueError(f"Schedule name '{schedule}' does not exist.")
        if hyperepoch not in self._stm_schedules[schedule].schedule.hyperepochs:
            raise ValueError(f"Hyperepoch name '{hyperepoch}' does not exist.")
        if (
            epoch_name
            not in self._stm_schedules[schedule].schedule.hyperepochs[hyperepoch].epochs
        ):
            raise ValueError(f"Epoch name '{epoch_name}' does not exist.")

        dst = self._stm_schedules[schedule].schedule.hyperepochs[hyperepoch].epochs
        removal = dst.pop(epoch_name)

        self._self_is_modified = True

        return UndoContext(self, schedule, hyperepoch, removal)

    def insert_epoch_passes(
        self,
        schedule: str,
        hyperepoch: str,
        epoch_name: str,
        content: List[str],
        pipeline_phase: int = 0,
        insert_at: Optional[int] = None,
    ) -> None:
        """insert some epoch passes (undo-able)."""
        if schedule not in self._stm_schedules:
            raise ValueError(f"Schedule name '{schedule}' does not exist.")
        if hyperepoch not in self._stm_schedules[schedule].schedule.hyperepochs:
            raise ValueError(f"Hyperepoch name '{hyperepoch}' does not exist.")
        if (
            epoch_name
            not in self._stm_schedules[schedule].schedule.hyperepochs[hyperepoch].epochs
        ):
            raise ValueError(f"Epoch name '{epoch_name}' does not exist.")
        if pipeline_phase < 0 or pipeline_phase >= len(
            self._stm_schedules[schedule]
            .schedule.hyperepochs[hyperepoch]
            .epochs[epoch_name]
            .passes
        ):
            raise ValueError(f"pipeline_phase '{pipeline_phase}' does not exist.")

        src = (
            self._stm_schedules[schedule]
            .schedule.hyperepochs[hyperepoch]
            .epochs[epoch_name]
            .passes[pipeline_phase]
        )
        self.__insert_into_list(
            src=src, content=content, insert_at=insert_at, remove_duplicate=True
        )

    def remove_epoch_passes(
        self, schedule: str, hyperepoch: str, epoch_name: str, content: List[str]
    ) -> None:
        """remove some epoch passes (undo-able)."""
        if schedule not in self._stm_schedules:
            raise ValueError(f"Schedule name '{schedule}' does not exist.")
        if hyperepoch not in self._stm_schedules[schedule].schedule.hyperepochs:
            raise ValueError(f"Hyperepoch name '{hyperepoch}' does not exist.")
        if (
            epoch_name
            not in self._stm_schedules[schedule].schedule.hyperepochs[hyperepoch].epochs
        ):
            raise ValueError(f"Epoch name '{epoch_name}' does not exist.")

        src = (
            self._stm_schedules[schedule]
            .schedule.hyperepochs[hyperepoch]
            .epochs[epoch_name]
            .passes
        )
        for value in content:
            for pass_list in src:
                if value in pass_list:
                    ind = pass_list.index(value)
                    self.__remove_from_list(src=pass_list, start_at=ind, length=1)

    @object_model_channel.pair
    def insert_hyperepoch(
        self, schedule: str, hyperepoch: HyperepochDefinition
    ) -> UndoContext:
        """Insert a hyperepoch (undo-able) (undo-able)."""
        if schedule not in self._stm_schedules:
            raise ValueError(f"Schedule name '{schedule}' does not exists.")
        if hyperepoch.name in self._stm_schedules[schedule].schedule.hyperepochs:
            raise ValueError(f"Hyperepoch name '{hyperepoch.name}' already exists.")

        dst = self._stm_schedules[schedule].schedule.hyperepochs
        dst[hyperepoch.name] = hyperepoch

        self._self_is_modified = True

        return UndoContext(self, schedule, hyperepoch.name)

    @insert_hyperepoch.pair
    def remove_hyperepoch(self, schedule: str, hyperepoch_name: str) -> UndoContext:
        """remove a hyperepoch (undo-able)."""
        if schedule not in self._stm_schedules:
            raise ValueError(f"Schedule name '{schedule}' does not exists.")
        if hyperepoch_name not in self._stm_schedules[schedule].schedule.hyperepochs:
            raise ValueError(f"Hyperepoch name '{hyperepoch_name}' does not exists.")

        dst = self._stm_schedules[schedule].schedule.hyperepochs
        hyperepoch = dst.pop(hyperepoch_name)

        self._self_is_modified = True

        return UndoContext(self, schedule, hyperepoch)

    @object_model_channel.pair
    def insert_pass_dependencies(
        self, schedule: str, pass_def: PassDependencyDefinition
    ) -> UndoContext:
        """insert pass dependencies (undo-able)."""
        if schedule not in self._stm_schedules:
            raise ValueError(f"Schedule name '{schedule}' does not exist.")
        if (
            pass_def.pass_name
            in self._stm_schedules[schedule].schedule.passDependencies
        ):
            raise ValueError(f"Pass name '{pass_def.pass_name}' already exists.")

        dst = self._stm_schedules[schedule].schedule.passDependencies
        dst[pass_def.pass_name] = pass_def

        self._self_is_modified = True

        return UndoContext(self, schedule, pass_def.pass_name)

    @insert_pass_dependencies.pair
    def remove_pass_dependencies(self, schedule: str, pass_name: str) -> UndoContext:
        """remove pass dependencies (undo-able)."""
        if schedule not in self._stm_schedules:
            raise ValueError(f"Schedule name '{schedule}' does not exist.")
        if pass_name not in self._stm_schedules[schedule].schedule.passDependencies:
            raise ValueError(f"Pass name '{pass_name}' does not exist.")

        dst = self._stm_schedules[schedule].schedule.passDependencies
        removal = dst.pop(pass_name)

        self._self_is_modified = True

        return UndoContext(self, schedule, removal)

    @object_model_channel.pair
    def insert_stm_external_runnable(
        self, runnable_def: STMExternalRunnableDefinition
    ) -> UndoContext:
        """insert a stm external runnable (undo-able)."""
        if runnable_def.name in self._stm_external_runnables:
            raise ValueError(f"Runnable name '{runnable_def.name}' already exists.")
        self._stm_external_runnables[runnable_def.name] = runnable_def

        self._self_is_modified = True

        return UndoContext(self, runnable_def.name)

    @insert_stm_external_runnable.pair
    def remove_stm_external_runnable(self, runnable_name: str) -> UndoContext:
        """remove a stm external runnable (undo-able)."""
        if runnable_name not in self._stm_external_runnables:
            raise ValueError(f"Runnable name '{runnable_name}' does not exist.")
        runnable = self._stm_external_runnables.pop(runnable_name)

        self._self_is_modified = True

        return UndoContext(self, runnable)

    @object_model_channel.pair_self
    def update_stm_external_runnable_name(
        self, runnable_name: str, new_name: str
    ) -> UndoContext:
        """update a stm external runnable's name (undo-able)."""
        if runnable_name not in self._stm_external_runnables:
            raise ValueError(f"Runnable name '{runnable_name}' does not exist.")
        if new_name in self._stm_external_runnables:
            raise ValueError(f"New runnable name '{new_name}' already exists.")
        runnable = self._stm_external_runnables.pop(runnable_name)
        runnable.name = new_name
        self._stm_external_runnables[new_name] = runnable

        self._self_is_modified = True

        return UndoContext(self, new_name, runnable_name)

    @object_model_channel.pair_self
    def update_stm_external_runnable_wcet(
        self, runnable_name: str, new_wcet: int
    ) -> UndoContext:
        """update a stm external runnable's wcet (undo-able)."""
        if runnable_name not in self._stm_external_runnables:
            raise ValueError(f"Runnable name '{runnable_name}' does not exist.")
        old_wcet = self._stm_external_runnables[runnable_name].wcet
        self._stm_external_runnables[runnable_name].wcet = new_wcet

        self._self_is_modified = True

        return UndoContext(self, runnable_name, old_wcet)

    def insert_stm_ext_runnable_proc_types(
        self, runnable_name: str, content: List[str], insert_at: Optional[int] = None
    ) -> None:
        """insert some stm external runnable's processor types (undo-able)."""
        if runnable_name not in self._stm_external_runnables:
            raise ValueError(f"Runnable name '{runnable_name}' does not exist.")

        src = self._stm_external_runnables[runnable_name].processorTypes
        self.__insert_into_list(
            src=src, content=content, insert_at=insert_at, remove_duplicate=True
        )

    def remove_stm_ext_runnable_proc_types(
        self, runnable_name: str, content: List[str]
    ) -> None:
        """remove some stm external runnable's processor types (undo-able)."""
        if runnable_name not in self._stm_external_runnables:
            raise ValueError(f"Runnable name '{runnable_name}' does not exist.")

        src = self._stm_external_runnables[runnable_name].processorTypes
        for value in content:
            if value in src:
                ind = src.index(value)
                self.__remove_from_list(src=src, start_at=ind, length=1)

    def insert_stm_ext_runnable_pass_dependencies(
        self, runnable_name: str, content: List[str], insert_at: Optional[int] = None
    ) -> None:
        """insert pass dependencies for a stm external runnable(undo-able)."""
        if runnable_name not in self._stm_external_runnables:
            raise ValueError(f"Runnable name '{runnable_name}' does not exist.")

        src = self._stm_external_runnables[runnable_name].passDependencies
        self.__insert_into_list(
            src=src, content=content, insert_at=insert_at, remove_duplicate=True
        )

    def remove_stm_ext_runnable_pass_dependencies(
        self, runnable_name: str, content: List[str]
    ) -> None:
        """remove pass dependencies for a stm external runnable(undo-able)."""
        if runnable_name not in self._stm_external_runnables:
            raise ValueError(f"Runnable name '{runnable_name}' does not exist.")

        src = self._stm_external_runnables[runnable_name].passDependencies
        for value in content:
            if value in src:
                ind = src.index(value)
                self.__remove_from_list(src=src, start_at=ind, length=1)

    @object_model_channel.pair
    def insert_schedule(self, schedule: ScheduleDefinition) -> UndoContext:
        """insert a schedule (undo-able)."""
        if schedule.name in self._stm_schedules:
            raise ValueError(f"Schedule name '{schedule.name}' already exists.")

        self._stm_schedules[schedule.name] = STMSchedule(schedule)

        self._self_is_modified = True

        return UndoContext(self, schedule.name)

    @insert_schedule.pair
    def remove_schedule(self, schedule_name: str) -> UndoContext:
        """remove a schedule (undo-able)."""
        if schedule_name not in self._stm_schedules:
            raise ValueError(f"Schedule name '{schedule_name}' does not exist.")

        schedule = self._stm_schedules.pop(schedule_name).schedule

        self._self_is_modified = True

        return UndoContext(self, schedule)

    @object_model_channel.pair_self
    def update_schedule_name(self, schedule_name: str, new_name: str) -> UndoContext:
        """update a schedule's name (undo-able)."""
        if schedule_name not in self._stm_schedules:
            raise ValueError(f"Schedule name '{schedule_name}' does not exist.")
        if new_name in self._stm_schedules:
            raise ValueError(f"Schedule name '{new_name}' already exists.")

        schedule = self._stm_schedules.pop(schedule_name)
        schedule.schedule.name = new_name
        self._stm_schedules[new_name] = schedule

        self._self_is_modified = True

        return UndoContext(self, new_name, schedule_name)

    @object_model_channel.pair_self
    def update_schedule_wcet(self, schedule_name: str, new_wcet: str) -> UndoContext:
        """update a schedule's wcet (undo-able)."""
        if schedule_name not in self._stm_schedules:
            raise ValueError(f"Schedule name '{schedule_name}' does not exist.")

        schedule = self._stm_schedules[schedule_name]
        old_wcet = str(schedule.schedule.wcet)
        schedule.schedule.wcet = Path(new_wcet)

        self._self_is_modified = True

        return UndoContext(self, schedule_name, old_wcet)

    @object_model_channel.pair
    def insert_state(self, state: StateDefinition) -> UndoContext:
        """insert a state (undo-able)."""
        if state.name in self._states:
            raise ValueError(f"State name '{state.name}' already exists.")

        self._states[state.name] = State(state)

        self._self_is_modified = True

        return UndoContext(self, state.name)

    @insert_state.pair
    def remove_state(self, state_name: str) -> UndoContext:
        """remove a state (undo-able)."""
        if state_name not in self._states:
            raise ValueError(f"State name '{state_name}' does not exists.")

        state = self._states.pop(state_name).state

        self._self_is_modified = True

        return UndoContext(self, state)

    @object_model_channel.pair_self
    def update_state_name(self, state_name: str, new_name: str) -> UndoContext:
        """update a state's name."""
        if state_name not in self._states:
            raise ValueError(f"State name '{state_name}' does not exists.")

        state = self._states.pop(state_name)
        state.state.name = new_name
        self._states[new_name] = state

        self._self_is_modified = True

        return UndoContext(self, new_name, state_name)

    @object_model_channel.pair_self
    def update_state_schedule(self, state_name: str, new_schedule: str) -> UndoContext:
        """update a state's schedule."""
        if state_name not in self._states:
            raise ValueError(f"State name '{state_name}' does not exists.")

        state = self._states[state_name].state
        old_schedule = state.stm_schedule_key
        state.stm_schedule_key = new_schedule

        self._self_is_modified = True

        return UndoContext(self, state_name, old_schedule)

    @object_model_channel.pair_self
    def select_default_state(
        self, state_name: str, new_default_state: bool
    ) -> UndoContext:
        """update a state's default state."""
        if state_name not in self._states:
            raise ValueError(f"State name '{state_name}' does not exists.")

        state = self._states[state_name].state
        old_default_state = state.is_default
        state.is_default = new_default_state

        self._self_is_modified = True

        return UndoContext(self, state_name, old_default_state)

    @object_model_channel.pair_self
    def update_pass_dependencies(
        self, schedule: str, pass_dependencies_new: dict
    ) -> UndoContext:
        """update pass dependencies (undo-able)."""
        if schedule not in self._stm_schedules:
            raise ValueError(f"Schedule name '{schedule}' does not exist.")
        pass_dependencies_old = self._stm_schedules[schedule].schedule.passDependencies

        if pass_dependencies_new is pass_dependencies_old:
            return UndoContext.NO_CHANGE
        ret = UndoContext(self, schedule, pass_dependencies_old)
        self._stm_schedules[schedule].schedule.passDependencies = pass_dependencies_new
        self._self_is_modified = True
        return ret

    @object_model_channel.pair_self
    def set_sensor_mapping_lookups(
        self, new_sensor_mapping_lookups: List[str]
    ) -> UndoContext:
        """set sensor mapping lookups(undo-able)."""
        assert isinstance(new_sensor_mapping_lookups, list)
        assert all(isinstance(e, str) for e in new_sensor_mapping_lookups)

        undo_context = []
        for path in self.sensor_mapping_lookups:
            undo_context.append(str(path))

        self.sensor_mapping_lookups.clear()
        for path_str in new_sensor_mapping_lookups:
            self.sensor_mapping_lookups.append(Path(path_str))

        self._self_is_modified = True

        return UndoContext(self, undo_context)

    @object_model_channel.pair_self
    def set_required_sensors(
        self, new_required_sensors: Optional[Path] = None
    ) -> UndoContext:
        """set required sensors(undo-able)."""
        if self.required_sensors == new_required_sensors:
            return UndoContext.NO_CHANGE

        orig_required_sensors = self.required_sensors
        self._required_sensors = new_required_sensors
        self._self_is_modified = True
        return UndoContext(self, orig_required_sensors)

    @object_model_channel.pair_self
    def set_rcs_enabled(self, process_name: str, enable: bool) -> UndoContext:
        """Set rcs enable or disable."""
        if process_name not in self._processes:
            raise ValueError(f"Can't disable as the specified process doesn't exist.")

        services = self._processes[process_name].services
        if services is None or services.rcs is None:
            raise ValueError(
                f"Can't insert as the RoadCastService is not defined for the process."
            )
        if enable == services.rcs.enabled:
            return UndoContext.NO_CHANGE
        origin_enabled = services.rcs.enabled
        services.rcs.enabled = enable
        self._self_is_modified = True
        return UndoContext(self, process_name, origin_enabled)

    @object_model_channel.pair
    def insert_rcs_port(
        self, process_name: str, port_name: str, port_value: dict
    ) -> UndoContext:
        """insert RoadCastService port (undo-able)."""
        if process_name not in self._processes:
            raise ValueError(f"Can't insert as the specified process doesn't exist.")

        services = self._processes[process_name].services
        if services is None or services.rcs is None:
            raise ValueError(
                f"Can't insert as the RoadCastService is not defined for the process."
            )
        if port_name in services.rcs.ports:
            raise ValueError(f"Port name '{port_name}' already exist.")

        services.rcs.ports[port_name] = SignalProducerPort.from_json_data(
            name=port_name, content=port_value
        )

        self._self_is_modified = True

        return UndoContext(self, process_name, port_name)

    @insert_rcs_port.pair
    def remove_rcs_port(self, process_name: str, port_name: str) -> UndoContext:
        """remove RoadCastService port (undo-able)."""
        if process_name not in self._processes:
            raise ValueError(f"Can't remove as the specified process doesn't exist.")

        services = self._processes[process_name].services
        if services is None or services.rcs is None:
            raise ValueError(
                f"Can't remove as the RoadCastService is not defined for the process."
            )
        if port_name not in services.rcs.ports:
            raise ValueError(f"Port name '{port_name}' does not exist.")

        removal = services.rcs.ports.pop(port_name)

        self._self_is_modified = True

        return UndoContext(self, process_name, port_name, removal)

    @object_model_channel.pair_self
    def update_rcs_serialize_mask(
        self, process_name: str, channel_id: str, serialize_mask: dict
    ) -> UndoContext:
        """update RoadCastService serialize mask for the specified channel."""
        if process_name not in self._processes:
            raise ValueError(f"Can't update as the specified process doesn't exist.")

        services = self._processes[process_name].services
        if services is None or services.rcs is None:
            raise ValueError(
                f"Can't update as the RoadCastService is not defined for the process."
            )
        if channel_id not in services.rcs.params.channel_params:
            raise ValueError(f"Channel id '{channel_id}' does not exists.")

        original_mask = {}
        ref = services.rcs.params.channel_params[channel_id].serial_mask
        for mask_name, mask_value in serialize_mask.items():
            if not isinstance(mask_name, str) or not isinstance(mask_value, bool):
                raise ValueError(f"Invalid serialize mask.")
            original_mask[mask_name] = ref[mask_name] if mask_name in ref else False
            ref[mask_name] = mask_value

        self._self_is_modified = True

        return UndoContext(self, process_name, channel_id, original_mask)
