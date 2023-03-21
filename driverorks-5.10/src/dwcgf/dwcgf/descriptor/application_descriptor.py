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
"""Data structures for Descriptor Reference Graph."""
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from .component_descriptor import ParameterDefinition
from .component_descriptor import PortDefinition
from .descriptor_factory import DescriptorFactory
from .descriptor_factory import DescriptorType
from .graphlet_descriptor import ConnectionDefinition
from .graphlet_descriptor import GraphletDescriptor
from .graphlet_descriptor import SubcomponentDefinition
from .schedule_definition import ScheduleDefinition


class StateDefinition:
    """Class for states descriptor."""

    def __init__(self, *, name: str, stm_scheudle_key: str, default: bool):
        """An entry in states section.

        @param name             name of the state
        @param stm_schedule_key the schedule name with which this state associate
        @param default          if this state is the "default" state
        """
        self._name = name
        self._stm_schedule_key = stm_scheudle_key
        self._default = default

    @property
    def name(self) -> str:
        """Return name of this state."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set the name of this state."""
        self._name = value

    @property
    def stm_schedule_key(self) -> str:
        """Return the schedule name with which this state associate."""
        return self._stm_schedule_key

    @stm_schedule_key.setter
    def stm_schedule_key(self, value: str) -> None:
        """Set the schedule name with which this state associate."""
        self._stm_schedule_key = value

    @property
    def is_default(self) -> bool:
        """Indicates if this state is "default" state."""
        return self._default

    @is_default.setter
    def is_default(self, value: bool) -> None:
        """Set the state of this state."""
        self._default = value

    def to_json_data(self) -> OrderedDict:
        """Dump StateDefinition to JSON data."""
        state_json: OrderedDict = OrderedDict()
        state_json["stmScheduleKey"] = self.stm_schedule_key
        if self.is_default:
            state_json["default"] = True

        return state_json

    def __eq__(self, __o: object) -> bool:
        """Compare the two instance of StateDefinition."""
        if not isinstance(__o, StateDefinition):
            return NotImplemented
        return self.to_json_data() == __o.to_json_data() and self.name == __o.name


class STMExternalRunnableDefinition:
    """Class for stm external runnable descriptor."""

    def __init__(
        self,
        *,
        name: str,
        wcet: int,
        processorTypes: List[str],
        passDependencies: Optional[List[str]] = None,
    ):
        """An entry in stm external runnable section."""
        self._name = name
        self._wcet = wcet
        self._processorTypes = processorTypes
        self._passDependencies = (
            passDependencies if passDependencies is not None else []
        )

    @property
    def name(self) -> str:
        """Return name of the stm external runnable."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set the name of the stm external runnable."""
        self._name = value

    @property
    def wcet(self) -> int:
        """Return wcet of the stm external runnable."""
        return self._wcet

    @wcet.setter
    def wcet(self, value: int) -> None:
        """Set the wcet of the stm external runnable."""
        self._wcet = value

    @property
    def processorTypes(self) -> List[str]:
        """Return processorTypes of the stm external runnable."""
        return self._processorTypes

    @property
    def passDependencies(self) -> List[str]:
        """Return passDependencies of the stm external runnable."""
        return self._passDependencies

    @classmethod
    def from_json_data(
        cls, name: str, content: Dict
    ) -> "STMExternalRunnableDefinition":
        """Load from JSON data."""
        return STMExternalRunnableDefinition(
            name=name,
            wcet=content.get("wcet", None),
            processorTypes=content.get("processorTypes", []),
            passDependencies=content.get("passDependencies", []),
        )

    def to_json_data(self) -> OrderedDict:
        """Dump to JSON data."""
        runnable_json: OrderedDict = OrderedDict()
        runnable_json["wcet"] = self.wcet
        runnable_json["processorTypes"] = self.processorTypes
        if len(self.passDependencies) != 0:
            runnable_json["passDependencies"] = self.passDependencies

        return runnable_json

    def __eq__(self, __o: object) -> bool:
        """Compare the two instance of STMExternalRunnableDefinition."""
        if not isinstance(__o, STMExternalRunnableDefinition):
            return NotImplemented
        return self.to_json_data() == __o.to_json_data() and self.name == __o.name


CMDArgumentValueType = Union[bool, str, List[str]]


class ProcessDefinition:
    """Class for processes descriptor."""

    def __init__(
        self,
        *,
        name: str,
        executable: str,
        run_on: str,
        log_spec: Optional[str],
        argv: Optional[Dict[str, CMDArgumentValueType]] = None,
        subcomponents: Optional[List[str]] = None,
        extra_info: Optional[Dict] = None,
    ):
        """An entry in processes section."""
        self._name = name
        self._executable = executable
        self._run_on = run_on
        self._log_spec = log_spec
        self._argv = argv
        self._subcomponents = subcomponents

        # this is for restore the extra attributes for the process
        # TODO(xingpengd) this is against schema, only for legacy
        # behavior, need to clean up later
        self._extra_info = extra_info

    @property
    def name(self) -> str:
        """Return name of the process."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set the name of the process."""
        self._name = value

    @property
    def executable(self) -> str:
        """Return executable name."""
        return self._executable

    @property
    def run_on(self) -> str:
        """Return the machine on which this process running."""
        return self._run_on

    @property
    def log_spec(self) -> Optional[str]:
        """Return the log spec of this process."""
        return self._log_spec

    @property
    def argv(self) -> Optional[Dict[str, CMDArgumentValueType]]:
        """Return the argument list of this process."""
        return self._argv

    @argv.setter
    def argv(self, value: Dict[str, CMDArgumentValueType]) -> None:
        """Set the argument list of this process."""
        self._argv = value

    @property
    def subcomponents(self) -> Optional[List[str]]:
        """Return the subcomponents for this process."""
        return self._subcomponents

    @subcomponents.setter
    def subcomponents(self, value: List[str]) -> None:
        """Set the subcomponents for this process."""
        self._subcomponents = value

    @property
    def extra_info(self) -> Optional[Dict]:
        """Return extra attributes specified in process descriptor."""
        return self._extra_info

    def to_json_data(self) -> OrderedDict:
        """Dump ProcessDefinition to JSON data."""
        process_json: OrderedDict = OrderedDict()
        process_json["executable"] = self.executable
        process_json["runOn"] = self.run_on
        if self.log_spec is not None:
            process_json["logSpec"] = self.log_spec
        if self.argv is not None:
            process_json["argv"] = self.argv
        if self.subcomponents is not None:
            process_json["subcomponents"] = self.subcomponents
        if self.extra_info is not None:
            for k, v in self.extra_info.items():
                process_json[k] = v

        return process_json

    def __eq__(self, __o: object) -> bool:
        """Compare the two instance of ProcessDefinition."""
        if not isinstance(__o, ProcessDefinition):
            return NotImplemented
        return self.to_json_data() == __o.to_json_data() and self.name == __o.name


@DescriptorFactory.register(DescriptorType.APPLICATION)
class ApplicationDescriptor(GraphletDescriptor):
    """class for app descriptors."""

    def __init__(
        self,
        file_path: Path,
        *,
        name: str,
        version: Optional[int] = None,
        log_spec: Optional[str],
        parameters: Optional[Dict[str, ParameterDefinition]] = None,
        input_ports: Optional[Dict[str, PortDefinition]] = None,
        output_ports: Optional[Dict[str, PortDefinition]] = None,
        subcomponents: Optional[Dict[str, SubcomponentDefinition]] = None,
        connections: Optional[List[ConnectionDefinition]] = None,
        processes: Optional[Dict[str, ProcessDefinition]] = None,
        states: Optional[Dict[str, StateDefinition]] = None,
        stm_schedules: Optional[Dict[str, ScheduleDefinition]] = None,
        rcs: Optional[Dict[str, Dict]] = None,
        framework_pass: Optional[Dict[str, Dict]] = None,
        stm_external_runnables: Optional[
            Dict[str, STMExternalRunnableDefinition]
        ] = None,
        required_sensors: Optional[Path] = None,
        multi_machine_required_sensors: Optional[Dict[str, Path]] = None,
        sensor_mapping_lookups: Optional[List[Path]] = None,
        extra_info: Optional[Path] = None,
        comment: Optional[str] = None,
    ):
        """Create an ApplicationDescriptor instance.

        @param file_path path of this application descriptor file
        """
        super().__init__(
            file_path,
            name=name,
            parameters=parameters,
            input_ports=input_ports,
            output_ports=output_ports,
            subcomponents=subcomponents,
            connections=connections,
            comment=comment,
        )

        if multi_machine_required_sensors is not None:
            assert isinstance(multi_machine_required_sensors, dict)
            assert all(
                isinstance(k, str) for k in multi_machine_required_sensors.keys()
            )
            assert all(
                isinstance(v, Path) for v in multi_machine_required_sensors.values()
            )

        if sensor_mapping_lookups is not None:
            assert isinstance(sensor_mapping_lookups, list)
            assert all(isinstance(e, Path) for e in sensor_mapping_lookups)

        self._log_spec = log_spec if log_spec is not None else "console"
        self._processes = processes if processes is not None else {}
        self._states = states if states is not None else {}
        self._stm_schedules = stm_schedules if stm_schedules is not None else {}
        self._rcs = rcs
        self._framework_pass = framework_pass
        self._stm_external_runnables = (
            stm_external_runnables if stm_external_runnables is not None else {}
        )
        self._required_sensors = required_sensors
        self._multi_machine_required_sensors = (
            multi_machine_required_sensors
            if multi_machine_required_sensors is not None
            else {}
        )
        self._sensor_mapping_lookups = (
            sensor_mapping_lookups if sensor_mapping_lookups is not None else []
        )
        self._extra_info = extra_info
        self._version = version

    @property
    def version(self) -> Optional[int]:
        """Return version of this app descriptor."""
        return self._version

    @property
    def log_spec(self) -> str:
        """Return log spec of this app."""
        return self._log_spec

    @property
    def required_sensors(self) -> Optional[Path]:
        """Return relative path of the required sensors."""
        return self._required_sensors

    @property
    def multi_machine_required_sensors(self) -> Dict[str, Path]:
        """Return relative path of multi machine required sensors."""
        return self._multi_machine_required_sensors

    @property
    def sensor_mapping_lookups(self) -> List[Path]:
        """Return relative path(s) of sensor mapping lookups."""
        return self._sensor_mapping_lookups

    @property
    def extra_info(self) -> Optional[Path]:
        """Return relative path of the extra info."""
        return self._extra_info

    @property
    def states(self) -> Dict[str, StateDefinition]:
        """Return states of the app."""
        return self._states

    @property
    def processes(self) -> Dict[str, ProcessDefinition]:
        """Return processes of the app."""
        return self._processes

    @property
    def stm_schedules(self) -> Dict[str, ScheduleDefinition]:
        """Return schedules of the app."""
        return self._stm_schedules

    @property
    def rcs(self) -> Optional[Dict[str, Dict]]:
        """Return RoadCastService of the app."""
        return self._rcs

    @property
    def framework_pass(self) -> Optional[Dict[str, Dict]]:
        """Return frameworkPass of the app."""
        return self._framework_pass

    @property
    def stm_external_runnables(self) -> Dict[str, STMExternalRunnableDefinition]:
        """Return stm external runnables of the app."""
        return self._stm_external_runnables

    @property
    def referenced_descriptors(self) -> List[Path]:
        """Return the descriptor files referenced by this descriptor.

        Return value is an array of tuple where first element indicates the descriptor type
        and second element indicates the file path
        """
        ret = []
        for desc in self.subcomponents.values():
            ret.append(self.dirname / desc.component_type)

        if self.required_sensors:
            ret.append(self.dirname / self.required_sensors)

        for path in self.multi_machine_required_sensors.values():
            ret.append(self.dirname / path)

        if self.extra_info:
            ret.append(self.dirname / self.extra_info)

        return ret

    @classmethod
    def from_json_data(
        cls, content: Dict, path: Union[str, Path]
    ) -> "ApplicationDescriptor":
        """Create ApplicationDescriptor from JSON data."""

        path = Path(path)

        processes = {}
        for name, process_raw in content.get("processes", {}).items():
            process_extra_info = deepcopy(process_raw)
            process_extra_info.pop("executable", None)
            process_extra_info.pop("runOn", None)
            process_extra_info.pop("logSpec", None)
            process_extra_info.pop("argv", None)
            process_extra_info.pop("subcomponents", None)
            processes[name] = ProcessDefinition(
                name=name,
                executable=process_raw["executable"],
                run_on=process_raw["runOn"],
                log_spec=process_raw.get("logSpec", None),
                argv=process_raw.get("argv", None),
                subcomponents=process_raw.get("subcomponents", None),
                extra_info=process_extra_info,
            )

        states = {
            name: StateDefinition(
                name=name,
                stm_scheudle_key=state_raw["stmScheduleKey"],
                default=state_raw.get("default", False),
            )
            for name, state_raw in content.get("states", {}).items()
        }

        stm_schedules = {
            name: ScheduleDefinition.from_json_data(name, schedule)
            for name, schedule in content.get("stmSchedules", {}).items()
        }

        # RoadCastService
        rcs = content.get("RoadCastService", None)
        # frameworkPass
        framework_pass = content.get("frameworkPass", None)
        stm_external_runnables = {
            name: STMExternalRunnableDefinition.from_json_data(name, runnable)
            for name, runnable in content.get("stmExternalRunnables", {}).items()
        }

        required_sensors_path = content.get("requiredSensors", None)
        multi_machine_required_sensors: Dict[str, Path] = {
            name: Path(path)
            for name, path in content.get("multiMachineRequiredSensors", {}).items()
        }
        sensor_mapping_lookups: List[Path] = [
            Path(path) for path in content.get("sensorMappingLookups", [])
        ]
        extra_info_path = content.get("extraInfo", None)

        comp = super().from_json_data(content, path)

        return ApplicationDescriptor(
            path,
            name=comp.name,
            version=content.get("version", None),
            log_spec=content.get("logSpec"),
            parameters=comp.parameters,
            # input_ports=comp.input_ports,
            # output_ports=comp.output_ports,
            subcomponents=comp.subcomponents,
            connections=comp.connections,
            processes=processes,
            states=states,
            stm_schedules=stm_schedules,
            rcs=rcs,
            framework_pass=framework_pass,
            stm_external_runnables=stm_external_runnables,
            required_sensors=required_sensors_path,
            multi_machine_required_sensors=multi_machine_required_sensors,
            sensor_mapping_lookups=sensor_mapping_lookups,
            extra_info=extra_info_path,
            comment=comp.comment,
        )

    def to_json_data(self) -> OrderedDict:
        """Dump ApplicationDescriptor to JSON data."""

        def dump_states(states: Dict[str, StateDefinition]) -> OrderedDict:
            """Dumps states."""
            states_json: OrderedDict = OrderedDict()
            for k, v in states.items():
                states_json[k] = OrderedDict(stmScheduleKey=v.stm_schedule_key)
                if v.is_default:
                    states_json[k]["default"] = True
            return states_json

        def dump_processes(processes: Dict[str, ProcessDefinition]) -> OrderedDict:
            processes_json: OrderedDict = OrderedDict()
            for process_name in sorted(processes.keys()):
                processes_json[process_name] = processes[process_name].to_json_data()

            return processes_json

        graphlet_json = super().to_json_data()
        app_json: OrderedDict = OrderedDict()

        if self.comment is not None:
            app_json["comment"] = self.comment

        if self.version is not None:
            app_json["version"] = self.version

        app_json["name"] = graphlet_json["name"]
        app_json["logSpec"] = self.log_spec
        # app_json["inputPorts"] = graphlet_json["inputPorts"]
        # app_json["outputPorts"] = graphlet_json["outputPorts"]
        app_json["parameters"] = graphlet_json["parameters"]
        if self.required_sensors is not None:
            app_json["requiredSensors"] = str(self.required_sensors)
        if self.multi_machine_required_sensors:
            app_json["multiMachineRequiredSensors"] = {
                name: str(path)
                for name, path in self.multi_machine_required_sensors.items()
            }
        if self.sensor_mapping_lookups:
            app_json["sensorMappingLookups"] = [
                str(path) for path in self.sensor_mapping_lookups
            ]

        app_json["subcomponents"] = graphlet_json["subcomponents"]
        app_json["connections"] = graphlet_json["connections"]

        app_json["states"] = dump_states(self.states)
        app_json["stmSchedules"] = {
            schedule_name: schedule.to_json_data()
            for schedule_name, schedule in self.stm_schedules.items()
        }

        stm_external_runnables_json = {
            name: runnable.to_json_data()
            for name, runnable in self.stm_external_runnables.items()
        }
        if len(stm_external_runnables_json) != 0:
            app_json["stmExternalRunnables"] = stm_external_runnables_json

        app_json["processes"] = dump_processes(self.processes)
        if self.extra_info is not None:
            app_json["extraInfo"] = str(self.extra_info)

        # roadCastService
        if self.rcs is not None:
            app_json["RoadCastService"] = self.rcs
        # frameworkPass
        if self.framework_pass is not None:
            app_json["frameworkPass"] = self.framework_pass

        return app_json
