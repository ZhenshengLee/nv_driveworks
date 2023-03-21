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
import os
from pathlib import Path
from typing import Any
from typing import cast
from typing import Dict
from typing import List

from dwcgf.action import Action
from dwcgf.action import ActionAttribute
from dwcgf.action.action_factory import ActionFactory
from dwcgf.descriptor import ConnectionDefinition
from dwcgf.descriptor import ProcessDefinition
from dwcgf.json_merge_patch import merge
from dwcgf.object_model import Application
from dwcgf.object_model import Component
from dwcgf.object_model import Graphlet
from dwcgf.object_model import Node
from dwcgf.object_model import Port

####################################################################################################
# Single Process
####################################################################################################


@ActionFactory.register("single-process")
class SingleProcess(Action):
    """class for single-process action."""

    extra_attributes = (
        ActionAttribute(
            name="multipleMachines",
            is_required=False,
            attr_type=bool,
            default=False,
            description="Wheather merge processes by machine.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """Transform any application into single process app."""
        # TODO(xingpengd) to implement
        pass


####################################################################################################
# Devviz
####################################################################################################
@ActionFactory.register("update-devviz-parameters")
class UpdateDevvizParameters(Action):
    """class for update-devviz-parameters action."""

    extra_attributes = (
        ActionAttribute(
            name="newPatch",
            is_required=True,
            attr_type=dict,
            default={},
            description="Json patch to update.",
        ),
    )

    def transform_impl(self, target: Graphlet) -> None:
        """
        action implementation.

        Update arbitrary devvizComponents value with help of json merge patch api.
        Support both inserting new value and updating existing value.
        """
        devviz_components_new = deepcopy(target.devviz_components)
        newPatch: dict = self.get_attribute("newPatch")

        devviz_components_new = merge(devviz_components_new, newPatch)

        target.update_devviz_components(devviz_components_new)


@ActionFactory.register("add-devviz-parameters")
class AddDevvizParameters(Action):
    """class for add-devviz-parameters action."""

    extra_attributes = (
        ActionAttribute(
            name="module",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of target module.",
        ),
        ActionAttribute(
            name="params",
            is_required=True,
            attr_type=dict,
            default=[],
            description="Parameter names to add.",
        ),
    )

    def transform_impl(self, target: Graphlet) -> None:
        """
        action implementation.

        Note: if params exist duplicate keys,
        only the last key works, e.g.
        params = {"a":0, "a": 1, "a", 2} in devviz.trans.json
        here params = {"a": 2}
        """
        devviz_components_new = deepcopy(target.devviz_components)

        params = self.get_attribute("params")
        module_name = self.get_attribute("module")

        # check module is in devviz components
        if module_name not in devviz_components_new.keys():
            raise ValueError(f"devviz module: {module_name} does not exist!")
        module = devviz_components_new[module_name]
        module["params"] = merge(module["params"], params)

        target.update_devviz_components(devviz_components_new)


####################################################################################################
# Required Sensors and Sensor Mappings
####################################################################################################
@ActionFactory.register("insert-multi-machine-required-sensors")
class InsertMultiMachineRequiredSensors(Action):
    """class for insert-multi-machine-required-sensors(undo-able)."""

    extra_attributes = (
        ActionAttribute(
            name="newMultiMachineRequiredSensors",
            is_required=True,
            attr_type=dict,
            default={},
            description="New multi machine required sensors to be inserted.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        target.set_required_sensors(new_required_sensors=None)
        target.insert_multi_machine_required_sensors(
            new_multi_machine_required_sensors=self.get_attribute(
                "newMultiMachineRequiredSensors"
            )
        )


@ActionFactory.register("remove-multi-machine-required-sensors")
class RemoveMultiMachineRequiredSensors(Action):
    """class for remove-multi-machine-required-sensors."""

    extra_attributes = (
        ActionAttribute(
            name="multiMachineRequiredSensors",
            is_required=True,
            attr_type=list,
            default=[],
            description="Required sensors to be removed from multi machine required sensors.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        target.set_required_sensors(new_required_sensors=None)
        target.remove_multi_machine_required_sensors(
            multi_machine_required_sensors=self.get_attribute(
                "multiMachineRequiredSensors"
            )
        )


@ActionFactory.register("set-sensor-mapping-lookups")
class SetSensorMappingLookups(Action):
    """class for set-sensor-mapping-lookups(undo-able)."""

    extra_attributes = (
        ActionAttribute(
            name="newSensorMappingLookups",
            is_required=True,
            attr_type=list,
            default=[],
            description="New sensor mapping lookups to be updated.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        target.set_sensor_mapping_lookups(
            new_sensor_mapping_lookups=self.get_attribute("newSensorMappingLookups")
        )


@ActionFactory.register("remove-sensor-mapping-lookups")
class RemoveSensorMappingLookups(Action):
    """class for remove-sensor-mapping-lookups(undo-able)."""

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        target.set_sensor_mapping_lookups(new_sensor_mapping_lookups=[])


@ActionFactory.register("update-required-sensors")
class UpdateRequiredSensors(Action):
    """class for update-required-sensors(undo-able)."""

    extra_attributes = (
        ActionAttribute(
            name="newRequiredSensors",
            is_required=True,
            attr_type=str,
            default="",
            description="New required sensors to be updated.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        target.set_required_sensors(
            new_required_sensors=Path(
                cast(str, self.get_attribute("newRequiredSensors"))
            )
        )


####################################################################################################
# Extra Info
####################################################################################################
@ActionFactory.register("update-extra-info")
class UpdateExtraInfo(Action):
    """class for update-extra-info."""

    extra_attributes = (
        ActionAttribute(
            name="newExtraInfo",
            is_required=True,
            attr_type=str,
            default="",
            description="New extra info file path.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        target.extra_info = Path(cast(str, self.get_attribute("newExtraInfo")))


####################################################################################################
# RoadCast Service
####################################################################################################
@ActionFactory.register("disable-rcs")
class DisableRCS(Action):
    """class for disable-rcs action."""

    extra_attributes = (
        ActionAttribute(
            name="disable",
            is_required=False,
            attr_type=bool,
            default=True,
            description="If disable RCS.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""

        disable = self.get_attribute("disable")
        if disable:
            target.set_rcs_enable(False)
            target.framework_pass = {}


@ActionFactory.register("insert-rcs-ports")
class InsertRcsPorts(Action):
    """class for insert-rcs-ports action."""

    extra_attributes = (
        ActionAttribute(
            name="ports",
            is_required=True,
            attr_type=dict,
            default=[],
            description="Name of the target ports.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        ports = self.get_attribute("ports")
        if isinstance(ports, dict):
            for port_name, port_value in ports.items():
                target.insert_rcs_port(port_name=port_name, port_value=port_value)
        else:
            raise TypeError("ports should be a dict.")


@ActionFactory.register("remove-rcs-ports")
class RemoveRcsPorts(Action):
    """class for remove-rcs-ports action."""

    extra_attributes = (
        ActionAttribute(
            name="ports",
            is_required=True,
            attr_type=list,
            default=[],
            description="Name of the target ports.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        port_names = self.get_attribute("ports")
        if isinstance(port_names, list):
            for port_name in port_names:
                target.remove_rcs_port(port_name=port_name)
        else:
            raise TypeError("ports should be a list of str.")


@ActionFactory.register("update-rcs-serialize-mask")
class UpdateRcsSerializeMask(Action):
    """class for update-rcs-serialize-mask."""

    extra_attributes = (
        ActionAttribute(
            name="channelId",
            is_required=True,
            attr_type=str,
            default=[],
            description="Name of the target ports.",
        ),
        ActionAttribute(
            name="serializeMask",
            is_required=True,
            attr_type=dict,
            default=[],
            description="Name of the target ports.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        channel_id = self.get_attribute("channelId")
        serialize_mask = self.get_attribute("serializeMask")
        if isinstance(channel_id, str) and isinstance(serialize_mask, dict):
            target.update_rcs_serialize_mask(
                channel_id=channel_id, serialize_mask=serialize_mask
            )
        else:
            raise TypeError(
                "channelId should be a str and serializeMask should be a dict."
            )


####################################################################################################
# Pipelining
####################################################################################################
@ActionFactory.register("disable-pipelining")
class DisablePipelining(Action):
    """class for disable-pipelining action."""

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
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""

        schedule_names: List[str] = self.get_attribute("scheduleNames")
        hyperepoch_name: str = self.get_attribute("hyperepochName")
        epoch_name: str = self.get_attribute("epochName")
        for schedule_name in schedule_names:
            if schedule_name not in target.schedules:
                raise ValueError(f"Cannot find schedule name: '{schedule_name}'")
            schedule = target.schedules[schedule_name].schedule
            if hyperepoch_name not in schedule.hyperepochs:
                raise ValueError(f"Cannot find hyperepoch name '{hyperepoch_name}'")
            hyperepoch = schedule.hyperepochs[hyperepoch_name]
            if epoch_name not in hyperepoch.epochs:
                raise ValueError(f"Cannot find epoch name '{epoch_name}'")
            epoch = hyperepoch.epochs[epoch_name]

            if len(epoch.passes) > 1 and len(epoch.passes[1]) > 0:
                phase1_passes = deepcopy(epoch.passes[1])
                target.remove_epoch_passes(
                    schedule_name, hyperepoch_name, epoch_name, phase1_passes
                )
                target.insert_epoch_passes(
                    schedule_name, hyperepoch_name, epoch_name, phase1_passes, 0
                )

            if len(epoch.passes) > 2 and len(epoch.passes[2]) > 0:
                phase2_passes = deepcopy(epoch.passes[2])
                target.remove_epoch_passes(
                    schedule_name, hyperepoch_name, epoch_name, phase2_passes
                )
                target.insert_epoch_passes(
                    schedule_name, hyperepoch_name, epoch_name, phase2_passes, 0
                )


@ActionFactory.register("enable-sensor-services")
class EnableSensorServices(Action):
    """class for enabling sensor services action."""

    extra_attributes = (
        ActionAttribute(
            name="ultrasonic",
            is_required=False,
            attr_type=bool,
            default=True,
            description="Move ultrasonic to sensor services",
        ),
        ActionAttribute(
            name="radar",
            is_required=False,
            attr_type=bool,
            default=True,
            description="Move radars to sensor services",
        ),
        ActionAttribute(
            name="imu",
            is_required=False,
            attr_type=bool,
            default=True,
            description="Move imu to sensor services",
        ),
        ActionAttribute(
            name="gps",
            is_required=False,
            attr_type=bool,
            default=True,
            description="Move gps to sensor services",
        ),
        ActionAttribute(
            name="vio",
            is_required=False,
            attr_type=bool,
            default=False,
            description="Move vio to sensor services",
        ),
        ActionAttribute(
            name="camera",
            is_required=False,
            attr_type=bool,
            default=False,
            description="Move cameras to sensor services",
        ),
        ActionAttribute(
            name="lidar",
            is_required=False,
            attr_type=bool,
            default=False,
            description="Move lidars to sensor services",
        ),
    )

    def iterate_subcomponents_by_type(
        component: Component, target_type: str
    ) -> Component:
        """iterate subcomponents of component, looking for components of type target_type."""
        if str(os.path.basename(component.descriptor_path)) == target_type:
            yield component
        if isinstance(component, Graphlet):
            for subcomponent in component.subcomponents.values():
                yield from EnableSensorServices.iterate_subcomponents_by_type(
                    subcomponent, target_type
                )

    def camel_case_to_snake_case(name: str) -> str:
        """turn thisSting to THIS_STRING."""
        output = ""
        for c in name:
            if c.isupper():
                output += "_"
            output += c.upper()
        return output

    def get_port_by_type(
        self, is_input: bool, component: Component, data_type: str
    ) -> Port:
        """get first port of given data type."""
        ports = None
        if is_input:
            ports = component.input_ports.values()
        else:
            ports = component.output_ports.values()
        for port in ports:
            if port.data_type == data_type:
                return port
        raise TypeError("Cannot get port by type: ", data_type)

    substitution_parameters: Dict[Any, Any] = {
        "radar": {
            "graphlet_type": "RadarSensor.graphlet.json",
            "old_node_type": "dwRadarNode.node.json",
            "new_node_type": "dwRadarChannelNode.node.json",
            "input_data_type": ["dwRadarScan"],
            "output_data_type": [],
            "ss_graphlet_type": "SensorServiceRadarSensor.graphlet.json",
            "fifo-size": 16,
            "parameter_mapping": {"sensorId": "radarIndex"},
            "removed_mappings": [],
            "enable": (
                "radarEnabled",
                "true, true, true, true, true, true, true, true, \
                    true, true, true, true, true, true, true, true",
            ),
        },
        "imu": {
            "graphlet_type": "ImuSensor.graphlet.json",
            "old_node_type": "dwIMUNode.node.json",
            "new_node_type": "dwIMUChannelNode.node.json",
            "input_data_type": ["dwIMUFrame"],
            "output_data_type": [],
            "ss_graphlet_type": "SensorServiceImuSensor.graphlet.json",
            "fifo-size": 16,
            "parameter_mapping": {},
            "removed_mappings": ["inputModeVal"],
            "enable": ("imuEnabled", "true"),
        },
        "gps": {
            "graphlet_type": "GpsSensor.graphlet.json",
            "old_node_type": "dwGPSNode.node.json",
            "new_node_type": "dwGPSChannelNode.node.json",
            "input_data_type": ["dwGPSFrame"],
            "output_data_type": [],
            "ss_graphlet_type": "SensorServiceGpsSensor.graphlet.json",
            "fifo-size": 16,
            "parameter_mapping": {},
            "removed_mappings": ["inputModeVal"],
            "enable": ("gpsEnabled", "true"),
        },
        "ultrasonic": {
            "graphlet_type": "UltrasonicSensor.graphlet.json",
            "old_node_type": "dwUltrasonicNode.node.json",
            "new_node_type": "dwUltrasonicChannelNode.node.json",
            "input_data_type": ["dwUltrasonicEnvelope"],
            "output_data_type": [],
            "ss_graphlet_type": "SensorServiceUltrasonicSensor.graphlet.json",
            "fifo-size": 16,
            "parameter_mapping": {},
            "removed_mappings": [],
            "enable": ("ultrasonicEnabled", "true"),
        },
        # TODO fill these in with appropriate values
        "vio": {
            "graphlet_type": "Control.graphlet.json",
            "old_node_type": "dwVehicleStateNode.node.json",
            "new_node_type": "dwVehicleStateChannelNode.node.json",
            "input_data_type": ["dwVehicleIOState"],
            "output_data_type": ["dwVehicleIOCommand", "dwVehicleIOMiscCommand"],
            "ss_graphlet_type": "SensorServiceVehicleStateSAL.graphlet.json",
            "fifo-size": 16,
            "parameter_mapping": {},
            "removed_mappings": [],
            "enable": ("vehicleSensorEnabled", "true"),
        },
        "lidar": {},
        "camera": {},
    }

    num_endpoints = 0
    id = 1234

    def make_component(self, module: Any, name: str, desc_path: str) -> Component:
        """make a new component by descriptor path."""
        desc = self._loader.get_descriptor_by_path(desc_path)
        if module == Node:
            return module.from_descriptor(name, desc)
        else:
            return module.from_descriptor(name, self._loader, desc)

    def wire_up_to_parent(
        self, is_input: bool, subcomponent: Component, data_type: str
    ) -> str:
        """make wire a subcomponent port to its parent, add port to parent."""
        port = self.get_port_by_type(is_input, subcomponent, data_type)
        port_name = (
            EnableSensorServices.camel_case_to_snake_case(subcomponent.name)
            + "_"
            + port.name
        )
        subcomponent.parent.insert_port(
            is_input, Port(name=port_name, data_type=data_type)
        )

        connection = None
        if is_input:
            connection = ConnectionDefinition(
                src=port_name,
                params={},
                dests={subcomponent.name + "." + port.name: {}},
            )
        else:
            connection = ConnectionDefinition(
                src=subcomponent.name + "." + port.name,
                params={},
                dests={port_name: {}},
            )
        subcomponent.parent.insert_connections(connection)

        return port_name

    def wire_connection_to_sensorservice(
        self, ss_graphlet: Graphlet, rr_graphlet: Graphlet, params: Dict[Any, Any]
    ) -> None:
        """wire a connection between RR graphlet and corresponding SS graphlet."""

        def wire_connection(is_input: bool, data_type: str) -> None:
            """wire the nvsci connection."""
            # add port to parent for the new graphlet output
            rr_port_name = self.wire_up_to_parent(is_input, rr_graphlet, data_type)
            rr_port_full_name = rr_graphlet.parent.name + "." + rr_port_name

            # add new port for the sensor
            ss_port_name = self.wire_up_to_parent(not is_input, ss_graphlet, data_type)
            ss_port_full_name = ss_graphlet.parent.name + "." + ss_port_name

            # wire together the SS node and the RR node
            endpoint = str(EnableSensorServices.num_endpoints)

            src_port_name = ss_port_full_name
            dst_port_name = rr_port_full_name
            if not is_input:
                src_port_name = rr_port_full_name
                dst_port_name = ss_port_full_name

            ss_rr_connection = ConnectionDefinition(
                src=src_port_name,
                dests={
                    dst_port_name: {
                        "consStreamName": "nvscistreamer_c" + endpoint,
                        "consEnabledComponents": "CPU",
                        "fifo-size": params["fifo-size"],
                    }
                },
                params={
                    "type": "nvsci",
                    "streamName": "nvscistreamer_p" + endpoint,
                    "id": str(EnableSensorServices.id),
                    "producer-fifo-size": 16,
                },
            )
            rr_graphlet.parent.parent.insert_connections(ss_rr_connection)
            EnableSensorServices.num_endpoints += 1
            EnableSensorServices.id += 1

        for data_type in params["input_data_type"]:
            wire_connection(True, data_type)
        for data_type in params["output_data_type"]:
            wire_connection(False, data_type)

    def replace_sensor_graphlet(self, app: Application, params: dict) -> None:
        """replace sensor graphlet in the given application."""
        for component in EnableSensorServices.iterate_subcomponents_by_type(
            app, params["graphlet_type"]
        ):
            # replace subcomponent and wire the new input ports for that node
            for subcomponent in EnableSensorServices.iterate_subcomponents_by_type(
                component, params["old_node_type"]
            ):
                new_desc_path = str(subcomponent.descriptor_path).replace(
                    params["old_node_type"], params["new_node_type"]
                )
                node = self.make_component(Node, subcomponent.name, new_desc_path)
                missing_inputs = (
                    subcomponent.input_ports.keys() - node.input_ports.keys()
                )
                missing_outputs = (
                    subcomponent.output_ports.keys() - node.output_ports.keys()
                )
                for connection in component.get_all_connections_for_subcomponent(
                    subcomponent.name
                ):
                    loc = connection[0].find(subcomponent.name)
                    if loc != -1:
                        port_name = connection[0][len(subcomponent.name) + 1 :]
                        if port_name in missing_outputs:
                            component.remove_connections(connection[0], connection[1])
                    for dst in connection[1]:
                        loc = dst[0].find(subcomponent.name)
                        if loc != 1:
                            port_name = dst[len(subcomponent.name) + 1 :]
                            if port_name in missing_inputs:
                                component.remove_connections(connection[0], [dst])

                component.replace_subcomponent(
                    node, component.parameter_mappings[subcomponent.name]
                )
                for input_data_type in params["input_data_type"]:
                    self.wire_up_to_parent(True, node, input_data_type)
                for output_data_type in params["output_data_type"]:
                    self.wire_up_to_parent(False, node, output_data_type)

                # remove parameter mappings for the subcomponent
                component.remove_parameter_mappings(
                    subcomponent.name, params["removed_mappings"]
                )

            # create the new sensor service sensor graphlet
            sensor_graphlet_desc_path = os.path.join(
                self._base_path, "descriptions", "graphlets", params["ss_graphlet_type"]
            )

            sensor_graphlet = self.make_component(
                Graphlet, component.name, sensor_graphlet_desc_path
            )

            self._sensorservices_graphlet.insert_subcomponent(sensor_graphlet)

            # get the parameters for the sensor
            mappings = {}
            for dst_param, src_param in params["parameter_mapping"].items():
                mappings[src_param] = component.parent.parameter_mappings[
                    component.name
                ][dst_param]
            self._sensorservices_graphlet.insert_parameter_mappings(
                component.name, mappings
            )

            self.wire_connection_to_sensorservice(sensor_graphlet, component, params)

    def make_sensorservice_empty_graphlet(self) -> Graphlet:
        """make sensorservice empty graphlet to be run by SS."""
        desc_path = os.path.join(
            self._base_path, "modules", "SensorServices.graphlet.json"
        )
        sensorservices_graphlet = self.make_component(
            Graphlet, "SensorServices", desc_path
        )
        return sensorservices_graphlet

    def make_sensorservice_process(
        self, c_extra_info: Dict[Any, Any]
    ) -> ProcessDefinition:
        """make SS process."""
        extra_info = {"controller": {"waitForExitSignal": "true"}}
        extra_info["controller"].update(c_extra_info)
        sensorservices = ProcessDefinition(
            name="sensorservices",
            run_on="machine0",
            log_spec=None,
            executable="sensorservices/sensorservices",
            subcomponents=["SensorServices"],
            extra_info=extra_info,
        )

        return sensorservices

    def set_paths(self, target: Application) -> None:
        """set the base path of the descriptions."""
        desc_path = str(target.descriptor_path)
        search = "graphs/"
        pos = desc_path.find(search) + len(search)
        self._base_path = desc_path[:pos]

    def transform_impl(self, target: Application) -> None:
        """transform the application, moving sensors to sensorservices."""
        self.set_paths(target)
        sensorservices_graphlet = self.make_sensorservice_empty_graphlet()
        self._sensorservices_graphlet = sensorservices_graphlet
        target.insert_subcomponent(sensorservices_graphlet)

        extra_info = {}

        for modality, params in EnableSensorServices.substitution_parameters.items():
            if self.get_attribute(modality):
                print("Moving sensors of type", modality, "to sensorservices")
                self.replace_sensor_graphlet(target, params)
                extra_info[params["enable"][0]] = params["enable"][1]

        # lastly, insert sernsorservices process
        target.insert_process(self.make_sensorservice_process(extra_info))
