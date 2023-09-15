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
from copy import deepcopy
import os
from pathlib import Path
from typing import Any, cast, Dict, Iterable, List, Union

from dwcgf.action import Action
from dwcgf.action import ActionAttribute
from dwcgf.action.action_factory import ActionFactory
from dwcgf.descriptor import ConnectionDefinition
from dwcgf.descriptor import ProcessDefinition
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
# Required Sensors and Sensor Mappings
####################################################################################################
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
            name="processName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the process.",
        ),
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

        process_name = self.get_attribute("processName")
        disable = self.get_attribute("disable")
        if disable:
            target.set_rcs_enabled(process_name, False)


@ActionFactory.register("insert-rcs-ports")
class InsertRcsPorts(Action):
    """class for insert-rcs-ports action."""

    extra_attributes = (
        ActionAttribute(
            name="processName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the process.",
        ),
        ActionAttribute(
            name="ports",
            is_required=True,
            attr_type=dict,
            default={},
            description="Name of the target ports.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        process_name = self.get_attribute("processName")
        ports = self.get_attribute("ports")
        if isinstance(ports, dict):
            for port_name, port_value in ports.items():
                target.insert_rcs_port(
                    process_name=process_name,
                    port_name=port_name,
                    port_value=port_value,
                )
        else:
            raise TypeError("ports should be a dict.")


@ActionFactory.register("remove-rcs-ports")
class RemoveRcsPorts(Action):
    """class for remove-rcs-ports action."""

    extra_attributes = (
        ActionAttribute(
            name="processName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the process.",
        ),
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
        process_name = self.get_attribute("processName")
        port_names = self.get_attribute("ports")
        if isinstance(port_names, list):
            for port_name in port_names:
                target.remove_rcs_port(process_name=process_name, port_name=port_name)
        else:
            raise TypeError("ports should be a list of str.")


@ActionFactory.register("update-rcs-serialize-mask")
class UpdateRcsSerializeMask(Action):
    """class for update-rcs-serialize-mask."""

    extra_attributes = (
        ActionAttribute(
            name="processName",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the process.",
        ),
        ActionAttribute(
            name="channelId",
            is_required=True,
            attr_type=str,
            default="",
            description="Name of the target ports.",
        ),
        ActionAttribute(
            name="serializeMask",
            is_required=True,
            attr_type=dict,
            default={},
            description="Name of the target ports.",
        ),
    )

    def transform_impl(self, target: Application) -> None:
        """action implementation."""
        process_name = self.get_attribute("processName")
        channel_id = self.get_attribute("channelId")
        serialize_mask = self.get_attribute("serializeMask")
        if isinstance(channel_id, str) and isinstance(serialize_mask, dict):
            target.update_rcs_serialize_mask(
                process_name=process_name,
                channel_id=channel_id,
                serialize_mask=serialize_mask,
            )
        else:
            raise TypeError(
                "channelId should be a str and serializeMask should be a dict."
            )


####################################################################################################
# Pipelining
####################################################################################################


def find_graphlets_by_type(
    component: Union[Application, Graphlet], target_type: str
) -> Iterable[Graphlet]:
    """Recursively search for subcomponents of a starting node with a given type.

    Args:
        component: A starting node to be searched in a depth-first fashion
        target_type: A representation of the desired graphlet node

    Yields:
        An iterable of graphlets matching the desired type
    """
    if target_type == "*":
        yield component
    elif (
        component.descriptor_path is not None
        and str(os.path.basename(component.descriptor_path)) == target_type
    ):
        yield component

    for subcomponent in component.subcomponents.values():
        if isinstance(subcomponent, Graphlet):
            yield from find_graphlets_by_type(subcomponent, target_type)


def filter_subcomponents_by_type(
    component: Graphlet, target_type: str
) -> List[Component]:
    """Filter subcomponents of a given graphlet, yielding components of type target_type.

    Args:
        component: A graphlet object to be searched
        target_type: The descriptor of the target type to be filtered.

    Returns:
        A a list of subcomponents matching the desired target type
    """

    def is_match(subcomponent: Component) -> bool:
        return (
            subcomponent.descriptor_path is not None
            and str(os.path.basename(subcomponent.descriptor_path)) == target_type
        )

    return list(filter(is_match, component.subcomponents.values()))


def camel_case_to_snake_case(name: str) -> str:
    """Turn thisSting to THIS_STRING."""
    output = ""
    for c in name:
        if c.isupper():
            output += "_"
        output += c.upper()
    return output


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


class GraphletUpdateHelper(Action):
    """Class with basic graphet transform methods."""

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

    def make_component(self, module: Any, name: str, desc_path: str) -> Component:
        """make a new component by descriptor path."""
        desc = self._loader.get_descriptor_by_path(desc_path)

        if module == Node:
            return module.from_descriptor(name, desc)
        else:
            return module.from_descriptor(name, self._loader, desc)

    def wire_up_to_parent(
        self, is_input: bool, subcomponent: Component, data_type: str, name: str = ""
    ) -> str:
        """make wire a subcomponent port to its parent, add port to parent."""
        port = None
        if name != "":
            if is_input:
                port = subcomponent.input_ports[name]
            else:
                port = subcomponent.output_ports[name]
            if port.data_type != data_type:
                raise ValueError(
                    "port data type " + port.data_type + " is not data_type: ",
                    data_type,
                )
        else:
            port = self.get_port_by_type(is_input, subcomponent, data_type)
        port_name = camel_case_to_snake_case(subcomponent.name) + "_" + port.name
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


@ActionFactory.register("enable-sensor-services")
class EnableSensorServices(GraphletUpdateHelper):
    """class for enabling sensor services action."""

    extra_attributes = (
        ActionAttribute(
            name="remove",
            is_required=False,
            attr_type=bool,
            default=False,
            description="Replace sensorservice consumers with original",
        ),
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
        ActionAttribute(
            name="enableRRSSVIOConnection",
            is_required=False,
            attr_type=bool,
            default=False,
            description="Enable RR to SS input VIO channels",
        ),
        ActionAttribute(
            name="defaultEnableSpyglass",
            is_required=False,
            attr_type=bool,
            default=False,
            description="Default enable spyglass connections",
        ),
    )

    substitution_parameters: Dict[Any, Any] = {
        "ultrasonic": {
            "graphlet_type": ["UltrasonicSensor.graphlet.json"],
            "old_node_type": "dwUltrasonicNode.node.json",
            "new_node_type": "dwUltrasonicChannelNode.node.json",
            "input_data_type": ["dwUltrasonicGroup", "dwUltrasonicMountingPositions"],
            "output_data_type": [],
            "ss_graphlet_type": "SensorServiceUltrasonicSensor.graphlet.json",
            "fifo-size": 16,
            "parameter_mapping": {},
            "removed_mappings": [],
            "assignEndpoint": True,
            "enable": ("ultrasonicEnabled", "true"),
        },
        "radar": {
            "graphlet_type": ["RadarSensor.graphlet.json"],
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
                "true, true, true, true, true, false, false, false, \
                    false, false, false, false, false, false, false, false",
            ),
            "ssr_connection": "true",
            "pdr_connection": "false",
        },
        "imu": {
            "graphlet_type": ["ImuSensor.graphlet.json"],
            "old_node_type": "dwIMUNode.node.json",
            "new_node_type": "dwIMUChannelNode.node.json",
            "input_data_type": ["dwIMUFrame"],
            "output_data_type": [],
            "ss_graphlet_type": "SensorServiceImuSensor.graphlet.json",
            "fifo-size": 32,
            "parameter_mapping": {},
            "removed_mappings": ["inputModeVal"],
            "enable": ("imuEnabled", "true"),
            "external_connection_enabled": {
                "dwIMUFrame": {"src": "spg_d_imu_src", "dst": "spg_d_imu_dst"}
            },
            "ssr_connection": "true",
            "pdr_connection": "false",
        },
        "gps": {
            "graphlet_type": ["GpsSensor.graphlet.json"],
            "old_node_type": "dwGPSNode.node.json",
            "new_node_type": "dwGPSChannelNode.node.json",
            "input_data_type": ["dwGPSFrame"],
            "output_data_type": [],
            "ss_graphlet_type": "SensorServiceGpsSensor.graphlet.json",
            "fifo-size": 32,
            "parameter_mapping": {},
            "removed_mappings": ["inputModeVal"],
            "enable": ("gpsEnabled", "true"),
            "external_connection_enabled": {
                "dwGPSFrame": {"src": "spg_d_gps_src", "dst": "spg_d_gps_dst"}
            },
            "ssr_connection": "true",
            "pdr_connection": "false",
        },
        # TODO fill these in with appropriate values
        "vio": {
            "graphlet_type": ["Control.graphlet.json"],
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
        "camera": {
            "graphlet_type": [
                "CameraSensorFront.graphlet.json",
                "CameraSensor.graphlet.json",
            ],
            "old_node_type": "dwCameraNode.node.json",
            "new_node_type": "dwCameraChannelNode.node.json",
            "input_data_type": ["dwImageHandle_t"],
            "output_data_type": [],
            "ss_graphlet_type": "SensorServiceCameraSensor.graphlet.json",
            "fifo-size": 4,
            "parameter_mapping": {
                "cameraIndex": "cameraIndex",
                "cameraIndex_2": "cameraIndex",
            },
            "removed_mappings": ["frameSkipMask"],
            "enable": (
                "cameraEnabled",
                "false, true, true, true, true, true, true, true, \
                    true, false, false, false, false, false, false, false",
            ),
            "ssr_connection": "true",
            "pdr_connection": "true",
        },
    }

    num_endpoints = 0
    dwstreamer_endpoints = 0
    nvscistreamer_endpoints = 0
    id = 1234

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
            endpoint = str(self.num_endpoints)

            src_port_name = ss_port_full_name
            dst_port_name = rr_port_full_name
            if not is_input:
                src_port_name = rr_port_full_name
                dst_port_name = ss_port_full_name

            dests_dict = {dst_port_name: {"fifo-size": params["fifo-size"]}}
            if "assignEndpoint" in params:
                dests_dict[dst_port_name]["destEndpoint"] = "nvscistreamer_c" + str(
                    self.nvscistreamer_endpoints
                )
                dests_dict[dst_port_name]["srcEndpoint"] = "nvscistreamer_p" + str(
                    self.nvscistreamer_endpoints
                )
                self.nvscistreamer_endpoints += 1
            if data_type in params.get("external_connection_enabled", {}):
                outbound_port_name = (
                    ("EXTERNAL:SENSOR_SERVICES_SD_" + ss_port_name)
                    if is_input
                    else ("EXTERNAL:TOP_SD_" + rr_port_name)
                )

                dests_dict[outbound_port_name] = {
                    "destEndpoint": params["external_connection_enabled"][data_type][
                        "dst"
                    ],
                    "srcEndpoint": params["external_connection_enabled"][data_type][
                        "src"
                    ],
                    "fifo-size": params["fifo-size"],
                    "limits": params["fifo-size"] // 2,
                    "disableConsumer": not (
                        self.get_attribute("defaultEnableSpyglass")
                    ),
                    "destName": "SpyglassDaemon",
                }

            ss_rr_connection = ConnectionDefinition(
                src=src_port_name,
                dests=dests_dict,
                params={
                    "type": "nvsci",
                    "id": "sensorservices_output" + endpoint,
                    "producer-fifo-size": params["fifo-size"],
                },
            )

            rr_graphlet.parent.parent.insert_connections(ss_rr_connection)

            outbound_port_name = (
                ("EXTERNAL:SENSOR_SERVICES_SSR_" + ss_port_name)
                if is_input
                else ("EXTERNAL:TOP_SSR_" + rr_port_name)
            )
            # SSR connections
            if params.get("ssr_connection", {}) == "true" and (
                "CAMERA_SENSOR_FRONT" in outbound_port_name
                or "CAMERA" not in outbound_port_name
            ):

                dests_dict = {
                    outbound_port_name: {
                        "destEndpoint": "dwstreamer_c" + str(self.dwstreamer_endpoints),
                        "srcEndpoint": "dwstreamer_p" + str(self.dwstreamer_endpoints),
                        "fifo-size": params["fifo-size"],
                        "limits": params["fifo-size"] // 2,
                        "disableConsumer": True,
                        "destName": "SSR",
                    }
                }
                ss_ssr_connection = ConnectionDefinition(
                    src=src_port_name,
                    dests=dests_dict,
                    params={
                        "type": "nvsci",
                        "id": "sensorservices_output" + endpoint,
                        "producer-fifo-size": params["fifo-size"],
                    },
                )
                self.dwstreamer_endpoints += 1
                rr_graphlet.parent.parent.insert_connections(ss_ssr_connection)

            outbound_port_name = (
                ("EXTERNAL:SENSOR_SERVICES_PDR_" + ss_port_name)
                if is_input
                else ("EXTERNAL:TOP_PDR_" + rr_port_name)
            )
            # PDR connections turn on only fisheye for camera for now
            if params.get("pdr_connection", {}) == "true" and (
                "FISHEYE_CAMERA" in outbound_port_name
                or "CAMERA" not in outbound_port_name
            ):
                dests_dict = {
                    outbound_port_name: {
                        "destEndpoint": "nvscistreamer_c"
                        + str(self.nvscistreamer_endpoints),
                        "srcEndpoint": "nvscistreamer_p"
                        + str(self.nvscistreamer_endpoints),
                        "fifo-size": params["fifo-size"],
                        "limits": params["fifo-size"] // 2,
                        "disableConsumer": True,
                        "destName": "PDR",
                    }
                }
                ss_pdr_connection = ConnectionDefinition(
                    src=src_port_name,
                    dests=dests_dict,
                    params={
                        "type": "nvsci",
                        "id": "sensorservices_output" + endpoint,
                        "producer-fifo-size": params["fifo-size"],
                    },
                )
                rr_graphlet.parent.parent.insert_connections(ss_pdr_connection)
                self.nvscistreamer_endpoints += 1

            # increasing end point number
            self.num_endpoints += 1
            self.id += 1

        for data_type in params["input_data_type"]:
            wire_connection(True, data_type)
        for data_type in params["output_data_type"]:
            wire_connection(False, data_type)

    def replace_sensor_graphlet(self, app: Application, params: dict) -> None:
        """replace sensor graphlet in the given application."""
        for gtype in params["graphlet_type"]:

            for component in find_graphlets_by_type(app, gtype):
                # replace subcomponent and wire the new input ports for that node
                for subcomponent in filter_subcomponents_by_type(
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
                                component.remove_connections(
                                    connection[0], connection[1]
                                )
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
                    self._base_path,
                    "descriptions",
                    "graphlets",
                    params["ss_graphlet_type"],
                )

                sensor_graphlet = self.make_component(
                    Graphlet, component.name, sensor_graphlet_desc_path
                )

                self._sensorservices_graphlet.insert_subcomponent(sensor_graphlet)

                # get the parameters for the sensor
                mappings = {}
                for dst_param, src_param in params["parameter_mapping"].items():
                    if dst_param in component.parent.parameter_mappings[component.name]:
                        mappings[src_param] = component.parent.parameter_mappings[
                            component.name
                        ][dst_param]
                self._sensorservices_graphlet.insert_parameter_mappings(
                    component.name, mappings
                )

                self.wire_connection_to_sensorservice(
                    sensor_graphlet, component, params
                )

    def add_uss_raw_channel(self, target: Application) -> None:
        """add external raw channel output from USS."""
        ss_graphlet = self._sensorservices_graphlet
        for subcomponent in ss_graphlet.subcomponents:
            if subcomponent.find("ultrasonic") == -1:
                continue

            existing_connections = ss_graphlet.get_all_connections_for_subcomponent(
                subcomponent
            )
            for src, dests in existing_connections:
                if src == subcomponent + ".RAW_DATA":
                    for dest in dests:
                        if dest == f"EXTERNAL:{subcomponent}_RAW_DATA":
                            return
            connection = ConnectionDefinition(
                src=subcomponent + ".RAW_DATA",
                dests={
                    f"EXTERNAL:{subcomponent}_RAW_DATA": {
                        "destEndpoint": "spg_d_uss_dst",
                        "srcEndpoint": "spg_d_uss_src",
                        "fifo-size": 16,
                        "limits": 8,
                    }
                },
                params={
                    "type": "nvsci",
                    "id": "spg_d_uss_src",
                    "producer-fifo-size": 16,
                    "disableProducer": True,
                },
            )
            self.num_endpoints += 1
            self.id += 1
            ss_graphlet.insert_connections(connection)

    def add_vio_channels(self, target: Application) -> None:
        """add external VIO channel inputs from VAL."""
        if "vehicleStateVAL" in self._sensorservices_graphlet.subcomponents:
            return
        desc_path = os.path.join(
            self._base_path,
            "descriptions",
            "graphlets",
            "SensorServiceVehicleStateVAL.graphlet.json",
        )
        vio_graphlet = self.make_component(Graphlet, "vehicleStateVAL", desc_path)
        self._sensorservices_graphlet.insert_subcomponent(vio_graphlet)

        infos = [
            {
                "name": "VEHICLE_SAFETY_STATE_EXTERNAL",
                "type": "dwVehicleIOSafetyState",
                "endpoint": "spg_d_vio_s_state",
            },
            {
                "name": "VEHICLE_NON_SAFETY_STATE_EXTERNAL",
                "type": "dwVehicleIONonSafetyState",
                "endpoint": "spg_d_vio_ns_state",
            },
            {
                "name": "VEHICLE_ACTUATION_FEEDBACK_EXTERNAL",
                "type": "dwVehicleIOActuationFeedback",
                "endpoint": "spg_d_vio_af",
            },
        ]

        for info in infos:
            ss_vio_input = self.wire_up_to_parent(True, vio_graphlet, info["type"])
            rr_vio_output = self.wire_up_to_parent(
                False,
                target.subcomponents["top"].subcomponents["planningAndControl"],
                info["type"],
                info["name"],
            )

            dests = {
                "sensorServices."
                + ss_vio_input: {
                    "fifo-size": 16,
                    "limits": 8,
                    "srcEndpoint": "dwstreamer_p" + str(self.dwstreamer_endpoints),
                    "destEndpoint": "dwstreamer_c" + str(self.dwstreamer_endpoints),
                }
            }
            connection = ConnectionDefinition(
                src="top." + rr_vio_output,
                dests=dests,
                params={
                    "type": "nvsci",
                    "producer-fifo-size": 16,
                    "id": "dwstreamer_p" + str(self.dwstreamer_endpoints),
                },
            )
            target.insert_connections(connection)
            self.id += 1
            self.dwstreamer_endpoints += 1

            ss_vio_output = self.wire_up_to_parent(False, vio_graphlet, info["type"])
            dests = {
                f"EXTERNAL:SENSOR_SERVICES_{ss_vio_output}": {
                    "fifo-size": 16,
                    "limits": 8,
                    "srcEndpoint": info["endpoint"] + "_src",
                    "destEndpoint": info["endpoint"] + "_dst",
                }
            }
            connection = ConnectionDefinition(
                src="sensorServices." + ss_vio_output,
                dests=dests,
                params={
                    "type": "nvsci",
                    "producer-fifo-size": 16,
                    "id": info["endpoint"] + "_src",
                    "disableProducer": True,
                },
            )
            target.insert_connections(connection)
            self.id += 1

    def make_sensorservice_empty_graphlet(self) -> Graphlet:
        """make sensorservice empty graphlet to be run by SS."""
        desc_path = os.path.join(
            self._base_path, "modules", "SensorServices.graphlet.json"
        )
        sensorservices_graphlet = self.make_component(
            Graphlet, "sensorServices", desc_path
        )
        return sensorservices_graphlet

    def make_sensorservice_process(self, dirname: Path) -> ProcessDefinition:
        """make SS process."""
        traceParams = {"enabled": False, "traceLevel": 20, "channelMask": 0x81}
        controller = {"waitForExitSignal": "true"}
        extra_info = {"controller": controller, "trace": traceParams}
        sensorservices = ProcessDefinition(
            dirname=dirname,
            name="sensorservices",
            run_on="machine0",
            log_spec=None,
            executable="sensorservices/sensorservices",
            subcomponents=["sensorServices"],
            extra_info=extra_info,
        )

        return sensorservices

    def set_paths(self, target: Application) -> None:
        """set the base path of the descriptions."""
        desc_path = str(target.descriptor_path)
        search = "graphs/"
        pos = desc_path.find(search) + len(search)
        self._base_path = desc_path[:pos]
        actualSearchPath = "apps/roadrunner-2.0/graphs/"
        if desc_path.find(search) != -1:
            pos = desc_path.find(search) + len(search)
            self._base_path = desc_path[:pos]
        else:
            self._base_path = actualSearchPath

    def replace_sensorservice_consumer_nodes(
        self, app: Application, params: dict
    ) -> None:
        """replace a node type in RR that consumer from SS back to SAL nodes."""
        if "graphlet_type" not in params:
            return
        for component in find_graphlets_by_type(app, "*"):
            # replace subcomponent and wire the new input ports for that node
            for subcomponent in filter_subcomponents_by_type(
                component, params["new_node_type"]
            ):
                old_desc_path = str(subcomponent.descriptor_path).replace(
                    params["new_node_type"], params["old_node_type"]
                )
                node = self.make_component(Node, subcomponent.name, old_desc_path)

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

    def replace_sensorservice_consumers(self, target: Application) -> None:
        """replace nodes in RR that consumer from SS back to SAL nodes."""
        for params in self.substitution_parameters.values():
            self.replace_sensorservice_consumer_nodes(target, params)

    def transform_impl(self, target: Application) -> None:
        """transform the application, moving sensors to sensorservices."""
        if self.get_attribute("remove"):
            self.replace_sensorservice_consumers(target)
            return
        self.set_paths(target)

        if "sensorservices" in target.processes:
            self.num_endpoints = target.processes["sensorservices"].extra_info[
                "controller"
            ]["num_endpoints"]
            self.nvscistreamer_endpoints = target.processes[
                "sensorservices"
            ].extra_info["controller"]["nvscistreamer_endpoints"]
            self.dwstreamer_endpoints = target.processes["sensorservices"].extra_info[
                "controller"
            ]["dwstreamer_endpoints"]

        if "sensorServices" not in target.subcomponents:
            self._sensorservices_graphlet = self.make_sensorservice_empty_graphlet()
            target.insert_subcomponent(self._sensorservices_graphlet)
        else:
            self._sensorservices_graphlet = target.subcomponents["sensorServices"]

        extra_info = {}
        for modality, params in self.substitution_parameters.items():
            if self.get_attribute(modality):
                print("Moving sensors of type", modality, "to sensorservices")
                self.replace_sensor_graphlet(target, params)
                extra_info[params["enable"][0]] = params["enable"][1]
                if modality == "ultrasonic":
                    self.add_uss_raw_channel(target)
                    self.add_vio_channels(target)

        extra_info["num_endpoints"] = self.num_endpoints
        extra_info["nvscistreamer_endpoints"] = self.nvscistreamer_endpoints
        extra_info["dwstreamer_endpoints"] = self.dwstreamer_endpoints

        if "sensorservices" not in target.processes:
            target.insert_process(
                self.make_sensorservice_process(target.descriptor_path.parent)
            )

        target.processes["sensorservices"].extra_info["controller"].update(extra_info)


####################################################################################################
# New vio node
####################################################################################################


@ActionFactory.register("new_vio_node")
class NewVioNode(GraphletUpdateHelper):
    """Class for swapping old vio node for three new vio node action."""

    extra_attributes = (
        ActionAttribute(
            name="newVioNode",
            is_required=False,
            attr_type=bool,
            default=False,
            description="Insert external connection to vcm",
        ),
        ActionAttribute(
            name="remove",
            is_required=False,
            attr_type=bool,
            default=False,
            description="Remove external connection to vcm",
        ),
    )

    vio_node_parameters: Dict[Any, Any] = {
        "vehicleStateNonSafetyStateNode": {
            "data_type": "dwVehicleIONonSafetyState",
            "destStreamName": "val_67_1",
            "srcStreamName": "val_67_0",
            "id": "val_67_1",
            "fifo_size": 12,
        },
        "vehicleStateSafetyStateNode": {
            "data_type": "dwVehicleIOSafetyState",
            "destStreamName": "val_65_1",
            "srcStreamName": "val_65_0",
            "id": "val_65_1",
            "fifo_size": 12,
        },
        "vehicleStateActuationFeedbackNode": {
            "data_type": "dwVehicleIOActuationFeedback",
            "destStreamName": "val_69_1",
            "srcStreamName": "val_69_0",
            "id": "val_69_1",
            "fifo_size": 12,
        },
    }

    def add_input_connection(
        self, target: Application, dataType: str, nodeName: str
    ) -> None:
        """Add input external connection."""
        rr_vio_input = self.wire_up_to_parent(
            True,
            target.subcomponents["top"].subcomponents["planningAndControl"],
            dataType,
        )
        dests = {
            "top."
            + rr_vio_input: {
                "srcEndpoint": self.vio_node_parameters[nodeName]["srcStreamName"],
                "destEndpoint": self.vio_node_parameters[nodeName]["destStreamName"],
                "disableConsumer": True,
            }
        }
        connection = ConnectionDefinition(
            src=f"EXTERNAL:TOP_{rr_vio_input}",
            dests=dests,
            params={
                "type": "nvsci",
                "producer-fifo-size": self.vio_node_parameters[nodeName]["fifo_size"],
                "id": self.vio_node_parameters[nodeName]["id"],
                "disableProducer": True,
            },
        )
        target.insert_connections(connection)

    def remove_input_connection(
        self, target: Application, dataType: str, nodeName: str
    ) -> None:
        """Transform to remove connections."""
        subcomponent = target.subcomponents["top"].subcomponents["planningAndControl"]
        port = self.get_port_by_type(True, subcomponent, dataType)
        port_name = camel_case_to_snake_case(subcomponent.name) + "_" + port.name
        port_full_name = "top." + port_name

        target.remove_connections(f"EXTERNAL:TOP_{port_name}", [port_full_name])

    def transform_impl(self, target: Application) -> None:
        """Transform to add or remove connections."""

        if self.get_attribute("newVioNode"):
            for nodeName, params in self.vio_node_parameters.items():
                # adding input connections
                self.add_input_connection(target, params["data_type"], nodeName)

        if self.get_attribute("remove"):
            for nodeName, params in self.vio_node_parameters.items():
                # remove input connections
                self.remove_input_connection(target, params["data_type"], nodeName)
