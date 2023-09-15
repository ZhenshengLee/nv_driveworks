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
"""For RoadCastService descriptor."""
from collections import OrderedDict
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from .descriptor import Descriptor
from .descriptor_factory import DescriptorFactory
from .descriptor_factory import DescriptorType


class RoadCastBroadcastChannelParam:
    """Class for the params field of RoadCastService."""

    def __init__(
        self,
        use_socket: bool,
        use_dds: bool,
        use_file: bool,
        port: int,
        domain_id: int,
        file_name: str,
        serial_mask: Dict[str, bool],
    ):
        """Create a RoadCastBroadcastChannelParam instance.

        @param use_socket    Whether enable socket broadcast
        @param use_dds       Whether enable DDS broadcast
        @param use_file      Whether enable file broadcast
        @param port          The port ID for socket broadcast
        @param domain_id     The domain ID for DDS broadcast
        @param file_name     The file for file broadcast
        @param serial_mask   The mask to enable or disable signals by serialization type
        """
        self._use_socket = use_socket
        self._use_dds = use_dds
        self._use_file = use_file
        self._port = port
        self._domain_id = domain_id
        self._file_name = file_name
        self._serial_mask = serial_mask

    @property
    def serial_mask(self) -> Dict[str, bool]:
        """Return serial_mask."""
        return self._serial_mask


class RoadCastServiceParam:
    """Class for the params field of RoadCastService."""

    def __init__(
        self,
        enable: bool,
        allow_av_msg: bool,
        use_aio_file_write: bool,
        channel_cnt: int,
        radar_scan_detection_ports: List[int],
        radar_track_sensor_name: str,
        wait_perception_enabled: bool,
        max_lane_graph_polyline_length: int,
        enable_profiling: bool,
        send_sys_timestamp: bool,
        enabled_lane_graphs: List[bool],
        register_to_context: bool,
        send_av_schema: bool,
        channel_params: Dict[str, RoadCastBroadcastChannelParam],
        topic_mask: Dict[str, bool],
    ):
        """Create a RoadCastBroadcastChannelParam instance.

        @param enable                          Whether to instantiate the dwRoadCastServer
        @param allow_av_msg                    Enable/disable AVMessage casting
        @param use_aio_file_write              Enable/disable async IO file write
        @param channel_cnt                     Count of RoadCastChannel
        @param radar_scan_detection_ports      Mask to cast which radar scan
        @param radar_track_sensor_name         Name for casting radar scan track type
        @param wait_perception_enabled         Enable/disable cast of wait condition object in
                                               dwObjectHistoryArray
        @param max_lane_graph_polyline_length  Max number of vertices per lane graph polyline
        @param enable_profiling                Enable/disable RoadCastServerEngine profiling
        @param send_sys_timestamp              Override avmessage.timestamp with current timestamp
                                               from context, used for debug
        @param enabled_lane_graphs             Mask to cast which LaneGraph in dwLaneGraphList
        @param register_to_context             Enable/disable RoadCast registration into context to
                                               enable RoadCast Producer
        @param send_av_schema                  Enable/disable RoadCast sending avprotos schema
        @param channel_params                  Base parameters for each RoadCastBroadcastChannel
        @param topic_mask                      Mask for the topic logging via the producer log() API
        """

        self._enable = enable
        self._allow_av_msg = allow_av_msg
        self._use_aio_file_write = use_aio_file_write
        self._channel_cnt = channel_cnt
        self._radar_scan_detection_ports = radar_scan_detection_ports
        self._radar_track_sensor_name = radar_track_sensor_name
        self._wait_perception_enabled = wait_perception_enabled
        self._max_lane_graph_polyline_length = max_lane_graph_polyline_length
        self._enable_profiling = enable_profiling
        self._send_sys_timestamp = send_sys_timestamp
        self._enabled_lane_graphs = enabled_lane_graphs
        self._register_to_context = register_to_context
        self._send_av_schema = send_av_schema
        self._channel_params = channel_params
        self._topic_mask = topic_mask

    @property
    def channel_params(self) -> Dict[str, RoadCastBroadcastChannelParam]:
        """Return channel_params."""
        return self._channel_params


class SignalProducerPort:
    """Class for the port of signal producer."""

    def __init__(
        self,
        name: str,
        serialization_type: str,
        low_frequency: bool,
        sensor_timestamp: Optional[bool] = None,
        channel_attr: Optional[Dict[str, Union[int, bool]]] = None,
    ):
        """Create a PortDefinition instance.

        @param name                The full name of the signal producer port
        @param serialization_type  The signal serialization type
        @param low_frequency       If the signal should be cast in low frequency
        @param sensor_timestamp    If the signal is used as a roadcast sensor time source
        @param channel_attr        Extra channel parameters to receive the signal
        """
        if serialization_type == "":
            raise ValueError("Empty serializationType is invalid")
        self._name = name
        self._serialization_type = serialization_type
        self._low_frequency = low_frequency
        self._sensor_timestamp = sensor_timestamp
        self._channel_attr = channel_attr

    @property
    def name(self) -> str:
        """Return the full name of this port."""
        return self._name

    @property
    def serialization_type(self) -> str:
        """Return the signal serialization type of this port."""
        return self._serialization_type

    @property
    def low_frequency(self) -> bool:
        """Return flag if the signal should be cast in low frequency of this port."""
        return self._low_frequency

    @property
    def sensor_timestamp(self) -> Optional[bool]:
        """Return the flag if the signal will be used as the sensor time source of this port."""
        return self._sensor_timestamp

    @property
    def channel_attr(self) -> Optional[Dict[str, Union[int, bool]]]:
        """Return extra channel paramters of this port."""
        return self._channel_attr

    @classmethod
    def from_json_data(cls, name: str, content: Dict) -> "SignalProducerPort":
        """Create an instance of SignalProducerPort from the json data."""
        return SignalProducerPort(
            name=name,
            serialization_type=content.get("serializationType", ""),
            low_frequency=content.get("lowFrequency", True),
            sensor_timestamp=content.get("sensorTimestamp", None),
            channel_attr=content.get("channelAttr", None),
        )


@DescriptorFactory.register(DescriptorType.ROADCAST_SERVICE)
class RoadCastServiceDescriptor(Descriptor):
    """class for RoadCastService."""

    def __init__(
        self,
        file_path: Path,
        name: str,
        enabled: bool,
        params: RoadCastServiceParam,
        ports: Dict[str, SignalProducerPort],
        pass_mapping: Dict[str, Dict[str, str]],
    ):
        """Create a RoadCastServiceDescriptor instance.

        @param file_path path of this descriptor file
        """
        super().__init__(file_path)
        self._name = name
        self._enabled = enabled
        self._params = params
        self._ports = ports
        self._pass_mapping = pass_mapping

    @property
    def name(self) -> str:
        """Return name."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set the name."""
        self._name = value

    @property
    def enabled(self) -> bool:
        """Return if enabled."""
        return self._enabled

    @property
    def params(self) -> RoadCastServiceParam:
        """Return param."""
        return self._params

    @property
    def ports(self) -> Dict[str, SignalProducerPort]:
        """Return ports."""
        return self._ports

    @property
    def pass_mapping(self) -> Dict[str, Dict[str, str]]:
        """Return pass mapping."""
        return self._pass_mapping

    @classmethod
    def from_json_data(
        cls, content: Dict, path: Union[str, Path]
    ) -> "RoadCastServiceDescriptor":
        """Create a RoadCastServiceDescriptor from JSON data."""
        path = Path(path)
        params_content = content.get("params")
        if not isinstance(params_content, dict):
            raise TypeError(
                "Wrong json content for RoadCastService: Missing params dict"
            )
        params = RoadCastServiceParam(
            enable=params_content.get("enable", False),
            allow_av_msg=params_content.get("allowAVMsg", False),
            use_aio_file_write=params_content.get("useAIOFileWrite", False),
            channel_cnt=params_content.get("channelCnt", 0),
            radar_scan_detection_ports=params_content.get(
                "radarScanDetectionPorts",
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ),
            radar_track_sensor_name=params_content.get("radarTrackSensorName", ""),
            wait_perception_enabled=params_content.get("waitPerceptionEnabled", False),
            max_lane_graph_polyline_length=params_content.get(
                "maxLaneGraphPolylineLength", 0
            ),
            enable_profiling=params_content.get("enableProfiling", False),
            send_sys_timestamp=params_content.get("sendSysTimestamp", False),
            enabled_lane_graphs=params_content.get(
                "enabledLaneGraphs",
                [False, False, False, False, False, False, False, False, False, False],
            ),
            register_to_context=params_content.get("registerToContext", False),
            send_av_schema=params_content.get("sendAVSchema", False),
            channel_params={
                key: RoadCastBroadcastChannelParam(
                    use_socket=value.get("useSocket"),
                    use_dds=value.get("useDDS"),
                    use_file=value.get("useFile"),
                    port=value.get("port"),
                    domain_id=value.get("domainId"),
                    file_name=value.get("fileName"),
                    serial_mask=value.get("serialMask"),
                )
                for key, value in params_content.get("channelParams", {}).items()
            },
            topic_mask=params_content.get("topicMask", {}),
        )
        producerPorts = {
            key: SignalProducerPort(
                name=key,
                serialization_type=value["serializationType"],
                low_frequency=value["lowFrequency"],
                sensor_timestamp=value.get("sensorTimestamp", None),
                channel_attr=value.get("channelAttr", None),
            )
            for key, value in content.get("ports", {}).items()
        }

        return RoadCastServiceDescriptor(
            path,
            name="RoadCastService",
            enabled=content.get("enabled", False),
            params=params,
            ports=producerPorts,
            pass_mapping=content.get("passMapping", {}),
        )

    def to_json_data(self) -> OrderedDict:
        """Dump GraphletDescriptor to JSON data."""

        def dump_params(params: RoadCastServiceParam) -> OrderedDict:
            params_json: OrderedDict = OrderedDict()
            params_json["enable"] = params._enable
            params_json["allowAVMsg"] = params._allow_av_msg
            params_json["useAIOFileWrite"] = params._use_aio_file_write
            params_json["channelCnt"] = params._channel_cnt
            params_json["radarScanDetectionPorts"] = params._radar_scan_detection_ports
            params_json["radarTrackSensorName"] = params._radar_track_sensor_name
            params_json["waitPerceptionEnabled"] = params._wait_perception_enabled
            params_json[
                "maxLaneGraphPolylineLength"
            ] = params._max_lane_graph_polyline_length
            params_json["enableProfiling"] = params._enable_profiling
            params_json["sendSysTimestamp"] = params._send_sys_timestamp
            params_json["enabledLaneGraphs"] = params._enabled_lane_graphs
            params_json["registerToContext"] = params._register_to_context
            params_json["sendAVSchema"] = params._send_av_schema
            params_json["channelParams"] = OrderedDict()
            for k, v in params._channel_params.items():
                params_json["channelParams"][k] = OrderedDict()
                params_json["channelParams"][k]
                params_json["channelParams"][k]["useSocket"] = v._use_socket
                params_json["channelParams"][k]["useDDS"] = v._use_dds
                params_json["channelParams"][k]["useFile"] = v._use_file
                params_json["channelParams"][k]["port"] = v._port
                params_json["channelParams"][k]["domainId"] = v._domain_id
                params_json["channelParams"][k]["fileName"] = v._file_name
                params_json["channelParams"][k]["serialMask"] = v._serial_mask
            params_json["topicMask"] = dict(params._topic_mask)
            return params_json

        def dump_ports(ports: Dict[str, SignalProducerPort]) -> OrderedDict:
            """Dump input and output ports."""
            ports_json: OrderedDict = OrderedDict()
            for k, v in ports.items():
                ports_json[k] = OrderedDict(serializationType=v.serialization_type)
                ports_json[k]["lowFrequency"] = v.low_frequency
                if v.sensor_timestamp:
                    ports_json[k]["sensorTimestamp"] = v.sensor_timestamp
                if v.channel_attr:
                    ports_json[k]["channelAttr"] = v.channel_attr
            return ports_json

        json_data: OrderedDict = OrderedDict()

        json_data["enabled"] = self.enabled
        json_data["params"] = dump_params(self.params)
        json_data["ports"] = dump_ports(self.ports)
        json_data["passMapping"] = self.pass_mapping
        return json_data
