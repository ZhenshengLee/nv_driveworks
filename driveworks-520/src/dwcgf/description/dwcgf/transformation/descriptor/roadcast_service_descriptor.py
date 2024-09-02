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
# Copyright (c) 2022-2024 NVIDIA Corporation. All rights reserved.
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
from typing import Dict, List, Union

from .descriptor import Descriptor
from .descriptor_factory import DescriptorFactory, DescriptorType


class RoadCastBroadcastChannelParam:
    """Class for the params field of RoadCastService."""

    def __init__(
        self,
        use_socket: bool,
        use_dds: bool,
        use_file: bool,
        use_nvsci: bool,
        port: int,
        domain_id: int,
        file_name: str,
        serial_mask: Dict[str, bool],
        channel_id: str,
        stream_name: str,
        fifo_size: int,
        payload_size: int,
    ):
        """Create a RoadCastBroadcastChannelParam instance.

        @param use_socket    Whether to enable socket broadcast
        @param use_dds       Whether to enable DDS broadcast
        @param use_file      Whether to enable file broadcast
        @param use_nvsci     Whether to enable NVSci channel broadcast
        @param port          The port ID for socket broadcast
        @param domain_id     The domain ID for DDS broadcast
        @param file_name     The file for file broadcast
        @param serial_mask   The mask to enable or disable signals by serialization type
        @param channel_id    Id for the NVSci channel
        @param stream_name   Stream name for the NVSci channel
        @param fifo_size     Max number of buffers stored in the NVSci channel
        @param payload_size  Size of each buffer in the NVSci channel (in bytes)
        """
        self._use_socket = use_socket
        self._use_dds = use_dds
        self._use_file = use_file
        self._use_nvsci = use_nvsci
        self._port = port
        self._domain_id = domain_id
        self._file_name = file_name
        self._serial_mask = serial_mask
        self._channel_id = channel_id
        self._stream_name = stream_name
        self._fifo_size = fifo_size
        self._payload_size = payload_size

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
        max_lane_graph_polyline_length: int,
        enable_profiling: bool,
        send_sys_timestamp: bool,
        roadcast_health_msg_log_freq_in_secs: int,
        enabled_lane_graphs: List[bool],
        register_to_context: bool,
        send_av_schema: bool,
        use_rc2: bool,
        send_timeout_us: int,
        channel_params: Dict[str, RoadCastBroadcastChannelParam],
        topic_mask: Dict[str, bool],
        latched_mask: Dict[str, bool],
        msg_queue_param: List[Dict],
    ):
        """Create a RoadCastBroadcastChannelParam instance.

        @param enable                          Whether to instantiate the dwRoadCastServer
        @param allow_av_msg                    Enable/disable AVMessage casting
        @param use_aio_file_write              Enable/disable async IO file write
        @param channel_cnt                     Count of RoadCastChannel
        @param max_lane_graph_polyline_length  Max lane graph polyline length
        @param enable_profiling                Enable/disable RoadCastServerEngine profiling
        @param send_sys_timestamp              Override avmessage.timestamp with current timestamp
                                               from context, used for debug
        @param roadcast_health_msg_log_freq_in_secs  frequency for logging RC health messages
        @param enabled_lane_graphs             Mask to cast which LaneGraph in dwLaneGraphList
        @param register_to_context             Enable/disable RoadCast registration into context to
                                               enable RoadCast Producer
        @param send_av_schema                  Enable/disable RoadCast sending avprotos schema
        @param use_rc2                         Use RoadCast 2.1 server backend
        @param send_timeout_us                 Max timeout value in us roadcast log API blocks
                                               when buffer is full
        @param channel_params                  Base parameters for each RoadCastBroadcastChannel
        @param topic_mask                      Mask for the topic logging via the producer log() API
        @param latched_mask                    Mask for latched messages to be cached by RCServer
        @param msg_queue_param                 List of msgCount and msgSize RC2.1 allocates
        """

        self._enable = enable
        self._allow_av_msg = allow_av_msg
        self._use_aio_file_write = use_aio_file_write
        self._channel_cnt = channel_cnt
        self._max_lane_graph_polyline_length = max_lane_graph_polyline_length
        self._enable_profiling = enable_profiling
        self._send_sys_timestamp = send_sys_timestamp
        self._roadcast_health_msg_log_freq_in_secs = (
            roadcast_health_msg_log_freq_in_secs
        )
        self._enabled_lane_graphs = enabled_lane_graphs
        self._register_to_context = register_to_context
        self._send_av_schema = send_av_schema
        self._use_rc2 = use_rc2
        self._channel_params = channel_params
        self._topic_mask = topic_mask
        self._send_timeout_us = send_timeout_us
        self._latched_mask = latched_mask
        self._msg_queue_param = msg_queue_param

    @property
    def channel_params(self) -> Dict[str, RoadCastBroadcastChannelParam]:
        """Return channel_params."""
        return self._channel_params


@DescriptorFactory.register(DescriptorType.ROADCAST_SERVICE)
class RoadCastServiceDescriptor(Descriptor):
    """class for RoadCastService."""

    def __init__(
        self,
        file_path: Path,
        name: str,
        enabled: bool,
        params: RoadCastServiceParam,
    ):
        """Create a RoadCastServiceDescriptor instance.

        @param file_path path of this descriptor file
        """
        super().__init__(file_path)
        self._name = name
        self._enabled = enabled
        self._params = params

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
            max_lane_graph_polyline_length=params_content.get(
                "maxLaneGraphPolylineLength", 150
            ),
            enable_profiling=params_content.get("enableProfiling", False),
            send_sys_timestamp=params_content.get("sendSysTimestamp", False),
            roadcast_health_msg_log_freq_in_secs=params_content.get(
                "roadCastHealthMsgLogFreqInSecs", 10
            ),
            send_timeout_us=params_content.get("sendTimeoutUs", 0),
            enabled_lane_graphs=params_content.get(
                "enabledLaneGraphs",
                [False, False, False, False, False, False, False, False, False, False],
            ),
            register_to_context=params_content.get("registerToContext", False),
            send_av_schema=params_content.get("sendAVSchema", False),
            use_rc2=params_content.get("useRC2", False),
            channel_params={
                key: RoadCastBroadcastChannelParam(
                    use_socket=value.get("useSocket"),
                    use_dds=value.get("useDDS"),
                    use_file=value.get("useFile"),
                    use_nvsci=value.get("useNVSci"),
                    port=value.get("port"),
                    domain_id=value.get("domainId"),
                    file_name=value.get("fileName"),
                    serial_mask=value.get("serialMask"),
                    fifo_size=value.get("channelFifoSize"),
                    payload_size=value.get("channelPayloadSize"),
                    channel_id=value.get("channelId"),
                    stream_name=value.get("streamName"),
                )
                for key, value in params_content.get("channelParams", {}).items()
            },
            topic_mask=params_content.get("topicMask", {}),
            latched_mask=params_content.get("latchedMask", {}),
            msg_queue_param=params_content.get("msgQueueParam", []),
        )
        return RoadCastServiceDescriptor(
            path,
            name="RoadCastService",
            enabled=content.get("enabled", False),
            params=params,
        )

    def to_json_data(self) -> OrderedDict:
        """Dump GraphletDescriptor to JSON data."""

        def dump_params(params: RoadCastServiceParam) -> OrderedDict:
            params_json: OrderedDict = OrderedDict()
            params_json["enable"] = params._enable
            params_json["allowAVMsg"] = params._allow_av_msg
            params_json["useAIOFileWrite"] = params._use_aio_file_write
            params_json["channelCnt"] = params._channel_cnt
            params_json[
                "maxLaneGraphPolylineLength"
            ] = params._max_lane_graph_polyline_length
            params_json["enableProfiling"] = params._enable_profiling
            params_json["sendSysTimestamp"] = params._send_sys_timestamp
            params_json[
                "roadCastHealthMsgLogFreqInSecs"
            ] = params._roadcast_health_msg_log_freq_in_secs
            params_json["enabledLaneGraphs"] = params._enabled_lane_graphs
            params_json["registerToContext"] = params._register_to_context
            params_json["sendAVSchema"] = params._send_av_schema
            params_json["useRC2"] = params._use_rc2
            params_json["sendTimeoutUs"] = params._send_timeout_us
            params_json["topicMask"] = dict(params._topic_mask)
            params_json["latchedMask"] = dict(params._latched_mask)
            params_json["msgQueueParam"] = params._msg_queue_param
            params_json["channelParams"] = OrderedDict()
            for k, v in params._channel_params.items():
                params_json["channelParams"][k] = OrderedDict()
                params_json["channelParams"][k]
                params_json["channelParams"][k]["useSocket"] = v._use_socket
                params_json["channelParams"][k]["useDDS"] = v._use_dds
                params_json["channelParams"][k]["useFile"] = v._use_file
                params_json["channelParams"][k]["useNVSci"] = v._use_nvsci
                params_json["channelParams"][k]["port"] = v._port
                params_json["channelParams"][k]["domainId"] = v._domain_id
                params_json["channelParams"][k]["fileName"] = v._file_name
                params_json["channelParams"][k]["serialMask"] = v._serial_mask
                params_json["channelParams"][k]["channelFifoSize"] = v._fifo_size
                params_json["channelParams"][k]["channelPayloadSize"] = v._payload_size
                params_json["channelParams"][k]["channelId"] = v._channel_id
                params_json["channelParams"][k]["streamName"] = v._stream_name
            return params_json

        json_data: OrderedDict = OrderedDict()

        json_data["enabled"] = self.enabled
        json_data["params"] = dump_params(self.params)
        return json_data
