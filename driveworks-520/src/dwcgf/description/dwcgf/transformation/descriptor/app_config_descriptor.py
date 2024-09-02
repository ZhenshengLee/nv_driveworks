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
"""For app-config descriptor."""
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

from .descriptor import Descriptor
from .descriptor_factory import DescriptorFactory, DescriptorType


@DescriptorFactory.register(DescriptorType.APP_CONFIG)
class AppConfigDescriptor(Descriptor):
    """class for app-config descriptor."""

    def __init__(
        self,
        file_path: Path,
        *,
        comment: Optional[str] = None,
        component_parameters: Optional[Dict[str, Dict]] = None,
        skip_nodes: Optional[Dict[str, List]] = None,
        processes: Optional[Dict[str, Dict]] = None,
        roadcast_service: Optional[Dict[str, Dict]] = None,
    ):
        """Create AppConfigDescriptor instance.

        @param file_path            path of this app-config descriptor file
        @param comment              optional description of this file
        @param component_parameters dict where key is component full ID and
                                    value is another dict contains parameters
                                    amend (where key is parameter name, value
                                    is amended value).
        @param skip_nodes           dict where key is components ID to be skipped
                                    and value is a list. (Current only empty list
                                    is supported)
        @param processes            dict where key is process name and value is dict
                                    to be used as JSON merge patch to amend the
                                    process arguments.
        @param roadcast_service     dict to amend RoadCastService parameters
        """
        super().__init__(file_path)

        self._comment = comment
        self._component_parameters = component_parameters
        self._skip_nodes = skip_nodes
        self._processes = processes
        self._roadcast_service = roadcast_service

    @property
    def comment(self) -> Optional[str]:
        """Return the optional description of this descriptor."""
        return self._comment

    @property
    def component_parameters(self) -> Optional[Dict[str, Dict]]:
        """Return the component parameter amends."""
        return self._component_parameters

    @property
    def skip_nodes(self) -> Optional[Dict[str, List]]:
        """Return the components get skipped."""
        return self._skip_nodes

    @property
    def processes(self) -> Optional[Dict[str, Dict]]:
        """Return the process command line argument amends."""
        return self._processes

    @property
    def roadcast_service(self) -> Optional[Dict[str, Dict]]:
        """Return the roadcast_service command line argument amends."""
        return self._roadcast_service

    def to_json_data(self) -> OrderedDict:
        """Dump the descriptor instance to JSON data."""

        data = OrderedDict()
        if self.comment is not None:
            data["comment"] = self.comment
        if self.component_parameters is not None:
            data["componentParameters"] = self.component_parameters  # type: ignore

        if self.skip_nodes is not None:
            data["skipNodes"] = self.skip_nodes  # type: ignore

        if self.processes is not None:
            data["processes"] = OrderedDict()  # type: ignore
            for proc_id, arg_amend in sorted(self.processes.items()):
                data["processes"][proc_id] = OrderedDict(argv=arg_amend)  # type: ignore

        if self.roadcast_service is not None:
            data["RoadCastService"] = OrderedDict()  # type: ignore
            for key, arg_amend in self.roadcast_service.items():
                data["RoadCastService"][key] = arg_amend  # type: ignore

        return data
