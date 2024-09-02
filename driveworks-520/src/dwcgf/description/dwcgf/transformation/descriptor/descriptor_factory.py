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
"""Data structures for Descriptor base class and factory."""
import enum
from pathlib import Path
from typing import Dict, Optional, Type, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .descriptor import Descriptor


@enum.unique
class DescriptorType(enum.Enum):
    """Indicates the type of descriptor files."""

    INVALID = 0
    APPLICATION = 1
    GRAPHLET = 2
    NODE = 3
    REQUIRED_SENSORS = 4
    EXTRA_INFO = 5  # to be deprecated
    TRANSFORMATION = 6
    APP_CONFIG = 7
    ROADCAST_SERVICE = 8
    SENSOR_MAPPINGS = 9


class DescriptorFactory:
    """Factory class for construct all registered descriptor implmentations."""

    Extensions = {
        DescriptorType.APPLICATION: ".app.json",
        DescriptorType.GRAPHLET: ".graphlet.json",
        DescriptorType.NODE: ".node.json",
        DescriptorType.REQUIRED_SENSORS: ".required-sensors.json",
        DescriptorType.TRANSFORMATION: ".trans.json",
        DescriptorType.APP_CONFIG: ".app-config.json",
        DescriptorType.ROADCAST_SERVICE: ".roadcastservice.json",
        DescriptorType.SENSOR_MAPPINGS: ".sensor-mappings.json",
    }

    _factory: Dict[DescriptorType, Type["Descriptor"]] = {}

    class register:
        """Descriptor implementations need to be registered into this factory."""

        def __init__(self, desc_type: DescriptorType):
            """Remebers implementation type.

            @param desc_type the DescriptorType of this implementation.
            """
            self._type = desc_type

        def __call__(self, desc_class: Type["Descriptor"]) -> Type["Descriptor"]:
            """Register a Descriptor subtype to the factory."""
            if self._type in DescriptorFactory._factory:
                raise ValueError(
                    f"Descriptor type {self._type.name} has already been registered"
                )
            DescriptorFactory._factory[self._type] = desc_class
            desc_class.desc_type = self._type
            return desc_class

    @staticmethod
    def determine_descriptor_type(file_path: Path) -> DescriptorType:
        """Deterine descriptor type from file path."""

        for desc_type, extension in DescriptorFactory.Extensions.items():
            if file_path.match("*" + extension):  # pattern like *.app.json
                return desc_type

        # doesn't have match in Descriptor.Extensions
        if file_path.match(
            "*.json"
        ):  # if we put this in the map, this will match everything
            return DescriptorType.EXTRA_INFO
        raise ValueError(
            f"Cannot determine descriptor type from the file path: '{file_path}'"
        )

    @staticmethod
    def check_descriptor_extension(file_path: Path, desc_type: DescriptorType) -> None:
        """For checking descriptor file extensions.

        Checking failure causes exception
        """

        # only check for the descriptor types listed in the Descriptor.Extensions
        if desc_type in DescriptorFactory.Extensions and not file_path.match(
            "*" + DescriptorFactory.Extensions[desc_type]
        ):
            raise ValueError(
                f"Descriptor type '{desc_type.name}' expect file extension"
                f" '{DescriptorFactory.Extensions[desc_type]}': {file_path}"
            )

        if desc_type is DescriptorType.EXTRA_INFO and not file_path.match("*.json"):
            raise ValueError(
                f"Descriptor type '{desc_type.name}' expect file extension"
                f" '.json': {file_path}"
            )

    @staticmethod
    def create(
        path: Union[str, Path], *, content: Optional[Dict] = None
    ) -> "Descriptor":
        """Factory function for create descriptor.

        @param path    path of the descriptor
        @param content instead of reading from file_path, use raw_data instead
        """
        if isinstance(path, str):
            path = Path(path)
        desc_type = DescriptorFactory.determine_descriptor_type(path)
        # registered implementation has static member desc_type added by factory
        if desc_type not in DescriptorFactory._factory:
            raise ValueError(
                f"Descriptor type '{desc_type.name}' has no implementation"
                " registered for it"
            )
        if content is None:
            descriptor = DescriptorFactory._factory[desc_type].from_json_file(path)
        else:
            descriptor = DescriptorFactory._factory[desc_type].from_json_data(
                content, path
            )
        return descriptor
