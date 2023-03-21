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
"""Data structures for Descriptor base class and factory."""
from collections import OrderedDict
import json
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from .descriptor_factory import DescriptorFactory
from .descriptor_factory import DescriptorType


class Descriptor:
    """Base class for all descriptor files in memory."""

    desc_type: DescriptorType = DescriptorType.INVALID

    def __init__(self, file_path: Path):
        """Create a Descriptor instance.

        @param file_path file path of the document
        """

        self._file_path = file_path

    @property
    def basename(self) -> str:
        """Return basename of the descriptor file."""
        return self._file_path.name

    @property
    def dirname(self) -> Path:
        """Return dirname of the descriptor file."""
        return self._file_path.parent

    @property
    def file_path(self) -> Path:
        """Return the file path of the descriptor file."""
        return self._file_path

    @file_path.setter
    def file_path(self, new_path: Path) -> None:
        """Set new file path."""
        self._file_path = new_path

    @property
    def referenced_descriptors(self) -> List[Path]:
        """Return the descriptor files referenced by this descriptor.

        Return value is an array where element is the file path
        """
        return []

    @classmethod
    def from_json_file(cls, path: Union[str, Path]) -> "Descriptor":
        """Create a Descriptor from JSON file.

        @param path path to the JSON file.
        """
        path = Path(path)
        DescriptorFactory.check_descriptor_extension(path, cls.desc_type)
        return cls.from_json_data(json.loads(path.read_text()), path)

    @classmethod
    def from_json_data(cls, content: Dict, path: Union[str, Path]) -> "Descriptor":
        """Create a Descriptor from JSON file.

        @param content content of the JSON.
        @param path    path indicates the content path, relative path
                       in passed in JSON is relative to this path.
        """
        raise NotImplementedError("from_json_data() is not implemented")

    def to_json_file(
        self, *, force_override: bool = False, dest_path: Optional[Path] = None
    ) -> None:
        """Dump the descriptor to JSON file.

        @param force_override override the exist file.
        @param dest_path      serialize to this file path instead.
        """
        dest_path = dest_path if dest_path is not None else self.file_path
        if not dest_path.exists() or force_override:
            json_data = self.to_json_data()
            dest_path.write_text(json.dumps(json_data, indent=2))

    def to_json_data(self) -> OrderedDict:
        """Convert the descriptor into JSON data."""
        raise NotImplementedError("to_json_data() is not implemented")

    def __eq__(self, other: object) -> bool:
        """Compare the two descriptor."""
        if not isinstance(other, Descriptor):
            return NotImplemented
        return self.to_json_data() == other.to_json_data()
