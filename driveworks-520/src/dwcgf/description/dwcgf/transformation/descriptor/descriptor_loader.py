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
"""Data structures for Descriptor Reference Graph."""
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from .application_descriptor import ApplicationDescriptor, ProcessDefinition
from .component_descriptor import ComponentDescriptor
from .descriptor import Descriptor
from .descriptor_factory import DescriptorFactory
from .graphlet_descriptor import GraphletDescriptor
from .roadcast_service_descriptor import RoadCastServiceDescriptor


class DescriptorLoader:
    """class for loading all referenced descriptors from files."""

    @staticmethod
    def _resolved_path(path: Path, absolute: bool = True) -> Path:
        """Return resolved path. (Only works when python > 3.6)."""
        # bazel rely on links, must not resolve soft links
        ret = os.path.abspath(path)
        if not absolute:
            ret = os.path.relpath(ret)
        return Path(ret)

    def _handle_path_offset(self, path: Union[Path, str]) -> Union[Path, str]:
        """If path does not exist, check if offseted path exists.

        If offseted path exists, return offseted one, otherwise return orignal one.
        The path passed in should be a relative path respect to current working directory.
        """
        if self._path_offset is None:
            return DescriptorLoader._resolved_path(Path(path))
        path = Path(path)
        if path.exists():
            return path

        path_str = str(path)
        offset_str = str(self._path_offset)
        if path_str.startswith(offset_str):
            return DescriptorLoader._resolved_path(
                Path(path_str[len(offset_str) + 1 :])
            )
        elif offset_str in path_str:
            idx = path_str.find(offset_str)
            return DescriptorLoader._resolved_path(
                Path(path_str[idx + len(offset_str) + 1 :])
            )
        else:
            return DescriptorLoader._resolved_path(Path(offset_str + "/" + path_str))

    def __init__(
        self,
        root_descriptor: Union[str, Path, Descriptor],
        path_offset: Optional[Path] = None,
    ):
        """Create a DescriptorLoader instance.

        @param root_descriptor root descriptor file path, or a descriptor
        @param path_offset     this is for bazel integration, bazel put source files and
                               generated files in different root directly, path_offset is
                               the relative path of the two roots.
        All descriptor files recursively referenced by the file will also be loaded
        """
        self._path_offset = path_offset
        self._all_descriptors: Dict[Path, Descriptor] = {}  # key is resolved path
        if isinstance(root_descriptor, (str, Path)):
            self._root_descriptor = DescriptorFactory.create(
                self._handle_path_offset(root_descriptor)
            )
        else:
            self._root_descriptor = root_descriptor
        self.add_descriptor(self._root_descriptor)
        self._load_referenced_descriptors(self._root_descriptor)

    def add_descriptor(self, descriptor: Descriptor) -> None:
        """Add newly created descripton to the map.

        @param descriptor newly created descriptor
        Add the descriptor to the map if the map doesn't contain the descriptor.
        """
        descriptor_resolved_path = DescriptorLoader._resolved_path(descriptor.file_path)
        if descriptor_resolved_path not in self._all_descriptors:
            self._all_descriptors[descriptor_resolved_path] = descriptor

    def _load_referenced_descriptors(self, descriptor: Descriptor) -> None:
        """Recursively load all referenced descriptors.

        @param descriptor the function will load all descriptors referenced by it
        """
        for file_path in descriptor.referenced_descriptors:
            referenced_descriptor = self.get_descriptor_by_path(file_path)
            self._load_referenced_descriptors(referenced_descriptor)

    @property
    def descriptors(self) -> Dict[Path, Descriptor]:
        """Return all descriptors loaded by this DescriptorLoader."""
        return self._all_descriptors

    def get_descriptor_by_path(self, file_path: Path) -> Descriptor:
        """Get a descriptor instance by descriptor file path.

        @param file_path absolute path or relative path relative to current working directory
        """
        resolved_path = DescriptorLoader._resolved_path(file_path)
        if resolved_path in self._all_descriptors:
            return self._all_descriptors[resolved_path]

        # the file_path is not loaded, try to load
        new_desc = DescriptorFactory.create(
            self._handle_path_offset(DescriptorLoader._resolved_path(file_path, False))
        )
        self.add_descriptor(new_desc)
        return new_desc

    def get_subcomponent_descriptors(
        self, desc: Union[ApplicationDescriptor, GraphletDescriptor]
    ) -> Dict[str, ComponentDescriptor]:
        """Get all subcomponent descriptor of an ApplicationDescriptor or GraphletDescriptor.

        @param desc application or graphlet descriptor,
                    desc must be a descriptor loaded by this loader

        Return is a dict where the key is the subcomponent relative name
        and value is the descriptor

        All subcomponent descriptors should have been loaded at init
        of this loader, so it's a fatel error if a subcomponent descriptor
        cannot be found.
        """
        subcomps = {}
        for k, v in desc.subcomponents.items():
            subcomp = self.get_descriptor_by_path(desc.dirname / v.component_type)
            if isinstance(subcomp, ComponentDescriptor):
                subcomps[k] = subcomp
            else:
                raise ValueError(
                    f"Cannot find descriptor for subcomponent '{k}' of"
                    f" descriptor '{desc.name}', wrong descriptor type"
                )
        return subcomps

    def get_roadcastservice_descriptor(
        self, desc: ProcessDefinition
    ) -> Union[RoadCastServiceDescriptor, None]:
        """Get the RoadCastService descriptor of an ApplicationDescriptor.

        @param desc application descriptor,
                    desc must be a descriptor loaded by this loader

        Return the RoadCastServiceDescriptor or None as this is optional
        """
        if desc.services is None or "RoadCastService" not in desc.services:
            return None
        p = desc.services["RoadCastService"].parameters
        if "configurationFile" not in p:
            raise KeyError(
                "The configuration file for RoadCastService is missing in the description."
            )
        rcs_config_file = p["configurationFile"]
        rcs_desc = self.get_descriptor_by_path(desc.dirname / rcs_config_file)
        if not isinstance(rcs_desc, RoadCastServiceDescriptor):
            raise ValueError("Cannot find descriptor for RoadCastService")
        return rcs_desc

    def get_referenced_descriptors(self, desc: Descriptor) -> List[Descriptor]:
        """Get all referenced descriptors of a descriptor.

        @param desc the descriptor must be a descriptor loaded by this loader
        Return is a list of descriptors referenced by input descriptor

        All referenced descriptors should have been loaded at init of this loader,
        so it's a fatel error if a referenced descriptor cannot be found.
        """
        referenced_desc = []
        for file_path in desc.referenced_descriptors:
            comp = self.get_descriptor_by_path(file_path)
            referenced_desc.append(comp)
        return referenced_desc

    @property
    def root_descriptor(self) -> Descriptor:
        """Return root descriptor."""
        return self._root_descriptor
