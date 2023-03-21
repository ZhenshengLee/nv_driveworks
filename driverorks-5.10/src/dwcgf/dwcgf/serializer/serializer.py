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
"""Data structures for serializer."""
from copy import deepcopy
import os
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from dwcgf.descriptor import ApplicationDescriptor
from dwcgf.descriptor import Descriptor
from dwcgf.descriptor import DescriptorFactory
from dwcgf.descriptor import DescriptorLoader
from dwcgf.descriptor import DescriptorType
from dwcgf.descriptor import ExtraInfoDescriptor
from dwcgf.descriptor import GraphletDescriptor
from dwcgf.descriptor import NodeDescriptor
from dwcgf.descriptor import PortDefinition
from dwcgf.descriptor import RequiredSensorsDescriptor
from dwcgf.descriptor import SubcomponentDefinition
from dwcgf.object_model import Application
from dwcgf.object_model import Component
from dwcgf.object_model import Graphlet
from dwcgf.object_model import Node
from dwcgf.object_model import Port
from dwcgf.object_model import PortArray


class Serializer:
    """For serialize object module back to descriptor and back to file."""

    def __init__(
        self,
        *,
        loader: DescriptorLoader,
        root: Union[Application, Graphlet],
        force_override: bool = False,
        output_dir: Optional[str] = None,
        path_offset: Optional[str] = None,
    ):
        """Create a serializer.

        @param loader         loader by which the root is loaded, transformed
                              descriptor will also be added to loader.
        @param root           the root graphlet or application.
        @param force_override override the existing file when serializing.
        @param output_dir     the output directory of the serializer.
        """
        self._loader = loader
        self._root = root
        self._force_override = force_override
        self._output_dir = output_dir
        self._path_offset = path_offset

        # key is the base name, value is a list of derived number
        self._derived_names: Dict[str, List[int]] = {}

        self._init_derived_name()

    @property
    def loader(self) -> DescriptorLoader:
        """get the loader of the serializer."""
        return self._loader

    def _init_derived_name(self) -> None:
        """Init _derived_name map for already loaded descriptors."""
        for descriptor in self._loader.descriptors.values():
            if not isinstance(
                descriptor, (RequiredSensorsDescriptor, ExtraInfoDescriptor)
            ):
                self._record_name(descriptor.name)

    def _split_name(self, name: str) -> Tuple[str, Optional[int]]:
        """Split name and extract the base name and derived number info."""
        if "_derived_" in name:
            names = name.split("_derived_")
            return (names[0], int(names[1]))
        else:
            return (name, None)

    def _record_name(self, new_name: str) -> None:
        """Record all descriptor names and update _derived_name."""
        base_name, derived_number = self._split_name(new_name)
        if base_name not in self._derived_names:
            self._derived_names[base_name] = (
                [] if derived_number is None else [derived_number]
            )
        else:
            if derived_number is not None:
                self._derived_names[base_name].append(derived_number)

    def _get_next_derived_suffix(self, name: str) -> str:
        """Get next derived name suffix.

        The suffix can be appended to descriptor name and descriptor file name.
        """
        base_name, _ = self._split_name(name)
        derived_numbers = self._derived_names[base_name]
        if len(derived_numbers) > 0:
            derived_numbers.sort()
            return f"_derived_{derived_numbers[-1] + 1}"
        else:
            return f"_derived_0"

    def serialize_to_file(
        self, change_dest_path: Optional[Callable[[Path], Path]] = None
    ) -> None:
        """Serialize the root and all subcomponents to files."""
        self.serialize_to_descriptor_impl(self._root)
        self.serialize_to_file_impl(self._root, change_dest_path)

    def serialize_to_file_impl(
        self,
        instance: Union[Application, Component],
        change_dest_path: Optional[Callable[[Path], Path]] = None,
    ) -> None:
        """Implementation of serialize_to_file."""
        desc = self._loader.get_descriptor_by_path(instance.descriptor_path)
        if change_dest_path is not None:
            desc.to_json_file(
                force_override=self._force_override,
                dest_path=change_dest_path(desc.file_path),
            )
        else:
            desc.to_json_file(force_override=self._force_override)

        if not isinstance(instance, Node):
            for subcomponent in instance.subcomponents.values():
                self.serialize_to_file_impl(subcomponent, change_dest_path)

    def serialize_to_descriptor(self) -> Path:
        """Serialize the root and all subcomponents to new Descriptor instance."""
        return self.serialize_to_descriptor_impl(self._root)

    def serialize_to_descriptor_impl(
        self, instance: Union[Application, Component]
    ) -> Path:
        """Implementation of the serialize_to_descriptor()."""
        if not instance.is_modified:
            return instance.descriptor_path

        subcomp_paths: Dict[str, Path] = {}
        if isinstance(instance, (Application, Graphlet)):
            subcomp_paths = {}
            for subcomp_name, subcomp in instance.subcomponents.items():
                subcomp_paths[subcomp_name] = self.serialize_to_descriptor_impl(subcomp)
        # serialize node, app or graphlet to descriptor
        desc = self.to_descriptor(instance, subcomp_paths)
        return desc.file_path

    def node_to_descriptor(self, instance: Node) -> NodeDescriptor:
        """Node is not supported to be changed at this point."""
        return self._loader.get_descriptor_by_path(instance.descriptor_path)

    def _create_port_definitions(
        self, ports: Dict[str, Port]
    ) -> Dict[str, PortDefinition]:
        """Construct port definitions from OM."""
        ret = {}
        for name, port in ports.items():
            if isinstance(port, PortArray):
                port_def = PortDefinition(
                    name=name,
                    data_type=port.data_type,
                    array_size=len(port.ports),
                    binding_required=port.binding_required,
                )
            else:
                port_def = PortDefinition(
                    name=name,
                    data_type=port.data_type,
                    array_size=None,
                    binding_required=port.binding_required,
                )
            ret[name] = port_def
        return ret

    def _remove_path_offset(self, path: Path) -> Path:
        """Remove path offset from the path.

        Only should be used when compute the relative path in generated file.
        Compute relative path as if they were under the same root.
        """
        if self._path_offset is None or self._path_offset == "":
            return path

        return Path(str(path).replace(self._path_offset + "/", ""))

    def _create_subcomponent_definitions(
        self,
        instance: Union[Application, Graphlet],
        instance_file_path: Path,
        subcomp_paths: Dict[str, Path],
    ) -> Dict[str, SubcomponentDefinition]:
        """Construct subcomponent definitions from OM.

        @param instance           the graphlet or application owns the subcomponents
        @param instance_file_path the file path of the descriptor of the graphlet or application
        @param subcomp_paths      the updated subcomponent paths
        """
        ret = {}
        for name in instance.subcomponents:
            subcomp_def = SubcomponentDefinition(
                name=name,
                component_type=os.path.relpath(
                    self._remove_path_offset(subcomp_paths[name]),
                    self._remove_path_offset(instance_file_path.parent),
                ),
                parameters=deepcopy(instance.parameter_mappings.get(name, None)),
            )
            ret[name] = subcomp_def
        return ret

    def graphlet_to_descriptor(
        self, instance: Graphlet, file_path: Path, subcomp_paths: Dict[str, Path]
    ) -> GraphletDescriptor:
        """Convert Graphlet to GraphletDescriptor.

        @param instance      the graphlet to be converted
        @param file_path     the file path of new descriptor
        @param subcomp_paths the updated subcomponent paths
        """
        original_desc = self._loader.get_descriptor_by_path(instance.descriptor_path)
        return GraphletDescriptor(
            file_path=original_desc.file_path,
            name=original_desc.name,
            parameters=deepcopy(instance.parameters),
            input_ports=self._create_port_definitions(instance.input_ports),
            output_ports=self._create_port_definitions(instance.output_ports),
            subcomponents=self._create_subcomponent_definitions(
                instance, file_path, subcomp_paths
            ),
            connections=deepcopy(instance.connection_definitions),
            devviz_components=deepcopy(instance.devviz_components),
            generated_from_gdl=instance.generated_from_gdl,
        )

    def application_to_descriptor(
        self, instance: Application, file_path: Path, subcomp_paths: Dict[str, Path]
    ) -> ApplicationDescriptor:
        """Convert Graphlet to GraphletDescriptor.

        @param instance      the application to be converted
        @param file_path     the file path of new descriptor
        @param subcomp_paths the updated subcomponent paths
        """

        original_desc = self._loader.get_descriptor_by_path(instance.descriptor_path)
        # TODO(hongwang, xingpengd): WAR by removing the first dot of the path.
        # This part will be removed when adding the logic to serialize and
        # deserialize the required_sensors and extra_info
        if instance.required_sensors is not None:
            required_sensors = str(instance.required_sensors)
            if self._output_dir and required_sensors[0:1] == "./":
                required_sensors = required_sensors[1:]
        else:
            required_sensors = ""
        if instance.extra_info is not None:
            extra_info = str(instance.extra_info)
            if self._output_dir and extra_info[0:1] == "./":
                extra_info = extra_info[1:]
        else:
            extra_info = ""
        if instance.sensor_mapping_lookups is not None:
            sensor_mapping_lookups = []
            for path in instance.sensor_mapping_lookups:
                path_str = str(path)
                if self._output_dir and path_str.startswith("../../"):
                    path_str = path_str[3:]
                sensor_mapping_lookups.append(Path(path_str))
        else:
            sensor_mapping_lookups = []

        process_desc = {
            process_name: instance.processes[process_name].to_descriptor()
            for process_name in instance.processes.keys()
        }

        return ApplicationDescriptor(
            file_path=original_desc.file_path,
            name=original_desc.name,
            version=instance.version,
            parameters=deepcopy(instance.parameters),
            input_ports=self._create_port_definitions(instance.input_ports),
            output_ports=self._create_port_definitions(instance.output_ports),
            subcomponents=self._create_subcomponent_definitions(
                instance, file_path, subcomp_paths
            ),
            connections=deepcopy(instance.connection_definitions),
            log_spec=instance.log_spec,
            states={
                name: deepcopy(state.state) for name, state in instance.states.items()
            },
            stm_schedules={
                name: deepcopy(sched.schedule)
                for name, sched in instance.schedules.items()
            },
            rcs=deepcopy(instance.rcs),
            framework_pass=deepcopy(instance.framework_pass),
            stm_external_runnables={
                name: deepcopy(runnable)
                for name, runnable in instance.stm_external_runnables.items()
            },
            processes=process_desc,
            required_sensors=required_sensors,
            multi_machine_required_sensors=deepcopy(
                instance.multi_machine_required_sensors
            ),
            sensor_mapping_lookups=deepcopy(sensor_mapping_lookups),
            extra_info=extra_info,
        )

    def to_descriptor(
        self, instance: Union[Application, Component], subcomp_paths: Dict[str, Path]
    ) -> Descriptor:
        """Convert node or graphlet or application to new Descriptor.

        @param instance      the node or graphlet or application to be converted
        @param subcomp_paths the updated subcomponent paths, can be empty dict
        """
        original_desc = self._loader.get_descriptor_by_path(instance.descriptor_path)

        # resolve derived descriptor file path and name
        base_file_name, _ = self._split_name(original_desc.file_path.stem.split(".")[0])
        base_name, _ = self._split_name(original_desc.name)

        if isinstance(instance, Application):
            file_ext = DescriptorFactory.Extensions[DescriptorType.APPLICATION]
        elif isinstance(instance, Graphlet):
            file_ext = DescriptorFactory.Extensions[DescriptorType.GRAPHLET]
        else:
            file_ext = DescriptorFactory.Extensions[DescriptorType.NODE]

        while True:
            derived_suffix = self._get_next_derived_suffix(original_desc.name)
            file_name = f"{base_file_name}{derived_suffix}{file_ext}"
            new_name = f"{base_name}{derived_suffix}"
            if self._output_dir is not None:
                # root file will be serialized separately
                if instance == self._root:
                    file_path = Path(self._output_dir + file_ext)
                    break
                # other files will be serialized into the output dir
                else:
                    file_path = Path(self._output_dir).joinpath(file_name)
            else:
                file_path = original_desc.file_path.with_name(file_name)
            self._record_name(new_name)
            if not file_path.exists():
                break
            # file already exists, skip the name

        if isinstance(instance, Application):
            new_desc = self.application_to_descriptor(
                instance, file_path, subcomp_paths
            )
        elif isinstance(instance, Graphlet):
            new_desc = self.graphlet_to_descriptor(instance, file_path, subcomp_paths)
        else:
            new_desc = self.node_to_descriptor(instance)

        new_desc.name = new_name
        new_desc.file_path = file_path

        # add the new descriptor to loader
        self._loader.add_descriptor(new_desc)
        # update the descriptor path that the instance references
        instance.descriptor_path = new_desc.file_path
        return new_desc
