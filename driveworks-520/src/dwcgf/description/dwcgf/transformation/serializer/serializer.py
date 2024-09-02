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
"""Data structures for serializer."""
from collections import OrderedDict
from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Callable, cast, Dict, List, Optional, Tuple, Union

from dwcgf.transformation.descriptor import (
    ApplicationDescriptor,
    ComponentDescriptor,
    Descriptor,
    DescriptorFactory,
    DescriptorLoader,
    DescriptorType,
    ExtraInfoDescriptor,
    GraphletDescriptor,
    NodeDescriptor,
    PortDefinition,
    ProcessDefinition,
    RequiredSensorsDescriptor,
    SubcomponentDefinition,
)
from dwcgf.transformation.object_model import (
    Application,
    Component,
    Graphlet,
    Node,
    Port,
    PortArray,
    RoadCastService,
)


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
            return "_derived_0"

    def serialize_to_json_data(self) -> OrderedDict:
        """Serialize the root and all subcomponents to single file.

        The root instance of this Serializer has to be an Application instance.
        """
        if not isinstance(self._root, Application):
            raise ValueError(
                "serialize_to_json_data() only supports Application instance."
            )
        self.serialize_to_descriptor_impl(self._root)

        path_to_name_mapping = {
            os.path.abspath(desc.file_path): desc.name
            for desc in self._loader.descriptors.values()
            if isinstance(desc, ComponentDescriptor)
        }

        def _convert_path_to_name(
            desc: ComponentDescriptor, component_type: str
        ) -> str:
            """Convert the component path to component type name."""
            comp_desc_path = desc.dirname / component_type

            # Cases:
            # 1. generated files refer to(->) generated ones / input files -> input files
            abs_path = os.path.abspath(comp_desc_path)
            if abs_path in path_to_name_mapping:
                return path_to_name_mapping[abs_path]
            # 2. generated -> input
            comp_desc_path = self._remove_path_offset(comp_desc_path)
            abs_path = os.path.abspath(comp_desc_path)
            if abs_path in path_to_name_mapping:
                return path_to_name_mapping[abs_path]
            # 3. input -> generated
            comp_desc_path = self._add_path_offset(comp_desc_path)
            abs_path = os.path.abspath(comp_desc_path)
            if abs_path in path_to_name_mapping:
                return path_to_name_mapping[abs_path]

            raise ValueError(f"'{abs_path}' is not in the loader")

        def _process_desc_json(
            desc: ComponentDescriptor, desc_json: OrderedDict
        ) -> None:
            """Replace subcomponent types."""
            for _, subcomponent in desc_json.get("subcomponents", {}).items():
                subcomponent["componentType"] = _convert_path_to_name(
                    desc, subcomponent["componentType"]
                )

        def _load_referenced_app_files(
            root_dirname: Path, root_desc_json: OrderedDict
        ) -> None:
            def _try_resolve_path_offset_all(
                dir_path: Path, relative_path: Path
            ) -> List[Path]:
                resolved_path: List[Path] = []
                possible_full_path: Path = relative_path
                # Cases:
                # 1. relative_path is actually absolute path
                if possible_full_path.exists():
                    resolved_path.append(possible_full_path)
                # 2. relative_path is not crossing path_offset
                possible_full_path = (dir_path / relative_path).resolve()
                if possible_full_path.exists():
                    resolved_path.append(possible_full_path)
                # 3. generated -> input
                possible_full_path = (
                    self._remove_path_offset(dir_path) / relative_path
                ).resolve()
                if possible_full_path.exists():
                    resolved_path.append(possible_full_path)
                if not resolved_path:
                    raise ValueError(
                        f"try_resolve_path_offset Failure: Path not exist: \
                        {str(dir_path / relative_path)}"
                    )
                return resolved_path

            def _try_resolve_path_offset(dir_path: Path, relative_path: Path) -> Path:
                return _try_resolve_path_offset_all(dir_path, relative_path)[0]

            def _load_referenced_file_from_path(
                path: str, desc_type: DescriptorType
            ) -> OrderedDict:
                resolved_path: Path = _try_resolve_path_offset(root_dirname, Path(path))
                assert desc_type == DescriptorFactory.determine_descriptor_type(
                    resolved_path
                ), f"The desc_tpye '{desc_type}' mismatches w/ path '{path}'"
                loaded_json: OrderedDict = json.loads(resolved_path.read_text())
                return loaded_json

            def _load_referenced_roadcast_service(
                processes: OrderedDict,
            ) -> OrderedDict:
                for process in processes.values():
                    configurationFile: str = (
                        process.get("services", {})
                        .get("RoadCastService", {})
                        .get("parameters", {})
                        .get("configurationFile", {})
                    )
                    if configurationFile:
                        resolved_path: Path = _try_resolve_path_offset(
                            root_dirname, Path(configurationFile)
                        )
                        DescriptorFactory.check_descriptor_extension(
                            resolved_path, DescriptorType.ROADCAST_SERVICE
                        )
                        process["services"]["RoadCastService"]["parameters"][
                            "configurationFile"
                        ] = json.loads(resolved_path.read_text())
                return processes

            def _load_referenced_sensor_mapping(
                sensor_mapping_lookups: List[str],
            ) -> OrderedDict:
                def _read_sensor_mapping_file(
                    path: Path, sensor_mapping_lookups_json: OrderedDict
                ) -> None:
                    def _is_sensor_mapping_file_exists(path: Path) -> bool:
                        return path.is_file() and path.match(
                            "*"
                            + DescriptorFactory.Extensions[
                                DescriptorType.SENSOR_MAPPINGS
                            ]
                        )

                    if _is_sensor_mapping_file_exists(path):
                        sensor_mapping_lookups_json.update(json.loads(path.read_text()))

                sensor_mapping_lookups_json: OrderedDict = OrderedDict()
                for path in sensor_mapping_lookups:
                    # Note the sensor_mapping_lookups can both from input and output,
                    # and both need to be traversed
                    possible_full_paths: List[Path] = _try_resolve_path_offset_all(
                        root_dirname, Path(path)
                    )
                    for resolved_path in possible_full_paths:
                        # 1. File
                        if resolved_path.is_file():
                            _read_sensor_mapping_file(
                                resolved_path, sensor_mapping_lookups_json
                            )
                        # 2. Directory
                        if resolved_path.is_dir():
                            for root, _, files in os.walk(resolved_path):
                                for filename in files:
                                    file_path: Path = Path(root) / Path(filename)
                                    _read_sensor_mapping_file(
                                        file_path, sensor_mapping_lookups_json
                                    )
                return sensor_mapping_lookups_json

            root_desc_json["processes"] = _load_referenced_roadcast_service(
                root_desc_json["processes"]
            )
            root_desc_json["extraInfo"] = _load_referenced_file_from_path(
                root_desc_json["extraInfo"], DescriptorType.EXTRA_INFO
            )
            root_desc_json["requiredSensors"] = _load_referenced_file_from_path(
                root_desc_json["requiredSensors"], DescriptorType.REQUIRED_SENSORS
            )
            if "sensorMappingLookups" in root_desc_json.keys():
                root_desc_json["sensorMappings"] = _load_referenced_sensor_mapping(
                    root_desc_json["sensorMappingLookups"]
                )
                # Remove section 'sensorMappingLookups'
                root_desc_json.pop("sensorMappingLookups")

        descriptions: OrderedDict = OrderedDict()
        self.serialize_to_json_data_impl(self._root, descriptions, _process_desc_json)

        root_desc = cast(
            ComponentDescriptor,
            self._loader.get_descriptor_by_path(self._root.descriptor_path),
        )
        root_desc_json = descriptions.pop(root_desc.name)
        _load_referenced_app_files(root_desc.dirname, root_desc_json)
        root_desc_json["description"] = descriptions
        return root_desc_json

    def serialize_to_json_data_impl(
        self,
        instance: Component,
        descriptions: Dict[str, Dict],
        callback: Callable[[ComponentDescriptor, OrderedDict], None],
    ) -> None:
        """Implementation of the serialize_to_json_data."""
        desc = self._loader.get_descriptor_by_path(instance.descriptor_path)
        if not isinstance(desc, ComponentDescriptor):
            raise ValueError(
                f"Graphlet '{instance.id}' must point to a ComponentDescriptor instance"
            )
        desc_json = desc.to_json_data()
        callback(desc, desc_json)
        if desc.name not in descriptions:
            descriptions[desc.name] = desc_json
        elif desc_json != descriptions[desc.name]:
            raise ValueError(f"Two descriptors use the same name '{desc.name}'")
        if not isinstance(instance, Node):
            for subcomponent in instance.subcomponents.values():
                self.serialize_to_json_data_impl(subcomponent, descriptions, callback)

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

        if isinstance(instance, Application):
            for _, proc_item in instance.processes.items():
                if proc_item.services:
                    self.serialize_rcs_to_file(proc_item.services.rcs, change_dest_path)

    def serialize_rcs_to_file(
        self,
        instance: RoadCastService,
        change_dest_path: Optional[Callable[[Path], Path]] = None,
    ) -> None:
        """Implementation of serialize_rcs_to_file."""
        if instance is None:
            return
        desc = self._loader.get_descriptor_by_path(instance.descriptor_path)
        if change_dest_path is not None:
            desc.to_json_file(
                force_override=self._force_override,
                dest_path=change_dest_path(desc.file_path),
            )
        else:
            desc.to_json_file(force_override=self._force_override)

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

    def _add_path_offset(self, path: Path) -> Path:
        """Add path offset from the path.

        Only should be used when compute the relative path in generated file.
        Compute relative path as if they were under the same root.
        """
        if self._path_offset is None or self._path_offset == "":
            return path

        return Path.joinpath(Path(self._path_offset), Path(path))

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

    def _create_process_definitions(
        self, instance: Application, instance_file_path: Path
    ) -> Dict[str, ProcessDefinition]:
        """Construct subcomponent definitions from OM.

        @param instance           the graphlet or application owns the subcomponents
        @param instance_file_path the file path of the descriptor of the graphlet or application
        """
        ret = {}
        for process_name in instance.processes.keys():
            ret[process_name] = instance.processes[process_name].to_descriptor()
            if (
                instance.processes[process_name].services
                and instance.processes[process_name].services.rcs
            ):
                new_desc_path = self._create_rcs(
                    instance.processes[process_name].services.rcs, instance_file_path
                )
                ret[process_name].services["RoadCastService"].parameters = {
                    "configurationFile": str(new_desc_path)
                }
        return ret

    def _create_rcs(
        self, instance: RoadCastService, instance_file_path: Path
    ) -> Optional[Path]:
        """Construct RoadCastService from OM.

        @param instance           the RoadCastService
        @param instance_file_path the file path of the descriptor of the graphlet or application
        """
        original_desc = self._loader.get_descriptor_by_path(instance.descriptor_path)
        # resolve derived descriptor file path and name
        base_file_name, _ = self._split_name(original_desc.file_path.stem.split(".")[0])
        base_name, _ = self._split_name(original_desc.name)
        file_ext = DescriptorFactory.Extensions[DescriptorType.ROADCAST_SERVICE]

        while True:
            derived_suffix = self._get_next_derived_suffix(original_desc.name)
            file_name = f"{base_file_name}{derived_suffix}{file_ext}"
            new_name = f"{base_name}{derived_suffix}"
            # serialize RoadCastService configuration file into the output dir
            if self._output_dir is not None:
                file_path = Path(self._output_dir).joinpath(file_name)
            else:
                file_path = original_desc.file_path.with_name(file_name)
            self._record_name(new_name)
            if not file_path.exists():
                break
            # file already exists, skip the name

        rcs_new_path = os.path.relpath(
            self._remove_path_offset(file_path),
            self._remove_path_offset(instance_file_path.parent),
        )

        # add new descriptor
        new_desc = instance.to_descriptor()
        new_desc.name = new_name
        new_desc.file_path = file_path

        # add the new descriptor to loader
        self._loader.add_descriptor(new_desc)
        # update the descriptor path that the instance references
        instance.descriptor_path = new_desc.file_path
        return Path(rcs_new_path)

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
            comment=original_desc.comment,
            parameters=deepcopy(instance.parameters),
            input_ports=self._create_port_definitions(instance.input_ports),
            output_ports=self._create_port_definitions(instance.output_ports),
            subcomponents=self._create_subcomponent_definitions(
                instance, file_path, subcomp_paths
            ),
            connections=deepcopy(instance.connection_definitions),
        )

    def application_to_descriptor(
        self, instance: Application, file_path: Path, subcomp_paths: Dict[str, Path]
    ) -> ApplicationDescriptor:
        """Convert Application to ApplicationDescriptor.

        @param instance      the application to be converted
        @param file_path     the file path of new descriptor
        @param subcomp_paths the updated subcomponent paths
        """

        original_desc = self._loader.get_descriptor_by_path(instance.descriptor_path)

        return ApplicationDescriptor(
            file_path=original_desc.file_path,
            name=original_desc.name,
            comment=original_desc.comment,
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
            stm_external_runnables={
                name: deepcopy(runnable)
                for name, runnable in instance.stm_external_runnables.items()
            },
            processes=self._create_process_definitions(instance, file_path),
            required_sensors=instance.required_sensors,
            sensor_mapping_lookups=deepcopy(instance.sensor_mapping_lookups),
            extra_info=instance.extra_info,
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
