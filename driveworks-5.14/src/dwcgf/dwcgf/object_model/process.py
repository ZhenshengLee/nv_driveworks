"""Data structures for process instance."""
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from dwcgf.descriptor import DescriptorLoader
from dwcgf.descriptor import ProcessDefinition
from dwcgf.descriptor import ProcessServiceDefinition
from dwcgf.descriptor import RoadCastServiceDescriptor
from dwcgf.descriptor import RoadCastServiceParam
from dwcgf.descriptor import SignalProducerPort
from dwcgf.transaction import UndoContext

from .component import Component
from .object_model_channel import object_model_channel


class RoadCastService:
    """class for RoadCastService."""

    def __init__(
        self,
        descriptor_path: Path,
        name: str,
        enabled: bool,
        params: RoadCastServiceParam,
        ports: Dict[str, SignalProducerPort],
        pass_mapping: Dict[str, Dict[str, str]],
    ):
        """Create RoadCastService Instance."""
        self.descriptor_path = descriptor_path
        self._name = name
        self._enabled = enabled
        self._params = params
        self._ports = ports
        self._pass_mapping = pass_mapping
        self._passes: List[str] = []

    @property
    def descriptor_path(self) -> Path:
        """Return descriptor_path."""
        return self._descriptor_path

    @descriptor_path.setter
    def descriptor_path(self, value: Path) -> None:
        """Set the descriptor_path."""
        self._descriptor_path = value

    @property
    def name(self) -> str:
        """Return name."""
        return self._name

    @property
    def enabled(self) -> bool:
        """Return enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Set the enabled."""
        self._enabled = value

    @property
    def params(self) -> RoadCastServiceParam:
        """Return params."""
        return self._params

    @property
    def ports(self) -> Dict[str, SignalProducerPort]:
        """Return ports."""
        return self._ports

    @property
    def pass_mapping(self) -> Dict[str, Dict[str, str]]:
        """Return pass_mapping."""
        return self._pass_mapping

    @property
    def passes(self) -> List[str]:
        """Return passes."""
        return self._passes

    @passes.setter
    def passes(self, value: List[str]) -> None:
        """Set the passes of the process service."""
        self._passes = value

    @staticmethod
    def from_descriptor(desc: RoadCastServiceDescriptor,) -> "RoadCastService":
        """Create a RoadCastService instance from a RoadCastServiceDescriptor."""

        return RoadCastService(
            descriptor_path=deepcopy(desc.file_path),
            name=deepcopy(desc.name),
            enabled=deepcopy(desc.enabled),
            params=deepcopy(desc.params),
            ports=deepcopy(desc.ports),
            pass_mapping=deepcopy(desc.pass_mapping),
        )

    def to_descriptor(self) -> RoadCastServiceDescriptor:
        """dump to RoadCastServiceDescriptor."""
        return RoadCastServiceDescriptor(
            file_path=deepcopy(self.descriptor_path),
            name=deepcopy(self.name),
            enabled=deepcopy(self.enabled),
            params=deepcopy(self.params),
            ports=deepcopy(self.ports),
            pass_mapping=deepcopy(self._pass_mapping),
        )


class EpochSyncService:
    """class for EpochSyncService."""

    def __init__(self, parameters: Dict[str, str], passes: List[str]):
        """Create EpochSyncService Instance."""
        self._parameters = parameters
        self._passes = passes

    @property
    def parameters(self) -> Dict[str, str]:
        """Return parameters."""
        return self._parameters

    @property
    def passes(self) -> List[str]:
        """Return passes."""
        return self._passes


class ProcessServices:
    """class for process services."""

    def __init__(
        self,
        rcs: Optional[RoadCastService] = None,
        sync: Optional[EpochSyncService] = None,
    ):
        """An entry in processe services section."""
        self._rcs = rcs
        self._sync = sync

    @property
    def rcs(self) -> Optional[RoadCastService]:
        """Return RoadCastService of the process services."""
        return self._rcs

    @property
    def sync(self) -> Optional[EpochSyncService]:
        """Return EpochSyncService of the process services."""
        return self._sync

    @staticmethod
    def from_definition(
        pro_def: ProcessDefinition, loader: Optional[DescriptorLoader]
    ) -> Optional["ProcessServices"]:
        """Create a ProcessServices instance from a ProcessDefinition."""
        if pro_def.services is None:
            return None
        if loader is None:
            raise ValueError(
                f"Process service is defined but the given loader is None."
            )
        rcs = None
        if "RoadCastService" in pro_def.services:
            rcs = RoadCastService.from_descriptor(
                loader.get_roadcastservice_descriptor(pro_def)
            )
            # set passes list
            rcs.passes = pro_def.services["RoadCastService"].passes
        sync = None
        if "EpochSyncService" in pro_def.services:
            sync = EpochSyncService(
                parameters=deepcopy(pro_def.services["EpochSyncService"].parameters),
                passes=deepcopy(pro_def.services["EpochSyncService"].passes),
            )
        return ProcessServices(rcs=rcs, sync=sync)

    def to_definition(self) -> Dict[str, ProcessServiceDefinition]:
        """dump to definition."""
        services_def: OrderedDict = OrderedDict()
        if self.rcs:
            services_def["RoadCastService"] = ProcessServiceDefinition(
                name="RoadCastService",
                parameters=deepcopy(
                    {"configurationFile": str(self.rcs.descriptor_path)}
                ),
                passes=deepcopy(self.rcs.passes),
            )
        if self.sync:
            services_def["EpochSyncService"] = ProcessServiceDefinition(
                name="EpochSyncService",
                parameters=deepcopy(self.sync.parameters),
                passes=deepcopy(self.sync.passes),
            )
        return services_def


CMDArgumentValueType = Union[bool, str, List[str]]


class Process:
    """class for process."""

    def __init__(
        self,
        *,
        dirname: Path,
        name: str,
        executable: str,
        run_on: str,
        log_spec: Optional[str],
        argv: Optional[Dict[str, CMDArgumentValueType]] = None,
        services: Optional[ProcessServices] = None,
        desc_subcomponents: Optional[List[str]] = None,
        subcomponents: Optional[List[Component]] = None,
        extra_info: Optional[Dict] = None,
    ):
        """An entry in processes section."""
        self._dirname = dirname
        self._name = name
        self._executable = executable
        self._run_on = run_on
        self._log_spec = log_spec
        self._argv = argv
        self._services = services
        self._desc_subcomponents = desc_subcomponents
        self._subcomponents = subcomponents

        # this is for restore the extra attributes for the process
        # TODO(xingpengd) this is against schema, only for legacy
        # behavior, need to clean up later
        self._extra_info = extra_info

    @staticmethod
    def from_descriptor(
        desc: ProcessDefinition,
        subcomponent_instances: Optional[List[Component]],
        loader: Optional[DescriptorLoader] = None,
    ) -> "Process":
        """Create a Process instance from a ProcessDefinition."""
        return Process(
            dirname=deepcopy(desc.dirname),
            name=deepcopy(desc.name),
            executable=deepcopy(desc.executable),
            run_on=deepcopy(desc.run_on),
            log_spec=deepcopy(desc.log_spec) if desc.log_spec is not None else None,
            argv=deepcopy(desc.argv) if desc.argv is not None else None,
            services=ProcessServices.from_definition(desc, loader),
            desc_subcomponents=deepcopy(desc.subcomponents),
            subcomponents=subcomponent_instances,
            extra_info=deepcopy(desc.extra_info)
            if desc.extra_info is not None
            else None,
        )

    def to_descriptor(self) -> ProcessDefinition:
        """dump to definition."""
        subcomponent_ids = None
        if self.subcomponents is not None:
            subcomponent_ids = [component.id for component in self.subcomponents]

        return ProcessDefinition(
            dirname=deepcopy(self.dirname),
            name=deepcopy(self.name),
            executable=deepcopy(self.executable),
            run_on=deepcopy(self.run_on),
            log_spec=deepcopy(self.log_spec) if self.log_spec is not None else None,
            argv=deepcopy(self.argv) if self.argv is not None else None,
            services=self.services.to_definition() if self.services else None,
            subcomponents=subcomponent_ids,
            extra_info=deepcopy(self.extra_info)
            if self.extra_info is not None
            else None,
        )

    @property
    def dirname(self) -> Path:
        """Return dirname of the process."""
        return self._dirname

    @property
    def name(self) -> str:
        """Return name of the process."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set the name of the process."""
        self._name = value

    @property
    def executable(self) -> str:
        """Return executable name."""
        return self._executable

    @property
    def run_on(self) -> str:
        """Return the machine on which this process running."""
        return self._run_on

    @property
    def log_spec(self) -> Optional[str]:
        """Return the log spec of this process."""
        return self._log_spec

    @property
    def argv(self) -> Optional[Dict[str, CMDArgumentValueType]]:
        """Return the argument list of this process."""
        return self._argv

    @argv.setter
    def argv(self, value: Dict[str, CMDArgumentValueType]) -> None:
        """Set the argument list of this process."""
        self._argv = value

    @property
    def services(self) -> Optional[ProcessServices]:
        """Return the services for this process."""
        return self._services

    @property
    def desc_subcomponents(self) -> Optional[List[str]]:
        """Return the desc_subcomponents for this process."""
        return self._desc_subcomponents

    @property
    def subcomponents(self) -> Optional[List[Component]]:
        """Return the subcomponents for this process."""
        return self._subcomponents

    @subcomponents.setter
    def subcomponents(self, value: Optional[List[Component]]) -> None:
        """Set the subcomponents for this process."""
        self.__set_subcomponents(value)

    @object_model_channel.pair_self
    def __set_subcomponents(self, value: Optional[List[Component]]) -> UndoContext:
        ret = UndoContext(self, self._subcomponents)
        self._subcomponents = value

        return ret

    @property
    def extra_info(self) -> Optional[Dict]:
        """Return extra attributes specified in process descriptor."""
        return self._extra_info
