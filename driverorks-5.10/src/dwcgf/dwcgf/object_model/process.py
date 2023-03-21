"""Data structures for process instance."""
from copy import deepcopy
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from dwcgf.descriptor import ProcessDefinition
from dwcgf.transaction import UndoContext

from .component import Component
from .object_model_channel import object_model_channel

CMDArgumentValueType = Union[bool, str, List[str]]


class Process:
    """class for process."""

    def __init__(
        self,
        *,
        name: str,
        executable: str,
        run_on: str,
        log_spec: Optional[str],
        argv: Optional[Dict[str, CMDArgumentValueType]] = None,
        subcomponents: Optional[List[Component]] = None,
        extra_info: Optional[Dict] = None,
    ):
        """An entry in processes section."""
        self._name = name
        self._executable = executable
        self._run_on = run_on
        self._log_spec = log_spec
        self._argv = argv
        self._subcomponents = subcomponents

        # this is for restore the extra attributes for the process
        # TODO(xingpengd) this is against schema, only for legacy
        # behavior, need to clean up later
        self._extra_info = extra_info

    @staticmethod
    def from_descriptor(
        desc: ProcessDefinition, subcomponent_instances: Optional[List[Component]]
    ) -> "Process":
        """Create a Process instance from a ProcessDefinition."""
        return Process(
            name=deepcopy(desc.name),
            executable=deepcopy(desc.executable),
            run_on=deepcopy(desc.run_on),
            log_spec=deepcopy(desc.log_spec) if desc.log_spec is not None else None,
            argv=deepcopy(desc.argv) if desc.argv is not None else None,
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
            name=deepcopy(self.name),
            executable=deepcopy(self.executable),
            run_on=deepcopy(self.run_on),
            log_spec=deepcopy(self.log_spec) if self.log_spec is not None else None,
            argv=deepcopy(self.argv) if self.argv is not None else None,
            subcomponents=subcomponent_ids,
            extra_info=deepcopy(self.extra_info)
            if self.extra_info is not None
            else None,
        )

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
