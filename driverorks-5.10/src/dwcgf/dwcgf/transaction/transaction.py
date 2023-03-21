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
"""Decorators for mark transaction API pairs."""
from contextlib import contextmanager
from functools import wraps
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union


class UndoContext:
    """Return type of the redo-able APIs."""

    NO_CHANGE: "UndoContext" = None  # type: ignore

    def __init__(self, *args: Any, **kwargs: Any):
        """Captures the return value of undo-able API.

        The return value will be passed into its pair API to undo the change
        """
        self.args = args
        self.kwargs = kwargs


UndoContext.NO_CHANGE = UndoContext()


class EditItem:
    """Wraps an redo-able API call."""

    def __init__(
        self, name: str, redo: Callable, args: Tuple, kwargs: Dict, ret: UndoContext
    ):
        """Wraps an API call and expose the undo/redo."""

        if "undo" not in redo.__dict__:
            raise ValueError("Cannot record the API calls without undo")
        self._name = name
        self._redo = redo
        self._args = args
        self._kwargs = kwargs
        self._ret = ret
        self._finished = True

    def undo(self) -> UndoContext:
        """Undo the API call."""
        if self._finished:
            self._finished = False
            return self._redo.undo(*self._ret.args, **self._ret.kwargs)  # type: ignore
        else:
            raise ValueError(
                f"{self._name}: undo cannot be called when finished is False"
            )

    def redo(self) -> UndoContext:
        """Redo the API call."""
        if not self._finished:
            self._finished = True
            return self._redo(*self._args, **self._kwargs)
        raise ValueError(f"{self._name}: redo cannot be called when finished is True")


class Edit:
    """Captures all undo-able API calls from Transaction.begin() to Transaction.end()."""

    def __init__(self) -> None:
        """Record the calls forms an edit."""
        self._callstack: List[EditItem] = []

    def record(self, call: EditItem) -> None:
        """Record an API call."""
        self._callstack.append(call)

    def redo(self) -> Optional[UndoContext]:
        """Redo all API calls."""
        ret = None
        for call in self._callstack:
            ret = call.redo()
        return ret

    def undo(self) -> Optional[UndoContext]:
        """Undo all API calls."""
        ret = None
        for call in reversed(self._callstack):
            ret = call.undo()
        return ret

    def size(self) -> int:
        """Return call stack size."""
        return len(self._callstack)


class Transaction:
    """Used to mark an redo/undo API pair.

    Redo/undo API pair should follow following conventions:
    1. The APIs form a redo/undo pair. One of the API can be
       used to revert change made by another API.
    2. The APIs should report any error with exception. If
       exception happens, nothing should be changed.
    3. If the API call doesn't cause any change, UndoContext.NO_CHANGE
       should be returned.
    4. Redo/undo APIs should return UndoContext contains all the parameters to
       be called by paired undo function to undo the changes made by the API.

    Example: suppose there is an API pair:
      - insert_port(*args, **kwargs) -> UndoContext
      - remove_port(*args, **kwargs) -> UndoContext
      - such API pair should be implemented in a way, such that the original
        instance should be given back after following operation
      ```
      instance = Instance()                                # original instance
      ret1 = instance.insert_port(param1, param2, param3)  # instance changed
      # ret1 contains the parameters needed to call paired function
      # to undo the changes done by insert_port()
      ret2 = instance.remove_port(*ret1.args, **ret1.kwargs)
      # after above call, instance should be exactly the same
      # as original instance. (undo)
      # And ret2 with ret2.args = (param1, param2, param3) and re2.kwargs = {},
      # which means following call redos the change:
      instance.insert_port(*ret2.args, **ret2.kwargs)
      ```

    A large portion of the cases, the modifications to the data structure can be
    categorized as following cases:
    - Update an object attribute to a new value.
    - Insert new item to a list.
    - Remove an item from a list.
    - Update a list item with new value. (Replaced item itself is not changed)
    - Insert new key value pair to a dict.
    - Remove a key from a dict.
    - Update a key with new value in a dict. (Replaced value itself is not changed)
    Transaction module provides 8 helper functions `attr_update()`, `list_insert()`,
    `list_remove()`, `list_remove_by_value()`, `list_update`, `dict_insert()`,
    `dict_remove()`, `dict_update()`, and they are already paired and managed by the
    transaction channel.

    Custom classes can delegate the redo/undo management to these helper functions,
    and most cases custom classes don't need to have paired APIs but still can
    achieve undo/redo functionality.
    (Example: tests/dwcgf/dwcgf/transaction/transaction_tests.py)
    """

    def __init__(self, name: str):
        """Create a Transaction channel with name."""
        self._recording: bool = False
        self._depth: int = 0
        self._edit: Optional[Edit] = Edit()
        self._name = name
        self._utils = {}

        # redo-able API for attribute update
        @self.pair_self
        def attr_update(obj: Any, attr_name: str, new_value: Any) -> UndoContext:
            """Redo-able util managed by this transaction channel for updating an attribute.

            @param obj       The target object.
            @param attr_name The attribute name to update.
            @param new_value The new value for the attribute.
            """
            if not hasattr(obj, attr_name):
                raise ValueError(
                    f"Failed to update the object: '{attr_name}' is not an attribute of the object."
                )

            ret = UndoContext(obj, attr_name, getattr(obj, attr_name))
            setattr(obj, attr_name, new_value)

            return ret

        self._utils["attr_update"] = attr_update

        # redo-able API for list insert/removal
        @self.pair
        def list_insert(
            array: List, value: Any, insert_at: Optional[int] = None
        ) -> UndoContext:
            """Redo-able util managed by this transaction channel for list insertion.

            @param array     The target list.
            @param value     The value to insert.
            @param insert_at The index the new value will be insert at. If the index is
                             not provided or the index is out of bound, the value will
                             be appended.
            """

            if insert_at is not None:
                if insert_at > len(array):
                    insert_at = len(array)
                array.insert(insert_at, value)
            else:
                insert_at = len(array)
                array.append(value)

            return UndoContext(array, insert_at)

        self._utils["list_insert"] = list_insert

        @list_insert.pair  # type: ignore
        def list_remove(array: List, remove_at: int) -> UndoContext:
            """Redo-able util managed by this transaction channel for list removal.

            @param array     The target list.
            @param remove_at The index of the item to be removed. Throws if
                             the index is out of bound.
            """

            if remove_at < 0 or remove_at >= len(array):
                raise ValueError(
                    f"Failed to remove an item from list: index '{remove_at}' out of bound"
                )

            value = array.pop(remove_at)
            return UndoContext(array, value, remove_at)

        self._utils["list_remove"] = list_remove

        @self.pair_self
        def list_update(array: List, index: int, new_value: Any) -> UndoContext:
            """Redo-able util managed by this transaction channel for list updata.

            @param array     The target list.
            @param index     The index to update. If index is out of bound, throws.
            @param new_value The new value for the index.
            """

            if index < 0 or index >= len(array):
                raise ValueError(
                    f"Failed to update an item in list: index '{index}' out of bound"
                )

            ret = UndoContext(array, index, array[index])
            array[index] = new_value

            return ret

        self._utils["list_update"] = list_update

        # redo-able API for dict insert/removal
        @self.pair
        def dict_insert(obj: Dict, key: Any, value: Any) -> UndoContext:
            """Redo-able util managed by this transaction channel for dict insertion.

            @param obj   The target dict.
            @param key   The key to insert, if the key exits already exists, API throws.
            @param value The new value will be inserted.
            """
            if key in obj:
                raise ValueError(
                    f"Failed to insert '{key}' into dict: key already exists."
                )
            obj[key] = value
            return UndoContext(obj, key)

        self._utils["dict_insert"] = dict_insert

        @dict_insert.pair  # type: ignore
        def dict_remove(obj: Dict, key: Any) -> UndoContext:
            """Redo-able util managed by this transaction channel for dict removal.

            @param obj The target dict.
            @param key The key to remove, if the key dosen't exits exists, API throws.
            """
            if key not in obj:
                raise ValueError(
                    f"Failed to remove '{key}' from dict: key dosen't exist."
                )
            value = obj.pop(key)
            return UndoContext(obj, key, value)

        self._utils["dict_remove"] = dict_remove

        @self.pair_self
        def dict_update(obj: Dict, key: Any, new_value: Any) -> UndoContext:
            """Redo-able util managed by this transaction channel for dict update.

            @param obj       The target dict.
            @param key       The key to update, if the key dosen't exits exists, API throws.
            @param new_value The new value for the key.
            """
            if key not in obj:
                raise ValueError(
                    f"Failed to update '{key}' from dict: key dosen't exist."
                )
            old_value = obj[key]
            obj[key] = new_value
            return UndoContext(obj, key, old_value)

        self._utils["dict_update"] = dict_update

    def attr_update(self, obj: Any, attr_name: str, new_value: Any) -> None:
        """Redo-able API managed by this transaction channel for updating an attribute.

        @param obj       The target object.
        @param attr_name The attribute name to attr_update.
        @param new_value The new value for the attribute.
        """
        self._utils["attr_update"](obj, attr_name, new_value)

    def list_insert(
        self, array: List, value: Any, insert_at: Optional[int] = None
    ) -> None:
        """Redo-able util managed by this transaction channel for list insertion.

        @param array     The target list.
        @param value     The value to insert.
        @param insert_at The index the new value will be insert at. If the index is
                         not provided or the index is out of bound, the value will
                         be appended.
        """
        self._utils["list_insert"](array, value, insert_at)

    def list_remove(self, array: List, remove_at: int) -> None:
        """Redo-able util managed by this transaction channel for list removal.

        @param array     The target list.
        @param remove_at The index of the item to be removed. Throws if the index is
                         out of bound.
        """
        self._utils["list_remove"](array, remove_at)

    def list_remove_by_value(self, array: List, value: Any) -> None:
        """Redo-able util managed by this transaction channel for list removal by value.

        @param array The target list.
        @param value The value to be removed from the list. Throws if the value dosen't exist.

        Note: Only first occurrence of the value will be removed.
        """
        index = array.index(value)  # if cannot find the value, index() throws
        self._utils["list_remove"](array, index)

    def list_update(self, array: List, index: int, new_value: Any) -> None:
        """Redo-able util managed by this transaction channel for list updata.

        @param array     The target list.
        @param index     The index to update. If index is out of bound, throws.
        @param new_value The new value for the index.
        """
        self._utils["list_update"](array, index, new_value)

    def dict_insert(self, obj: Dict, key: Any, value: Any) -> None:
        """Redo-able util managed by this transaction channel for dict insertion.

        @param obj   The target dict.
        @param key   The key to insert, if the key exits already exists, API throws.
        @param value The new value will be inserted.
        """
        self._utils["dict_insert"](obj, key, value)

    def dict_remove(self, obj: Dict, key: Any) -> None:
        """Redo-able util managed by this transaction channel for dict removal.

        @param obj The target dict.
        @param key The key to remove, if the key dosen't exits exists, API throws.
        """
        self._utils["dict_remove"](obj, key)

    def dict_update(self, obj: Dict, key: Any, new_value: Any) -> None:
        """Redo-able util managed by this transaction channel for dict update.

        @param obj       The target dict.
        @param key       The key to update, if the key dosen't exits exists, API throws.
        @param new_value The new value for the key.
        """
        self._utils["dict_update"](obj, key, new_value)

    @property
    def name(self) -> str:
        """Return name of this transaction channel."""
        return self._name

    def pair(self, api: Callable) -> Callable:
        """Mark one of the API.

        Add a method called `undo` to `api` function object,
        and `undo` points to another API decorated by `@api.pair`.
        """

        def another_pair(another_api: Callable) -> Callable:
            """Mark another API.

            Add a method called `undo` to `another_api` function object,
            and `undo` points to the API decorated by `@transaction.pair`.
            """

            @wraps(another_api)
            def wraps_another_api(*args: Any, **kwargs: Any) -> UndoContext:
                self._depth = self._depth + 1
                try:
                    ret = another_api(*args, **kwargs)
                    if (
                        self._recording
                        and self._depth == 1
                        and ret is not UndoContext.NO_CHANGE
                    ):
                        self._edit.record(  # type: ignore
                            EditItem(
                                another_api.__name__, another_api, args, kwargs, ret
                            )
                        )
                finally:
                    self._depth = self._depth - 1
                return ret

            if "undo" in api.__dict__:
                raise ValueError("undo function already exists!!")
            another_api.undo = api  # type: ignore
            api.undo = another_api  # type: ignore
            return wraps_another_api

        @wraps(api)
        def wraps_api(*args: Any, **kwargs: Any) -> UndoContext:
            self._depth = self._depth + 1
            try:
                ret = api(*args, **kwargs)
                if (
                    self._recording
                    and self._depth == 1
                    and ret is not UndoContext.NO_CHANGE
                ):
                    self._edit.record(  # type: ignore
                        EditItem(api.__name__, api, args, kwargs, ret)
                    )
            finally:
                self._depth = self._depth - 1
            return ret

        wraps_api.pair = another_pair  # type: ignore

        return wraps_api

    def pair_self(self, api: Callable) -> Callable:
        """Mark the API itself as the undo function.

        Add a method called `undo` to `api` function object,
        and `undo` points to itself.
        """

        @wraps(api)
        def wraps_api(*args: Any, **kwargs: Any) -> UndoContext:
            self._depth = self._depth + 1
            try:
                ret = api(*args, **kwargs)
                if (
                    self._recording
                    and self._depth == 1
                    and ret is not UndoContext.NO_CHANGE
                ):
                    self._edit.record(  # type: ignore
                        EditItem(api.__name__, api, args, kwargs, ret)
                    )
            finally:
                self._depth = self._depth - 1
            return ret

        api.undo = api  # type: ignore

        return wraps_api

    def forbidden(self, api: Callable) -> Callable:
        """Mark the API is forbidden to be called during transation recording."""

        @wraps(api)
        def wraps_api(*args: Any, **kwargs: Any) -> Any:
            if self._recording and self._depth == 0:
                raise ValueError(
                    f"during transaction recording, method '{api.__name__}' is"
                    " forbidden to be called outside of transaction managed API."
                )
            return api(*args, **kwargs)

        return wraps_api

    def ignore(self, api: Callable) -> Callable:
        """Mark the function in which the calls will be ignored by transaction system.

        Right now the only reasonable use case is to mark __init__(), because the
        changes happened in __init__() is for constructing, should not view it as
        undo-able change.

        Note: it's not required to mark the __init__ as ignored, the difference
        is the object constructed inside the `with channel.recording()`, if an undo()
        is called, the changes inside __init__ will also be reverted, and the undo()
        may lead to unconsistent state.
        """

        @wraps(api)
        def wraps_api(*args: Any, **kwargs: Any) -> Any:
            original_recording = self._recording
            self._recording = False  # disable recording
            try:
                ret = api(*args, **kwargs)
            finally:
                self._recording = original_recording  # restore recording
            return ret

        return wraps_api

    def capture_exception(
        self,
        exception_types: Union[Exception, Tuple[Exception]],
        exception_handler: Callable[[Exception], Optional[bool]],
    ) -> None:
        """Capture the exception with a callback."""
        self._exception_types: Union[
            None, Exception, Tuple[Exception]
        ] = exception_types
        self._exception_handler: Optional[
            Callable[[Exception], Optional[bool]]
        ] = exception_handler

    @contextmanager
    def recording(self, edit: Optional[Edit] = None) -> Generator:
        """Begin the transation recording."""

        self._edit = edit if edit is not None else Edit()
        self._recording = True
        try:
            yield self._edit
        except Exception as e:
            if (
                hasattr(self, "_exception_handler")
                and hasattr(self, "_exception_types")
                and self._exception_handler is not None
                and self._exception_types is not None
                and isinstance(e, self._exception_types)  # type: ignore
            ):
                if self._exception_handler(e):
                    # customer exception handler instruct to rethrow
                    raise e
                # exception handled by customer handler
            else:
                # no exception handler setup, rethrow
                raise e
        finally:
            self._exception_handler = None
            self._exception_types = None
            self._recording = False
            self._edit = None
