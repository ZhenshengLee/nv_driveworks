#!/usr/bin/env python3

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

from jsonschema import Draft7Validator as Validator
from jsonschema import validate
from jsonschema.exceptions import ValidationError
from jsonschema.validators import Draft7Validator


SCHEMAS = {
    ".app.json": "app.schema.json",
    ".app-config.json": "app-config.schema.json",
    ".graphlet.json": "graphlet.schema.json",
    ".node.json": "node.schema.json",
    ".node-params.json": "node-params.schema.json",
    ".node-wcets.json": "node-wcets.schema.json",
    ".required-sensors.json": "required-sensors.schema.json",
    ".sensor-mappings.json": "sensor-mappings.schema.json",
    ".trans.json": "trans.schema.json",
}
VALIDATED_FILES = {}


def main():
    global SCHEMAS
    global VALIDATED_FILES

    parser = argparse.ArgumentParser(
        description="Validate ( .app | .app-config | .graphlet | .node | .node-params | .node-wcets | .required-sensors | .sensor-mappings | .trans ) .json files against their schema and additional constraints"
    )
    parser.add_argument(
        "basepaths",
        nargs="+",
        type=Path,
        help="Base path to recusively search for JSON files",
    )
    parser.add_argument(
        "--ignore-order",
        action="store_true",
        help="Ignore if the order of keys doesn't match the schema",
    )
    parser.add_argument(
        "--ignore-indentation",
        action="store_true",
        help="Ignore if the indentation is not a multiple of 4",
    )
    args = parser.parse_args()

    for basepath in args.basepaths:
        if not basepath.exists():
            print(f"Passed path '{basepath}' doesn't exist", file=sys.stderr)
            return 1

    all_valid = True

    # read all schemas and instantiate validators
    for extension, schema_filename in SCHEMAS.items():
        schema = readSchema(schema_filename)
        if not schema:
            all_valid = False
        else:
            SCHEMAS[extension] = instantiateValidator(schema)

    for basepath in args.basepaths:
        if basepath.is_file():
            all_valid &= validateFileAgainstSchema(basepath)
        else:
            for ext in SCHEMAS.keys():
                all_valid &= validateFilesAgainstSchema(basepath, f"**/*{ext}")

    all_valid &= checkFilesAgainstAdditionalConstraints(
        checkFileAgainstAdditionalConstraints
    )

    if not args.ignore_order:
        all_valid &= checkFilesAgainstAdditionalConstraints(checkFileForStrictOrder)
    if not args.ignore_indentation:
        all_valid &= checkFilesAgainstAdditionalConstraints(
            checkFileForStrictIndentation
        )

    print()
    print(
        "Validated "
        + str(len(VALIDATED_FILES))
        + " files successfully against their schema"
    )
    print()

    return 0 if all_valid else 1


def readSchema(schema_name):
    node_schema_path = Path(__file__).parent / schema_name
    try:
        content = node_schema_path.read_text()
        schema = json.loads(content)
    except Exception as e:
        print(f"Schema '{schema_name}' failed to read/load: {e}", file=sys.stderr)
        return None
    try:
        Validator.check_schema(schema)
        return schema
    except Exception as e:
        print(f"Schema '{schema_name}' failed to validate: {e}", file=sys.stderr)
        return None


def instantiateValidator(schema):
    return Draft7Validator(schema)


def validateFilesAgainstSchema(basepath, pattern):
    all_valid = True
    paths = basepath.glob(pattern)
    for path in sorted(paths):
        all_valid &= validateFileAgainstSchema(path)
    return all_valid


def validateFileAgainstSchema(path):
    global VALIDATED_FILES

    validator = getValidator(path)
    if validator is None:
        return False

    try:
        content = path.read_text()
        instance = json.loads(content)
    except Exception as e:
        print(f"! JSON file '{path}' failed to read/load: {e}", file=sys.stderr)
        return False

    errors = list(validator.iter_errors(instance=instance))
    if not errors:
        print(f"\N{check mark} {path}", flush=True)
        VALIDATED_FILES[path.resolve()] = (path, instance)
        return True

    print(f"! {path}", file=sys.stderr)
    for e in errors:
        p = " -> ".join(str(x) for x in e.path)
        print(f"  {p}: {e.message}", file=sys.stderr)
    return False


def getValidator(path):
    for extension, validator in SCHEMAS.items():
        if path.name.endswith(extension):
            return validator
    print(f"No validator found for filename '{path.name}'", file=sys.stderr)
    return None


def checkFilesAgainstAdditionalConstraints(func):
    global VALIDATED_FILES

    all_valid = True
    # VALIDATED_FILES might be added to as part of the constraint checking
    checkedKeys = set()
    while True:
        validatedKeys = set(VALIDATED_FILES.keys())
        validatedKeys -= checkedKeys
        if not validatedKeys:
            break
        key = list(sorted(validatedKeys))[0]
        checkedKeys.add(key)
        path, instance = VALIDATED_FILES[key]
        try:
            all_valid &= func(path, instance)
        except Exception:
            print("Exception while validating:", key, file=sys.stderr)
            raise
    return all_valid


def checkFileAgainstAdditionalConstraints(path, instance):
    errors = []

    for k, v in instance.get("parameters", {}).items():
        if "default" in v:
            errors += checkParameterTypeAndValueMatches(
                f"The default value of", k, v["type"], v.get("array"), v["default"]
            )

    for pass_ in instance.get("passes", []):
        if "dependencies" in pass_:
            assert (
                False
            ), "To be implemented: check that dependency names are in the collection of previous pass names"

    for sc_name, sc in instance.get("subcomponents", {}).items():
        sc_path = path.parent / sc["componentType"]
        for extension in SCHEMAS:
            if sc_path.name.endswith(extension):
                break
        else:
            errors.append(
                f"Subcomponent '{sc_name}' with not recognized file extension '{sc_path}'"
            )
            continue

        sc_path, sc_instance = VALIDATED_FILES.get(sc_path.resolve(), (sc_path, None))
        if sc_instance is None:
            # print(f"Subcomponent '{sc_name}' in '{sc_path}' not validated before - doing it now")
            if not validateFileAgainstSchema(sc_path):
                errors.append(
                    f"Subcomponent '{sc_name}' in '{sc_path}' failed validation"
                )
                continue
            sc_path, sc_instance = VALIDATED_FILES.get(sc_path.resolve())
        for k, v in sc.get("parameters", {}).items():
            # check that wildcard has fixed value
            if k == "*":
                if v != "$*":
                    errors.append(
                        f"Subcomponent '{sc_name}' contains a parameter '*' but the value must be '$*'"
                    )
                continue
            # check subcomponent parameters are defined in the subcomponents descriptor
            if k not in sc_instance.get("parameters", {}):
                errors.append(
                    f"Subcomponent '{sc_name}' contains a parameter '{k}' but the referenced descritor doesn't contain a parameter with that name"
                )
                continue
            # check that $ values have a matching parameter in the parent component
            if str(v).startswith("$"):
                # TODO(dirkt) check array details
                paramName = v[1:].split("[")[0]
                if paramName not in instance.get("parameters", {}):
                    errors.append(
                        f"Subcomponent '{sc_name}' contains a parameter value '{v}' but the parent component doesn't have a parameter '{paramName}'"
                    )
                    continue
                errors += checkParameterTypesMatch(
                    instance, paramName, sc_name, sc_instance, k
                )
            else:
                errors += checkParameterTypeAndValueMatches(
                    f"The value of the subcomponent '{sc_name}'",
                    k,
                    sc_instance["parameters"][k]["type"],
                    sc_instance["parameters"][k].get("array"),
                    v,
                )
        # check all parameter types when wildcard is used
        if "*" in sc.get("parameters", {}):
            for k in sc_instance.get("parameters", {}).keys():
                if k in instance.get("parameters", {}):
                    errors += checkParameterTypesMatch(
                        instance, k, sc_name, sc_instance, k
                    )

    connections = instance.get("connections")
    if connections is not None:
        srcs_to_params = {}  # inbound srcs are not included.
        for i, connection in enumerate(connections):
            src = connection["src"]
            is_inbound = src == ""
            if not is_inbound:
                if src in srcs_to_params and srcs_to_params[src] == connection.get(
                    "params"
                ):
                    errors.append(
                        f"Multiple connection entries with the src '{src}' and the same parameters should be merged into a single entry with multiple destinations"
                    )
                else:
                    # if src in srcs_to_params:
                    #     print(f"Note: multiple connection entries in '{path}' with the src '{src}' but different parameters:")
                    #     print("- " + str(srcs_to_params[src]))
                    #     print("- " + str(connection.get("params")))
                    srcs_to_params[src] = connection.get("params")
                portErrors, srcPortType = checkPort(path, instance, i, "src", src, True)
                errors += portErrors
            for i, dest in enumerate(connection["dests"]):
                is_outbound = dest == ""
                if not is_outbound:
                    portErrors, destPortType = checkPort(
                        path, instance, i, f"dests[{i}]", dest, False
                    )
                    errors += portErrors
                if is_inbound and is_outbound:
                    errors.append(
                        f"Connection #{i}: source port is inbound port and connects to a outbound port. Not allowed."
                    )
                # skip the port type check when is_inbound or is_outbound.
                if (
                    not is_inbound
                    and srcPortType is not None
                    and not is_outbound
                    and destPortType is not None
                    and srcPortType != destPortType
                ):
                    errors.append(
                        f"Connection #{i}: source port '{src}' has different type than destination port '{dest}' - {srcPortType} vs. {destPortType}"
                    )

    errors += checkConnectionsForCycles(instance)

    errors += checkStatesAndSchedules(instance)

    if errors:
        print(f"! JSON file '{path}' violates additional constraints:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)

    return not errors


def checkParameterTypesMatch(instance, param, sc_name, sc_instance, sc_param):
    errors = []
    instanceParameter = instance.get("parameters")[param]
    sc_instanceParameter = sc_instance.get("parameters")[sc_param]
    if instanceParameter["type"] != sc_instanceParameter["type"]:
        errors.append(
            f"Subcomponent '{sc_name}' contains a parameter '{sc_param}' but the parameter types mismatch: '{instanceParameter['type']}' vs. '{sc_instanceParameter['type']}'"
        )
    return errors


def checkParameterTypeAndValueMatches(error_prefix, name, type_, array_size, value):
    errors = []
    if array_size is None:
        if isinstance(value, list):
            errors.append(
                f"{error_prefix} parameter '{name}' should be a '{type_}', not an array: {repr(value)} [{type(value).__name__}]"
            )
            return errors
        values = [value]
    else:
        if not isinstance(value, list):
            errors.append(
                f"{error_prefix} parameter '{name}' should be an array of '{type_}', not: {repr(value)} [{type(value).__name__}]"
            )
            return errors
        values = value

    for i, v in enumerate(values):
        if str(v).startswith("$"):
            continue
        array_index = "" if array_size is None else f"[{i}]"
        if type_ == "bool":
            if not isinstance(v, bool):
                errors.append(
                    f"{error_prefix} parameter '{name}{array_index}' doesn't match the parameter type '{type_}': {repr(v)} [{type(v).__name__}]"
                )
        elif type_ in (
            "int8_t",
            "int16_t",
            "int32_t",
            "int64_t",
            "uint8_t",
            "uint16_t",
            "uint32_t",
            "uint64_t",
            "size_t",
            "dwTime_t",
        ):
            if not isinstance(v, int):
                errors.append(
                    f"{error_prefix} parameter '{name}{array_index}' doesn't match the parameter type '{type_}': {repr(v)} [{type(v).__name__}]"
                )
        elif type_ in ("float32_t", "float64_t"):
            if type(v) not in (float, int):
                errors.append(
                    f"{error_prefix} parameter '{name}{array_index}' doesn't match the parameter type '{type_}': {repr(v)} [{type(v).__name__}]"
                )
        else:
            if not isinstance(v, str):
                errors.append(
                    f"{error_prefix} parameter '{name}{array_index}' doesn't match the parameter type '{type_}': {repr(v)} [{type(v).__name__}]"
                )

    return errors


def checkConnectionsForCycles(instance):
    errors = []
    connections = instance.get("connections")
    if connections is None:
        return errors

    # build graph of all subcomponent-to-subcomponent connections
    graph = defaultdict(set)
    for connection in connections:
        src = connection["src"]
        if src == "":
            continue
        if "." not in src:
            continue
        if connection.get("params", {}).get("indirect"):
            continue

        srcComp = src.split(".")[0]

        for dest, destParams in connection["dests"].items():
            if dest == "":
                continue
            if "." not in dest:
                continue
            if destParams.get("indirect"):
                continue
            destComp = dest.split(".")[0]
            graph[srcComp].add(destComp)

    # check for cycles
    try:
        import graphlib

        ts = graphlib.TopologicalSorter(graph)
        try:
            tuple(ts.static_order())
        except graphlib.CycleError as e:
            errors.append(
                f"At least one cycle within connections detected: {e.args[1]}"
            )
    except ImportError:
        # fallback implementation until graphlib introduced in Python 3.9 can be used
        def checkCycle(graph, node, stack):
            for successor in sorted(graph[node]):
                if successor not in graph:
                    continue
                if successor in stack:
                    return stack[stack.index(successor) :] + [successor]
                cycle = checkCycle(graph, successor, stack + [successor])
                if cycle:
                    return cycle
            return None

        for node in sorted(graph.keys()):
            cycle = checkCycle(graph, node, [node])
            if cycle:
                errors.append(
                    f"At least one cycle within connections detected: {cycle}"
                )
                break

    return errors


def checkStatesAndSchedules(instance):
    errors = []
    states = instance.get("states")
    if states is None:
        return errors

    for state_name, stm_schedule_info in instance.get("states", {}).items():
        stmScheduleKey = stm_schedule_info["stmScheduleKey"]
        if stmScheduleKey not in instance.get("stmSchedules", {}):
            errors.append(
                f" stmSchedules does not contain '{stmScheduleKey} ' required by state: '{state_name}'"
            )
    return errors


def checkPort(path, instance, connection_index, key, port_name, is_source):
    errors = []
    parts = port_name.split(".")
    portnameAndIndex = parts[-1].split("[")
    if len(parts) == 2:
        # port of a subcomponent
        sc = instance["subcomponents"].get(parts[0])
        if sc is None:
            errors.append(
                f"Connection #{connection_index}: {key} '{port_name}' is referencing an unknown subcomponent '{sc}'"
            )
            return errors, None

        sc_path = path.parent / sc["componentType"]
        sc_path, sc_instance = VALIDATED_FILES.get(sc_path.resolve(), (sc_path, None))
        if sc_instance is not None:
            port = sc_instance["outputPorts" if is_source else "inputPorts"].get(
                portnameAndIndex[0]
            )
            if port is None:
                port_type = "output" if is_source else "input"
                errors.append(
                    f"Connection #{connection_index}: {key} '{parts[-1]}' is not an {port_type} port of the subcomponent '{parts[0]}'"
                )
                return errors, None
        else:
            port = None
        component_type = "subcomponent"

    else:
        # port of the graphlet
        port = instance["inputPorts" if is_source else "outputPorts"].get(
            portnameAndIndex[0]
        )
        component_type = "graphlet"

    msg_suffix = ("input" if is_source else "output") + f" port of the {component_type}"
    if port is None:
        errors.append(
            f"Connection #{connection_index}: {key} '{port_name}' is not an {msg_suffix}"
        )
        return errors, None

    portSize = port.get("array")
    if len(portnameAndIndex) == 1:
        if portSize is not None:
            errors.append(
                f"Connection #{connection_index}: {key} '{port_name}' doesn't use an array index on an array {msg_suffix}"
            )
    else:
        if portSize is None:
            errors.append(
                f"Connection #{connection_index}: {key} '{port_name}' uses array index on a non-array {msg_suffix}"
            )
        elif int(portnameAndIndex[1][:-1]) >= portSize:
            errors.append(
                f"Connection #{connection_index}: {key} '{port_name}' uses out-of-bound array index on an {msg_suffix}"
            )

    return errors, port.get("type")


def checkFileForStrictOrder(path, instance):
    errors = []

    validator = getValidator(path)
    errors += checkObjectForStrictOrder(instance, validator.schema)

    if errors:
        print(f"! JSON file '{path}' violates strict order of keys:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)

    return not errors


def checkObjectForStrictOrder(instance, root_schema, sub_schema=None, breadcrumbs=None):
    if sub_schema is None:
        sub_schema = root_schema
    if breadcrumbs is None:
        breadcrumbs = []

    errors = []

    if sub_schema["type"] not in ("array", "object"):
        return errors

    if "properties" in sub_schema:
        desired_order = list(sub_schema["properties"].keys())
        current_keys = list(instance.keys())
        sorted_keys = sorted(current_keys, key=desired_order.index)
        if current_keys != sorted_keys:
            breadcrumbs_str = (
                ("under " + " -> ".join(f'"{x}"' for x in breadcrumbs))
                if breadcrumbs
                else "on the first level"
            )
            errors.append(
                f"The keys {breadcrumbs_str} should have the order: "
                + ", ".join(sorted_keys)
            )

    if sub_schema["type"] == "object":
        if "properties" in sub_schema:
            for key, subinstance in instance.items():
                errors += checkObjectForStrictOrder(
                    subinstance,
                    root_schema,
                    sub_schema["properties"][key],
                    breadcrumbs + [key],
                )
        elif "patternProperties" in sub_schema:
            assert len(sub_schema["patternProperties"]) == 1
            for sub_sub_schema in sub_schema["patternProperties"].values():
                if "$ref" not in sub_sub_schema:
                    continue
                assert sub_sub_schema["$ref"].startswith("#/$defs/"), sub_sub_schema[
                    "$ref"
                ]
                sub_sub_schema = root_schema["$defs"][sub_sub_schema["$ref"][8:]]
                for key, subinstance in instance.items():
                    errors += checkObjectForStrictOrder(
                        subinstance, root_schema, sub_sub_schema, breadcrumbs + [key]
                    )

    return errors


def checkFileForStrictIndentation(path, instance):
    errors = []

    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        stripped = line.lstrip(" ")
        if (len(line) - len(stripped)) % 4:
            errors.append(f"{index+1}")

    if errors:
        print(
            f"! JSON file '{path}' violates strict indentation level on the following lines: "
            + ", ".join(errors),
            file=sys.stderr,
        )

    return not errors


if __name__ == "__main__":
    sys.exit(main())
