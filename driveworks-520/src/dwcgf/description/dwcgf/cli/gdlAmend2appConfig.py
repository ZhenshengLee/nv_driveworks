#!/usr/bin/env python3
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
"""Entry point for CGF gdlAmend2appConfig tool."""
import argparse
from collections import OrderedDict
import json
import os
from pathlib import Path
import sys


def check_file_exists(file_path: str) -> Path:
    """Convert the command line argument into file Path.

    If the file doesn't exist, throws.
    """
    fpath = Path(file_path)
    if not fpath.is_file():
        raise ValueError(f"Input '{file_path}' doesn't exist.")
    return fpath


def check_folder_exists(file_path: str) -> Path:
    """Convert the command line argument into folder Path.

    If the file doesn't exist, throws.
    """
    fpath = Path(file_path)
    if not fpath.is_dir():
        raise ValueError(f"Input '{file_path}' doesn't exist.")
    return fpath


def config_arguments() -> argparse.ArgumentParser:
    """Configure the command line arguments of CGF gdlAmend2appConfig CLI."""

    parser = argparse.ArgumentParser(description="CGF gdlAmend2appConfig CLI.")
    parser.add_argument(
        "--amend_file",
        nargs="+",
        type=check_file_exists,
        required=False,
        help="amend file path, usage: --amend_file=<absolute path of amend file>",
    )
    parser.add_argument(
        "--amend_folder",
        nargs="+",
        type=check_folder_exists,
        required=False,
        help="amend files folder path, \
              usage: --amend_folder=<absolute folder path contains amend files>",
    )
    return parser


def parse_val(val: str):  # type: ignore
    """Parse the type of input val."""

    vals = val.split(",")
    res = []

    def is_int(s: str) -> bool:
        try:
            int(s)
            return True
        except ValueError:
            return False

    def is_float(s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False

    for v in vals:
        if v in ("true", "True"):
            res.append(True)
        elif v in ("false", "False"):
            res.append(False)
        elif "." in v and is_float(v):
            res.append(float(v))  # type: ignore
        elif is_int(v):
            res.append(int(v))  # type: ignore
        else:
            res.append(v)  # type: ignore
    if len(res) == 1:
        return res[0]
    return res


def convert(amend_file: str) -> None:
    """function perform conversion from amend to app-config.json."""

    # there is a placeholder file valAmend.json which is empty file.
    if os.path.getsize(amend_file) == 0:
        print(f"\033[31mSkip empty file: \033[32m{amend_file}\033[0m")
        return
    output = OrderedDict()  # type: ignore
    with open(amend_file) as f:
        amends = json.load(f)
        if "RoadRunner" not in amends:
            print(f"\033[31mSkip non-GDL amend file: \033[32m{amend_file}\033[0m")
            return
        for field in amends["RoadRunner"].keys():
            if field not in (
                "apps",
                "RoadCastService",
                "Recorder",
                "sensorLayouts",
                "extraInfo",
            ):
                print(f"[Warning] unhandled key: {field} in amend: {amend_file}!")
            # the field order is important to validate against
            # src/dwcgf/description/schema/app-config.schema.json
            process_amend_section = OrderedDict()  # type: ignore
            if field == "apps":
                for app in amends["RoadRunner"]["apps"].keys():
                    # handle camera_master/camera_b
                    if app == "camera_master" or app == "camera_b":
                        camera_master_amend = amends["RoadRunner"]["apps"][app][
                            "config"
                        ]
                        # sort with alphabetical order for schema
                        camera_master_amend_items = sorted(camera_master_amend.items())
                        for module_name, change_field in camera_master_amend_items:
                            for k, v in change_field.items():
                                if k == "params":
                                    if "componentParameters" not in output.keys():
                                        output["componentParameters"] = OrderedDict()
                                    # sort with alphabetical order for schema
                                    v_items = sorted(v.items())
                                    for param_name, param_val in v_items:
                                        param_name_stripped = param_name.replace(
                                            "params::", ""
                                        )
                                        module = "top." + module_name
                                        if module not in output["componentParameters"]:
                                            output["componentParameters"][
                                                module
                                            ] = OrderedDict()
                                        output["componentParameters"][module][
                                            param_name_stripped
                                        ] = parse_val(str(param_val))
                                elif k == "load":
                                    if "skipNodes" not in output.keys():
                                        output["skipNodes"] = OrderedDict()
                                    module = "top." + module_name
                                    output["skipNodes"][module] = []
                                else:
                                    print(
                                        f"[Warning] unhandled option of \
                                        app {app} field:{k} in amend!"
                                    )
                    # handle stm_master/ssm/sensor_sync_server
                    elif app in ("stm_master", "ssm", "sensor_sync_server"):
                        process_amend = amends["RoadRunner"]["apps"][app]
                        accepted_option = ("keyValueArgs", "switchArgs")
                        # sort with alphabetical order for schema
                        process_amend_items = sorted(process_amend.items())
                        # write output outside the for loop to fit schema
                        for option, change_field in process_amend_items:
                            if option in accepted_option:
                                for k, v in change_field.items():
                                    if app not in process_amend_section.keys():
                                        process_amend_section[app] = OrderedDict()
                                    if "argv" not in process_amend_section[app].keys():
                                        process_amend_section[app][
                                            "argv"
                                        ] = OrderedDict()
                                    process_amend_section[app]["argv"][k] = v
                            else:
                                print(
                                    f"[Warning] unhandled option of \
                                    app {app} field:{option} in amend!"
                                )
                    else:
                        print(
                            f"[Warning] unhandled app field: {app} in amend: {amend_file}!"
                        )
            # process section
            if process_amend_section:
                output["processes"] = process_amend_section
            # handle sensor_layout
            if field == "sensorLayouts":
                print(
                    f"[Warning] unhandled key:{field} in amend! \
                    Need to update .sensor-mappings.json files manually"
                )
            # handle RoadCastService/Recorder/extraInfo
            if field == "extraInfo":
                output[field] = amends["RoadRunner"][field]
            if field == "Recorder":
                output[field] = amends["RoadRunner"][field]
            if field == "RoadCastService":
                output[field] = amends["RoadRunner"][field]
    output_file = str(amend_file).replace(".json", ".app-config.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, sort_keys=False, ensure_ascii=False, indent=4)


def main() -> int:
    """Entry point of gdlAmend2appConfig CLI."""

    # handle command line arguments
    parser = config_arguments()
    argvs = parser.parse_args()
    if argvs.amend_file is None and argvs.amend_folder is None:
        parser.print_help()
        return -1
    if argvs.amend_file:
        print(f"\033[31mConverting \033[32m{argvs.amend_file[0]}\033[0m")
        convert(argvs.amend_file[0])
    if argvs.amend_folder:
        for root, _, files in os.walk(argvs.amend_folder[0], topdown=False):
            for name in files:
                if name.endswith(".json") and not name.endswith(".app-config.json"):
                    print(
                        f"\033[31mConverting \033[32m{os.path.join(root, name)}\033[0m"
                    )
                    convert(os.path.join(root, name))

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
