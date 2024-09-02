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
"""Entry point for CGF transformation tool."""
import argparse
from importlib.machinery import SourceFileLoader
import inspect
import json
from pathlib import Path
import sys

from dwcgf.transformation.action import Action, ActionQueue
from dwcgf.transformation.descriptor import DescriptorLoader, DescriptorType
from dwcgf.transformation.object_model import Application, Graphlet
from dwcgf.transformation.serializer import Serializer


def check_file_exists(file_path: str) -> Path:
    """Convert the command line argument into Path.

    If the file doesn't exist, throws.
    """
    fpath = Path(file_path)
    if not fpath.is_file():
        raise ValueError(f"Input '{file_path}' doesn't exist.")
    return fpath


def config_arguments() -> argparse.ArgumentParser:
    """Configure the command line arguments of CGF transformation CLI."""

    parser = argparse.ArgumentParser(description="CGF transformation CLI.")
    parser.add_argument(
        "base", type=check_file_exists, help="Base application or graphlet file path."
    )
    parser.add_argument(
        "-t",
        "--transformations",
        nargs="+",
        type=check_file_exists,
        required=True,
        help="Transformation file paths",
    )
    parser.add_argument(
        "--actions",
        nargs="+",
        type=str,
        help="Customer implemented action python module, must be accessible in cli",
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Derived application or graphlet file name."
    )
    parser.add_argument(
        "--path-offset",
        type=str,
        help="This argument denotes the relative path between generated files \
              and source files and should only be used for bazel integration.",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="If this flag is set, will also generate a file with all descriptors\
              in it. Only supported when the root is application descriptor.",
    )
    return parser


def main() -> int:
    """Entry point of CGF transformation CLI."""

    # handle command line arguments
    parser = config_arguments()
    argvs = parser.parse_args()

    # load base descriptors
    if argvs.path_offset:
        base_loader = DescriptorLoader(argvs.base, path_offset=Path(argvs.path_offset))
    else:
        base_loader = DescriptorLoader(argvs.base)
    # construct base instance tree
    if base_loader.root_descriptor.desc_type is DescriptorType.APPLICATION:
        base = Application.from_descriptor(base_loader, base_loader.root_descriptor)
    elif base_loader.root_descriptor.desc_type is DescriptorType.GRAPHLET:
        base = Graphlet.from_descriptor(
            "root", base_loader, base_loader.root_descriptor
        )
    else:
        raise ValueError(
            "Base application has to be either application or graphlet." ""
        )

    # load annd register customer actions
    if argvs.actions:
        for customer_actions in argvs.actions:
            module_name = customer_actions.split("/")[-1].split(".")[0]
            for _, cls in inspect.getmembers(
                SourceFileLoader(  # pylint: disable=W1505
                    module_name, customer_actions
                ).load_module(),
                inspect.isclass,
            ):
                # only pay attention to derived class of Action
                if (
                    issubclass(cls, Action)
                    and cls is not Action
                    and hasattr(cls, "action_type")
                ):
                    print(
                        "\033[32mRegister customer action:\033[0m",
                        cls.action_type,
                        ", object:",
                        cls,
                    )

    # for each transformation file
    for transformation_file in argvs.transformations:
        # load transformation descriptors
        trans_loader = DescriptorLoader(transformation_file)
        # construct action queue
        action_queue = ActionQueue.from_descriptor(
            base_loader, trans_loader, trans_loader.root_descriptor
        )
        # perform actions/transformations
        action_queue.transform([base])

    # dump the transformed descriptors
    serializer = Serializer(
        loader=base_loader,
        root=base,
        force_override=False,
        output_dir=argvs.output,
        path_offset=argvs.path_offset,
    )
    serializer.serialize_to_file()

    if argvs.combine:
        json_data = serializer.serialize_to_json_data()
        combined_file_name = (
            f"{base.descriptor_path.stem.split('.')[0]}.combined-app.json"
        )
        combined_file_path = base.descriptor_path.with_name(combined_file_name)
        combined_file_path.write_text(json.dumps(json_data, indent=2) + "\n")

    print("Transformed system generated:")
    print("=============================")
    print(Path.cwd() / base.descriptor_path)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
