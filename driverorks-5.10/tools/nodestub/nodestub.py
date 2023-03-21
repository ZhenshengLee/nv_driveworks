#!/usr/bin/env python3

################################################################################
#
# Notice
# ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS"
# NVIDIA MAKES NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR
# OTHERWISE WITH RESPECT TO THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED
# WARRANTIES OF NONINFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR
# PURPOSE.
#
# NVIDIA CORPORATION & AFFILIATES assumes no responsibility for the consequences
# of use of such information or for any infringement of patents or other rights
# of third parties that may result from its use. No license is granted by
# implication or otherwise under any patent or patent rights of NVIDIA
# CORPORATION & AFFILIATES. No third party distribution is allowed unless
# expressly authorized by NVIDIA. Details are subject to change without notice.
# This code supersedes and replaces all information previously supplied. NVIDIA
# CORPORATION & AFFILIATES products are not authorized for use as critical
# components in life support devices or systems without express written approval
# of NVIDIA CORPORATION & AFFILIATES.
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this material and related documentation without an express
# license agreement from NVIDIA CORPORATION or its affiliates is strictly
# prohibited.
#
################################################################################

import argparse
import json
import pathlib
import sys
import tempfile

# only used in release to find packaged dependencies
lib_path = pathlib.Path(__file__).parent / "lib"
if lib_path.exists():
    sys.path.insert(0, str(lib_path))

import jinja2

TEMPLATES = ["Node.thpp", "Node.tcpp", "NodeImpl.thpp", "NodeImpl.tcpp"]
BASE_CLASSES = {
    "dw::framework::ExceptionSafeProcessNode": "dw::framework::SimpleProcessNodeT",
    "dw::framework::ExceptionSafeSensorNode": "dw::framework::dwSensorNodeImplTemplate",
}
DATATYPE_HEADERS = {
    "dwAutoEmergencyBrakingRequest": "dw/autoemergencybraking/AutoEmergencyBraking.h",
    "dwCalibratedExtrinsics": "dwframework/dwnodes/common/SelfCalibrationTypes.hpp",
    "dwEgomotionStateHandle_t": "dw/egomotion/EgomotionState.h",
    "dwFreespaceDetection": "dw/perception/freespace/camera/FreespaceDetector.h",
    "dwImageHandle_t": "dw/image/Image.h",
    "dwLaneSupportRequest": "dwexperimental/lanesupport/LaneSupport.h",
    "dwLidarPose": "dwframework/dwnodes/common/PointCloudProcessingCommonTypes.hpp",
    "dwObjectArray": "dw/world/ObjectArray.h",
    "dwObjectHistoryArray": "dw/world/ObjectHistoryArray.h",
    "dwObstacleLinkArray": "dw/worldmodel/Obstacle.h",
    "dwParkDetection": "dw/perception/parking/camera/ParkingSpaceDetector.h",
    "dwPathDetection": "dw/perception/path/camera/PathDetector.h",
    "dwPointCloud": "dw/pointcloudprocessing/pointcloud/PointCloud.h",
    "dwRadarDopplerMotion": "dw/egomotion/radar/DopplerMotionEstimator.h",
    "dwRadarScan": "dw/sensors/radar/Radar.h",
    "dwSensorNodeProperties": "dwframework/dwnodes/common/SensorCommonTypes.hpp",
    "dwSensorNodeRawData": "dwframework/dwnodes/common/SensorCommonTypes.hpp",
    "dwSurroundMonitorLaneChangeSafeDecision": "dw/advancedfunctions/surroundmonitor/SurroundMonitor.h",
    "dwTime_t": "dw/core/base/Types.h",
    "dwVehicleIOState": "dw/control/vehicleio/VehicleIOLegacyStructures.h",
}


def main():
    global BASE_CLASSES
    parser = argparse.ArgumentParser(
        description="Generate node source stubs from a .node.json descriptor"
    )
    parser.add_argument(
        "json_path",
        type=_is_valid_file,
        metavar="NODE_JSON_FILE",
        help="The .node.json descriptor",
    )
    parser.add_argument(
        "base_class",
        metavar="BASE_CLASS",
        choices=BASE_CLASSES.keys(),
        help="The desired base class for the node: " + ", ".join(BASE_CLASSES.keys()),
    )
    parser.add_argument(
        "--output-path",
        type=pathlib.Path,
        help="The directory to generate the node sources in",
    )
    parser.add_argument(
        "--overwrite-existing-files",
        action="store_true",
        help="The directory to generate the node sources in",
    )
    args = parser.parse_args()

    node_descriptor = json.loads(args.json_path.read_text())
    if args.output_path is None:
        args.output_path = pathlib.Path(tempfile.mkdtemp(prefix="nodestub_"))
    else:
        args.output_path.mkdir(parents=True, exist_ok=True)
    print(f"Generating files in: {args.output_path}")

    generate_nodestubs(
        node_descriptor,
        (args.base_class, BASE_CLASSES[args.base_class]),
        args.output_path,
        args.overwrite_existing_files,
    )


def _is_valid_file(arg):
    p = pathlib.Path(arg)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"The file '{p}' does not exist")
    return p


def generate_nodestubs(
    node_descriptor, base_classes, output_basepath, overwrite_existing_files
):
    global TEMPLATES
    file_mapping = {}
    for template_name in TEMPLATES:
        assert template_name.startswith("Node")
        assert template_name.endswith((".thpp", ".tcpp"))
        template_path = pathlib.Path(__file__).parent / template_name
        output_path = output_basepath / (
            node_descriptor["name"].split("::")[-1]
            + template_name[4:-5]
            + "."
            + template_name[-3:]
        )
        file_mapping[template_path] = output_path

    if not overwrite_existing_files and any(p.exists() for p in file_mapping.values()):
        raise RuntimeError(
            "To-be-generated files already exist, use "
            "--overwrite-existing-files to overwrite them"
        )

    for template_path, output_path in file_mapping.items():
        base_class = base_classes[1 if "Impl." in template_path.name else 0]
        generate_nodestub(node_descriptor, base_class, template_path, output_path)


def generate_nodestub(node_descriptor, base_class, template_path, output_path):
    print("*", template_path, "->", output_path)
    content = expand_template(template_path.read_text(), node_descriptor, base_class)
    output_path.write_text(content)


def expand_template(template_string, node_descriptor, base_class):
    import datetime

    env = jinja2.Environment(trim_blocks=True)
    t = env.from_string(template_string)
    return t.render(
        json=node_descriptor,
        base_class=base_class,
        datatype_includes=get_additional_includes(node_descriptor),
        datetime=datetime,
        toLowerCamelCase=screamingSnakeCase2LowerCamelCase,
    )


def get_additional_includes(node_descriptor):
    global DATATYPE_HEADERS
    datatype_includes = set()
    for key in ("inputPorts", "outputPorts"):
        for port in node_descriptor[key].values():
            datatype = port["type"]
            if datatype in DATATYPE_HEADERS:
                datatype_includes.add(DATATYPE_HEADERS[datatype])
            elif datatype not in get_additional_includes.warnings:
                get_additional_includes.warnings.add(datatype)
                print(
                    f"Unknown data type '{datatype}', additional #include "
                    "directives might be needed in the node header",
                    file=sys.stderr,
                )
    return datatype_includes


get_additional_includes.warnings = set(("bool", "void*"))


def screamingSnakeCase2LowerCamelCase(value):
    parts = value.lower().split("_")
    return parts[0] + "".join(x.title() for x in parts[1:])


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
