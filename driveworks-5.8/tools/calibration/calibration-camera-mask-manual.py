#!/usr/bin/python3

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
# SPDX-FileCopyrightText: Copyright (c) 2018-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import argparse
import json
import subprocess
import math
import sys
from collections import namedtuple
import numpy as np
from PyInquirer import prompt
from tabulate import tabulate
from PIL import Image, ImageDraw, ImageOps

SUPPORTED_INPUT_IMAGE_EXTENSIONS = ['.jpg', '.png']

def main():
    args = parse_arguments()

    config = initConfig(args)

    printWelcome()

    path_to_rig = args.rig

    # interactive command line: main loop
    while True:
        if path_to_rig is None:
            path_to_rig = selectRigFile()

        if(path_to_rig is None or not os.path.isfile(path_to_rig)):
            print("The rig-file is not found: %s" % path_to_rig)
            break

        main_step_index = selectMainMenu()
        if main_step_index == "1":
            # show path to the current rig-file
            path_to_rig = selectRigFile()
        elif main_step_index == "2":
            # print camera sensors and their mask availability'
            printCameraSensors(config, path_to_rig)
        elif main_step_index == "3":
            # create a new mask
            createCarMask(path_to_rig, config)
        elif main_step_index == "4":
            # show an existing camera mask'
            showExistingCarMask(config, path_to_rig)
        elif main_step_index == "5":
            # print config
            showConfig(config)
        elif main_step_index == "6":
            # save all car masks to files
            exportAllCarMasks(config, path_to_rig)
        elif main_step_index == "7":
            # exit
            print("\nExit. Thank you for using the car mask extraction tool.")
            break
        else:
            print("The option '%s' is not supported. Please select a correct option.\n" % main_step_index)

    return

def printWelcome():
    print("============================================================================================================")
    print("Welcome to the car mask extraction tool. The tool helps you to create, modify and visualize image car masks.")
    print("============================================================================================================\n")

def selectRigFile():
    questions = [
        {
            'type': 'input',
            'name': 'path_to_rig',
            'message': 'Please enter the path to the input rig file (it will be modified in-place): ',
        }
    ]
    answers = prompt(questions)
    print("The current rig-file is: %s\n" % answers['path_to_rig'])
    return answers['path_to_rig']

def selectMainMenu():
    questions = [
        {
            'type': 'input',
            'name': 'main_step_index',
            'message': 'Please select one of the following options:\n'
                       '  1 - select another rig-file\n'
                       '  2 - show existing camera sensors and their mask statuses\n'
                       '  3 - create a car mask\n'
                       '  4 - show an existing camera mask\n'
                       '  5 - show the current config\n'
                       '  6 - export all car masks to individual files\n'
                       '  7 - exit\n'

        }
    ]
    answers = prompt(questions)
    return answers['main_step_index']

def getListOfCameraSensors(config, path_to_rig):
    path_to_mask_tool = config.camera_mask_tool
    if not os.path.isfile(path_to_mask_tool):
        raise ValueError("Could not find a camera-mask-tool: %s" % path_to_mask_tool)

    path_to_mask_info_file = os.path.join(os.path.dirname(path_to_rig), "mask-info.json")
    if os.path.isfile(path_to_mask_info_file):
        os.remove(path_to_mask_info_file)

    # call a C++ tool which exports camera info to a file
    command = path_to_mask_tool + \
              " --mode \"export-cameras-with-mask-availability\"" + \
              " --rig-file " + path_to_rig + \
              " --mask-info-file " + path_to_mask_info_file
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()

    if not os.path.isfile(path_to_mask_info_file) or process.returncode != 0:
        raise ValueError("Fail to get information about cameras from the rig file: %s" % path_to_rig)

    with open(path_to_mask_info_file) as f:
        camera_mask_info = json.load(f)
    if os.path.isfile(path_to_mask_info_file):
        os.remove(path_to_mask_info_file)

    return camera_mask_info

def printCameraSensors(config, path_to_rig):
    camera_mask_info = getListOfCameraSensors(config, path_to_rig)

    cameras = list()
    for iCameraSensorCounter, (camera_sensor_name, camera_sensor_data) in enumerate(camera_mask_info.items()):
        cameras.append([iCameraSensorCounter,
                        camera_sensor_name,
                        camera_sensor_data['sensor_resolution'],
                        camera_sensor_data['mask_status'] == 1,
                        camera_sensor_data['file']])

    print("")
    print(tabulate(cameras, headers=['ID', 'Name', 'Sensor Resolution', 'Mask is available', 'File (optional)']))
    print("")
    return cameras

def createCarMask(path_to_rig, config):
    cameras, camera_id = selectCamera(config, path_to_rig)
    if cameras is None or camera_id is None:
        return

    camera = cameras[camera_id]
    camera_name = camera[1]
    path_to_camera_image = getPathToCameraImage(path_to_rig, camera_name)

    # get reference resolution of camera sensor
    sensor_resolution = getListOfCameraSensors(config, path_to_rig)[camera_name]['sensor_resolution']

    # create a template for the mask
    path_to_mask_image = getPathToCameraMask(path_to_camera_image)
    input_image = Image.open(path_to_camera_image)

    # make sure image matches sensor resolution
    if input_image.size[0] != sensor_resolution[0] or input_image.size[1] != sensor_resolution[1]:
        raise ValueError('Resolution of camera image file ({}) does not match sensor resolution in rig file ({}), '
                         'please check both for consistency'.format(
            input_image.size, sensor_resolution))

    new_size = getDownsampledImageSize(
        input_image.size, int(config.target_mask_dimension))
    resized_input_image = input_image.copy()
    resized_input_image.thumbnail(new_size, Image.ANTIALIAS)
    resized_input_image.save(path_to_mask_image)

    # start an image editor for mask creation
    # '-n' -> to start a new window
    command = config.image_editor + " -n " + path_to_mask_image
    process = subprocess.Popen(command, shell=True)
    process.wait()
    if process.returncode == 0:
        generated_mask = Image.open(path_to_mask_image)
        generated_mask_np = np.array(generated_mask)
        if len(generated_mask_np.shape) != 2:
            os.remove(path_to_mask_image)
            print("Error. The mask image has 3 color channels after image editing. Do you forget to overwrite the mask?\n")
            return

        numberOfValidPixels = np.sum(generated_mask_np == 0) + np.sum(generated_mask_np == 255)
        expectedNumberOfValidPixels = generated_mask_np.shape[0] * generated_mask_np.shape[1]
        isValidBinaryMask = (numberOfValidPixels == expectedNumberOfValidPixels)
        if not isValidBinaryMask:
            corrected_mask_np = makeMaskBinary(generated_mask_np)
            corrected_mask = Image.fromarray(np.uint8(corrected_mask_np)).convert('L')
            with open(path_to_mask_image, "wb") as fImage:
                corrected_mask.save(fImage)
            print("Warning. The new mask contains some invalid pixels. All valid pixels should be equal 0 or 255.")
            print("Adjust %.2f %% image pixels to correct the mask.\n" % (100.0 * (expectedNumberOfValidPixels - numberOfValidPixels)/expectedNumberOfValidPixels))

        # update the mask in the rig file
        command = config.camera_mask_tool + \
                        " --mode \"update-camera-mask\" " + \
                        " --rig-file " + path_to_rig + \
                        " --image-mask-file \"" + path_to_mask_image + "\" " + \
                        " --camera-sensor-name \"" + camera_name +"\""
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        process.wait()
        if process.returncode == 0:
            print("The rig file section for the camera '%s' mask has been updated.\n" % camera_name)
        else:
            print("Fail to update the mask in the rig file.")
    else:
        print("Internal error of the GIMP editor.")

    if os.path.isfile(path_to_mask_image):
        os.remove(path_to_mask_image)

    displayImage(path_to_rig, config, camera_name)

    return

def showExistingCarMask(config, path_to_rig):
    cameras, camera_id = selectCamera(config, path_to_rig)
    if cameras is None or camera_id is None:
        return

    camera = cameras[camera_id]
    camera_name = camera[1]
    if not camera[2]:
        print("The car mask is not available for the camera: %s\n" % camera_name)
        return

    displayImage(path_to_rig, config, camera_name)
    return

def displayImage(path_to_rig, config, camera_name):
    path_to_camera_image = getPathToCameraImage(path_to_rig, camera_name)

    path_to_mask_image = getPathToCameraMask(path_to_camera_image)
    if os.path.isfile(path_to_mask_image):
        os.remove(path_to_mask_image)

    # get the mask as a file from the rig file
    command = config.camera_mask_tool + \
                    " --rig-file " + path_to_rig + \
                    " --mode \"export-camera-mask\" " + \
                    " --image-mask-file \"" + path_to_mask_image + "\" " + \
                    " --camera-sensor-name \"" + camera_name +"\""
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    if process.returncode != 0:
        raise ValueError("Fail to export the mask from the rig file.")
    if not os.path.isfile(path_to_mask_image):
        print("Could not find th exported mask file: %s" % path_to_mask_image)
        return

    input_image = Image.open(path_to_camera_image).convert('RGBA')
    mask_image = Image.open(path_to_mask_image)
    os.remove(path_to_mask_image)

    # resize image mask back to match an original RGB-image size
    mask_image = mask_image.resize(input_image.size, Image.ANTIALIAS)
    unmasked_image = mask_image.point(lambda i: i * 255 if i == 0 or i == 1 else 255)
    masked_image   = mask_image.point(lambda i: i * 255 if not (i == 0 or i == 1) else 255)

    # draw the mask on top of the RGB image
    overlay = Image.new('RGBA', input_image.size, (255, 255, 255, 0))
    drawing = ImageDraw.Draw(overlay)
    drawing.bitmap((0, 0), masked_image, fill=(128, 0, 0, 40))
    drawing.bitmap((0, 0), unmasked_image, fill=(0, 128, 0, 128))
    input_image = Image.alpha_composite(input_image, overlay)
    input_image.show()

def exportAllCarMasks(config, path_to_rig):
    path_to_single_mask_folder = getPathToSingleCameraMaskFolder(path_to_rig)
    if(not os.path.isdir(path_to_single_mask_folder)):
        os.mkdir(path_to_single_mask_folder)

    # get the mask as a file from the rig file
    command = config.camera_mask_tool + \
                    " --rig-file " + path_to_rig + \
                    " --mode \"export-all-camera-masks-to-json\" " + \
                    " --image-mask-file \"" + path_to_single_mask_folder + "\""

    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    if process.returncode != 0:
        raise ValueError("Fail to export the mask from the rig file.")
    else:
        print("Car masks are exported to the directory: %s\n" % path_to_single_mask_folder)

    return

def selectCamera(config, path_to_rig):
    cameras = printCameraSensors(config, path_to_rig)
    questions = [
        {
            'type': 'input',
            'name': 'ID',
            'message': 'Please select a camera ID: ',
        }
    ]
    answers = prompt(questions)

    camera_id = answers["ID"]
    if not camera_id.isdigit() or int(camera_id) < 0 or int(camera_id) >= len(cameras):
        print("The entered camera ID '%s' is not correct.\n" % camera_id)
        return None, None

    camera_id = int(camera_id)
    return cameras, camera_id

def getPathToImageFolder(path_to_rig):
    return os.path.join(os.path.dirname(path_to_rig), "extrinsics")

def getPathToCameraImage(path_to_rig, camera_name):
    path_to_image_folder = getPathToImageFolder(path_to_rig)
    if not os.path.isdir(path_to_image_folder):
        print("The directory with camera images does not exist: %s" % path_to_image_folder)
        return None

    base_path_to_image = os.path.join(path_to_image_folder, camera_name)
    path_to_camera_image = None
    for extension in SUPPORTED_INPUT_IMAGE_EXTENSIONS:
        if os.path.isfile(base_path_to_image + extension):
            path_to_camera_image = base_path_to_image + extension
            break

    if path_to_camera_image is None:
        print("The source camera image is not found in the directory: %s" % path_to_image_folder)
        print("Supported input image extensions are: %s" % SUPPORTED_INPUT_IMAGE_EXTENSIONS)

    return path_to_camera_image

def getPathToCameraMask(path_to_camera_image):
    path_to_mask_image = path_to_camera_image[:path_to_camera_image.rfind(".")] + "_mask.pgm"
    return path_to_mask_image

def getPathToSingleCameraMaskFolder(path_to_rig):
    return os.path.join(os.path.dirname(path_to_rig), "car_masks/")

def showConfig(config):
    print("\nThe current config:")
    config_info = list()
    for key, values in config._asdict().items():
        config_info.append([key, values])
    print(tabulate(config_info, headers=['Key', 'Value']))
    print("")

def makeMaskBinary(generated_mask_np):
    # input: a numpy mask, values are in range 0...255
    # output: a binary numpy mask, values are in range 0...255
    corrected_mask_np = 255 * (generated_mask_np > 128)
    return corrected_mask_np

def parse_arguments():
    parser = argparse.ArgumentParser(__file__,
                                     description="The tool helps to extract car masks from image data.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--image-editor", default="gimp", help="Executable name / path of the image-editor (default: %(default)s).")
    parser.add_argument("--camera-mask-tool", nargs='+', default=['../calibration-camera-mask-manual', './calibration-camera-mask-manual'], help="Executable name / path of the `calibration-camera-mask-manual` tool (default: %(default)s).")
    parser.add_argument("--target-mask-dimension", default=400, type=int, help="Target dimension of the image masks. Original images will be downscaled to be close to this target value in either x/y dimension, while exactly preserving the original aspect ratio (default: %(default)s).")
    parser.add_argument("-r", "--rig", required=True,
                        help="Path to the rig file (for example, <DWROOT>/data/tools/calibration/graph/camera-mask-manual-sample-rig.json).")
    return parser.parse_args()

def initConfig(args):
    Config = namedtuple('Config', ['image_editor', 'camera_mask_tool', 'target_mask_dimension'])
    config = Config(args.image_editor, get_executable(args.camera_mask_tool), args.target_mask_dimension)

    if not float(config.target_mask_dimension).is_integer() or not config.target_mask_dimension > 0:
        raise ValueError(
            "The value of the 'target-mask-dimension' parameter is not correct. It should be a positive integer number.")

    return config

def get_executable(candidates):
    """ Simple wrapper to determine valid executable path """
    for candidate in candidates:

        # Evaluate relative paths relative to script
        if not os.path.isabs(candidate):
            candidate = os.path.join(os.path.dirname(sys.argv[0]), candidate)

        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return os.path.abspath(candidate)

    raise RuntimeError('No valid executable found in candidates ' + str(candidates))


def getDownsampledImageSize(input_image_size, target_image_dimension):
    """ Returns the downsampled image size (with *same aspect ratio as input*) that best matches the target image dimension

        >>> getDownsampledImageSize((3848, 2168), 400)
        (481, 271)
        >>> getDownsampledImageSize((3848, 2168), 200)
        (481, 271)
        >>> getDownsampledImageSize((1920, 1208), 400)
        (480, 302)
        >>> getDownsampledImageSize((1920, 1208), 200)
        (240, 151)
    """

    if any([input_image_size[0] <= 0, input_image_size[1] <= 0]):
        raise RuntimeError('Invalid negative input images sizes')

    def costs(image_size):
        # Returns the costs associated with image_size by evaluating the minimum absolute distance to the requested target_image_dimension
        return min(abs(image_size[0] - target_image_dimension), abs(image_size[1] - target_image_dimension))

    def candidatesGenerator(image_size):
        # Yields all valid full downsample dimensions (including the original dimension)
        Candidate = namedtuple('Candidate', ['image_size', 'costs'])
        while True:
            yield Candidate(image_size, costs(image_size))

            # Downsample by halving resolutions
            next_image_size = image_size[0] / 2, image_size[1] / 2

            # Make sure resolution is still valid after downsampling
            if not all([next_image_size[0].is_integer(), next_image_size[1].is_integer()]):
                break

            # At this point we know that resolutions are integers and floats can be casted
            image_size = int(next_image_size[0]), int(next_image_size[1])

    # Select candidate with smallest costs
    candidates = candidatesGenerator(input_image_size)
    ret = next(candidates)
    for candidate in candidates:
        if candidate.costs < ret.costs:
            ret = candidate

    return ret.image_size


if __name__ == '__main__':
    main()
