# Car Mask Extraction Tool

## Introduction
This tool helps to create, edit and manipulate car masks in the DriveWorks rig-files. 

## Prerequisites

### Python

Python 3.6 is required.

### Installation

Please run the following command to install required python packages:
```
python3.6 -m pip install --user PyInquirer==1.0.3 tabulate==0.8.7 Pillow==7.2 numpy==1.19.4 --upgrade
```

### GIMP
Please install GIMP image editor before running this tool (`http://www.gimp.org/`).

### Parameters
The car mask extraction tool can be configured using command line parameters. 

The table below explains all supported optional command line parameters:

| Parameter | Description |
|---|---|
| `--image-editor` | the name of the image-editor |
| `--camera-mask-tool` | a path to the `calibration-camera-mask-manual` tool |
| `--target-mask-dimension` | a target dimension of the image masks. Original images will be downscaled to be close to this target value in either x/y dimension, while exactlay preserving the original aspect ratio  |

## Input Data
We assume the tool is executed after the extrinsic static camera calibration process. 
At this point the following directory structure is available:
```
<dataset-root>
   - <other subfolders, unused by this tool>
   - extrinsics/
   - rig.json
```
We assume the file `rig.json` contains all camera sensors for the car mask extraction.
In addition we assume a sample image is available in the `extrinsics` subfolder for each camera sensor in the `rig.json`.
These image are used for mask extraction. 
The name of the image file should correspond to the camera sensor name in the rig-file.

## Run the Tool
Please run the following command to start the tool:
```
python3.6 calibration-camera-mask-manual.py --rig <path-to-rig-file>
``` 

A sample of the rig-file is available here:
```
<PATH_TO_DW>/data/tools/calibration/graph/camera-mask-manual-sample-rig.json
```

Please follow the corresponding tool's menu options for the car mask extraction.  

### GIMP Image Processing Pipeline
This section describes how to use GIMP for mask selection (tested with GIMP 2.10.6).
The tool will automatically open an image to process in GIMP. Follow these steps to select a mask:
1. Choose the `Free Select Tool` (menu `Tools` -> `Selection Tools` -> `Free Select`)
    * All masks are stored in a lower resolution relative to the original image (see the `target-mask-dimension` parameter for details). 
      The tool automatically adjusts the size of the input image before starting GIMP.
    * Check if an `antialiasing` mode is switched off, correct if needed.
2. Draw the car mask (double-click to finish the polygon)
    * Depending on the camera location it might be easier to draw to car area or actual image area.
    * One can use an option `Invert` (menu `Select` -> `Invert`) to invert the selection mask 
3. Choose the `Bucket Fill Tool` (menu `Tools` -> `Paint Tools` -> `Bucket Fill`)
4. Fill the car area with a white color and all remaining area with a black color. 
    * Check if "Fill whole selection" is checked, correct if needed
    * Check if the Opacity value is 100.0, correct if needed.
5. Save the updated mask: menu `File` -> `Overwrite <mask image name>`
6. Close GIMP and press `Discard changes`

The sample video on the page `https://confluence.nvidia.com/display/DS/How+to+Self-Calibrate+Sensors` illustrates a typical pipeline in GIMP.
