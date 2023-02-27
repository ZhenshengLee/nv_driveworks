# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page virtual_can_file_updater CAN Recording Update Tool

The CAN Recording Update Tool upgrades CAN sensor recordings from the old 22 byte dwCanMessage structure to the new 78 byte dwCANMessage structure.<br>
The tool will create a new version of the specified file.

If seek files are present, then these are also updated.

### Usage

## Typical Usage
- Run the application

        ./virtual_can_file_updater  --file=/path/to/file/to/be/updated

## Options
###Required

####Existing file name

    --file=/path/to/file/to/be/updated

###Optional

####New file version

    --newVersion=newVersionNumber

If no new version number is specified, the the old version number is read from the
existing file and +1 added.

####New file name

    --newFile=/path/to/new/file

If no new file is specified, then the new files will be created with _v{newVersionNumber}.

####Radar CAN Mode

    --radarMode=[True,False]

Default value = `False`

Continental ARS430 CAN Radar files that have a 12 byte, non CAN entry before the CAN entries of the Radar event.<br>
The contents are a 4 byte unsigned int `size` indicating how many bytes are in the CAN radar event, and an 8 byte time stamp.<br>
The `size` value is updated to reflect the new `dwCANMessage` size.

`True` Use Radar CAN Mode

`False` Don't use Radar CAN Mode
