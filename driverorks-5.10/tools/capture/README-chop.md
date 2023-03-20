# SPDX-FileCopyrightText: Copyright (c) 2017-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_recording_chopping_tool Recording Chopping Tool

This tool is available on the x86 Host System and NVIDIA DRIVE<sup>&trade;</sup> OS Linux.

A tool for clipping recorded data. Allows to cut information from input recordings
based on given timestamp or event range.

# Usage

    ./chop  [--config=path_to_json_config_file]
            [--camera=path_to_camera_data]
            [--camera-out=output_path]
            [--timestamp_file=path_to_file_containing_timestamps]
            [--gps=path_to_gps_data]
            [--gps-out=output_path]
            [--imu=path_to_imu_data]
            [--imu-out=output_path]
            [--can=path_to_can_data]
            [--can-out=output_path]
            [--lidar=path_to_lidar_data]
            [--lidar-out=output_path]
            [--radar=path_to_radar_data]
            [--radar-out=output_path]
            [--ultrasonic=path_to_ultrasonic_data]
            [--ultrasonic-out=output_path]
            [--data=path_to_data_sensor]
            [--data-out=output_path]
            [--time=path_to_time_sensor]
            [--time-out=output_path]
            [--input-session=path1,path2,path3 or path/*]
            [--output-session=/path]
            [--useInputSessionDirName=true/false]
            [--result-file=/path/result.json]
            [--timestamp_start=start_timestamp_value]
            [--timestamp_end=end_timestamp_value]
            [--event_start=start_event_index]
            [--event_end=end_event_index]
            [--num_events=number_of_events_to_capture]
            [--force-create-seek]
            [--force-create-seek-regex]
            [--disable-output-verification]
            [--ignore-missing-timestamp]
            [--disable-cuda]

# Examples

    ./chop --camera=/path_to_video --timestamp_file=/path_to_existing_timestamp_file --num_events=20
    Capture the first 20 frames from provided video. Cut associated timestamp file.


    ./chop --camera=/path_to_video.h264 --ignore-missing-timestamp --num_events=20 --disable-cuda
    Capture the first 20 frames from provided video. Force cut of the video file only and ignore missing timestamp file without cuda environment.

    ./chop --camera=/path_to_video.h264 --ignore-missing-timestamp --num_events=20
    Capture the first 20 frames from provided video. Force cut of the video file only and ignore missing timestamp file.

    ./chop --lidar=/lidar_file --radar=/radar_file --timestamp_start=1506453354917264
    Cut all lidar and radar packets starting timestamp 1506453354917264 to the end of the stream.

    ./chop --can=/can_file --timestamp_start=1506453354917264 --num_events=50
    Chop 50 can messages starting from timestamp 1506453354917264.

    ./chop --imu=/imu_file --event_start=10 --event_end=11
    Capture events 10 and 11.

    ./chop --imu=/imu_file --event_start=10 --event_end=11 --num_events=20
    Capture 20 events from event 10. --event_end argument is ignored.

    ./chop --config=/path_to_config_file
    Chop recordings listed in provided config file, or create template config if specified file does not exist.

    ./chop --input-session=/mnt/SSD1,/mntSSD2 --output-session=/mnt/CHOP --timestamp_start=1506453354917264 --timestamp_end=1506453355017264 --generate-result=/mnt/CHOP/chopping_result.json --force-create-seek-regex=.*xsens.*
    Chop a session to produce another session with a new UUID and generate a json file with a summary of the chopping result.

# Configuration file

Template configuration file looks like:

```
{
    "version": "0.1",
    "timestamp_start": 0,
    "timestamp_end": 0,
    "event_start": 0,
    "event_end": 0,
    "num_events": 0,

    "camera" : [
          {
              "input_file": "",
              "output_file": "",
              "timestamp_file": ""
          }
    ],

    "can" : [
          {
              "input_file": "",
              "output_file": ""
          }
    ],

    "gps" : [
         {
             "input_file": "",
             "output_file": ""
         }
    ],

    "imu" : [
         {
             "input_file": "",
             "output_file": ""
         }
    ],

    "lidar" : [
         {
             "input_file": "",
             "output_file": ""
         }
    ],

    "radar" : [
         {
             "input_file": "",
             "output_file": ""
         }
    ],

    "ultrasonic" : [
         {
             "input_file": "",
             "output_file": ""
         }
    ],

    "data" : [
         {
             "input_file": "",
             "output_file": ""
         }
    ]
}
```

Parameters provided in configuration file are always primary with respect to those specified in command line.
That means if the range specified in both command line and configuration file the latter are chosen even if
they incorrect or zero. Thus if you want to use first variant then completely remove corresponding sections
from json config.

# Limitations

- For H264/H265/mp4 videos the interval is adjusted to start from the previous IDR frame and end at the next IDR frame.
- Chopping H264/H265 videos requires AUD units
- NMEA ASCII gps recordings are no longer supported
