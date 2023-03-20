/*
 @licstart  The following is the entire license notice for the JavaScript code in this file.

 The MIT License (MIT)

 Copyright (C) 1997-2020 by Dimitri van Heesch

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 and associated documentation files (the "Software"), to deal in the Software without restriction,
 including without limitation the rights to use, copy, modify, merge, publish, distribute,
 sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all copies or
 substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 @licend  The above is the entire license notice for the JavaScript code in this file
*/
var NAVTREE =
[
  [ "DriveWorks SDK Reference", "index.html", [
    [ "Getting Started With the NVIDIA DriveWorks SDK", "dwx_devguide_getting_started.html", [
      [ "Using the NVIDIA DRIVE SDK NGC Docker Container", "dwx_devguide_getting_started_drive_sdk_ngc_sdk_docker_container.html", null ],
      [ "Using the NVIDIA DRIVE SDK Debian Package Repository", "dwx_devguide_getting_started_drive_sdk_debian_package_repository.html", null ],
      [ "Using the NVIDIA SDK Manager", "dwx_devguide_getting_started_drive_sdk_nvidia_sdk_manager.html", null ],
      [ "Verifying the NVIDIA DriveWorks SDK Installation", "dwx_devguide_getting_started_verification.html", null ]
    ] ],
    [ "Modules", "dwx_modules.html", [
      [ "Core", "dwx_modules.html#dwx_core", [
        [ "Core", "core_mainsection.html", null ]
      ] ],
      [ "Sensor Abstraction Layer", "dwx_modules.html#dwx_sal", [
        [ "Sensors", "sensors_mainsection.html", [
          [ "Camera", "camera_mainsection.html", [
            [ "Supported Output Types", "camera_supported_output_types.html", null ],
            [ "IPP to NvSIPL Porting Guide", "camera_nvsipl_ipp_porting_guide.html", null ],
            [ "Cameras Supported", "supported_sensors.html", null ]
          ] ],
          [ "CAN Bus", "canbus_mainsection.html", null ],
          [ "GPS", "gps_mainsection.html", null ],
          [ "IMU", "imu_mainsection.html", null ],
          [ "Lidar", "lidar_mainsection.html", null ],
          [ "Radar", "radar_mainsection.html", null ],
          [ "Time", "time_mainsection.html", null ],
          [ "Sensor Manager", "sensormanager_mainsection.html", null ],
          [ "Integrating with Custom Sensors", "sensorplugins_mainsection.html", null ]
        ] ]
      ] ],
      [ "Vehicle And Motion Actuation", "dwx_modules.html#dwx_vehicle_motion_actuation", [
        [ "Rig Configuration", "rig_mainsection.html", null ],
        [ "VehicleIO", "vehicleio_mainsection.html", null ]
      ] ],
      [ "Image Processing", "dwx_modules.html#dwx_image_processing", [
        [ "Image", "image_mainsection.html", null ],
        [ "Image Transformation", "imagetransformation_mainsection.html", null ],
        [ "Color Correction", "colorcorrection_mainsection.html", null ],
        [ "Connected Components", "connectedcomponents_mainsection.html", null ],
        [ "Rectifier", "rectifier_mainsection.html", null ],
        [ "FeatureDetector", "imageprocessing_featuredetector_mainsection.html", null ],
        [ "Features", "imageprocessing_features_mainsection.html", null ],
        [ "Filtering", "imageprocessing_filtering_mainsection.html", null ],
        [ "Box Tracking", "imageprocessing_tracking_boxtracker2d_mainsection.html", null ],
        [ "Feature Tracking", "imageprocessing_tracking_featuretracker_mainsection.html", null ],
        [ "Template Tracking", "imageprocessing_tracking_templatetracker_mainsection.html", null ],
        [ "SFM", "sfm_mainsection.html", null ],
        [ "Stereo", "stereo_mainsection.html", null ],
        [ "Pose estimation", "imageprocessing_geometry_pose_mainsection.html", null ]
      ] ],
      [ "Point Cloud Processing", "dwx_modules.html#dwx_point_processing", [
        [ "Point Cloud Processing", "pointcloudprocessing_mainsection.html", null ]
      ] ],
      [ "Deep Neural Network (DNN) Framework", "dwx_modules.html#dwx_dnn_framework", [
        [ "Data Conditioner", "dataconditioner_mainsection.html", null ],
        [ "DNN", "dnn_mainsection.html", null ],
        [ "Clusterer", "clusterer_mainsection.html", null ]
      ] ],
      [ "Calibration", "dwx_modules.html#dwx_calibration", [
        [ "Intrinsic Camera Model", "cameramodel_mainsection.html", null ],
        [ "Self-Calibration", "calibration_mainsection.html", null ],
        [ "Egomotion", "egomotion_mainsection.html", null ]
      ] ],
      [ "Communication", "dwx_modules.html#dwx_comms", [
        [ "Inter-process Communication (IPC)", "ipc_mainsection.html", null ]
      ] ],
      [ "Utility", "dwx_modules.html#dwx_utility", [
        [ "Renderer", "renderer_mainsection.html", null ]
      ] ]
    ] ],
    [ "Samples", "dwx_samples_section.html", [
      [ "Introductory Samples", "dwx_samples_section.html#dwx_introductory_samples_group", [
        [ "Hello World Sample", "dwx_hello_world_sample.html", null ]
      ] ],
      [ "Sensor Abstraction Layer Samples", "dwx_samples_section.html#dwx_sensor_abstraction_layer", [
        [ "Camera Samples", "dwx_samples_section.html#dwx_camera_sensor_samples_group", [
          [ "USB Camera Capture Sample", "dwx_camera_usb_sample.html", null ],
          [ "Camera Sample", "dwx_camera_sample.html", null ],
          [ "Camera Replay Sample", "dwx_camera_replay_sample.html", null ],
          [ "Camera Seek Sample", "dwx_camera_seek_sample.html", null ]
        ] ],
        [ "Other Sensor Samples", "dwx_samples_section.html#dwx_other_sensor_samples_group", [
          [ "Sensor Enumeration Sample", "dwx_sensor_enum_sample.html", null ],
          [ "CAN Message Interpreter Sample", "dwx_canbus_message_sample.html", null ],
          [ "CAN Message Logger Sample", "dwx_canbus_logger_sample.html", null ],
          [ "CAN Plugin Sample", "dwx_canbus_plugin_sample.html", null ],
          [ "GPS Location Logger Sample", "dwx_gps_loc_sample.html", null ],
          [ "GPS Plugin Sample", "dwx_gps_plugin_sample.html", null ],
          [ "IMU Logger Sample", "dwx_imu_loc_sample.html", null ],
          [ "IMU Plugin Sample", "dwx_imu_plugin_sample.html", null ],
          [ "Lidar Replay Sample", "dwx_lidar_replay_sample.html", null ],
          [ "Lidar Plugin Sample", "dwx_lidar_plugin_sample.html", null ],
          [ "Radar Replay Sample", "dwx_radar_replay_sample.html", null ],
          [ "Radar Plugin Sample", "dwx_radar_plugin_sample.html", null ],
          [ "Simple Sensor Recording Sample", "dwx_record_sample.html", null ],
          [ "Time Sensor Sample", "dwx_time_sensor_sample.html", null ],
          [ "Data Sensor Sample", "dwx_data_sensor_sample.html", null ]
        ] ]
      ] ],
      [ "Vehicle And Motion Actuation Samples", "dwx_samples_section.html#dwx_vehicle_samples_group", [
        [ "Rig Configuration Sample", "dwx_rig_sample.html", null ],
        [ "VehicleIO Sample", "dwx_vehicleio_sample.html", null ],
        [ "VehicleIO Plugin Sample", "dwx_vehicleio_plugin_sample.html", null ],
        [ "Egomotion Sample", "dwx_egomotion_sample.html", null ],
        [ "Dataspeed Bridge Sample", "dwx_dataspeedBridge_sample.html", null ]
      ] ],
      [ "Image Processing Samples", "dwx_samples_section.html#dwx_image_processing_samples_group", [
        [ "Image Transformation Sample", "dwx_imagetransformation.html", null ],
        [ "Camera Color Correction Sample", "dwx_camera_color_correction_sample.html", null ],
        [ "Image Capture Sample", "dwx_image_capture_sample.html", null ],
        [ "Image Streamer Multi-Thread Sample", "dwx_image_streamer_multi_sample.html", null ],
        [ "Image Streamer Simple Sample", "dwx_image_streamer_simple_sample.html", null ],
        [ "Video Rectification Sample", "dwx_video_rectifier_sample.html", null ],
        [ "Video Rectification with LDC Sample", "dwx_video_rectifierLDC_sample.html", null ],
        [ "Stereo Disparity Sample", "dwx_stereo_disparity_sample.html", null ],
        [ "Connected Components Sample", "dwx_connected_components_sample.html", null ],
        [ "Feature Tracker Sample", "dwx_feature_tracker_sample.html", null ],
        [ "FAST9 Feature Detector Sample", "dwx_fast9_feature_detector_sample.html", null ],
        [ "Template Tracker Sample", "dwx_template_tracker_sample.html", null ],
        [ "Structure from Motion (SFM) Sample", "dwx_struct_from_motion_sample.html", null ]
      ] ],
      [ "Point Cloud Processing Samples", "dwx_samples_section.html#dwx_pcloud_processing_samples_group", [
        [ "Point Cloud Processing Sample", "dwx_pointcloudprocessing_sample.html", null ],
        [ "Iterative Closest Points (ICP) Sample", "dwx_sample_icp.html", null ]
      ] ],
      [ "Deep Neural Network (DNN) Framework Samples", "dwx_samples_section.html#dwx_dnn_samples_group", [
        [ "DNN Plugin Sample", "dwx_dnn_plugin_sample.html", null ],
        [ "Basic Object Detector and Tracker Sample", "dwx_object_detector_tracker_sample.html", null ],
        [ "Basic Object Detector Using DNN Tensor Sample", "dwx_sample_dnn_tensor.html", null ]
      ] ],
      [ "Calibration Samples", "dwx_samples_section.html#dwx_calibration_samples_group", [
        [ "Camera Calibration Sample", "dwx_camera_calibration_sample.html", null ],
        [ "IMU Calibration Sample", "dwx_imu_calibration_sample.html", null ],
        [ "Lidar Calibration Sample", "dwx_lidar_calibration_sample.html", null ],
        [ "Radar Calibration Sample", "dwx_radar_calibration_sample.html", null ],
        [ "Stereo Calibration Sample", "dwx_stereo_calibration_sample.html", null ],
        [ "Steering Calibration Sample", "dwx_vehicle_steering_calibration_sample.html", null ]
      ] ],
      [ "Communication Samples", "dwx_samples_section.html#dwx_communication_samples_group", [
        [ "Inter-process Communication (IPC) Sample", "dwx_ipc_socketclientserver_sample.html", null ]
      ] ],
      [ "Utility Samples", "dwx_samples_section.html#dwx_utility_samples_group", [
        [ "Rendering Sample", "dwx_renderer_sample.html", null ],
        [ "Rendering Engine Sample", "dwx_render_engine_sample.html", null ]
      ] ]
    ] ],
    [ "Tools", "dwx_tools_section.html", [
      [ "Sensor Tools", "dwx_tools_section.html#dwx_tools_sensors", [
        [ "Rig Viewer Tool ", "dwx_rig_viewer_tool.html", null ],
        [ "Rig Reserializer Tool ", "dwx_rig_json2json_tool.html", null ],
        [ "Sensor Indexer Tool", "dwx_sensor_indexer_tool.html", null ],
        [ "SIPL Query Tool", "dwx_sipl_query_tool.html", null ],
        [ "Sensors Initialization Tool", "dwx_sensor_initializer_tool.html", null ]
      ] ],
      [ "Recording Tools", "dwx_tools_section.html#dwx_tools_recording", [
        [ "General Recording Tools", "dwx_recording_tools.html", [
          [ "Basic Recording Tool", "dwx_recorder_tool.html", null ],
          [ "Text UI Recording Tool", "dwx_recorder_textui_tool.html", null ],
          [ "GUI Recording Tool", "dwx_gui_recording2_tool.html", null ],
          [ "Configuration Reference", "dwx_config_ref.html", null ]
        ] ]
      ] ],
      [ "Post-Recording Tools", "dwx_tools_section.html#dwx_tools_postrecording", [
        [ "Postrecord Checker", "dwx_postrecord_checker.html", null ],
        [ "Recording Chopping Tool", "dwx_recording_chopping_tool.html", null ],
        [ "Replayer Tool", "dwx_replayer_tool.html", null ],
        [ "Video Exporter Tool", "dwx_video_exporter_tool.html", null ],
        [ "Muxer mp4", "dwx_muxer_mp4_tool.html", null ],
        [ "Recording Header Dump", "dwx_headerdump_tool.html", null ],
        [ "LRAW to RAW Conversion Tool", "dwx_tools_lraw2raw.html", null ],
        [ "LRAW Preview Extraction Tool", "dwx_tools_extractlraw.html", null ],
        [ "CAN Recording Update Tool", "@ref virtual_can_file_updater", null ]
      ] ],
      [ "Calibration Tools", "dwx_tools_section.html#dwx_tools_calibration", [
        [ "Camera Calibration Tools", "dwx_camera_calibration_tools.html", [
          [ "Graph Calibration Tool", "dwx_calibration_graph_cli.html", null ],
          [ "Calibrated Graph to Rig File Tool", "dwx_calibration_graph_to_rig.html", null ],
          [ "Intrinsics Constraints Tool", "dwx_intrinsics_constraints.html", null ],
          [ "Intrinsics Validator Tool", "dwx_calibration_int_val.html", null ],
          [ "Camera Mask Tool", "dwx_calibration_camera_mask.html", null ],
          [ "Static Calibration Recorder Tool", "dwx_calibration_recorder.html", null ]
        ] ],
        [ "IMU Calibration Tool", "dwx_imu_calibration_tool.html", null ]
      ] ],
      [ "DNN Framework Tools", "dwx_tools_section.html#dwx_tensorRT_optimization", [
        [ "TensorRT Optimization Tool", "dwx_tensorRT_tool.html", null ]
      ] ],
      [ "General Tools", "dwx_tools_section.html#dwx_tools_general", [
        [ "DriveWorks Info Tool", "dwx_info_tool.html", null ]
      ] ]
    ] ],
    [ "Tutorials", "dwx_tutorials.html", [
      [ "Basic Tutorials", "basic_tutorials.html", [
        [ "DriveWorks 101", "basic_tutorials.html#dwx_101_tutorials", [
          [ "Conventions", "dwx_conventions.html", [
            [ "API Naming Conventions and General Structures", "dwx_naming_conventions.html", null ],
            [ "Coordinate Systems", "dwx_coordinate_systems.html", null ],
            [ "Alignment", "dwx_alignment.html", null ]
          ] ],
          [ "Context", "core_usecase5.html", null ],
          [ "System/Platform Information", "core_usecase2.html", null ],
          [ "Hello World Application", "dwx_hello_world.html", null ],
          [ "Choosing the GPU for Execution", "dwx_samples_gpu.html", null ],
          [ "Logging", "core_usecase4.html", null ],
          [ "Memory Management Policies and Multithreading", "core_usecase3.html", null ]
        ] ],
        [ "Accessing Sensors", "basic_tutorials.html#dwx_sensors_tutorials", [
          [ "Sensors Overview", "basic_tutorials.html#dwx_sensors_overview_tutorials", [
            [ "Sensors Life Cycle", "sensors_usecase1.html", null ],
            [ "Sensors Querying", "sensors_usecase2.html", null ],
            [ "Sensors Serialization", "sensors_usecase3.html", null ],
            [ "Sensors Timestamping", "sensors_usecase4.html", null ],
            [ "Replaying Sensors", "sensors_usecase5.html", null ]
          ] ],
          [ "Camera Usage", "basic_tutorials.html#dwx_sensors_camera_tutorials", [
            [ "Camera Workflow", "camera_usecase1.html", null ],
            [ "SIPL-based Image Sensors (Live)", "camera_usecase3.html", null ],
            [ "SIPL-based image sensors (Virtual)", "camera_usecase4.html", null ]
          ] ],
          [ "CANBUS Usage", "basic_tutorials.html#dwx_sensors_canbus_tutorials", [
            [ "Receiving and Sending Data", "canbus_usecase1.html", null ],
            [ "Reading CAN Messages from Raw Data (Binary)", "canbus_usecase2.html", null ],
            [ "CAN Interpreter", "canbus_usecase3.html", null ]
          ] ],
          [ "GPS Usage", "basic_tutorials.html#dwx_sensors_gps_tutorials", [
            [ "Reading GPS data from sensor", "gps_usecase1.html", null ],
            [ "Reading GPS data from raw data", "gps_usecase2.html", null ]
          ] ],
          [ "IMU Usage", "basic_tutorials.html#dwx_sensors_imu_tutorials", [
            [ "Reading IMU data from sensor", "imu_usecase1.html", null ],
            [ "Reading IMU data from raw data", "imu_usecase2.html", null ]
          ] ],
          [ "Lidar Usage", "basic_tutorials.html#dwx_sensors_lidar_tutorials", [
            [ "Reading Lidar data from sensor", "lidar_usecase1.html", null ],
            [ "Reading Lidar data from raw data", "lidar_usecase2.html", null ],
            [ "Increase Packet Queue Size ", "lidar_usecase3.html", null ]
          ] ],
          [ "Radar Usage", "basic_tutorials.html#dwx_sensors_radar_tutorials", [
            [ "Radar Workflow", "radar_usecase1.html", null ]
          ] ],
          [ "Time Usage", "basic_tutorials.html#dwx_sensors_time_tutorials", [
            [ "Get Time Synchronized Data", "time_usecase1.html", null ]
          ] ]
        ] ],
        [ "Accessing Rig Configurations", "basic_tutorials.html#dwx_rig_tutorials", [
          [ "Rig File Format", "rigconfiguration_usecase0.html", null ],
          [ "Accessing Vehicle Properties and Sensors From a Rig File", "rigconfiguration_usecase1.html", null ]
        ] ],
        [ "Utilities", "basic_tutorials.html#dwx_utility_tutorials", [
          [ "IPC Workflow", "ipc_usecase1.html", null ],
          [ "Renderer Workflow", "renderer_usecase1.html", null ],
          [ "Renderer Engine Workflow", "renderer_usecase2.html", null ]
        ] ],
        [ "Static Calibration and Recording Data", "basic_tutorials.html#dwx_calib_record_tutorials", [
          [ "Calibration Tools Overview", "dwx_calibration_overview.html", null ],
          [ "Static Camera Calibration", "dwx_camera_calibration.html", null ],
          [ "Recording Sensor Data", "dwx_recording_devguide_group.html", [
            [ "Basic Recording", "dwx_recording_devguide_basic_recording.html", null ],
            [ "Distributed Recording", "dwx_devguide_rec_distrec.html", null ],
            [ "High Throughput Recording", "dwx_recording_devguide_high_throughput_recording.html", null ]
          ] ]
        ] ]
      ] ],
      [ "Intermediate Tutorials", "intermediate_tutorials.html", [
        [ "Sensor Management", "intermediate_tutorials.html#dwx_intermediate_tutorials_sensors", [
          [ "Sensor Manager Workflow", "sensormanager_usecase1.html", null ]
        ] ],
        [ "Image Processing", "intermediate_tutorials.html#dwx_intermediate_tutorials_ip", [
          [ "Image Creation and Conversion", "image_usecase5.html", null ],
          [ "Image initialization using PNG file", "md_src_dw_image_doc_public_usecase6.html#image_usecase6", null ],
          [ "Image Scaling", "imagetransformation_usecase1.html", null ],
          [ "Image Streamer", "image_usecase2.html", null ],
          [ "Image Streamer Multi-Thread", "image_usecase4.html", null ],
          [ "Image Streamer Cross-Process", "image_usecase3.html", null ],
          [ "Image Capture", "image_usecase1.html", null ],
          [ "Camera Color Correction Workflow", "colorcorrection_usecase1.html", null ],
          [ "Connected Components Workflow", "connectedcomponents_usecase1.html", null ],
          [ "Rectifier Workflow", "rectifier_usecase1.html", null ],
          [ "Single Camera Feature Tracking", "@ref imageprocessing_tracking_usecase1", null ],
          [ "Single Camera Template Tracking", "@ref imageprocessing_tracking_usecase2", null ],
          [ "2D Box Tracking", "@ref imageprocessing_tracking_usecase3", null ],
          [ "Disparity Computation Workflow", "stereo_usecase1.html", null ],
          [ "Disparity Computation Workflow on PVA and NVENC", "stereo_usecase2.html", null ],
          [ "Structure from Motion (SFM) Workflow", "sfm_usecase1.html", null ],
          [ "Pose Estimation Workflow", "imageprocessing_geometry_pose_usecase1.html", null ]
        ] ],
        [ "Point Cloud Processing", "intermediate_tutorials.html#dwx_intermediate_tutorials_pc_processing", [
          [ "Point Cloud Memory Management", "pointcloudprocessing_usecase1.html", null ],
          [ "Point Cloud Accumulation", "pointcloudprocessing_usecase2.html", null ],
          [ "Point Cloud Stitching", "pointcloudprocessing_usecase3.html", null ],
          [ "Point Cloud Range Image Creation", "pointcloudprocessing_usecase4.html", null ],
          [ "Point Cloud ICP", "pointcloudprocessing_usecase5.html", null ],
          [ "Point Cloud Plane Extraction", "pointcloudprocessing_usecase6.html", null ],
          [ "Point Cloud Filter", "pointcloudprocessing_usecase7.html", null ]
        ] ],
        [ "Deep Neural Networks (DNN)", "intermediate_tutorials.html#dwx_intermediate_tutorials_dnn", [
          [ "Data Conditioner Workflow", "dataconditioner_usecase1.html", null ],
          [ "DNN Workflow", "dnn_usecase1.html", null ],
          [ "DNN Tensors", "dnn_usecase2.html", null ],
          [ "DNN with Safe DLA", "dnn_usecase3.html", null ],
          [ "Clusterer Workflow", "clusterer_usecase1.html", null ]
        ] ],
        [ "Vehicle Actuation", "intermediate_tutorials.html#dwx_intermediate_tutorials_vehicle", [
          [ "VehicleIO Workflow", "vehicleio_usecase1.html", null ],
          [ "Egomotion Workflow", "egomotion_usecase1.html", null ],
          [ "Absolute Egomotion Workflow", "egomotion_usecase2.html", null ]
        ] ],
        [ "Self-Calibration", "intermediate_tutorials.html#dwx_intermediate_tutorials_selfcalib", [
          [ "Ray-to-Pixel and Pixel-to-Ray", "cameramodel_usecase0.html", null ],
          [ "Feature-based Camera Self-Calibration", "calibration_usecase_features.html", null ],
          [ "IMU Self-Calibration", "calibration_usecase_imu.html", null ],
          [ "Lidar Self-Calibration", "calibration_usecase_lidar.html", null ],
          [ "Radar Self-Calibration", "calibration_usecase_radar.html", null ],
          [ "Epipolar-based Stereo Self-Calibration", "calibration_usecase_stereo.html", null ],
          [ "Steering Self-Calibration", "calibration_usecase_vehicle.html", null ]
        ] ]
      ] ],
      [ "Advanced Tutorials", "advanced_tutorials.html", [
        [ "Sensor Customizations", "advanced_tutorials.html#dwx_advanced_tutorials_custom", [
          [ "Integrating with Custom Boards", "camera_usecase2.html", null ],
          [ "Custom Lidars (Decoder Only)", "sensorplugins_lidardecoder.html", null ],
          [ "Custom Lidars (Comprehensive)", "sensorplugins_lidarsensor.html", null ],
          [ "Custom Radars (Decoder Only)", "sensorplugins_radardecoder.html", null ],
          [ "Custom Radars (Comprehensive)", "sensorplugins_radarsensor.html", null ],
          [ "Custom IMUs (Comprehensive)", "sensorplugins_imusensor.html", null ],
          [ "Custom GPSs (Comprehensive)", "sensorplugins_gpssensor.html", null ],
          [ "Custom CANs (Comprehensive)", "sensorplugins_canbussensor.html", null ],
          [ "Custom Cameras (SIPL)", "sensorplugins_camerasipl.html", null ],
          [ "DNN Plugins", "dwx_dnn_plugins.html", null ],
          [ "VehicleIO Plugins", "dwx_vehicleio_plugins.html", null ]
        ] ]
      ] ]
    ] ],
    [ "SDK Porting Guide", "dwx_porting_guide.html", [
      [ "From SDK 5.8 to SDK 5.10", "@ref dwx_porting_guide_5_10", null ],
      [ "From SDK 5.6 to SDK 5.8", "@ref dwx_porting_guide_5_8", null ],
      [ "From SDK 5.4 to SDK 5.6", "dwx_porting_guide_5_6.html", null ],
      [ "From SDK 5.2 to SDK 5.4", "md_doc_portingguide_5_2_to_5_4.html#dwx_porting_guide_5_4", null ],
      [ "From SDK 5.0 to SDK 5.2", "dwx_porting_guide_5_2.html", null ],
      [ "From SDK 4.0 to SDK 5.0", "dwx_porting_guide_5_0.html", null ],
      [ "From SDK 3.5 to SDK 4.0", "dwx_porting_guide_4_0.html", null ],
      [ "From SDK 3.0 to SDK 3.5", "dwx_porting_guide_3_5.html", null ],
      [ "From SDK 2.2 to SDK 3.0", "dwx_porting_guide_3_0.html", null ],
      [ "From SDK 2.0 to SDK 2.2", "dwx_porting_guide_2_2.html", null ],
      [ "From SDK 1.5 to SDK 2.0", "dwx_porting_guide_2_0.html", null ]
    ] ],
    [ "DriveWorks API", "modules.html", "modules" ],
    [ "Frequently Asked Questions", "dwx_faq.html", null ],
    [ "More", "usergroup0.html", [
      [ "Open Source and Third-Party Licenses", "dwx_open_source_attribution.html", null ],
      [ "NVIDIA Legal Information", "nvidia_legal.html", null ],
      [ "File List", "files.html", [
        [ "List", "files.html", "files_dup" ]
      ] ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"@ref dwx_porting_guide_5_10",
"MatrixTypes_8h.html#structdwVector2d",
"Tonemap_8h.html#a1bb713edbc6bef3a1eb33d831f758210ab2d1e9c1b94beb5749854e5adfcccba6",
"dir_901113639e66742d31cba0896b0c8ccf.html",
"dwx_vehicleio_plugins.html",
"group__VehicleIO__actuators__group.html#a415681e9449bba8bef9b2c18ec5128d6",
"group__VehicleIO__actuators__group.html#a8479f38f6a86b064536587b8b7f088ed",
"group__VehicleIO__actuators__group.html#ad3179bf38f6dbfd7cbc337df65d6b600",
"group__VehicleIO__actuators__group.html#ga8c7013975cc62f31a980e147200e7d80",
"group__VehicleIO__actuators__group.html#gga44d8220df31bf834963f5b09d32319c6a4d1282f8376508cbcf7d81e07790e38e",
"group__VehicleIO__actuators__group.html#gga8ee0f8b75b2c830f66008a24233cc39bafcb1d6f9e2c2c5d122fe984e8f00bb0d",
"group__VehicleIO__actuators__group.html#ggad582ef62ff8e6641323bb3baa96a1b3ead4e2f85e33037e26c8f55eff00e146a6",
"group__calibration__types__group.html#ggaab568f481f03e5a5dc14fb3ada6d9b40a669fbc422595b6dc378b571f02b340f3",
"group__color__correct.html#ad2cd9d099d222d7690400af27b75d682",
"group__core__types__group.html#ggaa47781681c93aac1f28bc9bef7b07960a26b83e423481535ae5819e59cb24d854",
"group__egomotion__group.html#aeb563a1f04ec7ac61704aa0cbee6736d",
"group__featureDetector__group.html#af81c8330b5436a09eec0cd5a58f75ce2",
"group__image__group.html#a70ab6cec58e0196187f23804b8964958",
"group__imu__group.html#ga1e2d921bd835f50e7bd55efac12984d9",
"group__pnp__group.html#gaab7c6056752f703d7e99e3e057bf32d7",
"group__pointcloudrangeimagecreator__group.html#ga9ef52b6954de59fc635751cd5bc87357",
"group__render__engine__group.html#ga430c59d51b644178e2eddebd78d28e54",
"group__renderer__group.html#ggaa5b4ce4113f5bd03ccd99b6033cf8c86ae28df26bb0595aad0c2468ae13996dd9",
"group__sensor__plugins__ext__common__group.html#ga38fe79550a16684becc46a6c6b111156",
"group__sensors__common__group.html#gga824ba1e3d8ed1cfa0e0e131c40ccc6a7a1bdfbc698f71d64a9744912226db4db3",
"interop_2streamer_2ImageStreamer_8h_source.html"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';