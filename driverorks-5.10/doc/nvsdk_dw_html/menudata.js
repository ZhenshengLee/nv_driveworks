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
var menudata={children:[
{text:"Welcome",url:"index.html"},
{text:"Getting Started With the NVIDIA DriveWorks SDK",url:"dwx_devguide_getting_started.html"},
{text:"Modules",url:"dwx_modules.html",children:[
{text:"Core",url:"dwx_modules.html#dwx_core",children:[
{text:"Core",url:"core_mainsection.html"}]},
{text:"Sensor Abstraction Layer",url:"dwx_modules.html#dwx_sal",children:[
{text:"Sensors",url:"sensors_mainsection.html",children:[
{text:"Camera",url:"camera_mainsection.html"},
{text:"CAN Bus",url:"canbus_mainsection.html"},
{text:"GPS",url:"gps_mainsection.html"},
{text:"IMU",url:"imu_mainsection.html"},
{text:"Lidar",url:"lidar_mainsection.html"},
{text:"Radar",url:"radar_mainsection.html"},
{text:"Time",url:"time_mainsection.html"},
{text:"Sensor Manager",url:"sensormanager_mainsection.html"},
{text:"Integrating with Custom Sensors",url:"sensorplugins_mainsection.html"}]}]},
{text:"Vehicle And Motion Actuation",url:"dwx_modules.html#dwx_vehicle_motion_actuation",children:[
{text:"Rig Configuration",url:"rig_mainsection.html"},
{text:"VehicleIO",url:"vehicleio_mainsection.html"}]},
{text:"Image Processing",url:"dwx_modules.html#dwx_image_processing",children:[
{text:"Image",url:"image_mainsection.html"},
{text:"Image Transformation",url:"imagetransformation_mainsection.html"},
{text:"Color Correction",url:"colorcorrection_mainsection.html"},
{text:"Connected Components",url:"connectedcomponents_mainsection.html"},
{text:"Rectifier",url:"rectifier_mainsection.html"},
{text:"FeatureDetector",url:"imageprocessing_featuredetector_mainsection.html"},
{text:"Features",url:"imageprocessing_features_mainsection.html"},
{text:"Filtering",url:"imageprocessing_filtering_mainsection.html"},
{text:"Box Tracking",url:"imageprocessing_tracking_boxtracker2d_mainsection.html"},
{text:"Feature Tracking",url:"imageprocessing_tracking_featuretracker_mainsection.html"},
{text:"Template Tracking",url:"imageprocessing_tracking_templatetracker_mainsection.html"},
{text:"SFM",url:"sfm_mainsection.html"},
{text:"Stereo",url:"stereo_mainsection.html"},
{text:"Pose estimation",url:"imageprocessing_geometry_pose_mainsection.html"}]},
{text:"Point Cloud Processing",url:"dwx_modules.html#dwx_point_processing",children:[
{text:"Point Cloud Processing",url:"pointcloudprocessing_mainsection.html"}]},
{text:"Deep Neural Network (DNN) Framework",url:"dwx_modules.html#dwx_dnn_framework",children:[
{text:"Data Conditioner",url:"dataconditioner_mainsection.html"},
{text:"DNN",url:"dnn_mainsection.html"},
{text:"Clusterer",url:"clusterer_mainsection.html"}]},
{text:"Calibration",url:"dwx_modules.html#dwx_calibration",children:[
{text:"Intrinsic Camera Model",url:"cameramodel_mainsection.html"},
{text:"Self-Calibration",url:"calibration_mainsection.html"},
{text:"Egomotion",url:"egomotion_mainsection.html"}]},
{text:"Communication",url:"dwx_modules.html#dwx_comms",children:[
{text:"Inter-process Communication (IPC)",url:"ipc_mainsection.html"}]},
{text:"Utility",url:"dwx_modules.html#dwx_utility",children:[
{text:"Renderer",url:"renderer_mainsection.html"}]}]},
{text:"Samples",url:"dwx_samples_section.html",children:[
{text:"Introductory Samples",url:"dwx_samples_section.html#dwx_introductory_samples_group"},
{text:"Sensor Abstraction Layer Samples",url:"dwx_samples_section.html#dwx_sensor_abstraction_layer",children:[
{text:"Camera Samples",url:"dwx_samples_section.html#dwx_camera_sensor_samples_group"},
{text:"Other Sensor Samples",url:"dwx_samples_section.html#dwx_other_sensor_samples_group"}]},
{text:"Vehicle And Motion Actuation Samples",url:"dwx_samples_section.html#dwx_vehicle_samples_group"},
{text:"Image Processing Samples",url:"dwx_samples_section.html#dwx_image_processing_samples_group"},
{text:"Point Cloud Processing Samples",url:"dwx_samples_section.html#dwx_pcloud_processing_samples_group"},
{text:"Deep Neural Network (DNN) Framework Samples",url:"dwx_samples_section.html#dwx_dnn_samples_group"},
{text:"Calibration Samples",url:"dwx_samples_section.html#dwx_calibration_samples_group"},
{text:"Communication Samples",url:"dwx_samples_section.html#dwx_communication_samples_group"},
{text:"Utility Samples",url:"dwx_samples_section.html#dwx_utility_samples_group"}]},
{text:"Tools",url:"dwx_tools_section.html",children:[
{text:"Sensor Tools",url:"dwx_tools_section.html#dwx_tools_sensors"},
{text:"Recording Tools",url:"dwx_tools_section.html#dwx_tools_recording",children:[
{text:"General Recording Tools",url:"dwx_recording_tools.html"}]},
{text:"Post-Recording Tools",url:"dwx_tools_section.html#dwx_tools_postrecording"},
{text:"Calibration Tools",url:"dwx_tools_section.html#dwx_tools_calibration",children:[
{text:"Camera Calibration Tools",url:"dwx_camera_calibration_tools.html"}]},
{text:"DNN Framework Tools",url:"dwx_tools_section.html#dwx_tensorRT_optimization"},
{text:"General Tools",url:"dwx_tools_section.html#dwx_tools_general"}]},
{text:"Tutorials",url:"dwx_tutorials.html",children:[
{text:"Basic Tutorials",url:"basic_tutorials.html",children:[
{text:"DriveWorks 101",url:"basic_tutorials.html#dwx_101_tutorials",children:[
{text:"Conventions",url:"dwx_conventions.html"}]},
{text:"Accessing Sensors",url:"basic_tutorials.html#dwx_sensors_tutorials",children:[
{text:"Sensors Overview",url:"basic_tutorials.html#dwx_sensors_overview_tutorials"},
{text:"Camera Usage",url:"basic_tutorials.html#dwx_sensors_camera_tutorials"},
{text:"CANBUS Usage",url:"basic_tutorials.html#dwx_sensors_canbus_tutorials"},
{text:"GPS Usage",url:"basic_tutorials.html#dwx_sensors_gps_tutorials"},
{text:"IMU Usage",url:"basic_tutorials.html#dwx_sensors_imu_tutorials"},
{text:"Lidar Usage",url:"basic_tutorials.html#dwx_sensors_lidar_tutorials"},
{text:"Radar Usage",url:"basic_tutorials.html#dwx_sensors_radar_tutorials"},
{text:"Time Usage",url:"basic_tutorials.html#dwx_sensors_time_tutorials"}]},
{text:"Accessing Rig Configurations",url:"basic_tutorials.html#dwx_rig_tutorials"},
{text:"Utilities",url:"basic_tutorials.html#dwx_utility_tutorials"},
{text:"Static Calibration and Recording Data",url:"basic_tutorials.html#dwx_calib_record_tutorials",children:[
{text:"Recording Sensor Data",url:"dwx_recording_devguide_group.html"}]}]},
{text:"Intermediate Tutorials",url:"intermediate_tutorials.html",children:[
{text:"Sensor Management",url:"intermediate_tutorials.html#dwx_intermediate_tutorials_sensors"},
{text:"Image Processing",url:"intermediate_tutorials.html#dwx_intermediate_tutorials_ip",children:[
{text:"Image Scaling",url:"imagetransformation_usecase1.html"}]},
{text:"Point Cloud Processing",url:"intermediate_tutorials.html#dwx_intermediate_tutorials_pc_processing"},
{text:"Deep Neural Networks (DNN)",url:"intermediate_tutorials.html#dwx_intermediate_tutorials_dnn"},
{text:"Vehicle Actuation",url:"intermediate_tutorials.html#dwx_intermediate_tutorials_vehicle"},
{text:"Self-Calibration",url:"intermediate_tutorials.html#dwx_intermediate_tutorials_selfcalib"}]},
{text:"Advanced Tutorials",url:"advanced_tutorials.html",children:[
{text:"Sensor Customizations",url:"advanced_tutorials.html#dwx_advanced_tutorials_custom"}]}]},
{text:"SDK Porting Guide",url:"dwx_porting_guide.html",children:[
{text:"From SDK 5.8 to SDK 5.10",url:"@ref dwx_porting_guide_5_10"},
{text:"From SDK 5.6 to SDK 5.8",url:"@ref dwx_porting_guide_5_8"},
{text:"From SDK 5.4 to SDK 5.6",url:"dwx_porting_guide_5_6.html"},
{text:"From SDK 5.2 to SDK 5.4",url:"md_doc_portingguide_5_2_to_5_4.html#dwx_porting_guide_5_4"},
{text:"From SDK 5.0 to SDK 5.2",url:"dwx_porting_guide_5_2.html"},
{text:"From SDK 4.0 to SDK 5.0",url:"dwx_porting_guide_5_0.html"},
{text:"From SDK 3.5 to SDK 4.0",url:"dwx_porting_guide_4_0.html"},
{text:"From SDK 3.0 to SDK 3.5",url:"dwx_porting_guide_3_5.html"},
{text:"From SDK 2.2 to SDK 3.0",url:"dwx_porting_guide_3_0.html"},
{text:"From SDK 2.0 to SDK 2.2",url:"dwx_porting_guide_2_2.html"},
{text:"From SDK 1.5 to SDK 2.0",url:"dwx_porting_guide_2_0.html"}]},
{text:"DriveWorks API",url:"modules.html"},
{text:"More",url:"usergroup0.html",children:[
{text:"Open Source and Third-Party Licenses",url:"dwx_open_source_attribution.html"},
{text:"NVIDIA Legal Information",url:"nvidia_legal.html"},
{text:"File List",url:"files.html",children:[
{text:"List",url:"files.html"}]}]}]}
