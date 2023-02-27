# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

@page dwx_sensor_enum_sample Sensor Enumeration Sample
@tableofcontents

@section dwx_sensor_enum_sample_description Description

The Sensor Enumeration sample is a minimal sample using the sensor abstraction
layer to enumerate all available sensors drivers in SDK.

@section dwx_sensor_enum_sample_running Running the Sample

The command line for the Sensor Enumeration sample, sample_sensors_info, is:

    ./sample_sensors_info

@section dwx_sensor_enum_sample_output Output

On execution, the sample outputs a list of sensors in the following format:

        Platform: OS_LINUX - CURRENT:
           Sensor [0] : time.virtual ? file=/path/to/file.bin[,create_seek]
           Sensor [1] : can.socket ? device=can0[,fifo-size=1024]
           Sensor [2] : can.virtual ? file=/path/to/file.can[,create_seek,default_timeout_us,time-offset=0]
           Sensor [3] : camera.virtual ? video/file=filepath.{h264,raw,lraw}[,timestamp=file.txt][,create_seek][,time-offset=0]
           Sensor [4] : camera.nvidia-ip ? host=<ip_addr>, port=<TCP port number>
           Sensor [5] : camera.usb ? device=0[,mode={0,a,b}]
           Sensor [6] : gps.uart ? device=/dev/ttyXXX[,baud={1200,2400,4800,9600,19200,38400,57600,115200}[,format=nmea0183][,fifo-size=1024]]
           Sensor [7] : gps.virtual ? file=filepath.bin[xsens-raw-gps=true,create_seek,default_timeout_us,time-smoothing=false,time-offset=0]
           Sensor [8] : gps.xsens ? device=0[,frequency=100,xsens-raw-gps=true,time-smoothing=false,fifo-size=1024,baudrate=115200,stop-bits=2]
           Sensor [9] : gps.novatel ? [fifo-size=1024]
           Sensor [10] : gps.dataspeed ? can-proto=can.socket,can-params=device=canX[,fifo-size=1024]
           Sensor [11] : imu.uart ? device=/dev/ttyXXX[,baud={1200,2400,4800,9600,19200,38400,57600,115200}[,format=xsens_nmea][,fifo-size=1024]]
           Sensor [12] : imu.xsens ? device=0[,frequency=100,time-smoothing=true,fifo-size=1024,baudrate=115200,stop-bits=2]
           Sensor [13] : imu.virtual ? file=filepath.bin[,create_seek,default_timeout_us,time-smoothing=true][,fifo-size=1024][,time-offset=0]
           Sensor [14] : imu.novatel ? [fifo-size=1024]
           Sensor [15] : imu.dataspeed ? can-proto=can.socket,can-params=device=canX[,fifo-size=1024]
           Sensor [16] : imu.bosch ? can-proto=can.socket,can-params=device=canX[,fifo-size=1024]
           Sensor [17] : imu.continental ? can-proto=can.socket,can-params=device=canX[,fifo-size=1024]
           Sensor [18] : lidar.virtual ? file=filepath.bin[,create_seek,default_timeout_us,decoder=filepath.so,time-smoothing=false,decoding=true,time-offset=0]
           Sensor [19] : lidar.socket ? ip=X.X.X.X,port=XXXX,device={VELO_VLP16, VELO_VLP16HR, VELO_HDL32E, VELO_VLP32C, VELO_HDL64E, VELO_VLS128, OUSTER_OS1, OUSTER_OS2_128, LUMINAR_H, CUSTOM},scan-frequency=XX.X[,protocol=xxx,multicast-ip=X.X.X.X,decoder=filepath.so,time-smoothing=false]
           Sensor [20] : radar.virtual ? file=filepath.bin,time-smoothing=true[,decoder=filepath.so,create_seek,default_timeout_us,decoding=true,time-offset=0]
           Sensor [21] : radar.socket ? ip=X.X.X.X,port=XXXX,device={DELPHI_ESR2_5, CONTINENTAL_ARS430, CONTINENTAL_ARS430_RDI, CUSTOM},multicast-ip=X.X.X.X],time-smoothing=true,isInverted=false,slave=false[,decoder=filepath.so,protocol=xxx]
           Sensor [22] : radar.can ? can-driver=can.xxx,can-bus=xxx,can-base-id=0x460,device={CONTINENTAL_ARS430_CAN}[,aurix-can-ip=X.X.X.X.X],[aurix-can-aport=XXXX][,aurix-can-bport=XXXX][,virtual-file=filepath.bin][,time-smoothing=false]

        Platform: OS_DRIVE_V5L:
           Sensor [0] : time.virtual ? file=/path/to/file.bin[,create_seek]
           Sensor [1] : can.socket ? device=can0[,fifo-size=1024]
           Sensor [2] : can.aurix ? ip=10.0.0.1,bus={a,b,c,d}[,aport=50000,bport=60395][config-file=/path/to/EasyCanConfigFile.conf][,fifo-size=1024]
           Sensor [3] : can.virtual ? file=/path/to/file.can[,create_seek,default_timeout_us,time-offset=0]
           Sensor [4] : camera.gmsl ? camera-type={ar0231-rccb-bae-sf3324, ar0231-rccb-bae-sf3325, ar0144-cccc-none-gazet1},output-format={yuv+raw+data}[,slave={0,1}][,fifo-size={3..20}][,custom-board=0]camera-group={a,b,c,d},camera-count={1,2,3,4},[,camera-mask={0001|0010|0011|..|1111}][,warn-per-frame={0,1}]
           Sensor [5] : camera.client ? host=<ip_addr>,port=<tcp_port_num>,siblingIndex=<sibling-num>,mode={fifo,mailbox},output-format={yuv+raw+data},camera-group={a,b,c},fifo-size={3..16}
           Sensor [6] : camera.virtual ? video/file=filepath.{h264,raw,lraw}[,timestamp=file.txt][,create_seek][,time-offset=0]
           Sensor [7] : camera.usb ? device=0[,mode={0,a,b}]
           Sensor [8] : camera.nvidia-ip ? host=[ip_addr], port=[TCP port number]
           Sensor [9] : gps.uart ? device=/dev/ttyXXX[,baud={1200,2400,4800,9600,19200,38400,57600,115200}[,format=nmea0183]][,fifo-size=1024]
           Sensor [10] : gps.virtual ? file=filepath.bin[,xsens-raw-gps=true,create_seek,default_timeout_us,time-smoothing=false,time-offset=0]
           Sensor [11] : gps.xsens ? device=0[,frequency=100,xsens-raw-gps=true,time-smoothing=false,fifo-size=1024,baudrate=115200,stop-bits=2]
           Sensor [12] : gps.novatel ? [fifo-size=1024]
           Sensor [13] : gps.dataspeed ? can-proto=can.socket,can-params=device=canX[,fifo-size=1024]
           Sensor [14] : imu.uart ? device=/dev/ttyXXX[,baud={1200,2400,4800,9600,19200,38400,57600,115200}[,format=xsens_nmea]][,fifo-size=1024]
           Sensor [15] : imu.xsens ? device=0[,frequency=100,time-smoothing=false,fifo-size=1024,baudrate=115200,stop-bits=2]
           Sensor [16] : imu.virtual ? file=filepath.bin[,create_seek,default_timeout_us,time-smoothing=false,time-offset=0]
           Sensor [17] : imu.novatel ? [fifo-size=1024]
           Sensor [18] : imu.dataspeed ? can-proto=can.socket,can-params=device=canX[,fifo-size=1024]
           Sensor [19] : imu.bosch ? can-proto=can.socket,can-params=device=canX[,fifo-size=1024]
           Sensor [20] : imu.continental ? can-proto=can.socket,can-params=device=canX[,fifo-size=1024]
           Sensor [21] : lidar.virtual ? file=filepath.bin[,create_seek,default_timeout_us,time-smoothing=false,decoding=true,time-offset=0]
           Sensor [22] : lidar.socket ? ip=X.X.X.X,port=XXXX,device={VELO_VLP16, VELO_VLP16HR, VELO_HDL32E, VELO_VLP32C, VELO_HDL64E, VELO_VLS128, OUSTER_OS1, OUSTER_OS2_128, LUMINAR_H, CUSTOM},scan-frequency=XX.X[,protocol=xxx,multicast-ip=X.X.X.X,decoder=filepath.so,time-smoothing=false]
           Sensor [23] : radar.virtual ? file=filepath.bin,time-smoothing=true[,decoder=filepath.so,create_seek,default_timeout_us,decoding=true,time-offset=0]
           Sensor [24] : radar.socket ? ip=X.X.X.X,port=XXXX,device={DELPHI_ESR2_5, CONTINENTAL_ARS430, CONTINENTAL_ARS430_RDI, CUSTOM},multicast-ip=X.X.X.X,time-smoothing=true,isInverted=false,slave=false[,decoder=filepath.so,protocol=xxx]
           Sensor [25] : radar.can ? can-driver=can.xxx,can-bus=xxx,can-base-id=0x460,device={CONTINENTAL_ARS430_CAN}[,aurix-can-ip=X.X.X.X.X],[aurix-can-aport=XXXX][,aurix-can-bport=XXXX][,virtual-file=filepath.bin]

        Platform: OS_QNX:
           Sensor [0] : time.virtual ? file=/path/to/file.bin[,create_seek]
           Sensor [1] : can.socket ? device=can0
           Sensor [2] : can.aurix ? ip=10.0.0.1,bus={a,b,c,d}[,aport=50000,bport=60395][config-file=/path/to/EasyCanConfigFile.conf][,fifo-size=1024]
           Sensor [3] : can.virtual ? file=/path/to/file.can[,create_seek][,time-offset=0]
           Sensor [4] : camera.gmsl ? camera-group={a,b,c,d},camera-count={1,2,3,4},camera-type={ar0231-rccb-bae-sf3324, ar0231-rccb-bae-sf3325},output-format={yuv+raw+data}[,slave={0,1}][,fifo-size={3..20}][,custom-board=0][,camera-mask={0001|0010|0011|..|1111}][,warn-per-frame={0,1}]
           Sensor [5] : camera.virtual ? video/file=filepath.{h264,raw}[,timestamp=file.txt][,create_seek][,time-offset=0]
           Sensor [6] : gps.uart ? device=/dev/ttyXXX[,baud={1200,2400,4800,9600,19200,38400,57600,115200}[,format=nmea0183]][,fifo-size=1024]
           Sensor [7] : gps.virtual ? file=filepath.bin[,xsens-raw-gps=true,create_seek,default_timeout_us,time-smoothing=false,time-offset=0]
           Sensor [8] : gps.xsens ? device=0[,frequency=100,xsens-raw-gps=true,time-smoothing=false,fifo-size=1024,baudrate=115200,stop-bits=2]
           Sensor [9] : gps.dataspeed ? can-proto=can.socket/aurix,can-params=device=canX[,fifo-size=1024]
           Sensor [10] : imu.uart ? device=/dev/ttyXXX[,baud={1200,2400,4800,9600,19200,38400,57600,115200}[,format=xsens_nmea]][,fifo-size=1024]
           Sensor [11] : imu.virtual ? file=filepath.bin[,create_seek,default_timeout_us,time-smoothing=false,time-offset=0]
           Sensor [12] : imu.xsens ? device=0[,frequency=100,time-smoothing=false,fifo-size=1024,baudrate=115200,stop-bits=2]
           Sensor [13] : imu.dataspeed ? can-proto=can.socket/aurix,can-params=device=canX[,fifo-size=1024]
           Sensor [14] : imu.bosch ? can-proto=can.socket/aurix,can-params=device=canX[,fifo-size=1024]
           Sensor [15] : imu.continental ? can-proto=can.socket/aurix,can-params=device=canX[,fifo-size=1024]
           Sensor [16] : lidar.virtual ? file=filepath.bin[,create_seek,default_timeout_us,time-smoothing=false,time-offset=0]
           Sensor [17] : lidar.socket ? ip=X.X.X.X,port=XXXX,device={VELO_VLP16, VELO_VLP16HR, VELO_HDL32E, VELO_VLP32C, VELO_HDL64E, VELO_VLS128, OUSTER_OS1, OUSTER_OS2_128, LUMINAR_H, CUSTOM},scan-frequency=XX.X[,protocol=xxx,multicast-ip=X.X.X.X,decoder=filepath.so,time-smoothing=false]
           Sensor [18] : radar.virtual ? file=filepath.bin,time-smoothing=true[,decoder=filepath.so,create_seek,default_timeout_us,decoding=false,time-offset=0]
           Sensor [19] : radar.socket ? ip=X.X.X.X,port=XXXX,device={DELPHI_ESR2_5, CONTINENTAL_ARS430, CONTINENTAL_ARS430_RDI, CUSTOM},multicast-ip=X.X.X.X],time-smoothing=true,isInverted=false[,decoder=filepath.so,protocol=xxx]
           Sensor [20] : radar.can ? can-driver=can.xxx,can-bus=xxx,can-base-id=0x460,device={CONTINENTAL_ARS430_CAN}[,aurix-can-ip=X.X.X.X.X],[aurix-can-aport=XXXX][,aurix-can-bport=XXXX][,virtual-file=filepath.bin]

The list of available sensors is grouped by the underlying platform
(Linux or NVIDIA DRIVE<sup>&trade;</sup> AGX) and on the sensor drivers currently
available/implemented in NVIDIA<sup>&reg;</sup> DriveWorks. A list indicates the name of
the sensor and the underlying protocol, as well as a set of string-based key-value
pairs that you can pass to a sensor as additional arguments.

In the example above, there is a sensor `camera.gmsl` available that expects as
a parameter the camera-group, which can be any value from the set {a,b,c,d}, a
number of cameras available at this port (i.e., 1, 2, 3 or 4), and the camera
type (i.e., `ar0231-rccb-bae-sf3324` or `ar0231-rccb-bae-sf3325`).

@section dwx_sensor_enum_sample_more Additional Information

For more details on using custom sensors, see @ref sensorplugins_mainsection.<br/>
For more details, see @ref sensors_usecase2.
