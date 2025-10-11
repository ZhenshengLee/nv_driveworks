# Nvidia Driveworks 的自驾中间件方案

## 背景

Nvidia销售算力平台和算力边缘设备，并且使用行业垂直整合战略，向行业用户提供整体解决方案。

Driveworks 是 Nvidia 通过DriveAGX边缘算力设备，向主机厂，Tier1等提供的自驾中间件，用于开发SDV自动驾驶汽车。

Driveworks 早期(4.x)提供自驾业务的可复用工具和算法应用库，后期(5.x)则提供了中间件层(包括通信中间件CGF和调度中间件STM), 并提供一套开发API和工具链

本文不讨论Driveworks4.x自驾算法相关的组件库，而是关注Driveworks5.x的新特性：中间件软件，其定位与ROS2核心组件一致。

## DriveOS5.x的自驾中间件方案

自驾中间件方案原先由DriveOS团队负责，自驾中间件方案经历两个阶段

- NvROS: DriveOS5.1版本提供的NvROS，即ROS1和nvmedia和eglstream的集成，见文档[NvROS-Integration-nvmedia-and-eglstream-in-ROS-2021.pdf](robotics-middleware/driveworks-ROS-doc/NvROS-Integration-nvmedia-and-eglstream-in-ROS-2021.pdf)
- OpenDDS: DriveOS5.2版本提供OpenDDS环境，见文档[Nvidia-OpenDDS-in-DriveOS-2021.pdf](robotics-middleware/driveworks-ROS-doc/Nvidia-OpenDDS in DriveOS-2021.pdf)

这两个方案在DriveOS6.x中都被废弃，取而代之的是Driveworks5.x提供的中间件层。

## Driveworks5.x的自驾中间件方案

Driveworks5.x与DriveOS6.x并列，在DriveAGX Orin产品上提供。

Driveworks5.x推出了通信中间件和调度中间件, 在其发布PPT中号称拥有ROS2的全面优势: 见文档[DRIVE-Platform-For-Developers-2302.pdf](drive-agx-orin-doc/0-overview/DRIVE-Platform-For-Developers-2302.pdf)

其中间件组件和文档包括:

- [Compute Graph Framework CGF](driverorks-5.10/doc/nvcgf_html)
- [Scheduling Middleware STM](drive-agx-orin-doc/3-drive-works/Nvidia-STM-Userguide-do6050.pdf)

其开发教程文档见：

- [CGF-presentation-2305.pdf](drive-agx-orin-doc/3-drive-works/CGF-presentation-2305.pdf)
- [building-reliable-av-app-with-cgf-2308.pdf](drive-agx-orin-doc/3-drive-works/building-reliable-av-app-with-cgf-2308.pdf)
- [Performance-Oriented-Scheduling-with-STM-2308.pdf](drive-agx-orin-doc/3-drive-works/Performance-Oriented-Scheduling-with-STM-2308.pdf)

## DriveOS SDK7.x的自驾中间件方案

Driveworks5.x最终合并到了DriveOS团队，和DriveOS一起统称DriveOS SDK，DriveOS-Driveworks7.x在DriveAGX Thor产品上提供。

该方案与上一代方案的最大差异是，移除了Compute Graph Framework CGF，而调度中间件STM继续保留。具体移除的组件见[Driveworks7.0.3-Upgrades and Migration](https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/drive-os-linux-sdk/embedded-software-components/DRIVE_AGX_SoC/DriveWorks/DriveWorks_SDK/migration/index.html)

按照论坛人员的说法：
> [We suggest to use DW APIs directly as CGF framework is removed.](https://forums.developer.nvidia.com/t/drive-os-7-cgf-support-camera-support/345023/6)

关于DriveOS SDK7.x新中间件架构，见文档[nvidia-drive-agx-thor-platform-for-developers-2509.pdf](drive-agx-orin-doc/0-overview/nvidia-drive-agx-thor-platform-for-developers-2509.pdf)

一句话描述：DriveOS SDK7.x移除了CGF，而只保留了STM，而STM的功能和其他业务组件之间没有恰当的集成，而需要开发者自己实现。

## 自驾中间件的开源方案

各个具有自研能力的汽车厂商和Tier1均有自己的自驾中间件方案，但其思想均来自于开源社区的主流方案，本仓库收集了以下方案的部分文档

- [Eclipse eCAL](robotics-middleware/eclipse-ecal-doc)
- [Nvidia ISAAC-ROS](robotics-middleware/isaac-ros-doc)

## 本仓库数据来源

仓库包含文件来自NV 官方deb包

```sh
# driveworks-5.8
driveworks_5.8.82-317146970_amd64.deb
driveworks-samples_5.8.82-317146970_amd64.deb
driveworks-cgf_5.8.82-317146970_amd64.deb
driveworks-cgf-samples_5.8.82-317146970_amd64.deb
driveworks-cgf-doc_5.8.82-317146970_all.deb
driveworks-stm_5.8.82-317146970_amd64.deb
driveworks-stm-samples_5.8.82-317146970_amd64.deb
# driveworks-cgf-ui_5.8.82-317146970_amd64.deb
# driveworks-data_5.8.82-317146970_all.deb
```

```sh
# driveworks-5.10
driveworks_5.10.87-323457480_amd64.deb
driveworks-samples_5.10.87-323457480_amd64.deb
# driveworks-doc_5.10.87-323457480_all.deb
# cgf not found
# stm
driveworks-stm_5.10.87-323457480_amd64.deb
```

### 文件拷贝方法

```sh
# 多线程压缩解压缩
sudo apt install pigz

cd /gw_demo
tar --use-compress-program=pigz -h --exclude=*/data/* -cvpf  driveworks-520.tgz /usr/local/driveworks/*

tar --use-compress-program=pigz -xvpf driveworks-520.tgz
```

### 目录对应关系

`/usr/local/driveworks-5.8` 改成`/driveworks-5.8`

`/driveworks-5.8/targets/x86_64-Linux/include` 改成 `/driveworks-5.8/include`
