/////////////////////////////////////////////////////////////////////////////////////////
//
// Notice
// ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS" NVIDIA MAKES
// NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//
// NVIDIA CORPORATION & AFFILIATES assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of NVIDIA CORPORATION & AFFILIATES. No third party distribution is allowed unless
// expressly authorized by NVIDIA. Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// NVIDIA CORPORATION & AFFILIATES products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// NVIDIA CORPORATION & AFFILIATES.
//
// SPDX-FileCopyrightText: Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

#include <string>

#include <framework/SamplesDataPath.hpp>
#include <framework/ProgramArguments.hpp>

#include "DNNTensorSample.hpp"

int main(int argc, const char** argv)
{
    // -------------------
    // define all arguments used by the application
    ProgramArguments args(argc, argv,
                          {
#ifdef VIBRANTE
                              ProgramArguments::Option_t("camera-type", "ar0231-rccb-bae-sf3324", "camera gmsl type (see sample_sensors_info for all available camera types on this platform)"),
                              ProgramArguments::Option_t("camera-group", "a", "input port"),
                              ProgramArguments::Option_t("camera-index", "0", "camera index within the camera-group 0-3"),
                              ProgramArguments::Option_t("input-type", "video", "input type either video or camera"),
                              ProgramArguments::Option_t("cudla", "0", "run inference on cudla"),
                              ProgramArguments::Option_t("dla-engine", "0", "dla engine number to run on if --cudla=1"),
#endif
                              ProgramArguments::Option_t("video", (dw_samples::SamplesDataPath::get() + "/samples/sfm/triangulation/video_0.h264").c_str(), "path to video"),
                              ProgramArguments::Option_t("tensorRT_model", "", (std::string("path to TensorRT model file. By default: ") + dw_samples::SamplesDataPath::get() + "/samples/detector/<gpu-architecture>/tensorRT_model.bin").c_str())},
                          "DNN Tensor sample which detects and tracks cars.");

    DNNTensorSample app(args);
    app.initializeWindow("DNN Tensor Sample", 1280, 800, args.enabled("offscreen"));

    if (!args.enabled("offscreen"))
        app.setProcessRate(30);

    return app.run();
}
