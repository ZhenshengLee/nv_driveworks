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
// SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_DWTRACE_GLOBALTRACER_HPP_
#define DW_DWTRACE_GLOBALTRACER_HPP_

#include <dw/trace/Trace.hpp>

namespace dw
{
namespace trace
{

/**
 * Wrapper for DWTrace intended for internal use. See dw/trace/Trace.hpp
 * for function details.
 **/

#define DW_TRACE_CHANNEL dw::trace::TraceChannel::DW
#define DW_TRACE_TAG_CPU(name) DW_TRACE_TAG(#name, "Host")
#define DW_TRACE_TAG_GPU(name) DW_TRACE_TAG(#name, "GPU")
#define DW_TRACE_LEVEL dw::trace::Level::LEVEL_50

#define PROFILE_SECTION(name, stream) DW_TRACE_CUDA_SCOPE(dw::trace::TraceChannel::DW, \
                                                          DW_TRACE_TAG_GPU(name),      \
                                                          stream,                      \
                                                          dw::trace::Level::LEVEL_50);

#define PROFILE_CPU_SECTION(name) DW_TRACE_SCOPE(dw::trace::TraceChannel::DW, \
                                                 DW_TRACE_TAG_CPU(name),      \
                                                 dw::trace::Level::LEVEL_50);

#define TRACE_BEGIN(name) DW_TRACE_BEGIN(dw::trace::TraceChannel::DW, DW_TRACE_TAG_CPU(name), dw::trace::Level::LEVEL_50);
#define TRACE_END(name) DW_TRACE_END(dw::trace::TraceChannel::DW, DW_TRACE_TAG_CPU(name), dw::trace::Level::LEVEL_50);
#define TRACE_ASYNC_BEGIN(name) DW_TRACE_ASYNC_BEGIN(dw::trace::TraceChannel::DW, DW_TRACE_TAG_CPU(name), dw::trace::Level::LEVEL_50);
#define TRACE_ASYNC_END(name) DW_TRACE_ASYNC_END(dw::trace::TraceChannel::DW, DW_TRACE_TAG_CPU(name), dw::trace::Level::LEVEL_50);
#define TRACE_CUDA_BEGIN(name, stream) DW_TRACE_CUDA_BEGIN(dw::trace::TraceChannel::DW, DW_TRACE_TAG_GPU(name), stream, dw::trace::Level::LEVEL_50);
#define TRACE_CUDA_END(name, stream) DW_TRACE_CUDA_END(dw::trace::TraceChannel::DW, DW_TRACE_TAG_GPU(name), stream, dw::trace::Level::LEVEL_50);
#define TRACE_CUDA_BEGIN_ASYNC(name, stream) DW_TRACE_CUDA_BEGIN_ASYNC(dw::trace::TraceChannel::DW, DW_TRACE_TAG_GPU(name), stream, dw::trace::Level::LEVEL_50);
#define TRACE_CUDA_RECORD_ASYNC(name, stream) DW_TRACE_CUDA_RECORD_ASYNC(dw::trace::TraceChannel::DW, DW_TRACE_TAG_GPU(name), stream, dw::trace::Level::LEVEL_50);
#define TRACE_CUDA_COLLECT_ASYNC(name, stream) DW_TRACE_CUDA_COLLECT_ASYNC(dw::trace::TraceChannel::DW, DW_TRACE_TAG_GPU(name), stream, dw::trace::Level::LEVEL_50);
#define TRACE_CUDA_COLLECT_ALL() DW_TRACE_CUDA_COLLECT_ALL()
}
}

#endif
