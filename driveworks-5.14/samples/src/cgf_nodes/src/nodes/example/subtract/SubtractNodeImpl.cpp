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
// SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "SubtractNodeImpl.hpp"
#include <dw/core/logger/Logger.hpp>

#include <nvtx3/nvToolsExt.h>

namespace dw
{
namespace framework
{

constexpr char SubtractNodeImpl::LOG_TAG[];

SubtractNodeImpl::SubtractNodeImpl(const SubtractNodeParams& params, const dwContextHandle_t ctx)
    : m_params(params)
    , m_ctx(ctx)
{
    nvtxRangePush(__PRETTY_FUNCTION__);
    NODE_INIT_INPUT_PORT("VALUE_0"_sv);
    NODE_INIT_INPUT_PORT("VALUE_1"_sv);

    // Init passes
    NODE_REGISTER_PASS("PROCESS"_sv, [this]() {
        return process();
    });

    DW_LOGD << "SubtractNodeImpl: created" << Logger::State::endl;
    nvtxRangePop();
}

///////////////////////////////////////////////////////////////////////////////////////
SubtractNodeImpl::~SubtractNodeImpl()
{
    DW_LOGD << "SubtractNodeImpl: destructed" << Logger::State::endl;
}

dwStatus SubtractNodeImpl::setState(const char* state)
{
    static_cast<void>(state);
    std::string _state(state);
    DW_LOGD << "SubtractNode change state: " << _state
            << Logger::State::endl;
            
    return DW_SUCCESS;
}

dwStatus SubtractNodeImpl::reset()
{
    return Base::reset();
}

///////////////////////////////////////////////////////////////////////////////////////
dwStatus SubtractNodeImpl::process()
{
    nvtxRangePush(__PRETTY_FUNCTION__);
    auto& inPort0 = NODE_GET_INPUT_PORT("VALUE_0"_sv);
    auto& inPort1 = NODE_GET_INPUT_PORT("VALUE_1"_sv);

    // underrun
    usleep(18000);
    // overrun
    // usleep(58000);

    if (inPort0.isBufferAvailable() && inPort1.isBufferAvailable())
    {
        auto inputValue0 = *inPort0.getBuffer();
        auto inputValue1 = *inPort1.getBuffer();
        DW_LOGD << "[Epoch " << m_epochCount << "]"
                << " Received " << inputValue0 << " from input VALUE_0"
                << ", received " << inputValue1 << " from input VALUE_1."
                << " Subtract together: " << (inputValue0 - inputValue1) << "!" << Logger::State::endl;
    }
    else
    {
        DW_LOGD << "[Epoch " << m_epochCount << "] inPort.buffer not available!" << Logger::State::endl;
    }

    std::string mark_str = "[" + std::to_string(m_epochCount) + "]";
    nvtxMark(mark_str.c_str());

    ++m_epochCount;
    nvtxRangePop();
    return DW_SUCCESS;
}

} // namespace framework
} // namespace dw
