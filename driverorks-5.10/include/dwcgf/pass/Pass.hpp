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
// SPDX-FileCopyrightText: Copyright (c) 2018-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_FRAMEWORK_PASS_HPP_
#define DW_FRAMEWORK_PASS_HPP_

#include <dwcgf/Types.hpp>
#include <dwcgf/Exception.hpp>
#include <dwcgf/channel/Channel.hpp>
#include <dw/core/container/StringView.hpp>

namespace dw
{
namespace framework
{

/// Forward declaration
class Node;

/// Pass is a runnable describes the metadata of a pass
class Pass
{
public:
    /// The maximum length of the runnable id.
    // coverity[autosar_cpp14_a0_1_1_violation]
    // coverity[autosar_cpp14_m0_1_4_violation]
    static constexpr size_t MAX_NAME_LEN{256U};

    /// Destructor
    virtual ~Pass() = default;
    /// Copy constructor.
    Pass(Pass const&) = delete;
    /// Move constructor.
    Pass(Pass&&) = delete;
    /// Copy assignment operator.
    Pass& operator=(Pass const&) & = delete;
    /// Move assignment operator.
    Pass& operator=(Pass&&) & = delete;

    /// Run the pass.
    virtual dwStatus run() = 0;

    /// Set the runnable id.
    virtual void setRunnableId(dw::core::StringView const& runnableId) = 0;
    /// Get the runnable id.
    virtual dw::core::FixedString<MAX_NAME_LEN> const& getRunnableId(bool isSubmitPass) const = 0;

    /// Get the node this pass belongs to.
    virtual Node& getNode() const = 0;

    /// The processor type this pass runs on.
    dwProcessorType m_processor;
    /// The process type this pass runs with.
    dwProcessType m_processType;

    /// @deprecated
    dwTime_t m_minTime;
    /// @deprecated
    dwTime_t m_avgTime;
    /// @deprecated
    dwTime_t m_maxTime;

    /// The cuda stream to use in case the processor type is GPU.
    cudaStream_t m_cudaStream;
    /// The dla engine to run on in case the processor type is GPU.
    NvMediaDla* m_dlaEngine;

    /// Keeps track of the 1st pass executed.
    /// WAR Added for enabling pipelining with schedule switching.
    /// WAR will be removed after STM resets epoch iteration count after schedule switch
    bool m_isFirstIteration;

protected:
    /// Constructor.
    // TODO(dwplc): FP -- This constructor is called by the PassImpl constructor below
    // coverity[autosar_cpp14_m0_1_10_violation]
    Pass(dwProcessorType const processor,
         dwProcessType const processType,
         dwTime_t const minTime, dwTime_t const avgTime, dwTime_t const maxTime) noexcept;
};

/// PassImpl contains the function to invoke on run().
template <typename PassFunctionT>
class PassImpl : public Pass
{
    static_assert(std::is_convertible<PassFunctionT, std::function<dwStatus()>>::value, "PassFunctionT must be callable without arguments and return dwStatus");

public:
    /// Constructor with a function running on a CPU.
    PassImpl(Node& node,
             PassFunctionT const passFunc,
             dwProcessorType const processor,
             dwProcessType const processType,
             dwTime_t const minTime, dwTime_t const avgTime, dwTime_t const maxTime)
        : Pass(processor, processType, minTime, avgTime, maxTime)
        , m_node(node)
        , m_functionInt(passFunc)
    {
    }

    /// Constructor with a function running on a GPU.
    PassImpl(Node& node,
             PassFunctionT const passFunc,
             dwProcessorType const processor,
             dwProcessType const processType,
             dwTime_t const minTime, dwTime_t const avgTime, dwTime_t const maxTime,
             cudaStream_t const cudaStream)
        : Pass(processor, processType, minTime, avgTime, maxTime)
        , m_node(node)
        , m_functionInt(passFunc)
    {
        if (processor == DW_PROCESSOR_TYPE_GPU)
        {
            m_cudaStream = cudaStream;
        }
        else
        {
            throw ExceptionWithStatus(DW_NOT_SUPPORTED, "PassImpl: Only GPU passes can use a cuda stream");
        }
    }

    /// Constructor with a function running on a DLA.
    PassImpl(Node& node,
             PassFunctionT const passFunc,
             dwProcessorType const processor,
             dwProcessType const processType,
             dwTime_t const minTime, dwTime_t const avgTime, dwTime_t const maxTime,
             NvMediaDla* const dlaEngine)
        : Pass(processor, processType, minTime, avgTime, maxTime)
        , m_node(node)
        , m_functionInt(passFunc)
    {
        if (processor == DW_PROCESSOR_TYPE_DLA_0 || processor == DW_PROCESSOR_TYPE_DLA_1)
        {
            m_dlaEngine = dlaEngine;
        }
        else
        {
            throw ExceptionWithStatus(DW_NOT_SUPPORTED, "PassImpl: Only DLA passes can use a DLA handle");
        }
    }

    /// @see Pass::run()
    dwStatus run() final
    {
        // TODO(DRIV-7184) - return code of the function shall not be ignored
        return Exception::guard([&] {
            m_functionInt();
        },
                                dw::core::Logger::Verbosity::WARN);
    }

    /// @see Pass::setRunnableId()
    void setRunnableId(dw::core::StringView const& runnableId) final
    {
        if (runnableId.size() >= 128 - 10 - 1)
        {
            throw ExceptionWithStatus(DW_BUFFER_FULL, "setRunnableId() runnable id exceeds capacity: ", runnableId);
        }
        m_runnableId       = dw::core::FixedString<128>(runnableId.data(), runnableId.size());
        m_runnableIdSubmit = dw::core::FixedString<128>(runnableId.data(), runnableId.size());
        m_runnableIdSubmit += "_submittee";
    }

    /// @see Pass::getRunnableId()
    dw::core::FixedString<MAX_NAME_LEN> const& getRunnableId(bool isSubmitPass) const final
    {
        if (isSubmitPass)
        {
            return m_runnableIdSubmit;
        }
        return m_runnableId;
    }

    /// @see Pass::getNode()
    Node& getNode() const final
    {
        return m_node;
    }

private:
    /// The node this pass belongs to.
    Node& m_node;
    /// The function to invoke on run().
    PassFunctionT m_functionInt;
    /// The runnable id.
    dw::core::FixedString<MAX_NAME_LEN> m_runnableId;
    /// The runnable submitter id.
    dw::core::FixedString<MAX_NAME_LEN> m_runnableIdSubmit;
};

} // namespace framework
} // namespace dw

#endif // DW_FRAMEWORK_PASS_HPP_
