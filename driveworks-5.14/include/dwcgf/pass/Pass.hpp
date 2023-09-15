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
// SPDX-FileCopyrightText: Copyright (c) 2018-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <dwcgf/Exception.hpp>
#include <dwcgf/channel/Channel.hpp>
#include <dwshared/dwfoundation/dw/core/container/StringView.hpp>
#include <dwshared/dwfoundation/dw/core/container/HashContainer.hpp>

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
    // coverity[autosar_cpp14_a0_1_1_violation] FP: nvbugs/2813925
    // coverity[autosar_cpp14_a5_1_1_violation] FP: nvbugs/4020293
    // coverity[autosar_cpp14_m0_1_4_violation] FP: nvbugs/2813925
    static constexpr const size_t MAX_NAME_LEN{256U};

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

    /// Get the name of the pass.
    const dw::core::StringView& getName() const;

    /// Run the pass.
    virtual dwStatus run() = 0;

    /// Set the runnable id.
    virtual void setRunnableId(dw::core::StringView const& runnableId) = 0;
    /// Get the runnable id.
    virtual dw::core::FixedString<MAX_NAME_LEN> const& getRunnableId(bool isSubmitPass) const = 0;
    /// Get the return status code to error map.
    dw::core::HeapHashMap<dwStatus, uint32_t> const& getPassReturnErrorMap();

    /// Get the node this pass belongs to.
    virtual Node& getNode() const = 0;

    /// The processor type this pass runs on.
    dwProcessorType m_processor;

    /// The cuda stream to use in case the processor type is GPU.
    cudaStream_t m_cudaStream;

    /// Keeps track of the 1st pass executed.
    /// WAR Added for enabling pipelining with schedule switching.
    /// WAR will be removed after STM resets epoch iteration count after schedule switch
    bool m_isFirstIteration;

protected:
    /// Constructor.
    Pass(const dw::core::StringView& name, dwProcessorType const processor,
         std::initializer_list<std::pair<dwStatus, uint32_t>> const& returnMapping = {});

private:
    /// The unique name within its node.
    dw::core::StringView m_name;

    /// Optional mapping of return codes to error codes
    dw::core::HeapHashMap<dwStatus, uint32_t> m_passReturnErrorMapping;
};

/// PassImpl contains the function to invoke on run().
template <typename PassFunctionT>
class PassImpl : public Pass
{
    static_assert(std::is_convertible<PassFunctionT, std::function<dwStatus()>>::value, "PassFunctionT must be callable without arguments and return dwStatus");

public:
    /// Constructor with a function running on the given processor type.
    PassImpl(Node& node,
             const dw::core::StringView& name,
             PassFunctionT const passFunc,
             dwProcessorType const processor,
             std::initializer_list<std::pair<dwStatus, uint32_t>> const& returnMapping = {})
        : Pass(name, processor, returnMapping)
        , m_node(node)
        , m_functionInt(passFunc)
    {
    }

    /// @see Pass::run()
    dwStatus run() final
    {
        // TODO(DRIV-7184) - return code of the function shall not be ignored
        return ExceptionGuard::guard([&] {
            m_functionInt();
        },
                                     dw::core::Logger::Verbosity::ERROR);
    }

    /// @see Pass::setRunnableId()
    void setRunnableId(dw::core::StringView const& runnableId) final
    {
        // coverity[autosar_cpp14_a5_1_1_violation]
        if (runnableId.size() >= MAX_NAME_LEN - 10U - 1U)
        {
            throw ExceptionWithStatus(DW_BUFFER_FULL, "setRunnableId() runnable id exceeds capacity: ", runnableId);
        }
        m_runnableId       = dw::core::FixedString<MAX_NAME_LEN>(runnableId.data(), runnableId.size());
        m_runnableIdSubmit = dw::core::FixedString<MAX_NAME_LEN>(runnableId.data(), runnableId.size());
        // coverity[autosar_cpp14_a5_1_1_violation]
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
