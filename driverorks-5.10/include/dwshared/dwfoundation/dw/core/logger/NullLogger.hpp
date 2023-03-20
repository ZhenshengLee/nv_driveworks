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
// SPDX-FileCopyrightText: Copyright (c) 2015-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_CORE_LOGGER_NULLLOGGER_HPP_
#define DW_CORE_LOGGER_NULLLOGGER_HPP_

#include "Logger.hpp"

namespace dw
{
namespace core
{
namespace log
{

/**
 * @brief Ignore any log output, i.e. log is sent to /dev/null
 */
class NullLogger final : public Logger
{
public:
    ~NullLogger() final           = default;
    NullLogger(NullLogger const&) = default;
    NullLogger(NullLogger&&)      = delete;
    NullLogger& operator=(NullLogger const&) = delete;
    NullLogger& operator=(NullLogger&&) = delete;
    NullLogger()
        : Logger(LoggerType::NullLogger) {}

    /**
     * Returns a pointer to the null logger
     *
     * Constructs a NullLogger when called for the first time
     */
    using Logger::get;
    static NullLogger& get();

    void logMessage(const Logger::LoggerMessageInfo&, char8_t const* const) final
    {
    }

    std::unique_ptr<Logger> clone() final;

private:
    static std::unique_ptr<NullLogger> m_instance;
    static std::once_flag m_initFlag;
};

} // namespace log
} // namespace core
} // namespace dw

#endif // DW_CORE_LOGGER_NULLLOGGER_HPP_