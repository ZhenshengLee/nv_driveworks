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

#ifndef DW_CORE_LOGGER_STREAMLOGGER_HPP_
#define DW_CORE_LOGGER_STREAMLOGGER_HPP_

#include "Logger.hpp"
#include <dwshared/dwfoundation/dw/core/base/ExceptionWithStackTrace.hpp>

#include <mutex>

namespace dw
{
namespace core
{
namespace log
{

/**
 * @brief Log output to the console stream
 */
class StreamLogger final : public Logger
{
public:
    explicit StreamLogger(std::ostream& stream);

    StreamLogger(StreamLogger const&) = default;
    StreamLogger(StreamLogger&&)      = delete;
    StreamLogger& operator=(StreamLogger const&) = delete;
    StreamLogger& operator=(StreamLogger&&) = delete;

    ~StreamLogger() final
    {
        StreamLogger::flush();
    }

    void logMessage(LoggerMessageInfo const& info, char8_t const* const message) final;
    void flush() final;

    void reset() final
    {
        flush();
    }

    //!@see std::basic_ostream::tellp
    size_t tellp();

    std::unique_ptr<Logger> clone() final;

private:
    static void streamIsoTimestamp(std::ostream& os);
    static void streamSourceCodeLocation(std::ostream& os, const SourceCodeLocation& loc);
    static void streamTag(std::ostream& os, StringBuffer<Logger::TAG_SIZE> const& tag);
    static void streamThreadId(std::ostream& os, StringBuffer<Logger::TID_SIZE> const& tid);
    static void streamVerbosity(std::ostream& os, Logger::Verbosity const level);

    std::ostream& m_stream;
    // Mutex for serializing access to m_stream. Not to be used for other purposes.
    std::shared_ptr<std::mutex> m_streamMutex;
};

} // namespace log
} // namespace core
} // namespace dw

#endif // DW_CORE_LOGGER_STREAMLOGGER_HPP_
