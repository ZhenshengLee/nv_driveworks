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
// SPDX-FileCopyrightText: Copyright (c) 2015-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_CORE_LOGGER_HPP_
#define DW_CORE_LOGGER_HPP_

#include <dwshared/dwfoundation/dw/core/logger/NullLogger.hpp>

/////////////////////////////////////////////////////////////
// Logging macros - deprecated
// clang-format off
//TODO(danielh): Check if this is ok with misra
namespace  // to be removed, because logging macros using it are deprecated
{
// TODO(dwplc): FP because useX() is used everytime logging macro is instantiated
//              The logging macros using it are deprecated, hence this function will be removed.
// coverity[autosar_cpp14_a0_1_3_violation]
constexpr char8_t const* useX(void const* x = nullptr)
{
    static_cast<void>(x);
    return "";
}
}

#define LOGSTREAM_VERBOSE(x) dw::core::Logger::get().log(dw::core::Logger::LoggerMessageInfo(dw::core::Logger::Verbosity::VERBOSE, \
                                                         /* coverity[autosar_cpp14_a5_1_1_violation] FP: nvbugs/2980306 */         \
                                                         "NO_TAG",                                                                 \
                                                         dw::core::SourceCodeLocation(__FILE__, __LINE__),                         \
                                                         useX(x)))

#define LOGSTREAM_DEBUG(x)   dw::core::Logger::get().log(dw::core::Logger::LoggerMessageInfo(dw::core::Logger::Verbosity::DEBUG,   \
                                                         /* coverity[autosar_cpp14_a5_1_1_violation] FP: nvbugs/2980306 */         \
                                                         "NO_TAG",                                                                 \
                                                         dw::core::SourceCodeLocation(__FILE__, __LINE__),                         \
                                                         useX(x)))

#define LOGSTREAM_INFO(x)    dw::core::Logger::get().log(dw::core::Logger::LoggerMessageInfo(dw::core::Logger::Verbosity::INFO,    \
                                                         /* coverity[autosar_cpp14_a5_1_1_violation] FP: nvbugs/2980306 */         \
                                                         "NO_TAG",                                                                 \
                                                         dw::core::SourceCodeLocation(__FILE__, __LINE__),                         \
                                                         useX(x)))

#define LOGSTREAM_WARN(x)    dw::core::Logger::get().log(dw::core::Logger::LoggerMessageInfo(dw::core::Logger::Verbosity::WARN,    \
                                                         /* coverity[autosar_cpp14_a5_1_1_violation] FP: nvbugs/2980306 */         \
                                                         "NO_TAG",                                                                 \
                                                         dw::core::SourceCodeLocation(__FILE__, __LINE__),                         \
                                                         useX(x)))

#define LOGSTREAM_ERROR(x)   dw::core::Logger::get().log(dw::core::Logger::LoggerMessageInfo(dw::core::Logger::Verbosity::ERROR,   \
                                                         /* coverity[autosar_cpp14_a5_1_1_violation] FP: nvbugs/2980306 */         \
                                                         "NO_TAG",                                                                 \
                                                         dw::core::SourceCodeLocation(__FILE__, __LINE__),                         \
                                                         useX(x)))

#define LOGSTREAM_FATAL(x)   dw::core::Logger::get().log(dw::core::Logger::LoggerMessageInfo(dw::core::Logger::Verbosity::FATAL,   \
                                                         /* coverity[autosar_cpp14_a5_1_1_violation] FP: nvbugs/2980306 */         \
                                                         "NO_TAG",                                                                 \
                                                         dw::core::SourceCodeLocation(__FILE__, __LINE__),                         \
                                                         useX(x)))

// extended logging macros
#define DW_LOG(verbosity) dw::core::Logger::get().log(dw::core::Logger::LoggerMessageInfo((verbosity),                             \
                                                      LOG_TAG,                                                                     \
                                                      dw::core::SourceCodeLocation(__FILE__, __LINE__),                            \
                                                      /* coverity[autosar_cpp14_a5_1_1_violation] FP: nvbugs/2980306 */            \
                                                      ""))
#define DW_LOGV dw::core::Logger::get().log(dw::core::Logger::LoggerMessageInfo((dw::core::Logger::Verbosity::VERBOSE),            \
                                                      LOG_TAG,                                                                     \
                                                      dw::core::SourceCodeLocation(__FILE__, __LINE__),                            \
                                                      /* coverity[autosar_cpp14_a5_1_1_violation] FP: nvbugs/2980306 */            \
                                                      ""))
#define DW_LOGD dw::core::Logger::get().log(dw::core::Logger::LoggerMessageInfo((dw::core::Logger::Verbosity::DEBUG),              \
                                                      LOG_TAG,                                                                     \
                                                      dw::core::SourceCodeLocation(__FILE__, __LINE__),                            \
                                                      /* coverity[autosar_cpp14_a5_1_1_violation] FP: nvbugs/2980306 */            \
                                                      ""))
#define DW_LOGI dw::core::Logger::get().log(dw::core::Logger::LoggerMessageInfo((dw::core::Logger::Verbosity::INFO),               \
                                                      LOG_TAG,                                                                     \
                                                      dw::core::SourceCodeLocation(__FILE__, __LINE__),                            \
                                                      /* coverity[autosar_cpp14_a5_1_1_violation] FP: nvbugs/2980306 */            \
                                                      ""))
#define DW_LOGW dw::core::Logger::get().log(dw::core::Logger::LoggerMessageInfo((dw::core::Logger::Verbosity::WARN),               \
                                                      LOG_TAG,                                                                     \
                                                      dw::core::SourceCodeLocation(__FILE__, __LINE__),                            \
                                                      /* coverity[autosar_cpp14_a5_1_1_violation] FP: nvbugs/2980306 */            \
                                                      ""))
#define DW_LOGE dw::core::Logger::get().log(dw::core::Logger::LoggerMessageInfo((dw::core::Logger::Verbosity::ERROR),              \
                                                      LOG_TAG,                                                                     \
                                                      dw::core::SourceCodeLocation(__FILE__, __LINE__),                            \
                                                      /* coverity[autosar_cpp14_a5_1_1_violation] FP: nvbugs/2980306 */            \
                                                      ""))
#define DW_LOGF dw::core::Logger::get().log(dw::core::Logger::LoggerMessageInfo((dw::core::Logger::Verbosity::FATAL),              \
                                                      LOG_TAG,                                                                     \
                                                      dw::core::SourceCodeLocation(__FILE__, __LINE__),                            \
                                                      /* coverity[autosar_cpp14_a5_1_1_violation] FP: nvbugs/2980306 */            \
                                                      ""))

#endif // DW_CORE_LOGGER_HPP_
