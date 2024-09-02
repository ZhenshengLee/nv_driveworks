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
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_CORE_SAFETY_SAFESTROPS_HPP_
#define DW_CORE_SAFETY_SAFESTROPS_HPP_

#include <dwshared/dwfoundation/dw/core/base/StringBuffer.hpp>
#include <dwshared/dwfoundation/dw/core/base/ExceptionWithStackTrace.hpp>

#include <cstring>

namespace dw
{
namespace core
{

// Need to use #ifdef on _GNU_SOURCE to define a proxy GNU_SOURCE_DEFINED to avoid triggering AUTOSAR M16-0-7
#ifdef _GNU_SOURCE
#define GNU_SOURCE_DEFINED 1
#else
#define GNU_SOURCE_DEFINED 0
#endif

constexpr size_t STRERROR_BUFFER_SIZE{256};

/**
 * @brief Threadsafe wrapper around @c strerror.  Returns a textual description of the system error code
          @c errnum.
 *
 * @param errnum integer value referring to an error code
 *
 * @return @c StringBuffer containing an error string corresponding to the @c errno code.
 */
inline StringBuffer<STRERROR_BUFFER_SIZE> safeStrerror(int errnum) noexcept
{
    char bufArg[STRERROR_BUFFER_SIZE];

#if (_POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600) && !GNU_SOURCE_DEFINED
    ::strerror_r(errnum, bufArg, STRERROR_BUFFER_SIZE);
    return StringBuffer<STRERROR_BUFFER_SIZE>{bufArg};
#else
    return StringBuffer<STRERROR_BUFFER_SIZE>{::strerror_r(errnum, bufArg, STRERROR_BUFFER_SIZE)};
#endif
}

/**
 * String conversion wrappers check and print relevant errors before returning
 */
inline float64_t safeStrtod(char8_t const* const str, char8_t** const endptr)
{
    errno = 0;
    float64_t const ret{strtod(str, endptr)};
    if (errno != 0)
    {
        // coverity[cert_str30_c_violation] FP: nvbugs/3964710
        throw InvalidConversionException("Invalid float64_t conversion errno: ", safeStrerror(errno).c_str());
    }
    return ret;
}

inline int64_t safeStrtol(char8_t const* const str, char8_t** const endptr, int32_t const base)
{
    errno = 0;
    int64_t const ret{strtol(str, endptr, base)};
    if (errno != 0)
    {
        throw InvalidConversionException("Invalid int64_t conversion errno: ", safeStrerror(errno).c_str());
    }
    return ret;
}

inline uint64_t safeStrtoul(char8_t const* const str, char8_t** const endptr, int32_t const base)
{
    errno = 0;
    uint64_t const ret{strtoul(str, endptr, base)};
    if (errno != 0)
    {
        throw InvalidConversionException("Invalid uint64_t conversion errno: ", safeStrerror(errno).c_str());
    }
    return ret;
}

inline float32_t safeStrtof(char8_t const* const str, char8_t** const endptr)
{
    errno = 0;
    float32_t const ret{strtof(str, endptr)};
    if (errno != 0)
    {
        throw InvalidConversionException("Invalid float32_t conversion errno: ", safeStrerror(errno).c_str());
    }
    return ret;
}

} // namespace core
} // namespace dw

#endif // DW_CORE_SAFETY_SAFESTROPS_HPP_
