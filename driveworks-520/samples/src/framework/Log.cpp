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

#include "Log.hpp"

#include <sys/time.h>
#include <unistd.h>

#include <ctime>
#include <cstdarg>
#include <cstdlib>
#include <sstream>
#include <string>

#include <dw/core/context/Context.h>

//------------------------------------------------------------------------------
bool shouldUseColor(bool streamIsTTY)
{
    // We rely on the TERM variable.
    const char* cterm = getenv("TERM");
    if (cterm == nullptr)
        return false;
    std::string term(cterm);
    const bool termOk =
        (term == "xterm") ||
        (term == "xterm-color") ||
        (term == "xterm-256color") ||
        (term == "screen") ||
        (term == "screen-256color") ||
        (term == "tmux") ||
        (term == "tmux-256color") ||
        (term == "rxvt-unicode") ||
        (term == "rxvt-unicode-256color") ||
        (term == "linux") ||
        (term == "cygwin");
    return streamIsTTY && termOk;
}

//------------------------------------------------------------------------------
// Returns the ANSI color code for the given color.  COLOR_DEFAULT is
// an invalid input.
const char* getAnsiColorCode(EConsoleColor color)
{
    switch (color)
    {
    case COLOR_RED: return "1";
    case COLOR_GREEN: return "2";
    case COLOR_YELLOW: return "3";
    default: return NULL;
    };
}

//------------------------------------------------------------------------------
void printColored(FILE* fd, EConsoleColor color, const char* msg)
{
    bool useColor = (color != COLOR_DEFAULT);

    if (useColor)
    {
        useColor = shouldUseColor(isatty(fileno(fd)) != 0);
    }

    if (!useColor)
    {
        fprintf(fd, "%s", msg);
        fflush(fd);
        return;
    }

    fprintf(fd, "\033[0;3%sm", getAnsiColorCode(color));
    fprintf(fd, "%s", msg);
    fprintf(fd, "\033[m"); // Resets the terminal to default.

    fflush(fd);
}

//------------------------------------------------------------------------------
dwLogCallback getConsoleLoggerCallback(bool useColors, bool disableBuffering)
{
    (void)useColors;
    (void)disableBuffering;

#ifndef DW_PROFILING
    if (disableBuffering)
        setbuf(stdout, NULL);

    if (useColors)
        return [](dwContextHandle_t, dwLoggerVerbosity level, const char* msg) {
            EConsoleColor color = COLOR_DEFAULT;
            FILE* fd            = stdout;
            switch (level)
            {
            case DW_LOG_SILENT:
            case DW_LOG_VERBOSE:
            case DW_LOG_DEBUG:
            case DW_LOG_INFO:
                break;
            case DW_LOG_WARN:
                color = COLOR_YELLOW;
                break;
            case DW_LOG_ERROR:
            case DW_LOG_FATAL:
                color = COLOR_RED;
                fd    = stderr;
                break;
            }
            printColored(fd, color, msg);
        };
    else
#endif
        return [](dwContextHandle_t, dwLoggerVerbosity, const char* msg) { printf("%s", msg); };
}

dwLoggerCallback getConsoleLoggerExtendedCallback(bool useColors, bool disableBuffering)
{
    static_cast<void>(useColors);
    static_cast<void>(disableBuffering);

#ifndef DW_PROFILING
    if (disableBuffering)
    {
        setbuf(stdout, NULL);
        return baseExtendedCallback();
    }
#endif

    if (useColors)
    {
        return [](dwLoggerMessage const* msg) {
            if (msg == nullptr)
            {
                return;
            }
            std::ostringstream oss;
            prependLoggerMessageInfo(oss, *msg);
            std::string s             = oss.str();
            const char* outputMessage = s.c_str();

            EConsoleColor color = COLOR_DEFAULT;
            FILE* fd            = stdout;
            switch (msg->verbosity)
            {
            case DW_LOG_SILENT:
            case DW_LOG_VERBOSE:
            case DW_LOG_DEBUG:
                break;
            case DW_LOG_WARN:
                color = COLOR_YELLOW;
                break;
            case DW_LOG_ERROR:
                color = COLOR_RED;
                fd    = stderr;
                break;
            default:
                break;
            }
            printColored(fd, color, outputMessage);
        };
    }
    else
    {
        return baseExtendedCallback();
    }
}

//------------------------------------------------------------------------------
dwLoggerCallback baseExtendedCallback()
{
    return [](dwLoggerMessage const* msg) {
        if (msg == nullptr)
        {
            return;
        }
        std::ostringstream oss;
        prependLoggerMessageInfo(oss, *msg);
        oss << " " << msg->msg;
        std::string s             = oss.str();
        const char* outputMessage = s.c_str();

        printf("%s", outputMessage);
    };
}
//------------------------------------------------------------------------------
void prependLoggerMessageInfo(std::ostream& os, dwLoggerMessage const& msg)
{
    streamIsoTimestamp(os);
    streamVerbosity(os, msg.verbosity);
    streamThreadId(os, msg.threadId);
    streamSourceCodeLocation(os, msg.fileName, msg.lineNum);
    streamTag(os, msg.tag);
    os << " " << msg.msg;
}

//------------------------------------------------------------------------------
void log(const char* format, ...) // NOLINT
{
    (void)format;
#ifndef DW_PROFILING
    char buffer[2048];
    va_list args;
    va_start(args, format);
    vsprintf(buffer, format, args);
    va_end(args);

    printColored(stdout, COLOR_DEFAULT, buffer);
#endif
}

//------------------------------------------------------------------------------
void logError(const char* format, ...) // NOLINT
{
    (void)format;
#ifndef DW_PROFILING
    char buffer[2048];
    va_list args;
    va_start(args, format);
    vsprintf(buffer, format, args);
    va_end(args);

    printColored(stderr, COLOR_RED, buffer);
#endif
}

//------------------------------------------------------------------------------
void logWarn(const char* format, ...) // NOLINT
{
    (void)format;
#ifndef DW_PROFILING
    char buffer[2048];
    va_list args;
    va_start(args, format);
    vsprintf(buffer, format, args);
    va_end(args);

    printColored(stdout, COLOR_YELLOW, buffer);
#endif
}

// -----------------------------------------------------------------------------
void streamIsoTimestamp(std::ostream& os)
{
    struct tm datetime = {};
    datetime.tm_mday   = 1;
    timespec ts{};
    clock_gettime(CLOCK_REALTIME, &ts);
    gmtime_r(&ts.tv_sec, &datetime);

    // Year
    os << "[" << (1900 + datetime.tm_year) << "-";
    // Month
    if (1 + datetime.tm_mon < 10)
    {
        os << "0" << 1 + datetime.tm_mon;
    }
    else
    {
        os << 1 + datetime.tm_mon;
    }
    os << "-";
    // Day of month
    if (datetime.tm_mday < 10)
    {
        os << "0" << datetime.tm_mday;
    }
    else
    {
        os << datetime.tm_mday;
    }
    os << "T";
    // Hour
    if (datetime.tm_hour < 10)
    {
        os << "0" << datetime.tm_hour;
    }
    else
    {
        os << datetime.tm_hour;
    }
    os << ":";
    // Minute
    if (datetime.tm_min < 10)
    {
        os << "0" << datetime.tm_min;
    }
    else
    {
        os << datetime.tm_min;
    }
    os << ":";
    // Second
    if (datetime.tm_sec < 10)
    {
        os << "0" << datetime.tm_sec;
    }
    else
    {
        os << datetime.tm_sec;
    }
    // Microsecond
    os << "." << (ts.tv_nsec / 1000);
    // Timezone UTC
    os << "Z]";
}

// -----------------------------------------------------------------------------
void streamTag(std::ostream& os, char const* tag)
{
    os << "[" << tag << "]";
}

// -----------------------------------------------------------------------------
void streamThreadId(std::ostream& os, char const* tid)
{
    os << "[tid:" << tid << "]";
}

// -----------------------------------------------------------------------------
void streamSourceCodeLocation(std::ostream& os, char const* fileName, int32_t lineNum)
{
    os << "[" << fileName << ":" << lineNum << "]";
}

// -----------------------------------------------------------------------------
void streamVerbosity(std::ostream& os, dwLoggerVerbosity const level)
{
    switch (level)
    {
    case DW_LOG_VERBOSE:
        os << "["
           << "VERBOSE"
           << "]";
        break;
    case DW_LOG_DEBUG:
        os << "["
           << "DEBUG"
           << "]";
        break;
    case DW_LOG_INFO:
        os << "["
           << "INFO"
           << "]";
        break;
    case DW_LOG_WARN:
        os << "["
           << "WARN"
           << "]";
        break;
    case DW_LOG_ERROR:
        os << "["
           << "ERROR"
           << "]";
        break;
    case DW_LOG_FATAL:
        os << "["
           << "FATAL"
           << "]";
        break;
    case DW_LOG_SILENT:
        // Since SILENT is set to the highest log level some modules are using it as log no matter what
        os << "["
           << "SILENT"
           << "]";
        break;
    default:
        break; // Unreachable w/ -Wswitch
    }
}
