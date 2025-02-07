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
// SPDX-FileCopyrightText: Copyright (c) 2015-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_CORE_LOGGER_BASE_HPP_
#define DW_CORE_LOGGER_BASE_HPP_

#include <dwshared/dwfoundation/dw/core/platform/CompilerSpecificMacros.hpp>
#include <dwshared/dwfoundation/dw/core/language/BasicTypes.hpp>
#include <dwshared/dwfoundation/dw/core/base/StringBuffer.hpp>
#include <dwshared/dwfoundation/dw/core/base/ExceptionWithStackTrace.hpp>

#include <atomic>
#include <ctime>
#include <memory>

/**
 * Important notice:
 * LoggerStream object returned by the DW logging APIs implements thread local reentrance protection to
 * prevent the possibility of an infinite recursions and stack overflows.
 * - Do not store LoggerStream objects long term.
 * - Do not destroy LoggerStream objects on a thread other than the one which has created it.
 * - Use one LoggerStream object at a time, with a minimum possible lifetime scope and without overlap with other LoggerStream objects.
 * Using DW_LOG macros inline implicitly complies with these constraints.
 */

/// Use this macro to define a gtest unit test as a friend of your class.
/// note: originally defined in <gtest.h>, but redelacring it here to not propagate <gtest.h> include everywhere
#ifndef FRIEND_TEST
// TODO(mpopovic): RFD - friend alias for test use case
// coverity[autosar_cpp14_a11_3_1_violation]
#define FRIEND_TEST(test_case_name, test_name) friend class test_case_name##_##test_name##_Test
#endif
// clang-format on

namespace dw
{
namespace core
{

/*  @brief Struct describing position of line of code
     */
struct SourceCodeLocation
{

    constexpr SourceCodeLocation()
        : lineNum{}, fileName{""} {}
    constexpr SourceCodeLocation(char8_t const* _fileName, int32_t _lineNum)
        : lineNum{_lineNum}, fileName{baseName(_fileName)} {}

    /*  @brief Extract basename of file from file path
        *   @param Character array of file path
        *   @return Character array of base name of file
        */
    constexpr char8_t const* baseName(char8_t const* filePath)
    {
        if (!filePath)
        {
            return nullptr;
        }

        char8_t const* baseName{filePath};
        while (*filePath)
        {
            if (*filePath++ == '/')
            {
                baseName = filePath;
            }
        }
        return baseName;
    }

    int32_t lineNum;
    const char8_t* fileName;
};

/**
 * @brief Basic logger interface class.<br>
 * Note: A logger can be allocated in the heap space of the application.
 *
 */
class Logger
{
public:
    static constexpr char8_t const* LOG_TAG{"Logger"};
    static constexpr char8_t const* NO_TAG{"NO_TAG"};

    /// Enumerative for the level of verbosity (VERBOSE, DEBUG, INFO, WARN, ERROR, SILENT)
    enum class Verbosity : int32_t
    {
        VERBOSE = 0x0000, ///< Show messages with any log level, dwLoggerVerbosity
        DEBUG   = 0x1000, ///< Show messages with log level dwLoggerVerbosity::DW_LOG_DEBUG and higher
        INFO    = 0x2000, ///< Show messages with log level dwLoggerVerbosity::DW_LOG_INFO and higher
        WARN    = 0x3000, ///< Show messages with log level dwLoggerVerbosity::DW_LOG_WARN and higher
        ERROR   = 0x4000, ///< Show messages with log level dwLoggerVerbosity::DW_LOG_ERROR and higher
        FATAL   = 0x5000, ///< Show messages with log level dwLoggerVerbosity::DW_LOG_FATAL and higher
        SILENT  = 0x7000  ///< Show messages with log level dwLoggerVerbosity::DW_LOG_SILENT
    };

    /// Enumerative for the type of logger (Logger, CallbackLogger, DefaultLogger, NullLogger, StreamLogger)
    enum class LoggerType : int32_t
    {
        Logger,
        CallbackLogger,
        DefaultLogger,
        NullLogger,
        StreamLogger
    };

    static constexpr int32_t NUM_VERBOSITY_LEVELS{7};
    static constexpr uint32_t TAG_SIZE{64}; //!< reasonably large length based on length of tags in use prior to change
    static constexpr uint32_t TID_SIZE{64}; //!< pthread id can be 16 and gave safety factor of 4

    /// Structure for the information of the Logger message
    struct LoggerMessageInfo
    {
        LoggerMessageInfo() = default;
        explicit LoggerMessageInfo(Verbosity const v)
            : verbosity{v}
            , tag{}
            , loc{"", 0}
            , threadId{}
        {
        }
        LoggerMessageInfo(Verbosity const v, char8_t const* t, SourceCodeLocation l, char8_t const* tid)
            : verbosity{v}, tag{t}, loc{l}, threadId{tid} {}

        Verbosity verbosity = Verbosity::VERBOSE;
        StringBuffer<TAG_SIZE> tag{};
        SourceCodeLocation loc = SourceCodeLocation("", 0);
        mutable StringBuffer<TID_SIZE> threadId{};
    };

    /// Default constructor of the class
    Logger() = default;
    /// Destructor of the class
    virtual ~Logger() = default;
    Logger(Logger&&)  = delete;
    Logger& operator=(Logger const&) = delete;
    Logger& operator=(Logger const&&) = delete;

    enum class State : int32_t
    {
        endl
    };

    /// Maximal size of the buffer that logger is capable to process at once.
    /// The log buffer is allocated in thread-local storage (TLS).
    // FP: nvbugs/2765391
    // coverity[autosar_cpp14_m0_1_4_violation]
    static constexpr size_t LOG_BUFFER_SIZE{16 * 1024};

    /// Logs a timestamp of the form [dd-mm-yyyy hh:mm:ss]
    /// No message is logged so a follow up call to logMessage() is
    /// expected to give meaning to the timestamp
    virtual void logTimestamp(LoggerMessageInfo const& info);

    /**
     * Enables or disables timestamp logging.
     *
     *  @param enabled true to enable, false to disable
     */
    inline void enableTimestamps(bool const enabled)
    {
        m_enableTimestamps = enabled;
    }

    /// @brief Check if timestamps are enabled
    /// @return true if enabled, flase if disabled
    bool getEnableTimestamps() const
    {
        return m_enableTimestamps;
    }

    /**
     * Pass a given message to the logger.
     *
     * @param info      Additional info that applies to the logged message
     * @param message   Null-terminated string containing the log message
     */
    virtual void logMessage(LoggerMessageInfo const& info, char8_t const* const message) = 0;

    /// @brief Set the level of Log
    /// @param level Verbosity level
    virtual void setLogLevel(Verbosity const level)
    {
        m_filterLevel = level;
    }

    /**
     * Give thread id an identifying string
     * Set default thread id if thread id hasn't been set and empty string is passed
     *
     * @param tid identifier for current thread
     * @note Any string will be automatically truncated to a length of TID_SIZE
     */
    static void setThreadId(char8_t const* tid);

    /**
     * Give thread id an identifying string from atomically incrementing counter
     */
    static void setDefaultThreadId();

    /// @brief Return the threat Id
    /// @return Threat Id
    static char8_t const* getThreadId();

    /// @brief Return the verbosity level
    /// @return Verbosity level
    Verbosity getLogLevel() const
    {
        return m_filterLevel;
    }

    virtual void flush()
    {
    }

    virtual void reset()
    {
    }

    /// Helper class to use stream operator
    class LoggerStream
    {
    public:
        explicit LoggerStream(Logger* const logger, LoggerMessageInfo&& info);
        Logger::LoggerStream& addLogger(Logger* const logger);

        // don't allow copy construction and assignment from const LoggerStream references,
        // to avoid double logging of m_buffer
        LoggerStream& operator=(const LoggerStream& other) = delete;
        LoggerStream(const LoggerStream& other)            = delete;

        /// @brief Constructor
        /// @param other Pointer to a pointer of the class
        LoggerStream(LoggerStream&& other);

        /// @brief Destructor
        ~LoggerStream() noexcept;

        /// @brief Copy operator
        /// @param other Pointer to a pointer of the class
        /// @return Pointer to the class
        Logger::LoggerStream& operator=(LoggerStream&& other);

        void disable() { m_enabled = false; }

        /// Return enabled status
        bool isEnabled() const { return m_enabled; }

        /// @brief Update the verbosity level in the log message structure information
        /// @param value verbosity level
        void setVerbosity(Verbosity const value)
        {
            m_info.verbosity = value;
        }

        /// @brief Enable or disable the use fo floating point
        /// @param value true to enable the floating point, false to disable floating point
        void setUseFloatingPoint(bool const value)
        {
            m_useFloatingPoint = value;
        }

        /// @brief Return the floating point status enable/disabled
        /// @return true floating point enabled, false floating point diabled
        bool getUseFloatingPoint() const
        {
            return m_useFloatingPoint;
        }

        dw::core::SourceCodeLocation getSourceCodeLocation() const
        {
            return m_info.loc;
        }

        /// @brief Pass the message in the buffer to all logger and the flush the logger and clear the buffer
        void flush() noexcept;

        /**
         * @brief Get current stream buffer capacity (excluding space reserved for truncation)
         *
         * @return Current buffer capacity in bytes.
         */
        size_t getCapacity() const;

        /**
         * @brief Output the data to the log stream.
         * If the internal log stream buffer of size @ref LOG_BUFFER_SIZE is exceeded,
         * the remaining data is truncated and a postfix ... is appended.
         * The remaning capacity can be queried with @ref getCapacity() API.
         * Calling @ref flush() dispatches the buffered data to the logger and clears the buffer.
         *
         * @param data Pointer to the data.
         * @param size Size of the data (excluding null termination character)
         */
        void outputTruncated(char8_t const* data, ::size_t size);

        /**
         * @brief Output the complete data to the @a LogStream.
         * The data is not truncated. If the internal buffer is exceeded, the functions
         * may perform multiple internal flush operations.
         * The API is intended for rig serialization.
         *
         * @param data Pointer to the data.
         * @param size Size of the data (excluding null termination character)
         * @return Number of written bytes. 0 if the LogStream object is disabled.
         */
        ::size_t write(const void* const data, ::size_t const size);

        /// @brief Return the internal buffer
        /// @return the internal buffer
        StringBuffer<LOG_BUFFER_SIZE>& getBuffer() { return m_buffer; }

    private:
        // FP: nvbugs/2765391
        // coverity[autosar_cpp14_m0_1_4_violation]
        constexpr static uint32_t MAX_LOGGER_COUNT{2};
        /// Enabled flag
        bool m_enabled{false};
        /// @brief Maximum number of logger
        Logger* m_loggers[MAX_LOGGER_COUNT] = {};
        /// @brief Counter of the logger in use
        uint32_t m_loggerCount{0};
        /// @brief Information of the message logger
        LoggerMessageInfo m_info{};
        /// @brief Floating point usage flag
        bool m_useFloatingPoint = false;

        // Thread-local static variable storing the stream object locked for logging for the lifetime of this object.
        // This prevents the possibility of logging API reentry, which could lead to infinite recursion and stack overflow.
        // User shall not capture and store logger stream object for a longer term.
        // User shall not destroy logger stream object on a thread other then the one which has created it.
        static DW_THREAD_LOCAL LoggerStream* m_localThreadLoggerStream;
        // Thread-local message buffer
        static DW_THREAD_LOCAL StringBuffer<Logger::LOG_BUFFER_SIZE> m_buffer;

        FRIEND_TEST(LoggerTest, NullLoggerTest_L0);
        FRIEND_TEST(LoggerTest, Logger_L0);
        FRIEND_TEST(LoggerTest, CurrentThreadLoggerStream_L0);
    };

    /// @brief Create a log with the specified level of verbosity.
    /// @param level Level of verbosity.
    LoggerStream log(Verbosity const level);
    /// @brief Create a log with the specified log info structure.
    /// @param info Structure of the message info
    LoggerStream log(LoggerMessageInfo&& info);

    // TODO(csketch): remove unused parameter
    /// @brief Create a log with verbosity = VERBOSE.
    LoggerStream logVerbose(const void* = nullptr)
    {
        return log(Verbosity::VERBOSE);
    }
    // TODO(csketch): remove unused parameter
    /// @brief Create a log with verbosity = DEBUG.
    LoggerStream logDebug(const void* = nullptr)
    {
        return log(Verbosity::DEBUG);
    }
    // TODO(csketch): remove unused parameter
    /// @brief Create a log with verbosity = INFO.
    LoggerStream logInfo(const void* = nullptr)
    {
        return log(Verbosity::INFO);
    }
    // TODO(csketch): remove unused parameter
    /// @brief Create a log with verbosity = WARN.
    LoggerStream logWarning(const void* = nullptr)
    {
        return log(Verbosity::WARN);
    }
    // TODO(csketch): remove unused parameter
    /// @brief Create a log with verbosity = ERROR.
    LoggerStream logError(const void* = nullptr)
    {
        return log(Verbosity::ERROR);
    }
    // TODO(csketch): remove unused parameter
    LoggerStream logFatal(const void* = nullptr)
    {
        return log(Verbosity::FATAL);
    }

    /// @brief If the flag m_shouldWarnDoubleLinkinge is true it return the default logger, otherwise it return the global logger instance (m_loggerInstance) and in case this last one is null the nulllogger.
    /// @return Pointer to the logger instance
    static Logger& get();

    /// @brief Assign the logger instance to the global logger instance (m_loggerInstance).
    /// @param instance Pointer to the logger
    static void set(std::unique_ptr<Logger> instance);

    /// @brief Clone this logger instance.
    /// @return The pointer to the logger.
    virtual std::unique_ptr<Logger> clone() = 0;

    /// @brief Set m_shouldWarnDoubleLinking to false.
    static void clearDoubleLinkingWarning();

    /// @brief Return the type of log.
    /// @return The type of log.
    inline LoggerType getLoggerType() const noexcept
    {
        return m_type;
    }

protected:
    /// @brief Default constructor
    /// @param other Pointer to the logger
    Logger(const Logger& other) = default;
    /// @brief Constructor using the type
    /// @param type Type of logger
    explicit Logger(LoggerType type)
        : m_filterLevel{Verbosity::VERBOSE}
        , m_enableTimestamps{true}
        , m_type{type}
    {
    }

protected:
    template <std::size_t N>
    /// @brief Get the time using the function clock_gettime(),
    ///        brake it down into the different element and store day, month, hours, minutes and seconds in the buffer (add a zero in front if the value is only one digit).
    /// @param buffer Pointer to the buffer of messages
    static void appendTimestamp(StringBuffer<N>& buffer)
    {
        timespec time{};
        if (0 != clock_gettime(CLOCK_REALTIME, &time))
        {
            throw ExceptionWithStackTrace("Logger: failed to read current time with clock_gettime()");
        }

        tm calendar{};
        if (gmtime_r(&time.tv_sec, &calendar) == nullptr)
        {
            throw ExceptionWithStackTrace("Logger: failed to convert to local time with gmtime_r()");
        }
        buffer << (calendar.tm_mday < 10 ? "[0" : "[") << calendar.tm_mday;
        buffer << (calendar.tm_mon < 9 ? "-0" : "-") << (calendar.tm_mon + 1) << "-" << (1900 + calendar.tm_year);
        buffer << (calendar.tm_hour < 10 ? " 0" : " ") << calendar.tm_hour;
        buffer << (calendar.tm_min < 10 ? ":0" : ":") << calendar.tm_min;
        buffer << (calendar.tm_sec < 10 ? ":0" : ":") << calendar.tm_sec << "]";
    }

    /// @brief Global logger instance
    static std::unique_ptr<Logger> m_loggerInstance;
    /// @brief Extended logger instance
    static std::unique_ptr<Logger> m_loggerExtendedInstance;
    /// @brief Flag for the double linking
    static bool m_shouldWarnDoubleLinking;

private:
    Verbosity m_filterLevel = Verbosity::VERBOSE;

    static DW_THREAD_LOCAL StringBuffer<TID_SIZE> m_threadId; //!< identifier for the current thread, defaults to an atomically incrementing integer
    static std::atomic<int32_t> m_threadIdCounter;            //!< an atomic counter that ensures each thread has its own unique id number starting at 0 and incrementing by 1
    bool m_enableTimestamps = true;
    LoggerType m_type{LoggerType::Logger};

    FRIEND_TEST(LoggerTest, Logger_L0);
};

/////////////////////////////////////////////////////////////
/// These operators are defined in the dw_shared::core namespace
/// so that they can be overloaded for custom types.
/// For example, dw/math/MathUtils.hpp defines overloads
/// to log eigen matrices directly.

// TODO(csketch) move operator<< out of dw::core namespace
Logger::LoggerStream& operator<<(Logger::LoggerStream& stream, char8_t const* const s);
Logger::LoggerStream& operator<<(Logger::LoggerStream& stream, char8_t const& c);
Logger::LoggerStream& operator<<(Logger::LoggerStream& stream, int32_t const v);
Logger::LoggerStream& operator<<(Logger::LoggerStream& stream, uint32_t const v);
Logger::LoggerStream& operator<<(Logger::LoggerStream& stream, uint16_t const v);
Logger::LoggerStream& operator<<(Logger::LoggerStream& stream, int64_t const v);
Logger::LoggerStream& operator<<(Logger::LoggerStream& stream, uint64_t const v);
Logger::LoggerStream& operator<<(Logger::LoggerStream& stream, float32_t const v);
Logger::LoggerStream& operator<<(Logger::LoggerStream& stream, float64_t const v);
Logger::LoggerStream& operator<<(Logger::LoggerStream& stream, Logger::State const s);

template <class ContainerT, std::enable_if_t<detail::has_c_str<ContainerT>::value, bool> = true>
Logger::LoggerStream& operator<<(Logger::LoggerStream& stream, ContainerT const& s)
{
    if (stream.isEnabled())
    {
        stream.outputTruncated(s.c_str(), s.length());
    }
    return stream;
}

/// These are special operators that receive an rvalue ref and return an lvalue ref
/// It is necessary because of the construct:
///    LOGSTREAM_WARN(this) << "Module had an error: " << errorCode << Logger::State::endl;
/// The macro creates a temporary object so the first operator<< will get an rvalue ref.
/// This temp will be destroyed when all the line is executed so the rest can receive an lvalue ref.
inline Logger::LoggerStream& operator<<(Logger::LoggerStream&& stream, char8_t const* const v)
{
    return stream << v;
}

inline Logger::LoggerStream& operator<<(Logger::LoggerStream&& stream, int32_t const v)
{
    return stream << v;
}

inline Logger::LoggerStream& operator<<(Logger::LoggerStream&& stream, uint32_t const v)
{

    return stream << v;
}

inline Logger::LoggerStream& operator<<(Logger::LoggerStream&& stream, uint16_t const v)
{
    return stream << v;
}

inline Logger::LoggerStream& operator<<(Logger::LoggerStream&& stream, int64_t const v)
{
    return stream << v;
}

inline Logger::LoggerStream& operator<<(Logger::LoggerStream&& stream, uint64_t const v)
{
    return stream << v;
}

inline Logger::LoggerStream& operator<<(Logger::LoggerStream&& stream, float32_t const v)
{
    return stream << v;
}

inline Logger::LoggerStream& operator<<(Logger::LoggerStream&& stream, float64_t const v)
{
    return stream << v;
}

inline Logger::LoggerStream& operator<<(Logger::LoggerStream&& stream, Logger::State const v)
{
    return stream << v;
}

template <std::size_t S>
inline Logger::LoggerStream& operator<<(Logger::LoggerStream&& stream, char8_t (&v)[S])
{
    return stream << v;
}

template <class ContainerT, std::enable_if_t<detail::has_c_str<ContainerT>::value, bool> = true>
inline Logger::LoggerStream& operator<<(Logger::LoggerStream&& stream, ContainerT const& v)
{
    return stream << v.c_str();
}

} // namespace core
} // namespace

#endif // DW_CORE_LOGGER_BASE_HPP_
