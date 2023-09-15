/*
 * Profiler.hpp
 *
 * Copyright(C) 2014-2019 University of Oulu, all rights reserved.
 * Copyright(C) 2014-2019 NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Contact: Daniel Herrera C. (dherrera@ee.oulu.fi),
 *          Kihwan Kim(kihwank@nvidia.com)
 * Author : Daniel Herrera C.
 */

#ifndef SAMPLES_COMMON_PROFILER_H_
#define SAMPLES_COMMON_PROFILER_H_

#include <memory>
#include <map>
#include <vector>
#include <stack>
#include <string>
#include <iomanip>
#include <thread>
#include <mutex>
#include <chrono>
#include <ostream>
#include <cassert>

#include <dw/core/base/Types.h>
#include <dw/core/context/Context.h>

#include "CudaTimer.hpp"
#include "StatsCounter.hpp"

// The define below enables the profiler!
#define ENABLE_PROFILER

namespace dw_samples
{
namespace common
{

/**
 * Class to keep track of runtimes for different sections of code (i.e. runtime profiling).
 * Can be used directly with tic(string) and toc(string), or through the ProfileSection() class. All
 * profiling code is removed if ENABLE_PROFILER is not defined.
 *
 * All profiling is done through cuda events. Thus, CPU time will not be measured.
 *
 * Profiler has a singleton instance and can be safely called from different threads. Timings from different
 * threads are kept separate.
 */
class ProfilerCUDA
{
public:
    ProfilerCUDA();

    inline void tic(const std::string& sectionKey, bool isTopLevelSection);
    inline void toc();

    /// Determines whether to show average or total timings
    void setShowTotals(bool value) { m_showTotals = value; }
    bool getShowTotals() const { return m_showTotals; }

    /// If total timings are shown, they will be divided by this constant factor
    int getTotalsFactor() const { return m_totalsFactor; }
    void setTotalsFactor(int value) { m_totalsFactor = value; }

    // Log all timings directly to a stream
    template <typename T>
    T& logStats(T& stream);

    bool empty();
    void reset();

    void setCurrentThreadName(const std::string& name);

    void collectTimers();

    ///////////////////////////////////////////////////////////
    // Child classes
    class ThreadData;
    class SectionData;

    class SectionData
    {
    public:
        SectionData(ThreadData* threadData, SectionData* parent, const std::string& key);

        SectionData* getParent() const { return m_parent; }
        const std::string& getKey() const { return m_key; }

        const std::map<std::string, std::unique_ptr<SectionData>>& getSubsections() const { return m_childSections; }

        inline SectionData* getSubsection(const std::string& subkey);

        void addTiming(dwTime_t timeCPU, CudaTimer* timer);
        std::vector<CudaTimer*> collectTimers();

        bool empty() const { return (m_statsGPU.getSampleCount() == 0) && m_pendingTimers.empty(); }
        void reset();

        const StatsCounter& getStatsCPU() const { return m_statsCPU; }
        const StatsCounter& getStatsGPU() const { return m_statsGPU; }

        template <typename T>
        T& logTotals(T& stream, const std::string& prefix, const float32_t parentTimeCPU, const float32_t parentTimeGPU);

        template <typename T>
        T& logStats(T& stream, const std::string& prefix, const float32_t parentTimeCPU, const float32_t parentTimeGPU);

    protected:
        ThreadData* m_threadData;
        SectionData* m_parent;

        std::string m_key;
        StatsCounter m_statsGPU;
        StatsCounter m_statsCPU;

        std::map<std::string, std::unique_ptr<SectionData>> m_childSections;

        std::vector<CudaTimer*> m_pendingTimers;
    };

    class ThreadData
    {
    public:
        ThreadData(ProfilerCUDA* profiler, std::thread::id id);

        const std::thread::id& getId() const { return m_id; }
        void setId(const std::thread::id& newId) { m_id = newId; }

        const std::string& getName() const { return m_name; }
        void setName(const std::string& name) { m_name = name; }

        ProfilerCUDA* getProfiler() const { return m_profiler; }
        SectionData* getRootSection() { return &m_rootSection; }
        const SectionData* getRootSection() const { return &m_rootSection; }

        CudaTimer* getTimer();
        void collectTimers();
        void collectTimers(SectionData* section);

        inline void tic(const std::string& sectionKey, bool isTopLevelSection);
        inline void toc();

        template <typename T>
        T& logStats(T& stream);

        void reset();

    protected:
        ProfilerCUDA* m_profiler;
        std::thread::id m_id;
        std::string m_name;

        SectionData m_rootSection;

        struct ActiveTimingData
        {
            SectionData* section;
            std::chrono::time_point<std::chrono::steady_clock> startTime;
            CudaTimer* timer;
        };
        std::stack<ActiveTimingData> m_activeSections;

        std::vector<std::unique_ptr<CudaTimer>> m_ownedTimers;
        std::stack<CudaTimer*> m_freeTimers;
    };

    ///////////////////////////////////////////////////////////
    // ProfilerCUDA member methods
    ThreadData* getThreadData();

    /// Note: The map returned is used by different threads and is therefore not thread-safe to use
    const std::map<std::thread::id, std::unique_ptr<ThreadData>>& getAllThreadData() const
    {
        return m_threads;
    }

protected:
    ///////////////////////////////////////////////////////////
    // ProfilerCUDA member variables
    std::mutex m_mutex;
    std::map<std::thread::id, std::unique_ptr<ThreadData>> m_threads;
    bool m_showTotals;
    int m_totalsFactor;
};

template <typename T>
T& operator<<(T& stream, ProfilerCUDA& profiler)
{
    return profiler.logStats(stream);
}

/**
 * @brief Used to profile a section by just constructing and destructing this object.
 * Example:
 *    {
 *      ProfileCUDASections s(&profiler, "cudaSection1");
 *      mykernel<<<N,K>>>();
 *    }
 */
class ProfileCUDASection
{
public:
    ProfileCUDASection(ProfilerCUDA* profiler, const std::string& sectionKey, bool isTopLevel)
        : m_profiler(profiler)
        , m_running(true)
    {
#ifdef ENABLE_PROFILER
        // Note: adding a sync here isolates the sections from any other work.
        //       This is good to profile the individual sections, but bad to profile
        //       the entire system and how the complete pipeline behaves.
        //       The sync must be commented in master but you may uncomment it to profile and compare.
        //cudaDeviceSynchronize();
        m_profiler->tic(sectionKey, isTopLevel);
#else
        (void)sectionKey;
        (void)isTopLevel;
#endif
    }
    ProfileCUDASection(ProfilerCUDA* profiler, const std::string& sectionKey)
        : ProfileCUDASection(profiler, sectionKey, false)
    {
    }
    ProfileCUDASection(ProfilerCUDA& profiler, const std::string& sectionKey)
        : ProfileCUDASection(&profiler, sectionKey)
    {
    }

    void stop()
    {
#ifdef ENABLE_PROFILER
        m_profiler->toc();
        m_running = false;
#endif
    }

    ~ProfileCUDASection()
    {
#ifdef ENABLE_PROFILER
        if (m_running)
            stop();
#endif
    }

private:
    ProfilerCUDA* m_profiler;
    bool m_running;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation of inline methods
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
ProfilerCUDA::SectionData* ProfilerCUDA::SectionData::getSubsection(const std::string& subkey)
{
    auto itSection = m_childSections.find(subkey);
    if (itSection == m_childSections.end())
    {
        SectionData* data;
        data = new SectionData(m_threadData, this, subkey);
        m_childSections.insert(std::make_pair(subkey, std::unique_ptr<ProfilerCUDA::SectionData>(data)));
        return data;
    }
    else
        return itSection->second.get();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
T& ProfilerCUDA::SectionData::logTotals(T& stream, const std::string& prefix, const float32_t parentTimeCPU, const float32_t parentTimeGPU)
{
    auto profiler = m_threadData->getProfiler();

    const float32_t factor = static_cast<float32_t>(profiler->getTotalsFactor());

    float32_t timeCPU = m_statsCPU.getSum();
    float32_t timeGPU = m_statsGPU.getSum();
    const char* units = "us";
    if (factor > 1)
    {
        timeCPU /= factor;
        timeGPU /= factor;
        units = "us/item";
    }

    //CPU
    stream << std::left << std::setw(25) << (prefix + m_key)
           << std::right
           << " CPU: " << std::setw(5) << timeCPU << units;
    if (parentTimeCPU > 0)
    {
        int percent = static_cast<int>(100 * timeCPU / parentTimeCPU);
        stream << " (" << std::setw(2) << percent << "%)";
    }
    else
    {
        // Print empty to keep columns aligned
        stream << std::setw(6) << " ";
    }

    //GPU
    stream << " | GPU: " << std::setw(5) << timeGPU << units;
    if (parentTimeGPU > 0)
    {
        int percent = static_cast<int>(100 * timeGPU / parentTimeGPU);
        stream << " (" << std::setw(2) << percent << "%)";
    }
    else
    {
        // Print empty to keep columns aligned
        stream << std::setw(6) << " ";
    }
    stream << "\n";

    std::string newPrefix = prefix + "-";
    for (auto& child : m_childSections)
    {
        child.second->logTotals(stream, newPrefix, timeCPU, timeGPU);
    }

    // Display "other" time
    if (m_childSections.size() > 1)
    {
        float32_t otherCPU = timeCPU;
        float32_t otherGPU = timeGPU;
        for (auto& child : m_childSections)
        {
            otherCPU -= child.second->getStatsCPU().getSum() / factor;
            otherGPU -= child.second->getStatsGPU().getSum() / factor;
        }

        stream << std::left << std::setw(25) << (newPrefix + "other") << std::right
               << " CPU: " << std::setw(5) << otherCPU << units;
        if (timeCPU > 0)
        {
            int percent = static_cast<int>(100 * otherCPU / timeCPU);
            stream << " (" << std::setw(2) << percent << "%)";
        }
        else
        {
            // Print empty to keep columns aligned
            stream << std::setw(6) << " ";
        }

        //GPU
        stream << " | GPU: " << std::setw(5) << otherGPU << units;
        if (timeGPU > 0)
        {
            int percent = static_cast<int>(100 * otherGPU / timeGPU);
            stream << " (" << std::setw(2) << percent << "%)";
        }
        else
        {
            // Print empty to keep columns aligned
            stream << std::setw(6) << " ";
        }

        stream << "\n";
    }

    return stream;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
T& ProfilerCUDA::SectionData::logStats(T& stream, const std::string& prefix, const float32_t parentTimeCPU, const float32_t parentTimeGPU)
{
    float32_t timeCPU = m_statsCPU.getMedian();
    float32_t timeGPU = m_statsGPU.getMedian();
    stream << std::left << std::setw(25) << (prefix + m_key)
           << std::right
           << " CPU: " << std::setw(5) << timeCPU << "us, std=" << std::setw(4) << m_statsCPU.getStdDev();
    if (parentTimeCPU > 0)
    {
        int percent = static_cast<int>(100 * timeCPU / parentTimeCPU);
        stream << " (" << std::setw(2) << percent << "%)";
    }
    else
    {
        // Print empty to keep columns aligned
        stream << std::setw(6) << " ";
    }

    stream << " | GPU: " << std::setw(5) << timeGPU << "us, std=" << std::setw(4) << m_statsGPU.getStdDev();
    if (parentTimeGPU > 0)
    {
        int percent = static_cast<int>(100 * timeGPU / parentTimeGPU);
        stream << " (" << std::setw(2) << percent << "%)";
    }
    else
    {
        // Print empty to keep columns aligned
        stream << std::setw(6) << " ";
    }
    stream << " | samples=" << m_statsGPU.getSampleCount() << "\n";

    std::string newPrefix = prefix + "-";
    for (auto& child : m_childSections)
        child.second->logStats(stream, newPrefix, timeCPU, timeGPU);

    return stream;
}

///////////////////////////////////////////////////////////////////////////////////////
void ProfilerCUDA::ThreadData::tic(const std::string& sectionKey, bool isTopLevelSection)
{
    ProfilerCUDA::SectionData* data;
    if (isTopLevelSection)
        data = m_rootSection.getSubsection(sectionKey);
    else
        data = m_activeSections.top().section->getSubsection(sectionKey);

    CudaTimer* timer = getTimer();

    auto timeCPU = std::chrono::steady_clock::now();

    m_activeSections.push(ActiveTimingData{data, timeCPU, timer});

    timer->start();
}

///////////////////////////////////////////////////////////////////////////////////////
void ProfilerCUDA::ThreadData::toc()
{
    auto data = m_activeSections.top();

    //Calculate ellapsed CPU time
    auto timeStopCPU = std::chrono::steady_clock::now();
    auto durationCPU = std::chrono::duration_cast<std::chrono::microseconds>(timeStopCPU - data.startTime).count();

    //Store ellapsed GPU time
    data.timer->stop();

    //Record data
    data.section->addTiming(durationCPU, data.timer);

    m_activeSections.pop();
}

///////////////////////////////////////////////////////////////////////////////////////
template <typename T>
T& ProfilerCUDA::ThreadData::logStats(T& stream)
{
    stream << "Thread ";
    if (m_name.empty())
        stream << m_id;
    else
        stream << m_name;
    stream << ":" << std::endl;

    if (m_profiler->getShowTotals())
    {
        for (auto& sections : m_rootSection.getSubsections())
            sections.second->logTotals(stream, "-", 0, 0);
    }
    else
    {
        for (auto& sections : m_rootSection.getSubsections())
            sections.second->logStats(stream, "-", 0, 0);
    }

    return stream;
}

///////////////////////////////////////////////////////////////////////////////////////
void ProfilerCUDA::tic(const std::string& sectionKey, bool isTopLevelSection)
{
    ThreadData* thread = getThreadData();
    thread->tic(sectionKey, isTopLevelSection);
}

///////////////////////////////////////////////////////////////////////////////////////
void ProfilerCUDA::toc()
{
    ThreadData* thread = getThreadData();
    thread->toc();
}

///////////////////////////////////////////////////////////////////////////////////////
template <typename T>
T& ProfilerCUDA::logStats(T& stream)
{
#ifdef ENABLE_PROFILER
    std::vector<ThreadData*> allData;
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        for (auto& thread : m_threads)
            allData.push_back(thread.second.get());
    }

    //Save flags
    std::ios oldState(nullptr);
    oldState.copyfmt(stream);

    stream << std::fixed << std::setprecision(0);

    //stream << "Profiler stats:" << std::endl;
    for (auto& thread : allData)
        thread->logStats(stream);

    //Restore flags
    stream.copyfmt(oldState);
#endif
    return stream;
}
}
}

#endif /* PROFILER_H_ */
