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
// SPDX-FileCopyrightText: Copyright (c) 2020-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.
//
// Parser Version: 0.7.1
// SSM Version:    0.8.2
//
/////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <ssm/SSMIncludes.hpp>
#include <ssm/comm.hpp>

namespace SystemStateManager
{

typedef struct _sampleTimestampStruct {
    int tag{0};     // Stores the id of the timestamp
    uint64_t ts{0}; // Stores the timestamp
} SampleTimestampStruct;


/*******************************************************************************************
Usage Instructions
==================

This class aids developers in the following ways
   - Latency Histogram of multiple passes
   - High latency code identification

  Latency Histogram
  -----------------
  SSMHistogram class has to be defined in a non-local scope.
  This object records the latest timestamp when the user calls 'startTimer()'.
  The user should cal the startTimer() at the start of a pass.
  At the end of the pass, the user should call the 'endTimer()'' at which point
  the object computes the difference between the prior timestamp and the latest timestamp.
  The difference constitutes the latency which is then accumulated in an hash map.
  The user can print the histogram by executing the 'printHistogram()' fucntion.

  High latency code identification
  --------------------------------
  We normally put printfs [such as printf("1111", timestamp)] identify code locations
  to identify code that has overhead. SSMHistogram provides a better way of managoing
  these printfs. The user can call 'addTimestamp()' and pass a tag (or a label/name) at
  different points in the code between the 'startTimer()' and the 'endTimer()' within
  each pass. SSMHistogram records latency between code points and then prints the
  timestamp instance if the latency of the whole pass crosses a certain threshold as
  indicated by the argument that user passes through 'endTimer()' api.

*******************************************************************************************/

class SSMHistogram
{
public:
    ~SSMHistogram();
    SSMHistogram(std::string name = "");
    void setName(std::string name);
    void addSample(uint64_t startTime,
                   uint64_t endTime,
                   uint64_t latencyThreshold = 0,
                   bool printLat = true);
    void startTimer();
    void endTimer(uint64_t latencyThreshold = 0, bool printLat = true);
    void resetTimer();
    void addTimestamp(int tag);
    void getHistogram(std::ostringstream &ostr, std::string str="");
    void printHistogram(std::string str = "");

private:
    bool m_areAnalyticsPresent{false};
    std::string m_histName;
    uint64_t m_sampleIndex{0};
    uint64_t m_startTime{0};
    uint64_t m_endTime{0};
    uint64_t *m_histArray{};
    SampleTimestampStruct *m_sampleAnalysisArray{};
    int m_sampleAnalysisIndex{0};
};

typedef std::shared_ptr<SSMHistogram> SSMHistogramPtr;

}
