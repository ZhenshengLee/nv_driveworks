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
// SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <functional>
#include <map>

namespace SystemStateManager
{

#define MAX_SIZE_GENERAL_STRINGS 64
#define MAX_VECTOR_ELEMENTS_ALLOWED 1024
#define TOTAL_ALLOWED_DEFERRED_LISTS 4

template<class T>
struct MyAllocator:std::allocator<T> {

    template <class U>
    struct rebind {
        typedef MyAllocator<U> other;
    };

    typedef typename std::allocator<T>::size_type size_type;
    MyAllocator(size_type sz=MAX_VECTOR_ELEMENTS_ALLOWED): m_maxsize(sz) {}

    size_type max_size() const {
        return m_maxsize;
    }

private:
    size_type m_maxsize;
};

template<typename T>
class FixedVector {
public:
    typedef std::vector<T,MyAllocator<T>> Vector;
    FixedVector() {
        m_vec.reserve(MAX_VECTOR_ELEMENTS_ALLOWED);
    }

    FixedVector(const FixedVector& o) {
        m_vec.reserve(MAX_VECTOR_ELEMENTS_ALLOWED);
        for (auto& s : o.m_vec){
            push_back(s);
        }
    }

    FixedVector(FixedVector&& o) {
        std::swap(m_vec, o.m_vec);
    }

    void push_back(const T& v) {

        // Throw exception if this condition becomes invalid
        if (m_vec.capacity() != MAX_VECTOR_ELEMENTS_ALLOWED) {
            throw std::length_error("Cannot allocate at runtime");
        }

        // Throw exception if the buffer is full
        if (m_vec.size() == m_vec.capacity() - 1) {
            throw std::length_error("Cannot allocate at runtime");
        }
        m_vec.push_back(v);
    }

    int size() {
        return m_vec.size();
    }

    T getLastElement() {
        T obj = m_vec.back();
        m_vec.pop_back();
        return obj;
    }

    void copyContents(FixedVector &vec) {
        for (auto &s:m_vec) {
            vec.push_back(s);
        }
    }

    void clearVector() {
        m_vec.clear();
    }

    T &getObject(int index) {return m_vec[index];}
    Vector &getVector() {return m_vec;}

    Vector m_vec;
};

// TODO: Replace std::map with a fixed Map [https://jirasw.nvidia.com/browse/AVRR-1383]
#define FixedMap std::map

typedef struct _swcStruct {
    std::string name;
    std::string ipaddress;
} SWCStruct;

typedef struct _charStruct{
    char name[MAX_SIZE_GENERAL_STRINGS];
} CharStruct;

class SMBaseClass;
typedef FixedVector<SWCStruct> SWCVector;
typedef FixedVector<std::string> StringVector;
typedef FixedVector<CharStruct> CharVector;

typedef std::function<void(SMBaseClass *ptr)> SSMFunctionHandler;

typedef FixedVector<CharStruct> CharVector;
typedef std::function<void(SMBaseClass *ptr, CharVector &argVector)> SSMQAFunctionHandler;

typedef struct _ssmfunctionHandlerStruct {
    int commandID;
    std::string stateName;
    SSMFunctionHandler handler;
    bool didFunctionExecute;
    bool isFunctionRequiredToExecute;
} SSMFunctionHandlerStruct;

typedef struct _ssmlockStepCmdStruct {
    bool isBroadCastComplete;
    uint32_t executionCompleteSignalCount;
    uint64_t broadCastTimestamp;
} SSMLockStepCmdStruct;

typedef struct _stateMachineStatus {
    std::string currentState;
} StateMachineStatusStruct;

typedef FixedVector<SSMFunctionHandlerStruct> SSMFunctionHandlerVector;
typedef FixedVector<SSMLockStepCmdStruct> LockSteppedCmdRecordVector;
typedef FixedMap<std::string, StateMachineStatusStruct> StateMachineStatusMap;

#define logStringVector(vec, title)  {\
                                        SystemStateManager::FixedStringStream ostr;\
                                        ostr << title << " : ";\
                                        int index=0;\
                                        for (auto a : vec.getVector()) { \
                                            if (index++ > 0) {\
                                                ostr << ", "; \
                                            }\
                                            ostr << a; \
                                        }\
                                        SSM_LOG(ostr.str());\
                                        cout << ostr.str() << endl;\
                                     }



#define removeStringFromVector(vec, str)    {\
                                                auto &v = vec.getVector(); \
                                                for (auto i = v.begin(); i != v.end(); ++i) { \
                                                    if (str == *i) { \
                                                        v.erase(i); \
                                                        break;\
                                                    } \
                                                } \
                                            }


#define ENTER_STATE        "ENTER"
#define EXIT_STATE         "EXIT"
#define PRE_INIT_STATE     "PRE_INIT"
#define INIT_STATE         "INIT"
#define POST_INIT_STATE    "POST_INIT"
#define PRE_READY_STATE    "PRE_READY"
#define READY_STATE        "READY"
#define INIT_CLONE_STATE   "INIT_CLONE"
#define PRE_SHUTDOWN_STATE "PRE_SHUTDOWN"
#define SHUTDOWN_STATE     "SHUTDOWN"

extern StringVector apiDeferredSMList;
extern void printInitBroadcast(std::string smName, int initPhase, uint64_t timestamp, StringVector &vec);
extern bool removeStateMachines(StringVector &sv);
void splitString(const std::string& s, char delimiter, CharVector &strvec);
}
