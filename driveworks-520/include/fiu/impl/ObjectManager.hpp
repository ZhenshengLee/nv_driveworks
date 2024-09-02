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
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.
//
// FI Tool Version:    0.2.0
//
/////////////////////////////////////////////////////////////////////////////////////////

#ifndef FIU_OBJECT_MANAGER_H
#define FIU_OBJECT_MANAGER_H
#include <fiu/FaultCommand.hpp>

namespace fiu2
{

////// Memory Manager related constants ////
constexpr int ONE_KB{1024};
constexpr int TEN_KB{10 * ONE_KB};
constexpr int HUNDRED_KB{10 * TEN_KB};
constexpr int ONE_MB{10 * HUNDRED_KB};
constexpr int TEN_MB{10 * ONE_MB};

constexpr int ONE_KB_BLOBS{30};
constexpr int TEN_KB_BLOBS{30};
constexpr int HUNDRED_KB_BLOBS{10};
constexpr int ONE_MB_BLOBS{2};
constexpr int TEN_MB_BLOBS{1};

constexpr int TOTAL_SCALAR_PARAMS_MEMORY_BLOBS{100};
constexpr int TOTAL_VECTOR_PARAMS_MEMORY_BLOBS{100};

// This structure keeps track of all the free memory blobs that the
// FIUCommand uses over the life time of the system. FIUCommand requests
// memory blobs from this structure and returns them back for reusage.
struct FIUObjNode
{
    // Size of each blob this memory holds
    size_t blobSize{};

    // Number of memory blobs this node holds
    uint32_t blobCount{};

    // Each index indicates if the associated blob is used or not
    bool* used{};

    // Array of fault command indexes that use the associated blob
    uint32_t* faultCommandID{};

    // Count of total memory nodes used
    uint32_t totalUsed{};

    // Double link list pointers
    FIUObjNode* next{};
    FIUObjNode* prev{};

    // Array of pointers to allocated blobs
    unsigned char** buffers{};
};

class ObjectManager
{
public:
    /**
     * Destructor
     */
    ~ObjectManager();

    /**
     * Constructor
     */
    ObjectManager();

    /**
     * Runs during construction phase... allocates memory for scalar & vector params
     *
     * @return true if memory allocated successfully; false otherwise
     */
    bool allocateMemoryForParams();

    /**
     * Runs during construction phase... add a blob of a certain size
     *
     * @param[in] blobSize Size of the blob
     * @param[in] blobCount No of blobs of that size within the array
     * @return true if memory allocated successfully; false otherwise
     */
    bool addBlob(size_t blobSize, uint32_t blobCount);

    /**
     * Helper function to display the memory stats
     */
    void printStats();

    /**
     * Returns an allocated buffer that fits a FaultScalarParameter
     *
     * @param[in] faultCommandID
     * @return Pointer to the object; NULL if object not available
     */
    FaultScalarParameter* getScalarParameter(uint32_t faultCommandID);

    /**
     * Returns an allocated buffer that fits a FaultVectorParameter
     *
     * @param[in] faultCommandID
     * @return Pointer to the object; NULL if object not available
     */
    FaultVectorParameter* getVectorParameter(uint32_t faultCommandID);

    /**
     * Returns a blob that is big enough to fit the given memory
     *
     * @param[out] **ptr Pointer that shall point to the allocated memory blob
     * @param[in] memorySizeRequested Size of memory requested
     * @param[in] faultCommandID id of the fault command
     * @param[out] poolIndex index of the pool where this was allocated
     * @return true if memory was allocated or not
     */
    bool getMemory(unsigned char** ptr, size_t memorySizeRequested, uint32_t faultCommandID, uint32_t& poolIndex);

    /**
     * Releases allcated memory back to the pool
     *
     * @param[in] *ptr Pointer that needs to be released
     * @param[in] faultCommandID id of the fault command
     * @param[in] poolIndex index of the pool where this was allocated initially
     * @return true if memory was released or not
     */
    bool releaseMemory(unsigned char* ptr, uint32_t faultCommandID, uint32_t poolIndex);

    /**
     * Releases allcated object for Scalar Parameter
     *
     * @param[in] *fsp Pointer to the scalar param object obj that needs to be released
     * @param[in] faultCommandID id of the fault command
     * @return true if object was released or not
     */
    bool releaseScalarParam(FaultScalarParameter* fsp, uint32_t faultCommandID);

    /**
     * Releases allcated object for vector Parameter
     *
     * @param[in] *fvp Pointer to the vector param object obj that needs to be released
     * @param[in] faultCommandID id of the fault command
     * @return true if object was released or not
     */
    bool releaseVectorParam(FaultVectorParameter* fvp, uint32_t faultCommandID);

    void resetFCObject(uint32_t mfcObjIndex);

    /**
     * Creates a new FIUObjNode
     *
     * @param[in] blobSize Size of each blob
     * @param[in] blobCount No of blobs of that size in this node
     * @return pointer to the allocated node
     */
    FIUObjNode* createNode(size_t blobSize, uint32_t blobCount);

    /**
     * Resets specific buffer of a node to 0
     *
     * @param[in] node to the buffer that needs to be reset
     * @param[in] index to the buffer that needs to be reset
     */
    void resetBuffer(FIUObjNode* node, uint32_t index);

    /**
     * Returns a new free bucket
     *
     * @param[out] Node from the bucket is requested
     * @param[in] id of the fault command
     * @return pointer to the allocated memory
     */
    unsigned char* getBucket(FIUObjNode* node, uint32_t faultCommandID);

    /**
     * Frees the relevant object node
     *
     * @param[in] Pointer to the node that should be freed.
     */
    void freeObjectStruct(FIUObjNode* node);

    /**
     * Releases the memory blob so that it can be reused
     *
     * @param[in] Node of the blob pointer that should be released.
     * @param[in] blob pointer that needs to be released for reuse.
     * @param[in] blob pointer that needs to be released for reuse.
     */
    bool releaseBlob(FIUObjNode* node, unsigned char* blob, uint32_t faultCommandID);

protected:
    /**
     * Garbage collect object pool
     */
    void freeObjectPool();

    SpinLock m_acquireReleaseLock{};

    // Member Variables
    FIUObjNode* m_headNode{};
    FIUObjNode* m_scalarParamNode{};
    FIUObjNode* m_vectorParamNode{};

    uint32_t m_totalScalarParams{TOTAL_SCALAR_PARAMS_MEMORY_BLOBS};
    uint32_t m_totalVectorParams{TOTAL_VECTOR_PARAMS_MEMORY_BLOBS};

    bool m_isMemoryAllocated{false};

    // Make the blob sizes array dyanmic so that it can be defined during
    // construction. The idea here is to allow test code to alter the sizes
    // for unit tests
    uint32_t* m_noOfBuckets{};
    size_t* m_bucketSize{};
    uint32_t m_objectPoolSize{};
    FIUObjNode** m_objectPool{};
};
};
#endif
