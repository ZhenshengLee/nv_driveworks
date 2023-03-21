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
// SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DW_FRAMEWORK_CHANNEL_PACKET_TYPES_HPP_
#define DW_FRAMEWORK_CHANNEL_PACKET_TYPES_HPP_

#include <typeinfo>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <iostream>
#include <nvscisync.h>
#include <dw/core/language/Function.hpp>
#include <dw/core/base/Types.h>

namespace dw
{
namespace framework
{

///////////////////////////////////////////////////////////////////////////////////////

struct ITypeInformation
{
    virtual const char* name() const = 0;
};

template <class T>
struct TypeInformation : ITypeInformation
{
    const char* name() const final
    {
        return typeid(T).name();
    }
};

/**
 * Weakly typed data buffer that couples together a generic pointer
 * with the size and type information of the memory it points to.
 */
class GenericData
{
public:
    /**
     * Default construct to point to nullptr.
     */
    GenericData()
        : GenericData(nullptr, 0)
    {
    }

    /**
     * Construct to point to data with only size information and no type information.
     *
     * @param data the pointer to the data
     * @param size the size of the data
     */
    GenericData(void* data, size_t size)
        : m_data{data}
        , m_size{size}
        , m_typeInfo{nullptr}
    {
    }

    /**
     * Construct to point to data of the given type.
     *
     * @tparam T the type to point to
     * @param data the pointer
     */
    template <typename T>
    GenericData(T* data)
        : m_data{data}
        , m_size{sizeof(T)}
    {
        static constexpr TypeInformation<T> g_typeInfo{};
        m_typeInfo = &g_typeInfo;
    }

    /**
     * Get the size of the data pointed to.
     * @return size_t
     */
    size_t size() const
    {
        return m_size;
    }

    /**
     * Get a pointer to the data with the given type.
     *
     * @note if no type information is available, this method will succeed if the
     * size of T and the size of the data match.
     *
     * @throws if the type T is incompatible with the value.
     *
     * @tparam T the type of data
     * @return T* the pointer to the data
     */
    template <typename T>
    T* getData() const
    {
        if (m_typeInfo == nullptr && m_size == sizeof(T))
        {
            // Type info not found but sizes match
            return static_cast<T*>(m_data);
        }
        else if (m_size != sizeof(T))
        {
            // Wrong size
            return nullptr;
        }
        else if (dynamic_cast<const TypeInformation<T>*>(m_typeInfo) != nullptr)
        {
            // Type info found and matched
            return static_cast<T*>(m_data);
        }
        else
        {
            // Wrong type
            return nullptr;
        }
    }

    /**
     * Get the pointer to the data.
     * @return void*
     */
    void* getPointer() const
    {
        return m_data;
    }

private:
    void* m_data;
    size_t m_size;
    const ITypeInformation* m_typeInfo;
};

/**
 * The unique id specifying how a data type should be allocated and transported. This id is used to verify that remote peers
 * are expecting the same data type over a connection.
 */
using ChannelPacketTypeID = uint32_t;
/**
 * The id of a data type that is flat and zero initialized. In this case, only the size will be checked
 * against remote peers.
 */
constexpr ChannelPacketTypeID DWFRAMEWORK_PACKET_ID_DEFAULT   = 0U;
constexpr uint32_t DWFRAMEWORK_METADATA_PACKET_TYPE_ID_OFFSET = 0x80000000;
constexpr uint32_t DWFRAMEWORK_MAX_INTERNAL_TYPES             = 0x400;

// clang-format off
#define DW_CHANNEL_PACKET_TYPES_LIST(_s) \
    _s(DW_IMAGE_HANDLE)                  \
    _s(DW_LATENCY)                       \
    _s(DW_PYRAMID_IMAGE)                 \
    _s(DW_FEATURE_ARRAY)                 \
    _s(DW_FEATURE_HISTORY_ARRAY)         \
    _s(DW_FEATURE_NCC_SCORES)            \
    _s(DW_SENSOR_NODE_RAW_DATA)          \
    _s(DW_RADAR_SCAN)                    \
    _s(DW_LIDAR_DECODE_PACKET)           \
    _s(DW_EGOMOTION_STATE_HANDLE)        \
    _s(DW_POINT_CLOUD)                   \
    _s(DW_LIDAR_PACKETS_ARRAY)           \
    _s(DW_TRACE_NODE_DATA)               \
    _s(DW_CODEC_PACKET)                  \
    _s(DW_SENSOR_SERVICE_RAW_DATA)
// clang-format on

// Enumerate the classes for channel packet types
// The DEFUALT class assumes the data is a contiguous chunk of memory
// Other classes are defined for specific complex objects.
// See packet_traits<> helper for mapping from this enum to the type the
// enum corresponds to.
// TODO: add APIs to allow external complex types to be registered.

#define DW_CHANNEL_GENERATE_ENUM(x) x,

enum class DWChannelPacketTypeID
{
    DEFAULT = DWFRAMEWORK_PACKET_ID_DEFAULT, // Assumes packets are contiguous chunk of memory
    DW_CHANNEL_PACKET_TYPES_LIST(DW_CHANNEL_GENERATE_ENUM)
        NUM_TYPES
};

enum class MetadataFlags : uint16_t
{
    METADATA_ITERATION_COUNT = 1 << 0,
    METADATA_EPOCH_TIMESTAMP = 1 << 1,
    METADATA_VALIDITY_STATUS = 1 << 2,
    METADATA_TIMESTAMP       = 1 << 3,
};

struct ChannelMetadata
{
    uint32_t producerId;             // id of the producer channel
    uint32_t sequenceNum;            // sequence number of packet
    dwTime_t timestamp;              // wall-clock time of the producer
    uint32_t iterationCount;         // node iteration count
    dwTime_t epochTimestamp;         // virtual time of the producer
    dwValidityStatus validityStatus; // validity info
    uint16_t validFields;
};

struct MetadataPayload
{
    ChannelMetadata header;
    GenericData data;
};

/**
 *  Function signature to call back to the application to set the sync attributes when needed.
 **/
using OnSetSyncAttrs = dw::core::Function<void(NvSciSyncAttrList)>;

///////////////////////////////////////////////////////////////////////////////////////

/**
 * Parameter used to initialize a producer/consumer endpoint.
 */
struct GenericDataReference
{
    std::shared_ptr<void> pCopy;          ///< a shared pointer to packet constructor parameters.
    GenericData data;                     ///< GenericData pointing to packet constructor parameters.
    ChannelPacketTypeID packetTypeID;     ///< The ID of the type of the endpoint.
    size_t typeSize;                      ///< The size of the type of the endpoint.
    OnSetSyncAttrs setWaiterAttributes;   ///< lambda to set the waiter attributes of the endpoint.
    OnSetSyncAttrs setSignalerAttributes; ///< lambda to set the signaler attributes of the endpoint.
};

template <typename T>
struct parameter_traits
{
    // TODO(chale): enable following code once we actually fix everywhere that POD is used with channel
    // but is not declared as such.
    // static_assert(!std::is_same<T, T>::value,
    //               "Attempting to use type with Port/Channel that has no declared packet handling. "
    //               "A packet handling must be declared with DWFRAMEWORK_DECLARE_PACKET_TYPE_RELATION.");
    using SpecimenT = T;
    static constexpr ChannelPacketTypeID PacketTID{DWFRAMEWORK_PACKET_ID_DEFAULT};
    static constexpr bool IsDeclared{false};
};

template <ChannelPacketTypeID PacketTID>
struct packet_traits
{
    // TODO(chale): enable following code once we actually fix everywhere that POD is used with channel
    // but is not declared as such.
    // static_assert(PacketTID != PacketTID,
    //               "Attempting to use type with Port/Channel that has no declared packet handling. "
    //               "A packet handling must be declared with DWFRAMEWORK_DECLARE_PACKET_TYPE_RELATION.");
};

// Creates a generic data speciemn wrapper for type T.
// The data specimen wrapper points to a copy of the passed specimen on the heap that it owns.
// This is necessary to make the specimen passable/copyable by value.
template <typename T>
static inline GenericDataReference make_specimen(typename parameter_traits<T>::SpecimenT* specimen)
{
    GenericDataReference result{};
    result.packetTypeID = parameter_traits<T>::PacketTID;
    result.typeSize     = sizeof(T);
    if (specimen != nullptr)
    {
        auto heapCopy = std::make_shared<typename parameter_traits<T>::SpecimenT>(*specimen);
        result.pCopy  = heapCopy;
        result.data   = GenericData(heapCopy.get());
    }
    else
    {
        result.pCopy = nullptr;
        result.data  = GenericData(specimen);
    }

    return result;
}

} // namespace framework
} // namespace dw

/**
 *  Define a mapping between the data type and the id under which the handling packet class will be registered.
 *  @param DATA_TYPE - the data type to register.
 *  @param SPECIMEN_TYPE - the reference information for the data type to be passed to packet implementation.
 *  @param PACKET_TYPE_ID - the id under which the handling packet class will be registered (should be convertible to ChannelPacketTypeID)
 *
 *  @note this macro must be used at global namespace to ensure it will work properly.
 **/
#define DWFRAMEWORK_DECLARE_PACKET_TYPE_RELATION(DATA_TYPE, SPECIMEN_TYPE, PACKET_TYPE_ID)                \
    namespace dw                                                                                          \
    {                                                                                                     \
    namespace framework                                                                                   \
    {                                                                                                     \
    template <>                                                                                           \
    struct parameter_traits<DATA_TYPE>                                                                    \
    {                                                                                                     \
        using SpecimenT = SPECIMEN_TYPE;                                                                  \
        static constexpr ChannelPacketTypeID PacketTID{static_cast<ChannelPacketTypeID>(PACKET_TYPE_ID)}; \
        static constexpr bool IsDeclared{true};                                                           \
    };                                                                                                    \
    template <>                                                                                           \
    struct packet_traits<static_cast<ChannelPacketTypeID>(PACKET_TYPE_ID)>                                \
    {                                                                                                     \
        using PacketT = DATA_TYPE;                                                                        \
    };                                                                                                    \
    }                                                                                                     \
    }

/**
 *  Declares a data type to be treated as plain old data (POD) by dwframework channels.
 * @param DATA_TYPE - the data type to be declared as POD.
 **/
#define DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(DATA_TYPE)                                 \
    namespace dw                                                                       \
    {                                                                                  \
    namespace framework                                                                \
    {                                                                                  \
    template <>                                                                        \
    struct parameter_traits<DATA_TYPE>                                                 \
    {                                                                                  \
        using SpecimenT = DATA_TYPE;                                                   \
        static constexpr ChannelPacketTypeID PacketTID{DWFRAMEWORK_PACKET_ID_DEFAULT}; \
        static constexpr bool IsDeclared{true};                                        \
    };                                                                                 \
    }                                                                                  \
    }

// Pre-declare some types as POD.
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(int32_t);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(uint32_t);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(int64_t);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(uint64_t);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(bool);
DWFRAMEWORK_DECLARE_PACKET_TYPE_POD(float);

#endif // DW_FRAMEWORK_CHANNEL_PACKET_TYPES_HPP_
