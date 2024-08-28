#ifndef CHANNEL_PACKET_TYPES_COMMON_AVENUMS_HPP_
#define CHANNEL_PACKET_TYPES_COMMON_AVENUMS_HPP_

// wraps around base framework macro but allows us to avoid boiler plating of dw::framework::DWChannelPacketTypeID
#define DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION(DATA_TYPE, SPECIMEN_TYPE, ENUM_SPEC)                                \
    DWFRAMEWORK_DECLARE_PACKET_TYPE_RELATION(DATA_TYPE, SPECIMEN_TYPE, dw::framework::DWChannelPacketTypeID::ENUM_SPEC)

// same as above, but covers simple case where the specimen for data type is data type itself
#define DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION_SIMPLE(DATA_TYPE, ENUM_SPEC)                                        \
    DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION(DATA_TYPE, DATA_TYPE, ENUM_SPEC)

#endif  // CHANNEL_PACKET_TYPES_COMMON_AVENUMS_HPP_
