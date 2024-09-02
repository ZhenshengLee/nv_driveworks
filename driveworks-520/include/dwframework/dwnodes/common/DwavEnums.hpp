#ifndef DWFRAMEWORK_DWNODES_COMMON_DWAVENUMS_HPP_
#define DWFRAMEWORK_DWNODES_COMMON_DWAVENUMS_HPP_

#include <dwcgf/channel/ChannelPacketTypes.hpp>

// wraps around base framework macro but allows us to avoid boiler plating of dw::framework::DWChannelPacketTypeID
#define DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION(DATA_TYPE, SPECIMEN_TYPE, ENUM_SPEC) \
    DWFRAMEWORK_DECLARE_PACKET_TYPE_RELATION(DATA_TYPE, SPECIMEN_TYPE, dw::framework::DWChannelPacketTypeID::ENUM_SPEC)

// same as above, but covers simple case where the specimen for data type is data type itself
#define DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION_SIMPLE(DATA_TYPE, ENUM_SPEC) \
    DWFRAMEWORK_DECLARE_PACKET_DWTYPE_RELATION(DATA_TYPE, DATA_TYPE, ENUM_SPEC)

#define DWFRAMEWORK_CUDA_STREAM_ID (0)

// TODO(csketch) why is DWChannelPacketTypeID inside dwcgf?

#endif // DWFRAMEWORK_DWNODES_COMMON_DWAVENUMS_HPP_
