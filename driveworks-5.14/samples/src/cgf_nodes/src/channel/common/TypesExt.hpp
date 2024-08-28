#ifndef DW_FRAMEWORK_TYPES_EXT_HPP_
#define DW_FRAMEWORK_TYPES_EXT_HPP_

namespace dw
{
namespace framework
{

// coverity[autosar_cpp14_a0_1_6_violation]
// coverity[autosar_cpp14_a7_2_3_violation]
enum dwSerializationTypeExt
{
    DW_FREESPACE_BOUNDARY = 257,
    // add here
    DW_GENERIC_BUFFER = 258,

    // ------end------
    DW_CUSTOM_RAW_BUFFER = 1024
};

}  // namespace framework
}  // namespace dw

#endif  // DW_FRAMEWORK_TYPES_EXT_HPP_
