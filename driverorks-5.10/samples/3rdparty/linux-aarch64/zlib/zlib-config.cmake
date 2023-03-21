## CMake configuration file
set(LIBNAME zlib)
if(${LIBNAME}_FOUND)
	return()
endif()

# library version information
set (${LIBNAME}_VERSION_STRING "1.2.11")
set (${LIBNAME}_VERSION_MAJOR  1)
set (${LIBNAME}_VERSION_MINOR  2)
set (${LIBNAME}_VERSION_PATCH  11)

# Compute the installation prefix relative to this file.
get_filename_component (_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/" ABSOLUTE)

# add library
add_library(${LIBNAME} SHARED IMPORTED)

set_target_properties(${LIBNAME} PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${_INSTALL_PREFIX}/include"
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_INSTALL_PREFIX}/lib/libz.so.1"
  IMPORTED_CONFIGURATIONS       RELEASE
  VERSION                       1.2.11
)

set (${LIBNAME}_FOUND TRUE)
set (${LIBNAME}_LIBRARIES "${_INSTALL_PREFIX}/lib")

# unset private variables
unset (_INSTALL_PREFIX)
