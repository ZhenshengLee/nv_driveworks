#.rst:
# FindPackageMessage
# ------------------
#
#
#
# FIND_PACKAGE_MESSAGE(<name> "message for user" "find result details")
#
# This macro is intended to be used in FindXXX.cmake modules files.  It
# will print a message once for each unique find result.  This is useful
# for telling the user where a package was found.  The first argument
# specifies the name (XXX) of the package.  The second argument
# specifies the message to display.  The third argument lists details
# about the find result so that if they change the message will be
# displayed again.  The macro also obeys the QUIET argument to the
# find_package command.
#
# Example:
#
# ::
#
#   if(X11_FOUND)
#     FIND_PACKAGE_MESSAGE(X11 "Found X11: ${X11_X11_LIB}"
#       "[${X11_X11_LIB}][${X11_INCLUDE_DIR}]")
#   else()
#    ...
#   endif()

#=============================================================================
# Copyright 2008-2023 Kitware, Inc.
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# (To distribute this file outside of CMake, substitute the full
#  License text for the above reference.)

function(FIND_PACKAGE_MESSAGE pkg msg details)
  # Avoid printing a message repeatedly for the same find result.
  if(NOT ${pkg}_FIND_QUIETLY)
    string(REPLACE "\n" "" details "${details}")
    set(DETAILS_VAR FIND_PACKAGE_MESSAGE_DETAILS_${pkg})
    if(NOT "${details}" STREQUAL "${${DETAILS_VAR}}")
      # The message has not yet been printed.
      message(STATUS "${msg}")

      # Save the find details in the cache to avoid printing the same
      # message again.
      set("${DETAILS_VAR}" "${details}"
        CACHE INTERNAL "Details about finding ${pkg}")
    endif()
  endif()
endfunction()
