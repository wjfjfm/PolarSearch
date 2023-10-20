# Install script for directory: /root/NGT/lib/NGT

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES
    "/root/NGT/build/lib/NGT/libngt.so.2.1.3"
    "/root/NGT/build/lib/NGT/libngt.so.2"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libngt.so.2.1.3"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libngt.so.2"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/root/NGT/build/lib/NGT/libngt.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libngt.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libngt.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libngt.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/root/NGT/build/lib/NGT/libngt.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/NGT" TYPE FILE FILES
    "/root/NGT/lib/NGT/ArrayFile.h"
    "/root/NGT/lib/NGT/Capi.h"
    "/root/NGT/lib/NGT/Clustering.h"
    "/root/NGT/lib/NGT/Command.h"
    "/root/NGT/lib/NGT/Common.h"
    "/root/NGT/lib/NGT/Graph.h"
    "/root/NGT/lib/NGT/GraphOptimizer.h"
    "/root/NGT/lib/NGT/GraphReconstructor.h"
    "/root/NGT/lib/NGT/HashBasedBooleanSet.h"
    "/root/NGT/lib/NGT/Index.h"
    "/root/NGT/lib/NGT/MmapManager.h"
    "/root/NGT/lib/NGT/MmapManagerDefs.h"
    "/root/NGT/lib/NGT/MmapManagerException.h"
    "/root/NGT/lib/NGT/MmapManagerImpl.hpp"
    "/root/NGT/lib/NGT/Node.h"
    "/root/NGT/lib/NGT/ObjectRepository.h"
    "/root/NGT/lib/NGT/ObjectSpace.h"
    "/root/NGT/lib/NGT/ObjectSpaceRepository.h"
    "/root/NGT/lib/NGT/Optimizer.h"
    "/root/NGT/lib/NGT/PrimitiveComparator.h"
    "/root/NGT/lib/NGT/SharedMemoryAllocator.h"
    "/root/NGT/lib/NGT/Thread.h"
    "/root/NGT/lib/NGT/Tree.h"
    "/root/NGT/lib/NGT/Version.h"
    "/root/NGT/lib/NGT/half.hpp"
    "/root/NGT/build/lib/NGT/defines.h"
    "/root/NGT/build/lib/NGT/version_defs.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/NGT/NGTQ" TYPE FILE FILES
    "/root/NGT/lib/NGT/NGTQ/Capi.h"
    "/root/NGT/lib/NGT/NGTQ/HierarchicalKmeans.h"
    "/root/NGT/lib/NGT/NGTQ/Matrix.h"
    "/root/NGT/lib/NGT/NGTQ/ObjectFile.h"
    "/root/NGT/lib/NGT/NGTQ/Optimizer.h"
    "/root/NGT/lib/NGT/NGTQ/QbgCli.h"
    "/root/NGT/lib/NGT/NGTQ/QuantizedBlobGraph.h"
    "/root/NGT/lib/NGT/NGTQ/QuantizedGraph.h"
    "/root/NGT/lib/NGT/NGTQ/Quantizer.h"
    )
endif()

