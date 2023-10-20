# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/NGT

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/NGT/build

# Utility rule file for versiondef.

# Include any custom commands dependencies for this target.
include lib/NGT/CMakeFiles/versiondef.dir/compiler_depend.make

# Include the progress variables for this target.
include lib/NGT/CMakeFiles/versiondef.dir/progress.make

lib/NGT/CMakeFiles/versiondef: lib/NGT/command

lib/NGT/command: ../lib/NGT/ArrayFile.cpp
lib/NGT/command: ../lib/NGT/Capi.cpp
lib/NGT/command: ../lib/NGT/Command.cpp
lib/NGT/command: ../lib/NGT/Graph.cpp
lib/NGT/command: ../lib/NGT/Index.cpp
lib/NGT/command: ../lib/NGT/MmapManager.cpp
lib/NGT/command: ../lib/NGT/NGTQ/Capi.cpp
lib/NGT/command: ../lib/NGT/NGTQ/HierarchicalKmeans.cpp
lib/NGT/command: ../lib/NGT/NGTQ/Optimizer.cpp
lib/NGT/command: ../lib/NGT/NGTQ/QbgCli.cpp
lib/NGT/command: ../lib/NGT/NGTQ/QuantizedGraph.cpp
lib/NGT/command: ../lib/NGT/Node.cpp
lib/NGT/command: ../lib/NGT/SharedMemoryAllocator.cpp
lib/NGT/command: ../lib/NGT/Thread.cpp
lib/NGT/command: ../lib/NGT/Tree.cpp
lib/NGT/command: ../lib/NGT/Version.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/root/NGT/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating command"
	cd /root/NGT/build/lib/NGT && sh /root/NGT/utils/mk_version_defs_h.sh /root/NGT version_defs.h

versiondef: lib/NGT/CMakeFiles/versiondef
versiondef: lib/NGT/command
versiondef: lib/NGT/CMakeFiles/versiondef.dir/build.make
.PHONY : versiondef

# Rule to build all files generated by this target.
lib/NGT/CMakeFiles/versiondef.dir/build: versiondef
.PHONY : lib/NGT/CMakeFiles/versiondef.dir/build

lib/NGT/CMakeFiles/versiondef.dir/clean:
	cd /root/NGT/build/lib/NGT && $(CMAKE_COMMAND) -P CMakeFiles/versiondef.dir/cmake_clean.cmake
.PHONY : lib/NGT/CMakeFiles/versiondef.dir/clean

lib/NGT/CMakeFiles/versiondef.dir/depend:
	cd /root/NGT/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/NGT /root/NGT/lib/NGT /root/NGT/build /root/NGT/build/lib/NGT /root/NGT/build/lib/NGT/CMakeFiles/versiondef.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : lib/NGT/CMakeFiles/versiondef.dir/depend
