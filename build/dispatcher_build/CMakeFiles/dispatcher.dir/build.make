# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

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
CMAKE_COMMAND = /home/bradenlock83/Environments/onnx/lib/python3.10/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/bradenlock83/Environments/onnx/lib/python3.10/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/bradenlock83/Projects/NanoGrad/nano_grad/csrc/cmake

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/bradenlock83/Projects/NanoGrad/build

# Include any dependencies generated for this target.
include dispatcher_build/CMakeFiles/dispatcher.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include dispatcher_build/CMakeFiles/dispatcher.dir/compiler_depend.make

# Include the progress variables for this target.
include dispatcher_build/CMakeFiles/dispatcher.dir/progress.make

# Include the compile flags for this target's objects.
include dispatcher_build/CMakeFiles/dispatcher.dir/flags.make

dispatcher_build/CMakeFiles/dispatcher.dir/dispatcher.cpp.o: dispatcher_build/CMakeFiles/dispatcher.dir/flags.make
dispatcher_build/CMakeFiles/dispatcher.dir/dispatcher.cpp.o: /home/bradenlock83/Projects/NanoGrad/nano_grad/csrc/Dispatcher/dispatcher.cpp
dispatcher_build/CMakeFiles/dispatcher.dir/dispatcher.cpp.o: dispatcher_build/CMakeFiles/dispatcher.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bradenlock83/Projects/NanoGrad/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object dispatcher_build/CMakeFiles/dispatcher.dir/dispatcher.cpp.o"
	cd /home/bradenlock83/Projects/NanoGrad/build/dispatcher_build && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT dispatcher_build/CMakeFiles/dispatcher.dir/dispatcher.cpp.o -MF CMakeFiles/dispatcher.dir/dispatcher.cpp.o.d -o CMakeFiles/dispatcher.dir/dispatcher.cpp.o -c /home/bradenlock83/Projects/NanoGrad/nano_grad/csrc/Dispatcher/dispatcher.cpp

dispatcher_build/CMakeFiles/dispatcher.dir/dispatcher.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dispatcher.dir/dispatcher.cpp.i"
	cd /home/bradenlock83/Projects/NanoGrad/build/dispatcher_build && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bradenlock83/Projects/NanoGrad/nano_grad/csrc/Dispatcher/dispatcher.cpp > CMakeFiles/dispatcher.dir/dispatcher.cpp.i

dispatcher_build/CMakeFiles/dispatcher.dir/dispatcher.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dispatcher.dir/dispatcher.cpp.s"
	cd /home/bradenlock83/Projects/NanoGrad/build/dispatcher_build && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bradenlock83/Projects/NanoGrad/nano_grad/csrc/Dispatcher/dispatcher.cpp -o CMakeFiles/dispatcher.dir/dispatcher.cpp.s

# Object files for target dispatcher
dispatcher_OBJECTS = \
"CMakeFiles/dispatcher.dir/dispatcher.cpp.o"

# External object files for target dispatcher
dispatcher_EXTERNAL_OBJECTS =

dispatcher_build/libdispatcher.a: dispatcher_build/CMakeFiles/dispatcher.dir/dispatcher.cpp.o
dispatcher_build/libdispatcher.a: dispatcher_build/CMakeFiles/dispatcher.dir/build.make
dispatcher_build/libdispatcher.a: dispatcher_build/CMakeFiles/dispatcher.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/bradenlock83/Projects/NanoGrad/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libdispatcher.a"
	cd /home/bradenlock83/Projects/NanoGrad/build/dispatcher_build && $(CMAKE_COMMAND) -P CMakeFiles/dispatcher.dir/cmake_clean_target.cmake
	cd /home/bradenlock83/Projects/NanoGrad/build/dispatcher_build && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/dispatcher.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
dispatcher_build/CMakeFiles/dispatcher.dir/build: dispatcher_build/libdispatcher.a
.PHONY : dispatcher_build/CMakeFiles/dispatcher.dir/build

dispatcher_build/CMakeFiles/dispatcher.dir/clean:
	cd /home/bradenlock83/Projects/NanoGrad/build/dispatcher_build && $(CMAKE_COMMAND) -P CMakeFiles/dispatcher.dir/cmake_clean.cmake
.PHONY : dispatcher_build/CMakeFiles/dispatcher.dir/clean

dispatcher_build/CMakeFiles/dispatcher.dir/depend:
	cd /home/bradenlock83/Projects/NanoGrad/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/bradenlock83/Projects/NanoGrad/nano_grad/csrc/cmake /home/bradenlock83/Projects/NanoGrad/nano_grad/csrc/Dispatcher /home/bradenlock83/Projects/NanoGrad/build /home/bradenlock83/Projects/NanoGrad/build/dispatcher_build /home/bradenlock83/Projects/NanoGrad/build/dispatcher_build/CMakeFiles/dispatcher.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : dispatcher_build/CMakeFiles/dispatcher.dir/depend

