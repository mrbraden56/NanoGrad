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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/bradenlock83/Projects/NanoGrad/nano_grad/csrc/cmake

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/bradenlock83/Projects/NanoGrad/build

# Include any dependencies generated for this target.
include autograd_build/CMakeFiles/autograd.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include autograd_build/CMakeFiles/autograd.dir/compiler_depend.make

# Include the progress variables for this target.
include autograd_build/CMakeFiles/autograd.dir/progress.make

# Include the compile flags for this target's objects.
include autograd_build/CMakeFiles/autograd.dir/flags.make

autograd_build/CMakeFiles/autograd.dir/autograd.cpp.o: autograd_build/CMakeFiles/autograd.dir/flags.make
autograd_build/CMakeFiles/autograd.dir/autograd.cpp.o: /home/bradenlock83/Projects/NanoGrad/nano_grad/csrc/Autograd/autograd.cpp
autograd_build/CMakeFiles/autograd.dir/autograd.cpp.o: autograd_build/CMakeFiles/autograd.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bradenlock83/Projects/NanoGrad/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object autograd_build/CMakeFiles/autograd.dir/autograd.cpp.o"
	cd /home/bradenlock83/Projects/NanoGrad/build/autograd_build && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT autograd_build/CMakeFiles/autograd.dir/autograd.cpp.o -MF CMakeFiles/autograd.dir/autograd.cpp.o.d -o CMakeFiles/autograd.dir/autograd.cpp.o -c /home/bradenlock83/Projects/NanoGrad/nano_grad/csrc/Autograd/autograd.cpp

autograd_build/CMakeFiles/autograd.dir/autograd.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/autograd.dir/autograd.cpp.i"
	cd /home/bradenlock83/Projects/NanoGrad/build/autograd_build && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bradenlock83/Projects/NanoGrad/nano_grad/csrc/Autograd/autograd.cpp > CMakeFiles/autograd.dir/autograd.cpp.i

autograd_build/CMakeFiles/autograd.dir/autograd.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/autograd.dir/autograd.cpp.s"
	cd /home/bradenlock83/Projects/NanoGrad/build/autograd_build && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bradenlock83/Projects/NanoGrad/nano_grad/csrc/Autograd/autograd.cpp -o CMakeFiles/autograd.dir/autograd.cpp.s

# Object files for target autograd
autograd_OBJECTS = \
"CMakeFiles/autograd.dir/autograd.cpp.o"

# External object files for target autograd
autograd_EXTERNAL_OBJECTS =

autograd_build/libautograd.a: autograd_build/CMakeFiles/autograd.dir/autograd.cpp.o
autograd_build/libautograd.a: autograd_build/CMakeFiles/autograd.dir/build.make
autograd_build/libautograd.a: autograd_build/CMakeFiles/autograd.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/bradenlock83/Projects/NanoGrad/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libautograd.a"
	cd /home/bradenlock83/Projects/NanoGrad/build/autograd_build && $(CMAKE_COMMAND) -P CMakeFiles/autograd.dir/cmake_clean_target.cmake
	cd /home/bradenlock83/Projects/NanoGrad/build/autograd_build && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/autograd.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
autograd_build/CMakeFiles/autograd.dir/build: autograd_build/libautograd.a
.PHONY : autograd_build/CMakeFiles/autograd.dir/build

autograd_build/CMakeFiles/autograd.dir/clean:
	cd /home/bradenlock83/Projects/NanoGrad/build/autograd_build && $(CMAKE_COMMAND) -P CMakeFiles/autograd.dir/cmake_clean.cmake
.PHONY : autograd_build/CMakeFiles/autograd.dir/clean

autograd_build/CMakeFiles/autograd.dir/depend:
	cd /home/bradenlock83/Projects/NanoGrad/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/bradenlock83/Projects/NanoGrad/nano_grad/csrc/cmake /home/bradenlock83/Projects/NanoGrad/nano_grad/csrc/Autograd /home/bradenlock83/Projects/NanoGrad/build /home/bradenlock83/Projects/NanoGrad/build/autograd_build /home/bradenlock83/Projects/NanoGrad/build/autograd_build/CMakeFiles/autograd.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : autograd_build/CMakeFiles/autograd.dir/depend

