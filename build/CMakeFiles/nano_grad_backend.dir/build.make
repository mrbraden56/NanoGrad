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
CMAKE_SOURCE_DIR = /home/bradenlock83/Projects/NanoGrad/nano_grad/csrc/cmake

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/bradenlock83/Projects/NanoGrad/build

# Include any dependencies generated for this target.
include CMakeFiles/nano_grad_backend.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/nano_grad_backend.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/nano_grad_backend.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/nano_grad_backend.dir/flags.make

# Object files for target nano_grad_backend
nano_grad_backend_OBJECTS =

# External object files for target nano_grad_backend
nano_grad_backend_EXTERNAL_OBJECTS = \
"/home/bradenlock83/Projects/NanoGrad/build/CMakeFiles/nano_grad_backend_obj.dir/home/bradenlock83/Projects/NanoGrad/nano_grad/csrc/main.cpp.o"

nano_grad_backend: CMakeFiles/nano_grad_backend_obj.dir/home/bradenlock83/Projects/NanoGrad/nano_grad/csrc/main.cpp.o
nano_grad_backend: CMakeFiles/nano_grad_backend.dir/build.make
nano_grad_backend: autograd_build/libautograd.a
nano_grad_backend: dispatcher_build/libdispatcher.so
nano_grad_backend: tensor_build/libpython_tensor.so
nano_grad_backend: /usr/lib/x86_64-linux-gnu/libpython3.10.so
nano_grad_backend: CMakeFiles/nano_grad_backend.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/bradenlock83/Projects/NanoGrad/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Linking CXX executable nano_grad_backend"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/nano_grad_backend.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/nano_grad_backend.dir/build: nano_grad_backend
.PHONY : CMakeFiles/nano_grad_backend.dir/build

CMakeFiles/nano_grad_backend.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/nano_grad_backend.dir/cmake_clean.cmake
.PHONY : CMakeFiles/nano_grad_backend.dir/clean

CMakeFiles/nano_grad_backend.dir/depend:
	cd /home/bradenlock83/Projects/NanoGrad/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/bradenlock83/Projects/NanoGrad/nano_grad/csrc/cmake /home/bradenlock83/Projects/NanoGrad/nano_grad/csrc/cmake /home/bradenlock83/Projects/NanoGrad/build /home/bradenlock83/Projects/NanoGrad/build /home/bradenlock83/Projects/NanoGrad/build/CMakeFiles/nano_grad_backend.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/nano_grad_backend.dir/depend

