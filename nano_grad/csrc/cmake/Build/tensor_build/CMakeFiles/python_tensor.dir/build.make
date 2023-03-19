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
CMAKE_BINARY_DIR = /home/bradenlock83/Projects/NanoGrad/nano_grad/csrc/cmake/Build

# Include any dependencies generated for this target.
include tensor_build/CMakeFiles/python_tensor.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tensor_build/CMakeFiles/python_tensor.dir/compiler_depend.make

# Include the progress variables for this target.
include tensor_build/CMakeFiles/python_tensor.dir/progress.make

# Include the compile flags for this target's objects.
include tensor_build/CMakeFiles/python_tensor.dir/flags.make

tensor_build/CMakeFiles/python_tensor.dir/python_tensor.cpp.o: tensor_build/CMakeFiles/python_tensor.dir/flags.make
tensor_build/CMakeFiles/python_tensor.dir/python_tensor.cpp.o: /home/bradenlock83/Projects/NanoGrad/nano_grad/csrc/Tensor/python_tensor.cpp
tensor_build/CMakeFiles/python_tensor.dir/python_tensor.cpp.o: tensor_build/CMakeFiles/python_tensor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/bradenlock83/Projects/NanoGrad/nano_grad/csrc/cmake/Build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tensor_build/CMakeFiles/python_tensor.dir/python_tensor.cpp.o"
	cd /home/bradenlock83/Projects/NanoGrad/nano_grad/csrc/cmake/Build/tensor_build && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tensor_build/CMakeFiles/python_tensor.dir/python_tensor.cpp.o -MF CMakeFiles/python_tensor.dir/python_tensor.cpp.o.d -o CMakeFiles/python_tensor.dir/python_tensor.cpp.o -c /home/bradenlock83/Projects/NanoGrad/nano_grad/csrc/Tensor/python_tensor.cpp

tensor_build/CMakeFiles/python_tensor.dir/python_tensor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/python_tensor.dir/python_tensor.cpp.i"
	cd /home/bradenlock83/Projects/NanoGrad/nano_grad/csrc/cmake/Build/tensor_build && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/bradenlock83/Projects/NanoGrad/nano_grad/csrc/Tensor/python_tensor.cpp > CMakeFiles/python_tensor.dir/python_tensor.cpp.i

tensor_build/CMakeFiles/python_tensor.dir/python_tensor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/python_tensor.dir/python_tensor.cpp.s"
	cd /home/bradenlock83/Projects/NanoGrad/nano_grad/csrc/cmake/Build/tensor_build && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/bradenlock83/Projects/NanoGrad/nano_grad/csrc/Tensor/python_tensor.cpp -o CMakeFiles/python_tensor.dir/python_tensor.cpp.s

# Object files for target python_tensor
python_tensor_OBJECTS = \
"CMakeFiles/python_tensor.dir/python_tensor.cpp.o"

# External object files for target python_tensor
python_tensor_EXTERNAL_OBJECTS =

tensor_build/libpython_tensor.so: tensor_build/CMakeFiles/python_tensor.dir/python_tensor.cpp.o
tensor_build/libpython_tensor.so: tensor_build/CMakeFiles/python_tensor.dir/build.make
tensor_build/libpython_tensor.so: /usr/lib/x86_64-linux-gnu/libpython3.10.so
tensor_build/libpython_tensor.so: tensor_build/CMakeFiles/python_tensor.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/bradenlock83/Projects/NanoGrad/nano_grad/csrc/cmake/Build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libpython_tensor.so"
	cd /home/bradenlock83/Projects/NanoGrad/nano_grad/csrc/cmake/Build/tensor_build && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/python_tensor.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tensor_build/CMakeFiles/python_tensor.dir/build: tensor_build/libpython_tensor.so
.PHONY : tensor_build/CMakeFiles/python_tensor.dir/build

tensor_build/CMakeFiles/python_tensor.dir/clean:
	cd /home/bradenlock83/Projects/NanoGrad/nano_grad/csrc/cmake/Build/tensor_build && $(CMAKE_COMMAND) -P CMakeFiles/python_tensor.dir/cmake_clean.cmake
.PHONY : tensor_build/CMakeFiles/python_tensor.dir/clean

tensor_build/CMakeFiles/python_tensor.dir/depend:
	cd /home/bradenlock83/Projects/NanoGrad/nano_grad/csrc/cmake/Build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/bradenlock83/Projects/NanoGrad/nano_grad/csrc/cmake /home/bradenlock83/Projects/NanoGrad/nano_grad/csrc/Tensor /home/bradenlock83/Projects/NanoGrad/nano_grad/csrc/cmake/Build /home/bradenlock83/Projects/NanoGrad/nano_grad/csrc/cmake/Build/tensor_build /home/bradenlock83/Projects/NanoGrad/nano_grad/csrc/cmake/Build/tensor_build/CMakeFiles/python_tensor.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tensor_build/CMakeFiles/python_tensor.dir/depend

