# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/clion-2018.1.3/bin/cmake/bin/cmake

# The command to remove a file.
RM = /opt/clion-2018.1.3/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lvfx/prog/opp_labs/lab5

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lvfx/prog/opp_labs/lab5/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/lab5.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/lab5.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/lab5.dir/flags.make

CMakeFiles/lab5.dir/lab5.c.o: CMakeFiles/lab5.dir/flags.make
CMakeFiles/lab5.dir/lab5.c.o: ../lab5.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lvfx/prog/opp_labs/lab5/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/lab5.dir/lab5.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/lab5.dir/lab5.c.o   -c /home/lvfx/prog/opp_labs/lab5/lab5.c

CMakeFiles/lab5.dir/lab5.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/lab5.dir/lab5.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/lvfx/prog/opp_labs/lab5/lab5.c > CMakeFiles/lab5.dir/lab5.c.i

CMakeFiles/lab5.dir/lab5.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/lab5.dir/lab5.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/lvfx/prog/opp_labs/lab5/lab5.c -o CMakeFiles/lab5.dir/lab5.c.s

CMakeFiles/lab5.dir/lab5.c.o.requires:

.PHONY : CMakeFiles/lab5.dir/lab5.c.o.requires

CMakeFiles/lab5.dir/lab5.c.o.provides: CMakeFiles/lab5.dir/lab5.c.o.requires
	$(MAKE) -f CMakeFiles/lab5.dir/build.make CMakeFiles/lab5.dir/lab5.c.o.provides.build
.PHONY : CMakeFiles/lab5.dir/lab5.c.o.provides

CMakeFiles/lab5.dir/lab5.c.o.provides.build: CMakeFiles/lab5.dir/lab5.c.o


# Object files for target lab5
lab5_OBJECTS = \
"CMakeFiles/lab5.dir/lab5.c.o"

# External object files for target lab5
lab5_EXTERNAL_OBJECTS =

lab5: CMakeFiles/lab5.dir/lab5.c.o
lab5: CMakeFiles/lab5.dir/build.make
lab5: CMakeFiles/lab5.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lvfx/prog/opp_labs/lab5/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable lab5"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/lab5.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/lab5.dir/build: lab5

.PHONY : CMakeFiles/lab5.dir/build

CMakeFiles/lab5.dir/requires: CMakeFiles/lab5.dir/lab5.c.o.requires

.PHONY : CMakeFiles/lab5.dir/requires

CMakeFiles/lab5.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/lab5.dir/cmake_clean.cmake
.PHONY : CMakeFiles/lab5.dir/clean

CMakeFiles/lab5.dir/depend:
	cd /home/lvfx/prog/opp_labs/lab5/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lvfx/prog/opp_labs/lab5 /home/lvfx/prog/opp_labs/lab5 /home/lvfx/prog/opp_labs/lab5/cmake-build-debug /home/lvfx/prog/opp_labs/lab5/cmake-build-debug /home/lvfx/prog/opp_labs/lab5/cmake-build-debug/CMakeFiles/lab5.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/lab5.dir/depend

