# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/boris/Documenti/Libraries/Optimization

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/boris/Documenti/Libraries/Optimization/build

# Include any dependencies generated for this target.
include CMakeFiles/my_DE.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/my_DE.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/my_DE.dir/flags.make

CMakeFiles/my_DE.dir/src/differential_evolution.c.o: CMakeFiles/my_DE.dir/flags.make
CMakeFiles/my_DE.dir/src/differential_evolution.c.o: ../src/differential_evolution.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/boris/Documenti/Libraries/Optimization/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/my_DE.dir/src/differential_evolution.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/my_DE.dir/src/differential_evolution.c.o   -c /home/boris/Documenti/Libraries/Optimization/src/differential_evolution.c

CMakeFiles/my_DE.dir/src/differential_evolution.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/my_DE.dir/src/differential_evolution.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/boris/Documenti/Libraries/Optimization/src/differential_evolution.c > CMakeFiles/my_DE.dir/src/differential_evolution.c.i

CMakeFiles/my_DE.dir/src/differential_evolution.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/my_DE.dir/src/differential_evolution.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/boris/Documenti/Libraries/Optimization/src/differential_evolution.c -o CMakeFiles/my_DE.dir/src/differential_evolution.c.s

CMakeFiles/my_DE.dir/src/differential_evolution.c.o.requires:

.PHONY : CMakeFiles/my_DE.dir/src/differential_evolution.c.o.requires

CMakeFiles/my_DE.dir/src/differential_evolution.c.o.provides: CMakeFiles/my_DE.dir/src/differential_evolution.c.o.requires
	$(MAKE) -f CMakeFiles/my_DE.dir/build.make CMakeFiles/my_DE.dir/src/differential_evolution.c.o.provides.build
.PHONY : CMakeFiles/my_DE.dir/src/differential_evolution.c.o.provides

CMakeFiles/my_DE.dir/src/differential_evolution.c.o.provides.build: CMakeFiles/my_DE.dir/src/differential_evolution.c.o


# Object files for target my_DE
my_DE_OBJECTS = \
"CMakeFiles/my_DE.dir/src/differential_evolution.c.o"

# External object files for target my_DE
my_DE_EXTERNAL_OBJECTS =

../lib/libmy_DE.a: CMakeFiles/my_DE.dir/src/differential_evolution.c.o
../lib/libmy_DE.a: CMakeFiles/my_DE.dir/build.make
../lib/libmy_DE.a: CMakeFiles/my_DE.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/boris/Documenti/Libraries/Optimization/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C static library ../lib/libmy_DE.a"
	$(CMAKE_COMMAND) -P CMakeFiles/my_DE.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/my_DE.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/my_DE.dir/build: ../lib/libmy_DE.a

.PHONY : CMakeFiles/my_DE.dir/build

CMakeFiles/my_DE.dir/requires: CMakeFiles/my_DE.dir/src/differential_evolution.c.o.requires

.PHONY : CMakeFiles/my_DE.dir/requires

CMakeFiles/my_DE.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/my_DE.dir/cmake_clean.cmake
.PHONY : CMakeFiles/my_DE.dir/clean

CMakeFiles/my_DE.dir/depend:
	cd /home/boris/Documenti/Libraries/Optimization/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/boris/Documenti/Libraries/Optimization /home/boris/Documenti/Libraries/Optimization /home/boris/Documenti/Libraries/Optimization/build /home/boris/Documenti/Libraries/Optimization/build /home/boris/Documenti/Libraries/Optimization/build/CMakeFiles/my_DE.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/my_DE.dir/depend
