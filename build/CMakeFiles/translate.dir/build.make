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
CMAKE_SOURCE_DIR = /home/fed/downward/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/fed/downward/build

# Utility rule file for translate.

# Include any custom commands dependencies for this target.
include CMakeFiles/translate.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/translate.dir/progress.make

translate: CMakeFiles/translate.dir/build.make
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Copying translator module into output directory"
	/usr/bin/cmake -E copy_directory /home/fed/downward/src/translate /home/fed/downward/build/bin/./translate
.PHONY : translate

# Rule to build all files generated by this target.
CMakeFiles/translate.dir/build: translate
.PHONY : CMakeFiles/translate.dir/build

CMakeFiles/translate.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/translate.dir/cmake_clean.cmake
.PHONY : CMakeFiles/translate.dir/clean

CMakeFiles/translate.dir/depend:
	cd /home/fed/downward/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/fed/downward/src /home/fed/downward/src /home/fed/downward/build /home/fed/downward/build /home/fed/downward/build/CMakeFiles/translate.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/translate.dir/depend
