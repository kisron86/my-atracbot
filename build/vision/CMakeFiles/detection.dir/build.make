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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/kisron/catkin_workspace/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kisron/catkin_workspace/build

# Include any dependencies generated for this target.
include vision/CMakeFiles/detection.dir/depend.make

# Include the progress variables for this target.
include vision/CMakeFiles/detection.dir/progress.make

# Include the compile flags for this target's objects.
include vision/CMakeFiles/detection.dir/flags.make

vision/CMakeFiles/detection.dir/src/detection.cpp.o: vision/CMakeFiles/detection.dir/flags.make
vision/CMakeFiles/detection.dir/src/detection.cpp.o: /home/kisron/catkin_workspace/src/vision/src/detection.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kisron/catkin_workspace/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object vision/CMakeFiles/detection.dir/src/detection.cpp.o"
	cd /home/kisron/catkin_workspace/build/vision && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/detection.dir/src/detection.cpp.o -c /home/kisron/catkin_workspace/src/vision/src/detection.cpp

vision/CMakeFiles/detection.dir/src/detection.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/detection.dir/src/detection.cpp.i"
	cd /home/kisron/catkin_workspace/build/vision && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kisron/catkin_workspace/src/vision/src/detection.cpp > CMakeFiles/detection.dir/src/detection.cpp.i

vision/CMakeFiles/detection.dir/src/detection.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/detection.dir/src/detection.cpp.s"
	cd /home/kisron/catkin_workspace/build/vision && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kisron/catkin_workspace/src/vision/src/detection.cpp -o CMakeFiles/detection.dir/src/detection.cpp.s

vision/CMakeFiles/detection.dir/src/detection.cpp.o.requires:

.PHONY : vision/CMakeFiles/detection.dir/src/detection.cpp.o.requires

vision/CMakeFiles/detection.dir/src/detection.cpp.o.provides: vision/CMakeFiles/detection.dir/src/detection.cpp.o.requires
	$(MAKE) -f vision/CMakeFiles/detection.dir/build.make vision/CMakeFiles/detection.dir/src/detection.cpp.o.provides.build
.PHONY : vision/CMakeFiles/detection.dir/src/detection.cpp.o.provides

vision/CMakeFiles/detection.dir/src/detection.cpp.o.provides.build: vision/CMakeFiles/detection.dir/src/detection.cpp.o


# Object files for target detection
detection_OBJECTS = \
"CMakeFiles/detection.dir/src/detection.cpp.o"

# External object files for target detection
detection_EXTERNAL_OBJECTS =

/home/kisron/catkin_workspace/devel/lib/vision/detection: vision/CMakeFiles/detection.dir/src/detection.cpp.o
/home/kisron/catkin_workspace/devel/lib/vision/detection: vision/CMakeFiles/detection.dir/build.make
/home/kisron/catkin_workspace/devel/lib/vision/detection: /home/kisron/catkin_workspace/devel/lib/libmy_roscpp_library.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /opt/ros/melodic/lib/libimage_transport.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /opt/ros/melodic/lib/libmessage_filters.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /opt/ros/melodic/lib/libclass_loader.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/libPocoFoundation.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libdl.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /opt/ros/melodic/lib/libroscpp.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /opt/ros/melodic/lib/librosconsole.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /opt/ros/melodic/lib/librosconsole_log4cxx.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /opt/ros/melodic/lib/librosconsole_backend_interface.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/liblog4cxx.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libboost_regex.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /opt/ros/melodic/lib/libxmlrpcpp.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /opt/ros/melodic/lib/libroslib.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /opt/ros/melodic/lib/librospack.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libpython2.7.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libboost_program_options.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libtinyxml2.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /opt/ros/melodic/lib/libroscpp_serialization.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /opt/ros/melodic/lib/librostime.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /opt/ros/melodic/lib/libcpp_common.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libboost_system.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libboost_thread.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libpthread.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libconsole_bridge.so.0.4
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/local/lib/libopencv_core.a
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/local/lib/libopencv_flann.a
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/local/lib/libopencv_imgproc.a
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/local/lib/libopencv_highgui.a
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/local/lib/libopencv_features2d.a
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/local/lib/libopencv_calib3d.a
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/local/lib/libopencv_ml.a
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/local/lib/libopencv_video.a
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/local/lib/libopencv_legacy.a
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/local/lib/libopencv_objdetect.a
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/local/lib/libopencv_photo.a
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/local/lib/libopencv_gpu.a
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/local/lib/libopencv_videostab.a
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/local/lib/libopencv_ocl.a
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/local/lib/libopencv_superres.a
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/local/lib/libopencv_nonfree.a
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/local/lib/libopencv_stitching.a
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/local/lib/libopencv_contrib.a
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/local/lib/libopencv_nonfree.a
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/local/lib/libopencv_gpu.a
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/local/lib/libopencv_legacy.a
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/local/lib/libopencv_photo.a
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/local/lib/libopencv_ocl.a
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/local/lib/libopencv_calib3d.a
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/local/lib/libopencv_features2d.a
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/local/lib/libopencv_flann.a
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/local/lib/libopencv_ml.a
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/local/lib/libopencv_video.a
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/local/lib/libopencv_objdetect.a
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/local/lib/libopencv_highgui.a
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/local/lib/libopencv_imgproc.a
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/local/lib/libopencv_core.a
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/local/share/OpenCV/3rdparty/lib/liblibjasper.a
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libjpeg.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libpng.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libtiff.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libImath.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libIlmImf.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libIex.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libHalf.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libIlmThread.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libjpeg.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libpng.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libtiff.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libImath.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libIlmImf.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libIex.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libHalf.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libIlmThread.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libz.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libQtOpenGL.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libQtGui.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libQtTest.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libQtCore.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libGL.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libGLU.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libGL.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: /usr/lib/x86_64-linux-gnu/libGLU.so
/home/kisron/catkin_workspace/devel/lib/vision/detection: vision/CMakeFiles/detection.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/kisron/catkin_workspace/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable /home/kisron/catkin_workspace/devel/lib/vision/detection"
	cd /home/kisron/catkin_workspace/build/vision && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/detection.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
vision/CMakeFiles/detection.dir/build: /home/kisron/catkin_workspace/devel/lib/vision/detection

.PHONY : vision/CMakeFiles/detection.dir/build

vision/CMakeFiles/detection.dir/requires: vision/CMakeFiles/detection.dir/src/detection.cpp.o.requires

.PHONY : vision/CMakeFiles/detection.dir/requires

vision/CMakeFiles/detection.dir/clean:
	cd /home/kisron/catkin_workspace/build/vision && $(CMAKE_COMMAND) -P CMakeFiles/detection.dir/cmake_clean.cmake
.PHONY : vision/CMakeFiles/detection.dir/clean

vision/CMakeFiles/detection.dir/depend:
	cd /home/kisron/catkin_workspace/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kisron/catkin_workspace/src /home/kisron/catkin_workspace/src/vision /home/kisron/catkin_workspace/build /home/kisron/catkin_workspace/build/vision /home/kisron/catkin_workspace/build/vision/CMakeFiles/detection.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : vision/CMakeFiles/detection.dir/depend

