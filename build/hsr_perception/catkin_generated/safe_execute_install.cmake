execute_process(COMMAND "/workspace/build/hsr_perception/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/workspace/build/hsr_perception/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
