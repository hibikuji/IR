execute_process(COMMAND "/workspace/build/wrs_algorithm/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/workspace/build/wrs_algorithm/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
