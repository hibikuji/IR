# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "detector_msgs: 2 messages, 3 services")

set(MSG_I_FLAGS "-Idetector_msgs:/workspace/src/detector_msgs/msg;-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg;-Igeometry_msgs:/opt/ros/noetic/share/geometry_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(detector_msgs_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/workspace/src/detector_msgs/msg/BBox.msg" NAME_WE)
add_custom_target(_detector_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "detector_msgs" "/workspace/src/detector_msgs/msg/BBox.msg" ""
)

get_filename_component(_filename "/workspace/src/detector_msgs/msg/BBoxArray.msg" NAME_WE)
add_custom_target(_detector_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "detector_msgs" "/workspace/src/detector_msgs/msg/BBoxArray.msg" "detector_msgs/BBox:std_msgs/Header"
)

get_filename_component(_filename "/workspace/src/detector_msgs/srv/SetTransformFromBBox.srv" NAME_WE)
add_custom_target(_detector_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "detector_msgs" "/workspace/src/detector_msgs/srv/SetTransformFromBBox.srv" "detector_msgs/BBox"
)

get_filename_component(_filename "/workspace/src/detector_msgs/srv/AddTransformToManager.srv" NAME_WE)
add_custom_target(_detector_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "detector_msgs" "/workspace/src/detector_msgs/srv/AddTransformToManager.srv" "geometry_msgs/Quaternion:geometry_msgs/TransformStamped:geometry_msgs/Transform:geometry_msgs/Vector3:std_msgs/Header"
)

get_filename_component(_filename "/workspace/src/detector_msgs/srv/GetObjectDetection.srv" NAME_WE)
add_custom_target(_detector_msgs_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "detector_msgs" "/workspace/src/detector_msgs/srv/GetObjectDetection.srv" "detector_msgs/BBox:std_msgs/Header:detector_msgs/BBoxArray"
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(detector_msgs
  "/workspace/src/detector_msgs/msg/BBox.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/detector_msgs
)
_generate_msg_cpp(detector_msgs
  "/workspace/src/detector_msgs/msg/BBoxArray.msg"
  "${MSG_I_FLAGS}"
  "/workspace/src/detector_msgs/msg/BBox.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/detector_msgs
)

### Generating Services
_generate_srv_cpp(detector_msgs
  "/workspace/src/detector_msgs/srv/SetTransformFromBBox.srv"
  "${MSG_I_FLAGS}"
  "/workspace/src/detector_msgs/msg/BBox.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/detector_msgs
)
_generate_srv_cpp(detector_msgs
  "/workspace/src/detector_msgs/srv/AddTransformToManager.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/TransformStamped.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Transform.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/detector_msgs
)
_generate_srv_cpp(detector_msgs
  "/workspace/src/detector_msgs/srv/GetObjectDetection.srv"
  "${MSG_I_FLAGS}"
  "/workspace/src/detector_msgs/msg/BBox.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/workspace/src/detector_msgs/msg/BBoxArray.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/detector_msgs
)

### Generating Module File
_generate_module_cpp(detector_msgs
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/detector_msgs
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(detector_msgs_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(detector_msgs_generate_messages detector_msgs_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/workspace/src/detector_msgs/msg/BBox.msg" NAME_WE)
add_dependencies(detector_msgs_generate_messages_cpp _detector_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/workspace/src/detector_msgs/msg/BBoxArray.msg" NAME_WE)
add_dependencies(detector_msgs_generate_messages_cpp _detector_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/workspace/src/detector_msgs/srv/SetTransformFromBBox.srv" NAME_WE)
add_dependencies(detector_msgs_generate_messages_cpp _detector_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/workspace/src/detector_msgs/srv/AddTransformToManager.srv" NAME_WE)
add_dependencies(detector_msgs_generate_messages_cpp _detector_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/workspace/src/detector_msgs/srv/GetObjectDetection.srv" NAME_WE)
add_dependencies(detector_msgs_generate_messages_cpp _detector_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(detector_msgs_gencpp)
add_dependencies(detector_msgs_gencpp detector_msgs_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS detector_msgs_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(detector_msgs
  "/workspace/src/detector_msgs/msg/BBox.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/detector_msgs
)
_generate_msg_eus(detector_msgs
  "/workspace/src/detector_msgs/msg/BBoxArray.msg"
  "${MSG_I_FLAGS}"
  "/workspace/src/detector_msgs/msg/BBox.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/detector_msgs
)

### Generating Services
_generate_srv_eus(detector_msgs
  "/workspace/src/detector_msgs/srv/SetTransformFromBBox.srv"
  "${MSG_I_FLAGS}"
  "/workspace/src/detector_msgs/msg/BBox.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/detector_msgs
)
_generate_srv_eus(detector_msgs
  "/workspace/src/detector_msgs/srv/AddTransformToManager.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/TransformStamped.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Transform.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/detector_msgs
)
_generate_srv_eus(detector_msgs
  "/workspace/src/detector_msgs/srv/GetObjectDetection.srv"
  "${MSG_I_FLAGS}"
  "/workspace/src/detector_msgs/msg/BBox.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/workspace/src/detector_msgs/msg/BBoxArray.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/detector_msgs
)

### Generating Module File
_generate_module_eus(detector_msgs
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/detector_msgs
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(detector_msgs_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(detector_msgs_generate_messages detector_msgs_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/workspace/src/detector_msgs/msg/BBox.msg" NAME_WE)
add_dependencies(detector_msgs_generate_messages_eus _detector_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/workspace/src/detector_msgs/msg/BBoxArray.msg" NAME_WE)
add_dependencies(detector_msgs_generate_messages_eus _detector_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/workspace/src/detector_msgs/srv/SetTransformFromBBox.srv" NAME_WE)
add_dependencies(detector_msgs_generate_messages_eus _detector_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/workspace/src/detector_msgs/srv/AddTransformToManager.srv" NAME_WE)
add_dependencies(detector_msgs_generate_messages_eus _detector_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/workspace/src/detector_msgs/srv/GetObjectDetection.srv" NAME_WE)
add_dependencies(detector_msgs_generate_messages_eus _detector_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(detector_msgs_geneus)
add_dependencies(detector_msgs_geneus detector_msgs_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS detector_msgs_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(detector_msgs
  "/workspace/src/detector_msgs/msg/BBox.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/detector_msgs
)
_generate_msg_lisp(detector_msgs
  "/workspace/src/detector_msgs/msg/BBoxArray.msg"
  "${MSG_I_FLAGS}"
  "/workspace/src/detector_msgs/msg/BBox.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/detector_msgs
)

### Generating Services
_generate_srv_lisp(detector_msgs
  "/workspace/src/detector_msgs/srv/SetTransformFromBBox.srv"
  "${MSG_I_FLAGS}"
  "/workspace/src/detector_msgs/msg/BBox.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/detector_msgs
)
_generate_srv_lisp(detector_msgs
  "/workspace/src/detector_msgs/srv/AddTransformToManager.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/TransformStamped.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Transform.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/detector_msgs
)
_generate_srv_lisp(detector_msgs
  "/workspace/src/detector_msgs/srv/GetObjectDetection.srv"
  "${MSG_I_FLAGS}"
  "/workspace/src/detector_msgs/msg/BBox.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/workspace/src/detector_msgs/msg/BBoxArray.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/detector_msgs
)

### Generating Module File
_generate_module_lisp(detector_msgs
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/detector_msgs
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(detector_msgs_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(detector_msgs_generate_messages detector_msgs_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/workspace/src/detector_msgs/msg/BBox.msg" NAME_WE)
add_dependencies(detector_msgs_generate_messages_lisp _detector_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/workspace/src/detector_msgs/msg/BBoxArray.msg" NAME_WE)
add_dependencies(detector_msgs_generate_messages_lisp _detector_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/workspace/src/detector_msgs/srv/SetTransformFromBBox.srv" NAME_WE)
add_dependencies(detector_msgs_generate_messages_lisp _detector_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/workspace/src/detector_msgs/srv/AddTransformToManager.srv" NAME_WE)
add_dependencies(detector_msgs_generate_messages_lisp _detector_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/workspace/src/detector_msgs/srv/GetObjectDetection.srv" NAME_WE)
add_dependencies(detector_msgs_generate_messages_lisp _detector_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(detector_msgs_genlisp)
add_dependencies(detector_msgs_genlisp detector_msgs_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS detector_msgs_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(detector_msgs
  "/workspace/src/detector_msgs/msg/BBox.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/detector_msgs
)
_generate_msg_nodejs(detector_msgs
  "/workspace/src/detector_msgs/msg/BBoxArray.msg"
  "${MSG_I_FLAGS}"
  "/workspace/src/detector_msgs/msg/BBox.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/detector_msgs
)

### Generating Services
_generate_srv_nodejs(detector_msgs
  "/workspace/src/detector_msgs/srv/SetTransformFromBBox.srv"
  "${MSG_I_FLAGS}"
  "/workspace/src/detector_msgs/msg/BBox.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/detector_msgs
)
_generate_srv_nodejs(detector_msgs
  "/workspace/src/detector_msgs/srv/AddTransformToManager.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/TransformStamped.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Transform.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/detector_msgs
)
_generate_srv_nodejs(detector_msgs
  "/workspace/src/detector_msgs/srv/GetObjectDetection.srv"
  "${MSG_I_FLAGS}"
  "/workspace/src/detector_msgs/msg/BBox.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/workspace/src/detector_msgs/msg/BBoxArray.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/detector_msgs
)

### Generating Module File
_generate_module_nodejs(detector_msgs
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/detector_msgs
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(detector_msgs_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(detector_msgs_generate_messages detector_msgs_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/workspace/src/detector_msgs/msg/BBox.msg" NAME_WE)
add_dependencies(detector_msgs_generate_messages_nodejs _detector_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/workspace/src/detector_msgs/msg/BBoxArray.msg" NAME_WE)
add_dependencies(detector_msgs_generate_messages_nodejs _detector_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/workspace/src/detector_msgs/srv/SetTransformFromBBox.srv" NAME_WE)
add_dependencies(detector_msgs_generate_messages_nodejs _detector_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/workspace/src/detector_msgs/srv/AddTransformToManager.srv" NAME_WE)
add_dependencies(detector_msgs_generate_messages_nodejs _detector_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/workspace/src/detector_msgs/srv/GetObjectDetection.srv" NAME_WE)
add_dependencies(detector_msgs_generate_messages_nodejs _detector_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(detector_msgs_gennodejs)
add_dependencies(detector_msgs_gennodejs detector_msgs_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS detector_msgs_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(detector_msgs
  "/workspace/src/detector_msgs/msg/BBox.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/detector_msgs
)
_generate_msg_py(detector_msgs
  "/workspace/src/detector_msgs/msg/BBoxArray.msg"
  "${MSG_I_FLAGS}"
  "/workspace/src/detector_msgs/msg/BBox.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/detector_msgs
)

### Generating Services
_generate_srv_py(detector_msgs
  "/workspace/src/detector_msgs/srv/SetTransformFromBBox.srv"
  "${MSG_I_FLAGS}"
  "/workspace/src/detector_msgs/msg/BBox.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/detector_msgs
)
_generate_srv_py(detector_msgs
  "/workspace/src/detector_msgs/srv/AddTransformToManager.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/TransformStamped.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Transform.msg;/opt/ros/noetic/share/geometry_msgs/cmake/../msg/Vector3.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/detector_msgs
)
_generate_srv_py(detector_msgs
  "/workspace/src/detector_msgs/srv/GetObjectDetection.srv"
  "${MSG_I_FLAGS}"
  "/workspace/src/detector_msgs/msg/BBox.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg;/workspace/src/detector_msgs/msg/BBoxArray.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/detector_msgs
)

### Generating Module File
_generate_module_py(detector_msgs
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/detector_msgs
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(detector_msgs_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(detector_msgs_generate_messages detector_msgs_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/workspace/src/detector_msgs/msg/BBox.msg" NAME_WE)
add_dependencies(detector_msgs_generate_messages_py _detector_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/workspace/src/detector_msgs/msg/BBoxArray.msg" NAME_WE)
add_dependencies(detector_msgs_generate_messages_py _detector_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/workspace/src/detector_msgs/srv/SetTransformFromBBox.srv" NAME_WE)
add_dependencies(detector_msgs_generate_messages_py _detector_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/workspace/src/detector_msgs/srv/AddTransformToManager.srv" NAME_WE)
add_dependencies(detector_msgs_generate_messages_py _detector_msgs_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/workspace/src/detector_msgs/srv/GetObjectDetection.srv" NAME_WE)
add_dependencies(detector_msgs_generate_messages_py _detector_msgs_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(detector_msgs_genpy)
add_dependencies(detector_msgs_genpy detector_msgs_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS detector_msgs_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/detector_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/detector_msgs
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(detector_msgs_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()
if(TARGET geometry_msgs_generate_messages_cpp)
  add_dependencies(detector_msgs_generate_messages_cpp geometry_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/detector_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/detector_msgs
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(detector_msgs_generate_messages_eus std_msgs_generate_messages_eus)
endif()
if(TARGET geometry_msgs_generate_messages_eus)
  add_dependencies(detector_msgs_generate_messages_eus geometry_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/detector_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/detector_msgs
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(detector_msgs_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()
if(TARGET geometry_msgs_generate_messages_lisp)
  add_dependencies(detector_msgs_generate_messages_lisp geometry_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/detector_msgs)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/detector_msgs
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(detector_msgs_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()
if(TARGET geometry_msgs_generate_messages_nodejs)
  add_dependencies(detector_msgs_generate_messages_nodejs geometry_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/detector_msgs)
  install(CODE "execute_process(COMMAND \"/home/developer/.pyenv/versions/catkin_py3/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/detector_msgs\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/detector_msgs
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(detector_msgs_generate_messages_py std_msgs_generate_messages_py)
endif()
if(TARGET geometry_msgs_generate_messages_py)
  add_dependencies(detector_msgs_generate_messages_py geometry_msgs_generate_messages_py)
endif()
