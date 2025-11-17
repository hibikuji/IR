; Auto-generated. Do not edit!


(cl:in-package detector_msgs-srv)


;//! \htmlinclude AddTransformToManager-request.msg.html

(cl:defclass <AddTransformToManager-request> (roslisp-msg-protocol:ros-message)
  ((transform
    :reader transform
    :initarg :transform
    :type geometry_msgs-msg:TransformStamped
    :initform (cl:make-instance 'geometry_msgs-msg:TransformStamped)))
)

(cl:defclass AddTransformToManager-request (<AddTransformToManager-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <AddTransformToManager-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'AddTransformToManager-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name detector_msgs-srv:<AddTransformToManager-request> is deprecated: use detector_msgs-srv:AddTransformToManager-request instead.")))

(cl:ensure-generic-function 'transform-val :lambda-list '(m))
(cl:defmethod transform-val ((m <AddTransformToManager-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader detector_msgs-srv:transform-val is deprecated.  Use detector_msgs-srv:transform instead.")
  (transform m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <AddTransformToManager-request>) ostream)
  "Serializes a message object of type '<AddTransformToManager-request>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'transform) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <AddTransformToManager-request>) istream)
  "Deserializes a message object of type '<AddTransformToManager-request>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'transform) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<AddTransformToManager-request>)))
  "Returns string type for a service object of type '<AddTransformToManager-request>"
  "detector_msgs/AddTransformToManagerRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'AddTransformToManager-request)))
  "Returns string type for a service object of type 'AddTransformToManager-request"
  "detector_msgs/AddTransformToManagerRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<AddTransformToManager-request>)))
  "Returns md5sum for a message object of type '<AddTransformToManager-request>"
  "bf7912ca51f884d0a3b0d663503f42cd")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'AddTransformToManager-request)))
  "Returns md5sum for a message object of type 'AddTransformToManager-request"
  "bf7912ca51f884d0a3b0d663503f42cd")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<AddTransformToManager-request>)))
  "Returns full string definition for message of type '<AddTransformToManager-request>"
  (cl:format cl:nil "geometry_msgs/TransformStamped transform~%~%================================================================================~%MSG: geometry_msgs/TransformStamped~%# This expresses a transform from coordinate frame header.frame_id~%# to the coordinate frame child_frame_id~%#~%# This message is mostly used by the ~%# <a href=\"http://wiki.ros.org/tf\">tf</a> package. ~%# See its documentation for more information.~%~%Header header~%string child_frame_id # the frame id of the child frame~%Transform transform~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Transform~%# This represents the transform between two coordinate frames in free space.~%~%Vector3 translation~%Quaternion rotation~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'AddTransformToManager-request)))
  "Returns full string definition for message of type 'AddTransformToManager-request"
  (cl:format cl:nil "geometry_msgs/TransformStamped transform~%~%================================================================================~%MSG: geometry_msgs/TransformStamped~%# This expresses a transform from coordinate frame header.frame_id~%# to the coordinate frame child_frame_id~%#~%# This message is mostly used by the ~%# <a href=\"http://wiki.ros.org/tf\">tf</a> package. ~%# See its documentation for more information.~%~%Header header~%string child_frame_id # the frame id of the child frame~%Transform transform~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: geometry_msgs/Transform~%# This represents the transform between two coordinate frames in free space.~%~%Vector3 translation~%Quaternion rotation~%~%================================================================================~%MSG: geometry_msgs/Vector3~%# This represents a vector in free space. ~%# It is only meant to represent a direction. Therefore, it does not~%# make sense to apply a translation to it (e.g., when applying a ~%# generic rigid transformation to a Vector3, tf2 will only apply the~%# rotation). If you want your data to be translatable too, use the~%# geometry_msgs/Point message instead.~%~%float64 x~%float64 y~%float64 z~%================================================================================~%MSG: geometry_msgs/Quaternion~%# This represents an orientation in free space in quaternion form.~%~%float64 x~%float64 y~%float64 z~%float64 w~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <AddTransformToManager-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'transform))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <AddTransformToManager-request>))
  "Converts a ROS message object to a list"
  (cl:list 'AddTransformToManager-request
    (cl:cons ':transform (transform msg))
))
;//! \htmlinclude AddTransformToManager-response.msg.html

(cl:defclass <AddTransformToManager-response> (roslisp-msg-protocol:ros-message)
  ((result
    :reader result
    :initarg :result
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass AddTransformToManager-response (<AddTransformToManager-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <AddTransformToManager-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'AddTransformToManager-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name detector_msgs-srv:<AddTransformToManager-response> is deprecated: use detector_msgs-srv:AddTransformToManager-response instead.")))

(cl:ensure-generic-function 'result-val :lambda-list '(m))
(cl:defmethod result-val ((m <AddTransformToManager-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader detector_msgs-srv:result-val is deprecated.  Use detector_msgs-srv:result instead.")
  (result m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <AddTransformToManager-response>) ostream)
  "Serializes a message object of type '<AddTransformToManager-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'result) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <AddTransformToManager-response>) istream)
  "Deserializes a message object of type '<AddTransformToManager-response>"
    (cl:setf (cl:slot-value msg 'result) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<AddTransformToManager-response>)))
  "Returns string type for a service object of type '<AddTransformToManager-response>"
  "detector_msgs/AddTransformToManagerResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'AddTransformToManager-response)))
  "Returns string type for a service object of type 'AddTransformToManager-response"
  "detector_msgs/AddTransformToManagerResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<AddTransformToManager-response>)))
  "Returns md5sum for a message object of type '<AddTransformToManager-response>"
  "bf7912ca51f884d0a3b0d663503f42cd")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'AddTransformToManager-response)))
  "Returns md5sum for a message object of type 'AddTransformToManager-response"
  "bf7912ca51f884d0a3b0d663503f42cd")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<AddTransformToManager-response>)))
  "Returns full string definition for message of type '<AddTransformToManager-response>"
  (cl:format cl:nil "bool result~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'AddTransformToManager-response)))
  "Returns full string definition for message of type 'AddTransformToManager-response"
  (cl:format cl:nil "bool result~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <AddTransformToManager-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <AddTransformToManager-response>))
  "Converts a ROS message object to a list"
  (cl:list 'AddTransformToManager-response
    (cl:cons ':result (result msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'AddTransformToManager)))
  'AddTransformToManager-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'AddTransformToManager)))
  'AddTransformToManager-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'AddTransformToManager)))
  "Returns string type for a service object of type '<AddTransformToManager>"
  "detector_msgs/AddTransformToManager")