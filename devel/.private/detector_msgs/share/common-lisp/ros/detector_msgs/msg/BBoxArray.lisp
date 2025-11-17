; Auto-generated. Do not edit!


(cl:in-package detector_msgs-msg)


;//! \htmlinclude BBoxArray.msg.html

(cl:defclass <BBoxArray> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (bboxes
    :reader bboxes
    :initarg :bboxes
    :type (cl:vector detector_msgs-msg:BBox)
   :initform (cl:make-array 0 :element-type 'detector_msgs-msg:BBox :initial-element (cl:make-instance 'detector_msgs-msg:BBox))))
)

(cl:defclass BBoxArray (<BBoxArray>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <BBoxArray>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'BBoxArray)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name detector_msgs-msg:<BBoxArray> is deprecated: use detector_msgs-msg:BBoxArray instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <BBoxArray>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader detector_msgs-msg:header-val is deprecated.  Use detector_msgs-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'bboxes-val :lambda-list '(m))
(cl:defmethod bboxes-val ((m <BBoxArray>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader detector_msgs-msg:bboxes-val is deprecated.  Use detector_msgs-msg:bboxes instead.")
  (bboxes m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <BBoxArray>) ostream)
  "Serializes a message object of type '<BBoxArray>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'bboxes))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'bboxes))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <BBoxArray>) istream)
  "Deserializes a message object of type '<BBoxArray>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'bboxes) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'bboxes)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'detector_msgs-msg:BBox))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<BBoxArray>)))
  "Returns string type for a message object of type '<BBoxArray>"
  "detector_msgs/BBoxArray")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'BBoxArray)))
  "Returns string type for a message object of type 'BBoxArray"
  "detector_msgs/BBoxArray")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<BBoxArray>)))
  "Returns md5sum for a message object of type '<BBoxArray>"
  "7ed748f248b539ee6ad03e80ae91120d")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'BBoxArray)))
  "Returns md5sum for a message object of type 'BBoxArray"
  "7ed748f248b539ee6ad03e80ae91120d")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<BBoxArray>)))
  "Returns full string definition for message of type '<BBoxArray>"
  (cl:format cl:nil "std_msgs/Header header~%detector_msgs/BBox[] bboxes~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: detector_msgs/BBox~%string label~%float64 score~%int64 x~%int64 y~%int64 w~%int64 h~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'BBoxArray)))
  "Returns full string definition for message of type 'BBoxArray"
  (cl:format cl:nil "std_msgs/Header header~%detector_msgs/BBox[] bboxes~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: detector_msgs/BBox~%string label~%float64 score~%int64 x~%int64 y~%int64 w~%int64 h~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <BBoxArray>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'bboxes) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <BBoxArray>))
  "Converts a ROS message object to a list"
  (cl:list 'BBoxArray
    (cl:cons ':header (header msg))
    (cl:cons ':bboxes (bboxes msg))
))
