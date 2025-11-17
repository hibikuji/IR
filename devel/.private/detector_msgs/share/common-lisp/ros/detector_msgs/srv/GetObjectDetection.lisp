; Auto-generated. Do not edit!


(cl:in-package detector_msgs-srv)


;//! \htmlinclude GetObjectDetection-request.msg.html

(cl:defclass <GetObjectDetection-request> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (type
    :reader type
    :initarg :type
    :type cl:string
    :initform ""))
)

(cl:defclass GetObjectDetection-request (<GetObjectDetection-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetObjectDetection-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetObjectDetection-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name detector_msgs-srv:<GetObjectDetection-request> is deprecated: use detector_msgs-srv:GetObjectDetection-request instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <GetObjectDetection-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader detector_msgs-srv:header-val is deprecated.  Use detector_msgs-srv:header instead.")
  (header m))

(cl:ensure-generic-function 'type-val :lambda-list '(m))
(cl:defmethod type-val ((m <GetObjectDetection-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader detector_msgs-srv:type-val is deprecated.  Use detector_msgs-srv:type instead.")
  (type m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetObjectDetection-request>) ostream)
  "Serializes a message object of type '<GetObjectDetection-request>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'type))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'type))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetObjectDetection-request>) istream)
  "Deserializes a message object of type '<GetObjectDetection-request>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'type) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'type) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetObjectDetection-request>)))
  "Returns string type for a service object of type '<GetObjectDetection-request>"
  "detector_msgs/GetObjectDetectionRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetObjectDetection-request)))
  "Returns string type for a service object of type 'GetObjectDetection-request"
  "detector_msgs/GetObjectDetectionRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetObjectDetection-request>)))
  "Returns md5sum for a message object of type '<GetObjectDetection-request>"
  "44386689b8f2bacd7dd66f142a8c65bc")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetObjectDetection-request)))
  "Returns md5sum for a message object of type 'GetObjectDetection-request"
  "44386689b8f2bacd7dd66f142a8c65bc")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetObjectDetection-request>)))
  "Returns full string definition for message of type '<GetObjectDetection-request>"
  (cl:format cl:nil "std_msgs/Header header~%string type~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetObjectDetection-request)))
  "Returns full string definition for message of type 'GetObjectDetection-request"
  (cl:format cl:nil "std_msgs/Header header~%string type~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetObjectDetection-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     4 (cl:length (cl:slot-value msg 'type))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetObjectDetection-request>))
  "Converts a ROS message object to a list"
  (cl:list 'GetObjectDetection-request
    (cl:cons ':header (header msg))
    (cl:cons ':type (type msg))
))
;//! \htmlinclude GetObjectDetection-response.msg.html

(cl:defclass <GetObjectDetection-response> (roslisp-msg-protocol:ros-message)
  ((successed
    :reader successed
    :initarg :successed
    :type cl:boolean
    :initform cl:nil)
   (error_code
    :reader error_code
    :initarg :error_code
    :type cl:string
    :initform "")
   (error_string
    :reader error_string
    :initarg :error_string
    :type cl:string
    :initform "")
   (bboxes
    :reader bboxes
    :initarg :bboxes
    :type detector_msgs-msg:BBoxArray
    :initform (cl:make-instance 'detector_msgs-msg:BBoxArray)))
)

(cl:defclass GetObjectDetection-response (<GetObjectDetection-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <GetObjectDetection-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'GetObjectDetection-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name detector_msgs-srv:<GetObjectDetection-response> is deprecated: use detector_msgs-srv:GetObjectDetection-response instead.")))

(cl:ensure-generic-function 'successed-val :lambda-list '(m))
(cl:defmethod successed-val ((m <GetObjectDetection-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader detector_msgs-srv:successed-val is deprecated.  Use detector_msgs-srv:successed instead.")
  (successed m))

(cl:ensure-generic-function 'error_code-val :lambda-list '(m))
(cl:defmethod error_code-val ((m <GetObjectDetection-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader detector_msgs-srv:error_code-val is deprecated.  Use detector_msgs-srv:error_code instead.")
  (error_code m))

(cl:ensure-generic-function 'error_string-val :lambda-list '(m))
(cl:defmethod error_string-val ((m <GetObjectDetection-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader detector_msgs-srv:error_string-val is deprecated.  Use detector_msgs-srv:error_string instead.")
  (error_string m))

(cl:ensure-generic-function 'bboxes-val :lambda-list '(m))
(cl:defmethod bboxes-val ((m <GetObjectDetection-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader detector_msgs-srv:bboxes-val is deprecated.  Use detector_msgs-srv:bboxes instead.")
  (bboxes m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <GetObjectDetection-response>) ostream)
  "Serializes a message object of type '<GetObjectDetection-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'successed) 1 0)) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'error_code))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'error_code))
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'error_string))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'error_string))
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'bboxes) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <GetObjectDetection-response>) istream)
  "Deserializes a message object of type '<GetObjectDetection-response>"
    (cl:setf (cl:slot-value msg 'successed) (cl:not (cl:zerop (cl:read-byte istream))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'error_code) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'error_code) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'error_string) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'error_string) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'bboxes) istream)
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<GetObjectDetection-response>)))
  "Returns string type for a service object of type '<GetObjectDetection-response>"
  "detector_msgs/GetObjectDetectionResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetObjectDetection-response)))
  "Returns string type for a service object of type 'GetObjectDetection-response"
  "detector_msgs/GetObjectDetectionResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<GetObjectDetection-response>)))
  "Returns md5sum for a message object of type '<GetObjectDetection-response>"
  "44386689b8f2bacd7dd66f142a8c65bc")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'GetObjectDetection-response)))
  "Returns md5sum for a message object of type 'GetObjectDetection-response"
  "44386689b8f2bacd7dd66f142a8c65bc")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<GetObjectDetection-response>)))
  "Returns full string definition for message of type '<GetObjectDetection-response>"
  (cl:format cl:nil "bool successed~%string error_code~%string error_string~%detector_msgs/BBoxArray bboxes~%~%~%================================================================================~%MSG: detector_msgs/BBoxArray~%std_msgs/Header header~%detector_msgs/BBox[] bboxes~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: detector_msgs/BBox~%string label~%float64 score~%int64 x~%int64 y~%int64 w~%int64 h~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'GetObjectDetection-response)))
  "Returns full string definition for message of type 'GetObjectDetection-response"
  (cl:format cl:nil "bool successed~%string error_code~%string error_string~%detector_msgs/BBoxArray bboxes~%~%~%================================================================================~%MSG: detector_msgs/BBoxArray~%std_msgs/Header header~%detector_msgs/BBox[] bboxes~%~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: detector_msgs/BBox~%string label~%float64 score~%int64 x~%int64 y~%int64 w~%int64 h~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <GetObjectDetection-response>))
  (cl:+ 0
     1
     4 (cl:length (cl:slot-value msg 'error_code))
     4 (cl:length (cl:slot-value msg 'error_string))
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'bboxes))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <GetObjectDetection-response>))
  "Converts a ROS message object to a list"
  (cl:list 'GetObjectDetection-response
    (cl:cons ':successed (successed msg))
    (cl:cons ':error_code (error_code msg))
    (cl:cons ':error_string (error_string msg))
    (cl:cons ':bboxes (bboxes msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'GetObjectDetection)))
  'GetObjectDetection-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'GetObjectDetection)))
  'GetObjectDetection-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'GetObjectDetection)))
  "Returns string type for a service object of type '<GetObjectDetection>"
  "detector_msgs/GetObjectDetection")