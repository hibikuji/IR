; Auto-generated. Do not edit!


(cl:in-package detector_msgs-srv)


;//! \htmlinclude SetTransformFromBBox-request.msg.html

(cl:defclass <SetTransformFromBBox-request> (roslisp-msg-protocol:ros-message)
  ((bbox
    :reader bbox
    :initarg :bbox
    :type detector_msgs-msg:BBox
    :initform (cl:make-instance 'detector_msgs-msg:BBox))
   (frame
    :reader frame
    :initarg :frame
    :type cl:string
    :initform ""))
)

(cl:defclass SetTransformFromBBox-request (<SetTransformFromBBox-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetTransformFromBBox-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetTransformFromBBox-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name detector_msgs-srv:<SetTransformFromBBox-request> is deprecated: use detector_msgs-srv:SetTransformFromBBox-request instead.")))

(cl:ensure-generic-function 'bbox-val :lambda-list '(m))
(cl:defmethod bbox-val ((m <SetTransformFromBBox-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader detector_msgs-srv:bbox-val is deprecated.  Use detector_msgs-srv:bbox instead.")
  (bbox m))

(cl:ensure-generic-function 'frame-val :lambda-list '(m))
(cl:defmethod frame-val ((m <SetTransformFromBBox-request>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader detector_msgs-srv:frame-val is deprecated.  Use detector_msgs-srv:frame instead.")
  (frame m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetTransformFromBBox-request>) ostream)
  "Serializes a message object of type '<SetTransformFromBBox-request>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'bbox) ostream)
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'frame))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'frame))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetTransformFromBBox-request>) istream)
  "Deserializes a message object of type '<SetTransformFromBBox-request>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'bbox) istream)
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'frame) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'frame) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetTransformFromBBox-request>)))
  "Returns string type for a service object of type '<SetTransformFromBBox-request>"
  "detector_msgs/SetTransformFromBBoxRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetTransformFromBBox-request)))
  "Returns string type for a service object of type 'SetTransformFromBBox-request"
  "detector_msgs/SetTransformFromBBoxRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetTransformFromBBox-request>)))
  "Returns md5sum for a message object of type '<SetTransformFromBBox-request>"
  "cf10ef4f67b53145111605419d2c8caa")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetTransformFromBBox-request)))
  "Returns md5sum for a message object of type 'SetTransformFromBBox-request"
  "cf10ef4f67b53145111605419d2c8caa")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetTransformFromBBox-request>)))
  "Returns full string definition for message of type '<SetTransformFromBBox-request>"
  (cl:format cl:nil "detector_msgs/BBox bbox~%string frame~%~%================================================================================~%MSG: detector_msgs/BBox~%string label~%float64 score~%int64 x~%int64 y~%int64 w~%int64 h~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetTransformFromBBox-request)))
  "Returns full string definition for message of type 'SetTransformFromBBox-request"
  (cl:format cl:nil "detector_msgs/BBox bbox~%string frame~%~%================================================================================~%MSG: detector_msgs/BBox~%string label~%float64 score~%int64 x~%int64 y~%int64 w~%int64 h~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetTransformFromBBox-request>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'bbox))
     4 (cl:length (cl:slot-value msg 'frame))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetTransformFromBBox-request>))
  "Converts a ROS message object to a list"
  (cl:list 'SetTransformFromBBox-request
    (cl:cons ':bbox (bbox msg))
    (cl:cons ':frame (frame msg))
))
;//! \htmlinclude SetTransformFromBBox-response.msg.html

(cl:defclass <SetTransformFromBBox-response> (roslisp-msg-protocol:ros-message)
  ((result
    :reader result
    :initarg :result
    :type cl:boolean
    :initform cl:nil))
)

(cl:defclass SetTransformFromBBox-response (<SetTransformFromBBox-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetTransformFromBBox-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetTransformFromBBox-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name detector_msgs-srv:<SetTransformFromBBox-response> is deprecated: use detector_msgs-srv:SetTransformFromBBox-response instead.")))

(cl:ensure-generic-function 'result-val :lambda-list '(m))
(cl:defmethod result-val ((m <SetTransformFromBBox-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader detector_msgs-srv:result-val is deprecated.  Use detector_msgs-srv:result instead.")
  (result m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetTransformFromBBox-response>) ostream)
  "Serializes a message object of type '<SetTransformFromBBox-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:if (cl:slot-value msg 'result) 1 0)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetTransformFromBBox-response>) istream)
  "Deserializes a message object of type '<SetTransformFromBBox-response>"
    (cl:setf (cl:slot-value msg 'result) (cl:not (cl:zerop (cl:read-byte istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetTransformFromBBox-response>)))
  "Returns string type for a service object of type '<SetTransformFromBBox-response>"
  "detector_msgs/SetTransformFromBBoxResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetTransformFromBBox-response)))
  "Returns string type for a service object of type 'SetTransformFromBBox-response"
  "detector_msgs/SetTransformFromBBoxResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetTransformFromBBox-response>)))
  "Returns md5sum for a message object of type '<SetTransformFromBBox-response>"
  "cf10ef4f67b53145111605419d2c8caa")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetTransformFromBBox-response)))
  "Returns md5sum for a message object of type 'SetTransformFromBBox-response"
  "cf10ef4f67b53145111605419d2c8caa")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetTransformFromBBox-response>)))
  "Returns full string definition for message of type '<SetTransformFromBBox-response>"
  (cl:format cl:nil "bool result~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetTransformFromBBox-response)))
  "Returns full string definition for message of type 'SetTransformFromBBox-response"
  (cl:format cl:nil "bool result~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetTransformFromBBox-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetTransformFromBBox-response>))
  "Converts a ROS message object to a list"
  (cl:list 'SetTransformFromBBox-response
    (cl:cons ':result (result msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'SetTransformFromBBox)))
  'SetTransformFromBBox-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'SetTransformFromBBox)))
  'SetTransformFromBBox-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetTransformFromBBox)))
  "Returns string type for a service object of type '<SetTransformFromBBox>"
  "detector_msgs/SetTransformFromBBox")