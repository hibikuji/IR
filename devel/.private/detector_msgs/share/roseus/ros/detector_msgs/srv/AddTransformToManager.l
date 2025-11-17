;; Auto-generated. Do not edit!


(when (boundp 'detector_msgs::AddTransformToManager)
  (if (not (find-package "DETECTOR_MSGS"))
    (make-package "DETECTOR_MSGS"))
  (shadow 'AddTransformToManager (find-package "DETECTOR_MSGS")))
(unless (find-package "DETECTOR_MSGS::ADDTRANSFORMTOMANAGER")
  (make-package "DETECTOR_MSGS::ADDTRANSFORMTOMANAGER"))
(unless (find-package "DETECTOR_MSGS::ADDTRANSFORMTOMANAGERREQUEST")
  (make-package "DETECTOR_MSGS::ADDTRANSFORMTOMANAGERREQUEST"))
(unless (find-package "DETECTOR_MSGS::ADDTRANSFORMTOMANAGERRESPONSE")
  (make-package "DETECTOR_MSGS::ADDTRANSFORMTOMANAGERRESPONSE"))

(in-package "ROS")

(if (not (find-package "GEOMETRY_MSGS"))
  (ros::roseus-add-msgs "geometry_msgs"))




(defclass detector_msgs::AddTransformToManagerRequest
  :super ros::object
  :slots (_transform ))

(defmethod detector_msgs::AddTransformToManagerRequest
  (:init
   (&key
    ((:transform __transform) (instance geometry_msgs::TransformStamped :init))
    )
   (send-super :init)
   (setq _transform __transform)
   self)
  (:transform
   (&rest __transform)
   (if (keywordp (car __transform))
       (send* _transform __transform)
     (progn
       (if __transform (setq _transform (car __transform)))
       _transform)))
  (:serialization-length
   ()
   (+
    ;; geometry_msgs/TransformStamped _transform
    (send _transform :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; geometry_msgs/TransformStamped _transform
       (send _transform :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; geometry_msgs/TransformStamped _transform
     (send _transform :deserialize buf ptr-) (incf ptr- (send _transform :serialization-length))
   ;;
   self)
  )

(defclass detector_msgs::AddTransformToManagerResponse
  :super ros::object
  :slots (_result ))

(defmethod detector_msgs::AddTransformToManagerResponse
  (:init
   (&key
    ((:result __result) nil)
    )
   (send-super :init)
   (setq _result __result)
   self)
  (:result
   (&optional (__result :null))
   (if (not (eq __result :null)) (setq _result __result)) _result)
  (:serialization-length
   ()
   (+
    ;; bool _result
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _result
       (if _result (write-byte -1 s) (write-byte 0 s))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _result
     (setq _result (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;;
   self)
  )

(defclass detector_msgs::AddTransformToManager
  :super ros::object
  :slots ())

(setf (get detector_msgs::AddTransformToManager :md5sum-) "bf7912ca51f884d0a3b0d663503f42cd")
(setf (get detector_msgs::AddTransformToManager :datatype-) "detector_msgs/AddTransformToManager")
(setf (get detector_msgs::AddTransformToManager :request) detector_msgs::AddTransformToManagerRequest)
(setf (get detector_msgs::AddTransformToManager :response) detector_msgs::AddTransformToManagerResponse)

(defmethod detector_msgs::AddTransformToManagerRequest
  (:response () (instance detector_msgs::AddTransformToManagerResponse :init)))

(setf (get detector_msgs::AddTransformToManagerRequest :md5sum-) "bf7912ca51f884d0a3b0d663503f42cd")
(setf (get detector_msgs::AddTransformToManagerRequest :datatype-) "detector_msgs/AddTransformToManagerRequest")
(setf (get detector_msgs::AddTransformToManagerRequest :definition-)
      "geometry_msgs/TransformStamped transform

================================================================================
MSG: geometry_msgs/TransformStamped
# This expresses a transform from coordinate frame header.frame_id
# to the coordinate frame child_frame_id
#
# This message is mostly used by the 
# <a href=\"http://wiki.ros.org/tf\">tf</a> package. 
# See its documentation for more information.

Header header
string child_frame_id # the frame id of the child frame
Transform transform

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

================================================================================
MSG: geometry_msgs/Transform
# This represents the transform between two coordinate frames in free space.

Vector3 translation
Quaternion rotation

================================================================================
MSG: geometry_msgs/Vector3
# This represents a vector in free space. 
# It is only meant to represent a direction. Therefore, it does not
# make sense to apply a translation to it (e.g., when applying a 
# generic rigid transformation to a Vector3, tf2 will only apply the
# rotation). If you want your data to be translatable too, use the
# geometry_msgs/Point message instead.

float64 x
float64 y
float64 z
================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w
---
bool result

")

(setf (get detector_msgs::AddTransformToManagerResponse :md5sum-) "bf7912ca51f884d0a3b0d663503f42cd")
(setf (get detector_msgs::AddTransformToManagerResponse :datatype-) "detector_msgs/AddTransformToManagerResponse")
(setf (get detector_msgs::AddTransformToManagerResponse :definition-)
      "geometry_msgs/TransformStamped transform

================================================================================
MSG: geometry_msgs/TransformStamped
# This expresses a transform from coordinate frame header.frame_id
# to the coordinate frame child_frame_id
#
# This message is mostly used by the 
# <a href=\"http://wiki.ros.org/tf\">tf</a> package. 
# See its documentation for more information.

Header header
string child_frame_id # the frame id of the child frame
Transform transform

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

================================================================================
MSG: geometry_msgs/Transform
# This represents the transform between two coordinate frames in free space.

Vector3 translation
Quaternion rotation

================================================================================
MSG: geometry_msgs/Vector3
# This represents a vector in free space. 
# It is only meant to represent a direction. Therefore, it does not
# make sense to apply a translation to it (e.g., when applying a 
# generic rigid transformation to a Vector3, tf2 will only apply the
# rotation). If you want your data to be translatable too, use the
# geometry_msgs/Point message instead.

float64 x
float64 y
float64 z
================================================================================
MSG: geometry_msgs/Quaternion
# This represents an orientation in free space in quaternion form.

float64 x
float64 y
float64 z
float64 w
---
bool result

")



(provide :detector_msgs/AddTransformToManager "bf7912ca51f884d0a3b0d663503f42cd")


