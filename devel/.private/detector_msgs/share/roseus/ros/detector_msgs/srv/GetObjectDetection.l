;; Auto-generated. Do not edit!


(when (boundp 'detector_msgs::GetObjectDetection)
  (if (not (find-package "DETECTOR_MSGS"))
    (make-package "DETECTOR_MSGS"))
  (shadow 'GetObjectDetection (find-package "DETECTOR_MSGS")))
(unless (find-package "DETECTOR_MSGS::GETOBJECTDETECTION")
  (make-package "DETECTOR_MSGS::GETOBJECTDETECTION"))
(unless (find-package "DETECTOR_MSGS::GETOBJECTDETECTIONREQUEST")
  (make-package "DETECTOR_MSGS::GETOBJECTDETECTIONREQUEST"))
(unless (find-package "DETECTOR_MSGS::GETOBJECTDETECTIONRESPONSE")
  (make-package "DETECTOR_MSGS::GETOBJECTDETECTIONRESPONSE"))

(in-package "ROS")

(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))




(defclass detector_msgs::GetObjectDetectionRequest
  :super ros::object
  :slots (_header _type ))

(defmethod detector_msgs::GetObjectDetectionRequest
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:type __type) "")
    )
   (send-super :init)
   (setq _header __header)
   (setq _type (string __type))
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:type
   (&optional __type)
   (if __type (setq _type __type)) _type)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; string _type
    4 (length _type)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; string _type
       (write-long (length _type) s) (princ _type s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; string _type
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _type (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass detector_msgs::GetObjectDetectionResponse
  :super ros::object
  :slots (_successed _error_code _error_string _bboxes ))

(defmethod detector_msgs::GetObjectDetectionResponse
  (:init
   (&key
    ((:successed __successed) nil)
    ((:error_code __error_code) "")
    ((:error_string __error_string) "")
    ((:bboxes __bboxes) (instance detector_msgs::BBoxArray :init))
    )
   (send-super :init)
   (setq _successed __successed)
   (setq _error_code (string __error_code))
   (setq _error_string (string __error_string))
   (setq _bboxes __bboxes)
   self)
  (:successed
   (&optional (__successed :null))
   (if (not (eq __successed :null)) (setq _successed __successed)) _successed)
  (:error_code
   (&optional __error_code)
   (if __error_code (setq _error_code __error_code)) _error_code)
  (:error_string
   (&optional __error_string)
   (if __error_string (setq _error_string __error_string)) _error_string)
  (:bboxes
   (&rest __bboxes)
   (if (keywordp (car __bboxes))
       (send* _bboxes __bboxes)
     (progn
       (if __bboxes (setq _bboxes (car __bboxes)))
       _bboxes)))
  (:serialization-length
   ()
   (+
    ;; bool _successed
    1
    ;; string _error_code
    4 (length _error_code)
    ;; string _error_string
    4 (length _error_string)
    ;; detector_msgs/BBoxArray _bboxes
    (send _bboxes :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; bool _successed
       (if _successed (write-byte -1 s) (write-byte 0 s))
     ;; string _error_code
       (write-long (length _error_code) s) (princ _error_code s)
     ;; string _error_string
       (write-long (length _error_string) s) (princ _error_string s)
     ;; detector_msgs/BBoxArray _bboxes
       (send _bboxes :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; bool _successed
     (setq _successed (not (= 0 (sys::peek buf ptr- :char)))) (incf ptr- 1)
   ;; string _error_code
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _error_code (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; string _error_string
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _error_string (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; detector_msgs/BBoxArray _bboxes
     (send _bboxes :deserialize buf ptr-) (incf ptr- (send _bboxes :serialization-length))
   ;;
   self)
  )

(defclass detector_msgs::GetObjectDetection
  :super ros::object
  :slots ())

(setf (get detector_msgs::GetObjectDetection :md5sum-) "44386689b8f2bacd7dd66f142a8c65bc")
(setf (get detector_msgs::GetObjectDetection :datatype-) "detector_msgs/GetObjectDetection")
(setf (get detector_msgs::GetObjectDetection :request) detector_msgs::GetObjectDetectionRequest)
(setf (get detector_msgs::GetObjectDetection :response) detector_msgs::GetObjectDetectionResponse)

(defmethod detector_msgs::GetObjectDetectionRequest
  (:response () (instance detector_msgs::GetObjectDetectionResponse :init)))

(setf (get detector_msgs::GetObjectDetectionRequest :md5sum-) "44386689b8f2bacd7dd66f142a8c65bc")
(setf (get detector_msgs::GetObjectDetectionRequest :datatype-) "detector_msgs/GetObjectDetectionRequest")
(setf (get detector_msgs::GetObjectDetectionRequest :definition-)
      "std_msgs/Header header
string type

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
---
bool successed
string error_code
string error_string
detector_msgs/BBoxArray bboxes


================================================================================
MSG: detector_msgs/BBoxArray
std_msgs/Header header
detector_msgs/BBox[] bboxes

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
MSG: detector_msgs/BBox
string label
float64 score
int64 x
int64 y
int64 w
int64 h
")

(setf (get detector_msgs::GetObjectDetectionResponse :md5sum-) "44386689b8f2bacd7dd66f142a8c65bc")
(setf (get detector_msgs::GetObjectDetectionResponse :datatype-) "detector_msgs/GetObjectDetectionResponse")
(setf (get detector_msgs::GetObjectDetectionResponse :definition-)
      "std_msgs/Header header
string type

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
---
bool successed
string error_code
string error_string
detector_msgs/BBoxArray bboxes


================================================================================
MSG: detector_msgs/BBoxArray
std_msgs/Header header
detector_msgs/BBox[] bboxes

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
MSG: detector_msgs/BBox
string label
float64 score
int64 x
int64 y
int64 w
int64 h
")



(provide :detector_msgs/GetObjectDetection "44386689b8f2bacd7dd66f142a8c65bc")


