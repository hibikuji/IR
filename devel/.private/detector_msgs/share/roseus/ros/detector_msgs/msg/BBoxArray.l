;; Auto-generated. Do not edit!


(when (boundp 'detector_msgs::BBoxArray)
  (if (not (find-package "DETECTOR_MSGS"))
    (make-package "DETECTOR_MSGS"))
  (shadow 'BBoxArray (find-package "DETECTOR_MSGS")))
(unless (find-package "DETECTOR_MSGS::BBOXARRAY")
  (make-package "DETECTOR_MSGS::BBOXARRAY"))

(in-package "ROS")
;;//! \htmlinclude BBoxArray.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass detector_msgs::BBoxArray
  :super ros::object
  :slots (_header _bboxes ))

(defmethod detector_msgs::BBoxArray
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:bboxes __bboxes) ())
    )
   (send-super :init)
   (setq _header __header)
   (setq _bboxes __bboxes)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
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
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; detector_msgs/BBox[] _bboxes
    (apply #'+ (send-all _bboxes :serialization-length)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; detector_msgs/BBox[] _bboxes
     (write-long (length _bboxes) s)
     (dolist (elem _bboxes)
       (send elem :serialize s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; detector_msgs/BBox[] _bboxes
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _bboxes (let (r) (dotimes (i n) (push (instance detector_msgs::BBox :init) r)) r))
     (dolist (elem- _bboxes)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;;
   self)
  )

(setf (get detector_msgs::BBoxArray :md5sum-) "7ed748f248b539ee6ad03e80ae91120d")
(setf (get detector_msgs::BBoxArray :datatype-) "detector_msgs/BBoxArray")
(setf (get detector_msgs::BBoxArray :definition-)
      "std_msgs/Header header
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



(provide :detector_msgs/BBoxArray "7ed748f248b539ee6ad03e80ae91120d")


