;; Auto-generated. Do not edit!


(when (boundp 'detector_msgs::SetTransformFromBBox)
  (if (not (find-package "DETECTOR_MSGS"))
    (make-package "DETECTOR_MSGS"))
  (shadow 'SetTransformFromBBox (find-package "DETECTOR_MSGS")))
(unless (find-package "DETECTOR_MSGS::SETTRANSFORMFROMBBOX")
  (make-package "DETECTOR_MSGS::SETTRANSFORMFROMBBOX"))
(unless (find-package "DETECTOR_MSGS::SETTRANSFORMFROMBBOXREQUEST")
  (make-package "DETECTOR_MSGS::SETTRANSFORMFROMBBOXREQUEST"))
(unless (find-package "DETECTOR_MSGS::SETTRANSFORMFROMBBOXRESPONSE")
  (make-package "DETECTOR_MSGS::SETTRANSFORMFROMBBOXRESPONSE"))

(in-package "ROS")





(defclass detector_msgs::SetTransformFromBBoxRequest
  :super ros::object
  :slots (_bbox _frame ))

(defmethod detector_msgs::SetTransformFromBBoxRequest
  (:init
   (&key
    ((:bbox __bbox) (instance detector_msgs::BBox :init))
    ((:frame __frame) "")
    )
   (send-super :init)
   (setq _bbox __bbox)
   (setq _frame (string __frame))
   self)
  (:bbox
   (&rest __bbox)
   (if (keywordp (car __bbox))
       (send* _bbox __bbox)
     (progn
       (if __bbox (setq _bbox (car __bbox)))
       _bbox)))
  (:frame
   (&optional __frame)
   (if __frame (setq _frame __frame)) _frame)
  (:serialization-length
   ()
   (+
    ;; detector_msgs/BBox _bbox
    (send _bbox :serialization-length)
    ;; string _frame
    4 (length _frame)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; detector_msgs/BBox _bbox
       (send _bbox :serialize s)
     ;; string _frame
       (write-long (length _frame) s) (princ _frame s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; detector_msgs/BBox _bbox
     (send _bbox :deserialize buf ptr-) (incf ptr- (send _bbox :serialization-length))
   ;; string _frame
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _frame (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass detector_msgs::SetTransformFromBBoxResponse
  :super ros::object
  :slots (_result ))

(defmethod detector_msgs::SetTransformFromBBoxResponse
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

(defclass detector_msgs::SetTransformFromBBox
  :super ros::object
  :slots ())

(setf (get detector_msgs::SetTransformFromBBox :md5sum-) "cf10ef4f67b53145111605419d2c8caa")
(setf (get detector_msgs::SetTransformFromBBox :datatype-) "detector_msgs/SetTransformFromBBox")
(setf (get detector_msgs::SetTransformFromBBox :request) detector_msgs::SetTransformFromBBoxRequest)
(setf (get detector_msgs::SetTransformFromBBox :response) detector_msgs::SetTransformFromBBoxResponse)

(defmethod detector_msgs::SetTransformFromBBoxRequest
  (:response () (instance detector_msgs::SetTransformFromBBoxResponse :init)))

(setf (get detector_msgs::SetTransformFromBBoxRequest :md5sum-) "cf10ef4f67b53145111605419d2c8caa")
(setf (get detector_msgs::SetTransformFromBBoxRequest :datatype-) "detector_msgs/SetTransformFromBBoxRequest")
(setf (get detector_msgs::SetTransformFromBBoxRequest :definition-)
      "detector_msgs/BBox bbox
string frame

================================================================================
MSG: detector_msgs/BBox
string label
float64 score
int64 x
int64 y
int64 w
int64 h
---
bool result

")

(setf (get detector_msgs::SetTransformFromBBoxResponse :md5sum-) "cf10ef4f67b53145111605419d2c8caa")
(setf (get detector_msgs::SetTransformFromBBoxResponse :datatype-) "detector_msgs/SetTransformFromBBoxResponse")
(setf (get detector_msgs::SetTransformFromBBoxResponse :definition-)
      "detector_msgs/BBox bbox
string frame

================================================================================
MSG: detector_msgs/BBox
string label
float64 score
int64 x
int64 y
int64 w
int64 h
---
bool result

")



(provide :detector_msgs/SetTransformFromBBox "cf10ef4f67b53145111605419d2c8caa")


