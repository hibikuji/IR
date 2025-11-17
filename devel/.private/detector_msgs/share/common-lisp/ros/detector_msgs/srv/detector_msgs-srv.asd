
(cl:in-package :asdf)

(defsystem "detector_msgs-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :detector_msgs-msg
               :geometry_msgs-msg
               :std_msgs-msg
)
  :components ((:file "_package")
    (:file "AddTransformToManager" :depends-on ("_package_AddTransformToManager"))
    (:file "_package_AddTransformToManager" :depends-on ("_package"))
    (:file "GetObjectDetection" :depends-on ("_package_GetObjectDetection"))
    (:file "_package_GetObjectDetection" :depends-on ("_package"))
    (:file "SetTransformFromBBox" :depends-on ("_package_SetTransformFromBBox"))
    (:file "_package_SetTransformFromBBox" :depends-on ("_package"))
  ))