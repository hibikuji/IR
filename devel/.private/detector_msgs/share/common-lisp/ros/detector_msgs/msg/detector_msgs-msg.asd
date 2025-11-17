
(cl:in-package :asdf)

(defsystem "detector_msgs-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :std_msgs-msg
)
  :components ((:file "_package")
    (:file "BBox" :depends-on ("_package_BBox"))
    (:file "_package_BBox" :depends-on ("_package"))
    (:file "BBoxArray" :depends-on ("_package_BBoxArray"))
    (:file "_package_BBoxArray" :depends-on ("_package"))
  ))