#!/bin/bash -eu

unset PYTHONPATH
source /opt/ros/noetic/setup.bash
source /workspace/devel/setup.bash
roslaunch hsr_perception pcd_to_tf.launch &
roslaunch wrs_detector frcnn_finetuned.launch &
# 元の start_task.launch は使わず、直接 Python ノードをテストモードで叩く
# （ & をつけるとバックグラウンド実行、つけないとそこで待機になります。
#   waitがあるのなら & をつけておくのが元の作法に近いです）
rosrun wrs_algorithm wrs_main_node.py _test_mode:=true &

wait
