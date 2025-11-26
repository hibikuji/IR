#!/bin/bash -eu

# 1. Python環境とROS環境のセットアップ
unset PYTHONPATH
source /opt/ros/noetic/setup.bash
source /workspace/devel/setup.bash

# 3. 認識系のノードをバックグラウンドで起動
# (WrsMainControllerの初期化で必要な場合があるため、元の記述を残しています)
roslaunch hsr_perception pcd_to_tf.launch &
roslaunch wrs_detector frcnn_finetuned.launch &

# ノードが立ち上がるまで少し待機
echo "Waiting for perception nodes to start..."
sleep 5

# 4. 【検証ツールの起動】
# ここは '&' を付けずに実行します（キーボード入力を受け付けるため）
# また、start_task.launch（自動タスク）は衝突するのでコメントアウトか削除します
rosrun wrs_algorithm tuning_drawer_abs.py

# ツールが終了(qで終了)したら、このスクリプトも終わる