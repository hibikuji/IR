#!/bin/bash -eu

unset PYTHONPATH
source /opt/ros/noetic/setup.bash
source /workspace/devel/setup.bash

# 1. 必須の裏方システム（目や認識機能）をバックグラウンドで起動
roslaunch hsr_perception pcd_to_tf.launch &
roslaunch wrs_detector frcnn_finetuned.launch &

# 2. それらが起動するのを少し待つ
echo "Waiting for systems to start..."
sleep 5

# 3. 最後に自分の調整ツールを起動する（ここを書き換えました）
# importエラーを防ぐため、ディレクトリを移動してから実行します
cd /workspace/src/wrs_algorithm21/scripts/wrs_algorithm/algorithm/
echo "Starting tuning_drawer.py..."
python3 tuning_drawer.py