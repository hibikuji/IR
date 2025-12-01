#!/bin/bash -eu

# 【重要】起動時に前回のゴミプロセスを強制終了する
# これにより「二重起動」によるメモリ枯渇を防ぎます
echo "Cleaning up previous processes..."
pkill -f frcnn_finetuned || true
pkill -f pcd_to_tf || true

# 【重要】このスクリプト終了時に、バックグラウンド起動したノードを全停止する設定
trap 'kill $(jobs -p)' EXIT

# 1. Python環境とROS環境のセットアップ
unset PYTHONPATH
source /opt/ros/noetic/setup.bash
source /workspace/devel/setup.bash

# 2. 認識系のノードをバックグラウンドで起動
echo "Starting perception nodes..."
roslaunch hsr_perception pcd_to_tf.launch &
roslaunch wrs_detector frcnn_finetuned.launch &

# ノードが立ち上がるまで待機
echo "Waiting 5 seconds for perception nodes..."
sleep 5

# 3. 【検証ツールの起動】
# ファイル名を正しいものに修正しました
echo "Starting Manual Tuning Tool..."
rosrun wrs_algorithm tuning_drawer_manual.py