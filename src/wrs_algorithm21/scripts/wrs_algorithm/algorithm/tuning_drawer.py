#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from wrs_main_node import WrsMainController
# wrs_main_nodeで使われている制御モジュールを直接インポート
from wrs_algorithm.util import omni_base, whole_body, gripper

def move_backward(distance):
    """
    現在の向きのまま、指定した距離(m)だけ後ろに下がる関数
    """
    rospy.loginfo(f"現在地から {distance}m 後退します...")
    # go_rel(x, y, yaw): xをマイナスにするとロボット後方へ進む
    omni_base.go_rel(-distance, 0, 0)

def pull_out_test(ctrl, x, y, z):
    """
    【重要】ロボットのベース移動(goto_name)を行わず、
    その場からアームだけを伸ばして引き出しを開けるテスト関数
    """
    # 角度は固定
    yaw = -90
    pitch = 0
    roll = 0
    
    rospy.loginfo(f"テスト実行: x={x}, y={y}, z={z} (アームのみ動作)")

    # 1. 構え（テーブル上の物体を掴む姿勢などを流用）
    ctrl.change_pose("grasp_on_table")
    gripper.command(1) # ハンドを開く

    # 2. アプローチ動作
    # ハンドルの「Y軸方向手前」に一度アームを持っていく
    # wrs_main_node内の定数と同じ 0.2m を使用
    approach_y = y + ctrl.TROFAST_Y_OFFSET
    
    # 手前へ移動 (ここは掴まない)
    whole_body.move_end_effector_pose(x, approach_y, z, yaw, pitch, roll)

    # 3. 掴みに行く (本番の座標へ)
    # ここでロボットが遠くにいれば、自然と腕が伸びきる
    whole_body.move_end_effector_pose(x, y, z, yaw, pitch, roll)

    # 4. 掴む
    gripper.command(0) # ハンドを閉じる

    # 5. 引っ張る (手前に戻る)
    whole_body.move_end_effector_pose(x, approach_y, z, yaw, pitch, roll)

    # 6. 放す
    gripper.command(1) 
    
    # 7. 姿勢を戻す
    ctrl.change_pose("all_neutral")

def main():
    rospy.init_node('drawer_tuning_tool_standalone')
    ctrl = WrsMainController()
    
    rospy.loginfo("初期化完了。まずは定位置へ移動します...")
    
    # 1. まずは通常の「近すぎる」定位置へ移動
    ctrl.goto_name("stair_like_drawer")
    ctrl.change_pose("look_at_near_floor")

    # ==========================================
    # ★ここが修正ポイント: 自動で距離をとる
    # 0.2m (20cm) ほど下がって、「腕を伸ばさないと届かない距離」にする
    # これにより低い位置(z<0.4)でも肘が体に当たらなくなるはずです
    # ==========================================
    BACK_DISTANCE = 0.20
    move_backward(BACK_DISTANCE)

    while not rospy.is_shutdown():
        print("\n========== 座標調整モード (後退済み) ==========")
        print(f"現在、定位置から {BACK_DISTANCE}m 後ろにいます。")
        print("試したいハンドルの座標を入力してください")
        print("入力例: 0.44 0.2 0.35  (終了は q )")
        
        user_input = input("x y z >> ")
        
        if user_input == 'q':
            break
            
        try:
            coords = [float(v) for v in user_input.split()]
            if len(coords) != 3:
                print("エラー: 数字を3つスペース区切りで入力してください")
                continue
                
            x, y, z = coords
            
            # 自作のテスト関数を呼ぶ（勝手に移動しないやつ）
            pull_out_test(ctrl, x, y, z)
            
            rospy.loginfo("動作完了。次の座標を入力できます。")
            
        except ValueError:
            print("エラー: 数値を正しく入力してください")
        except Exception as e:
            rospy.logerr("実行エラー: %s", e)

if __name__ == '__main__':
    main()


"""
1. ターミナルで実行権限を与える(chmod +x tuning_drawer.py)
2. シミュレータを立ち上げた上で rosrun wrs_algorithm tuning_drawer.py
   ※エラーが出た場合、rospack find wrs_algorithmと打って、出たパス名を使って書き直す。
3. ロボットが棚の前まで移動して待機する。
4. コンソールにx y z >>と出るので、そこに0.45 0.2 0.4のように入力してEnterする。
   すると、その場所から開ける動作を試す
5. 失敗した場合、プログラムを止めずに数値を書き換えて再トライする。
6. 終了する時はqと入力してEnterを押す。
"""