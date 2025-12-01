#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import math
import gc  # 【追加】メモリ掃除用
import traceback # 【追加】エラー表示用
# import tf  <-- これを削除しました
from wrs_main_node import WrsMainController
from wrs_algorithm.util import omni_base, whole_body, gripper

def quaternion_to_yaw_degree(q):
    """
    tfライブラリを使わずに、クォータニオン(x,y,z,w)から
    Yaw角（z軸周りの回転、度数法）を計算する関数
    """
    # クォータニオンの要素
    x = q.x
    y = q.y
    z = q.z
    w = q.w

    # Yaw (z軸回転) の計算式
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw_rad = math.atan2(siny_cosp, cosy_cosp)

    return math.degrees(yaw_rad)

def get_current_pose(ctrl):
    """
    現在地（マップ座標系での絶対座標 x, y, yaw）を取得して返す
    """
    # mapフレームからbase_footprint(ロボット足元)への座標変換を取得
    pose_stamped = ctrl.get_relative_coordinate("map", "base_footprint")
    
    if pose_stamped is None:
        rospy.logerr("現在地が取得できません！")
        return 0.0, 0.0, 0.0
    
    # TransformStamped型から座標を取り出す
    x = pose_stamped.translation.x
    y = pose_stamped.translation.y
    rotation = pose_stamped.rotation

    # 自作関数でYaw角を計算
    yaw_deg = quaternion_to_yaw_degree(rotation)
    
    return x, y, yaw_deg

def set_arm_extended():
    """アームを水平に伸ばす（初期姿勢）"""
    rospy.loginfo("アームを水平に展開します...")
    whole_body.move_to_joint_positions({
        "arm_lift_joint":  0.40,  # 昇降高さ
        "arm_flex_joint":  0.0,   # 肩: 0=水平
        "arm_roll_joint":  0.0,
        "wrist_flex_joint": -1.57, # 手首: 正面向き
        "wrist_roll_joint": 0.0,
        "head_pan_joint":  0.0,
        "head_tilt_joint": -0.5
    })
    gripper.command(1) # ハンドを開く

def move_shoulder(angle_deg):
    """肩の角度だけを絶対角度(deg)で指定して動かす"""
    angle_rad = math.radians(angle_deg)
    rospy.loginfo(f"肩角度を変更: {angle_deg} deg")
    whole_body.move_to_joint_positions({"arm_flex_joint": angle_rad})

def clean_memory(ctrl):
    """溜まったデータを捨ててメモリ掃除をする関数"""
    #リストの中身を空にする
    ctrl.instruction_list = []
    ctrl.detection_list   = []
    #Pythonのメモリ領域を強制開放
    gc.collect()
    rospy.loginfo("Memory cleaned.")

def main():
    rospy.init_node('drawer_tuning_abs')

    # ガベージコレクション有効化
    gc.enable()

    ctrl = WrsMainController()
    
    rospy.loginfo("初期化中... まずは定位置へ移動します")
    ctrl.goto_name("stair_like_drawer")
    set_arm_extended()

    while not rospy.is_shutdown():
        # =========================================================
        # 【重要】メモリリーク対策
        # wrs_main_node内で無限に追加され続けるリストをここで強制的に空にする
        # =========================================================
        ctrl.instruction_list = []
        ctrl.detection_list   = []
        
        # 不要メモリの強制回収
        gc.collect()
        rospy.loginfo("Memory cleaned.")
        
        # ループのたびに「今の絶対座標」を表示する
        curr_x, curr_y, curr_yaw = get_current_pose(ctrl)
        
        print("\n========== 絶対座標 検証モード (tf不使用版) ==========")
        print(f"【現在地 (Map座標)】")
        print(f"  X : {curr_x:.4f}")
        print(f"  Y : {curr_y:.4f}")
        print(f"  Yaw: {curr_yaw:.2f} deg")
        print("-----------------------------------------")
        print("操作を選択してください:")
        print(" 1: ロボット全体を移動 (入力: 絶対座標 X Y Yaw)")
        print(" 2: アーム角度(肩)を変更 (入力: 角度 deg)")
        print(" 3: 掴む (Grasp)")
        print(" 4: 放す (Open) & リセット")
        print(" q: 終了")
        
        try:
            try:
                cmd = input("Command >> ") # Python 3用
            except NameError:
                cmd = input("Command >>")
        except EOFError:
            break

        clean_memory(ctrl)
        
        if cmd == 'q':
            break
            
        elif cmd == '1':
            print("移動先の絶対座標を入力してください")
            print(f"ヒント: 現在は {curr_x:.2f} {curr_y:.2f} {curr_yaw:.0f} です")
            
            try:
                raw = input("Target X Y Yaw >> ")
                vals = [float(v) for v in raw.split()]
                if len(vals) == 3:
                    tgt_x, tgt_y, tgt_yaw = vals
                    rospy.loginfo(f"移動開始: Abs({tgt_x}, {tgt_y}, {tgt_yaw})")
                    omni_base.go_abs(tgt_x, tgt_y, tgt_yaw)
                else:
                    print("エラー: 数値を3つ入力してください")
            except Exception as e:
                print(f"入力エラー: {e}")

        elif cmd == '2':
            print("肩の角度を入力 (0=水平, マイナス=上, プラス=下)")
            try:
                val = input("Angle(deg) >> ")
                move_shoulder(float(val))
            except ValueError:
                print("数値を入力してください")

        elif cmd == '3':
            rospy.loginfo("ハンドを閉じます")
            gripper.command(0)
            
        elif cmd == '4':
            rospy.loginfo("ハンドを開き、アームを水平に戻します")
            gripper.command(1)
            # set_arm_extended()

if __name__ == '__main__':
    main()