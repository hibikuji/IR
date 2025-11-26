#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from wrs_main_node import WrsMainController # 既存のクラスをインポート

def main():
    rospy.init_node('drawer_tuning_tool')
    ctrl = WrsMainController()
    
    rospy.loginfo("初期化完了。引き出しの前まで移動します...")
    
    # まず定位置（引き出しの前）まで移動してしまう
    ctrl.goto_name("stair_like_drawer")
    ctrl.change_pose("look_at_near_floor")

    while not rospy.is_shutdown():
        print("\n========== 座標調整モード ==========")
        print("試したい座標を入力してください (例: 0.44 0.2 0.4)")
        print("終了するには 'q' を入力")
        
        user_input = input("x y z >> ") # Python 3なら input()
        
        if user_input == 'q':
            break
            
        try:
            # 入力を数値に変換
            coords = [float(v) for v in user_input.split()]
            if len(coords) != 3:
                print("エラー: 数字を3つスペース区切りで入力してください")
                continue
                
            x, y, z = coords
            rospy.loginfo("テスト実行: x=%.2f, y=%.2f, z=%.2f", x, y, z)
            
            # 指定した座標で引き出しを開ける動作を実行
            # 角度(-90, 0, 0)は固定にしていますが、必要ならここも変えられるようにします
            ctrl.pull_out_trofast(x, y, z, -90, 0, 0)
            
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