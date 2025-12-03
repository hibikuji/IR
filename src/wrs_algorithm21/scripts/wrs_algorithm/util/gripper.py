# -*- coding: utf-8 -*-
"""
HSRのハンドを制御するためのユーティリティモジュール
"""

from __future__ import unicode_literals, print_function, division, absolute_import
import moveit_commander
import rospy


# moveitでの制御対象としてハンドを指定
GRIPPER_CMD = moveit_commander.MoveGroupCommander(str("gripper"))


def command(value):
    """
    ハンドを制御

    Parameters
    ----------
        value (float): ハンドの開き具合 (0：閉じる、1:開く)

    Return
    ------
        正しく動作すればTrue, そうでなければFalse

    """

    GRIPPER_CMD.set_joint_value_target(str("hand_motor_joint"), value)
    success = GRIPPER_CMD.go()
    rospy.sleep(6)
    return success
<<<<<<< HEAD

def get_current_gap():
    """ sakura
    現在のハンドの開き具合を取得する

    Return
    ------
    float: 現在の関節値 (およそ0.0に近いほど閉じている)
    """
    # 現在の関節角度のリストを取得
    # HSRのgripperグループは通常1つの主要な関節(hand_motor_joint)を持つため、
    # リストの先頭[0]を取得すればOKです。
    current_values = GRIPPER_CMD.get_current_joint_values()
    
    # 安全のためリストが空でないか確認
    if len(current_values) > 0:
        return current_values[0]
    else:
        return 0.0
=======
>>>>>>> origin/IR-B
