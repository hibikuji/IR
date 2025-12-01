#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WRS環境内でロボットを動作させるためのメインプログラム
"""

from __future__ import unicode_literals, print_function, division, absolute_import
import json
import os
from select import select
import traceback
from turtle import pos
import rospy
import rospkg
import tf2_ros
from std_msgs.msg import String
from detector_msgs.srv import (
    SetTransformFromBBox, SetTransformFromBBoxRequest,
    GetObjectDetection, GetObjectDetectionRequest)
from wrs_algorithm.util import omni_base, whole_body, gripper
import math
import re


class WrsMainController(object):
    """
    WRSのシミュレーション環境内でタスクを実行するクラス
    """
    IGNORE_LIST = ["small_marker", "large_marker", "lego_duplo", "spatula", "nine_hole_peg_test"]
    GRASP_TF_NAME = "object_grasping"
    GRASP_BACK_SAFE = {"z": 0.05, "xy": 0.3}
    GRASP_BACK = {"z": 0.05, "xy": 0.1}
    HAND_PALM_OFFSET = 0.05  # hand_palm_linkは指の付け根なので、把持のために少しずらす必要がある
    HAND_PALM_Z_OFFSET = 0.075
    DETECT_CNT = 1
    TROFAST_Y_OFFSET = 0.2

    def __init__(self):
        # 物体検知の最大最小バウンディングボックス
        self.MIN_BBOX_AREA = 1600 # 40*40
        self.MAX_BBOX_AREA = 10000 # 100*100
        # 変数の初期化
        self.instruction_list = []
        self.detection_list   = []

        # configファイルの受信
        self.coordinates = self.load_json(self.get_path(["config", "coordinates.json"]))
        self.poses       = self.load_json(self.get_path(["config", "poses.json"]))

        food_cnt = 0
        tool_cnt = 0

        self.ORIENTATION_ITEM = [
            "large_marker", "small_marker", "fork", "spoon"
        ]
        self.FOOD_ITEM = [
            "cheez-it_cracker_box", "domino_suger box", "jell-o_chocolate_pudding_box",
            "jell-o_strawberry_gelatin_box", "spam_potted_meat_can", "master_chef_coffee_can",
            "starkist_tuna_fish_can", "pringles_chips_can", "french's_mustard_bottle",
            "tomato_soup_can", "plastic_banana", "plastic_strawberry",
            "plastic_apple", "plastic_lemon", "plastic_peach", "plastic_pear",
            "plastic_orange", "plastic_plum"
        ]
        self.KITCHEN_ITEM = [
            "windex_spray_bottle", "srub_cleanser_bottle", "scotch_brite_dobie_sponge",
            "pitcher_base", "pitcher_lid", "plate", "bowl", "fork", "spoon", "spatula",
            "wine_glass", "mug"
        ]
        self.TOOL_ITEM = [
            "large_marker", "small_marker", "keys", "bolt_and_nut", "clamps"
        ]
        self.SHAPE_ITEM = [
            "credit_card_blank", "mini_soccer_ball", "soft_ball", "baseball", "tennis_ball",
            "racquetball", "golf_ball", "marbles", "cups", "foam_bridk", "dice", "chain"
        ]
        self.TASK_ITEM = [
            "rubik's_cube", "colored_wood_blocks", "9-peg-hole_test", "toy_airplane", "lego_duplo",
            "magazine", "black_t-shirt", "timer"
        ]
        self.DISCARD = [
            "skillet", "skillet_lid", "table_cloth", "hammer", "adjustable_wrench", "wood_block",
            "power_drill", "washers", "nails", "knife", "scissors", "padlock", "phillips_screwdriver",
            "flat_screwdriver", "clear_box", "box_lid", "footlocker"
        ]

        self.obstacle_memory = []

        # ROS通信関連の初期化
        tf_from_bbox_srv_name = "set_tf_from_bbox"
        rospy.wait_for_service(tf_from_bbox_srv_name)
        self.tf_from_bbox_clt = rospy.ServiceProxy(tf_from_bbox_srv_name, SetTransformFromBBox)

        obj_detection_name = "detection/get_object_detection"
        rospy.wait_for_service(obj_detection_name)
        self.detection_clt = rospy.ServiceProxy(obj_detection_name, GetObjectDetection)

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.instruction_sub = rospy.Subscriber("/message",    String, self.instruction_cb, queue_size=10)
        self.detection_sub   = rospy.Subscriber("/detect_msg", String, self.detection_cb,   queue_size=10)

    @staticmethod
    def get_path(pathes, package="wrs_algorithm"):
        """
        ROSパッケージ名とファイルまでのパスを指定して、ファイルのパスを取得する
        """
        if not pathes:  # check if the list is empty
            rospy.logerr("Can NOT resolve file path.")
            raise ValueError("You must specify the path to file.")
        pkg_path = rospkg.RosPack().get_path(package)
        path = os.path.join(*pathes)
        return os.path.join(pkg_path, path)

    @staticmethod
    def load_json(path):
        """
        jsonファイルを辞書型で読み込む
        """
        with open(path, "r") as json_file:
            return json.load(json_file)

    def instruction_cb(self, msg):
        """
        指示文を受信する
        """
        rospy.loginfo("instruction received. [%s]", msg.data)
        self.instruction_list.append(msg.data)

    def detection_cb(self, msg):
        """
        検出結果を受信する
        """
        rospy.loginfo("received [Collision detected with %s]", msg.data)
        self.detection_list.append(msg.data)

    def get_relative_coordinate(self, parent, child):
        """
        tfで相対座標を取得する
        """
        try:
            # 4秒待機して各tfが存在すれば相対関係をセット
            trans = self.tf_buffer.lookup_transform(parent, child,rospy.Time.now(),rospy.Duration(4.0))
            return trans.transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            log_str = "failed to get transform between [{}] and [{}]\n".format(parent, child)
            log_str += traceback.format_exc()
            rospy.logerr(log_str)
            return None

    def goto_name(self, name):
        """
        waypoint名で指定された場所に移動する
        """
        if name in self.coordinates["positions"].keys():
            pos = self.coordinates["positions"][name]
            rospy.loginfo("go to [%s](%.2f, %.2f, %.2f)", name, pos[0], pos[1], pos[2])
            return omni_base.go_abs(pos[0], pos[1], pos[2])

        rospy.logerr("unknown waypoint name [%s]", name)
        return False

    def goto_pos(self, pos):
        """
        waypoint名で指定された場所に移動する
        """
        rospy.loginfo("go to [raw_pos](%.2f, %.2f, %.2f)", pos[0], pos[1], pos[2])
        return omni_base.go_abs(pos[0], pos[1], pos[2])

    def change_pose(self, name):
        """
        指定された姿勢名に遷移する
        """
        if name in self.poses.keys():
            rospy.loginfo("change pose to [%s]", name)
            return whole_body.move_to_joint_positions(self.poses[name])

        rospy.logerr("unknown pose name [%s]", name)
        return False

    def check_positions(self):
        """
        読み込んだ座標ファイルの座標を巡回する
        """
        whole_body.move_to_go()
        for wp_name in self.coordinates["routes"]["test"]:
            self.goto_name(wp_name)
            rospy.sleep(1)

    def get_latest_detection(self):
        """
        最新の認識結果が到着するまで待つ
        """
        res = self.detection_clt(GetObjectDetectionRequest())
        return res.bboxes

    def get_grasp_coordinate(self, bbox):
        """
        BBox情報から把持座標を取得する
        """
        # BBox情報からtfを生成して、座標を取得
        self.tf_from_bbox_clt.call(            SetTransformFromBBoxRequest(bbox=bbox, frame=self.GRASP_TF_NAME))
        rospy.sleep(1.0)  # tfが安定するのを待つ
        return self.get_relative_coordinate("map", self.GRASP_TF_NAME).translation

    @classmethod
    def get_most_graspable_bbox(cls, obj_list):
        """
        最も把持が行えそうなbboxを一つ返す。
        """
        # objが一つもない場合は、Noneを返す
        obj = cls.get_most_graspable_obj(obj_list)
        if obj is None: return None
        return obj["bbox"]

    @classmethod
    def get_most_graspable_obj(cls, obj_list):
        """
        把持すべきscoreが最も高い物体を返す。
        """
        extracted = []
        extract_str = "detected object list\n"
        ignore_str  = ""
        for obj in obj_list:
            info_str = "{:<15}({:.2%}, {:3d}, {:3d}, {:3d}, {:3d})\n".format(obj.label, obj.score, obj.x, obj.y, obj.w, obj.h)
            if obj.label in cls.IGNORE_LIST:
                ignore_str += "- ignored  : " + info_str
            else:
                score = cls.calc_score_bbox(obj)
                extracted.append({"bbox": obj, "score": score, "label": obj.label})
                extract_str += "- extracted: {:07.3f} ".format(score) + info_str

        rospy.loginfo(extract_str + ignore_str)

        # つかむべきかのscoreが一番高い物体を返す
        for obj_info in sorted(extracted, key=lambda x: x["score"], reverse=True):
            obj      = obj_info["bbox"]
            info_str = "{} ({:.2%}, {:3d}, {:3d}, {:3d}, {:3d})\n".format(obj.label, obj.score, obj.x, obj.y, obj.w, obj.h )
            rospy.loginfo("selected bbox: " + info_str)
            return obj_info

        # objが一つもない場合は、Noneを返す
        return None

    @classmethod
    def calc_score_bbox(cls, bbox):
        """
        detector_msgs/BBoxのスコアを計算する
        """
        gravity_x = bbox.x + bbox.w / 2
        gravity_y = bbox.y + bbox.h / 2
        xy_diff   = abs(320- gravity_x) / 320 + abs(360 - gravity_y) / 240

        return 1 / xy_diff

    @classmethod
    def get_most_graspable_bboxes_by_label(cls, obj_list, label):
        """
        label名が一致するオブジェクトの中から最も把持すべき物体のbboxを返す
        """
        match_objs = [obj for obj in obj_list if obj.label in label]
        if not match_objs:
            rospy.logwarn("Cannot find a object which labeled with similar name.")
            return None
        return cls.get_most_graspable_bbox(match_objs)

    @staticmethod
    def extract_target_obj_and_person(instruction):
        """
        指示文から対象となる物体名称を抽出する
        """
        #TODO: 関数は未完成です。引数のinstructionを利用すること
        rospy.loginfo("[extract_target_obj_and_person] instruction:"+  instruction)
        instruction_words = instruction.split()
        target_obj    = instruction_words[0]
        target_person = instruction_words[3]

        return target_obj, target_person

    def grasp_from_side(self, pos_x, pos_y, pos_z, yaw, pitch, roll, preliminary="-y"):
        """
        把持の一連の動作を行う

        NOTE: tall_tableに対しての予備動作を生成するときはpreliminary="-y"と設定することになる。
        """
        if preliminary not in [ "+y", "-y", "+x", "-x" ]: raise RuntimeError("unnkown graps preliminary type [{}]".format(preliminary))

        rospy.loginfo("move hand to grasp (%.2f, %.2f, %.2f)", pos_x, pos_y, pos_z)

        grasp_back_safe = {"x": pos_x, "y": pos_y, "z": pos_z + self.GRASP_BACK["z"]}
        grasp_back = {"x": pos_x, "y": pos_y, "z": pos_z + self.GRASP_BACK["z"]}
        grasp_pos = {"x": pos_x, "y": pos_y, "z": pos_z}

        if "+" in preliminary:
            sign = 1
        elif "-" in preliminary:
            sign = -1

        if "x" in preliminary:
            grasp_back_safe["x"] += sign * self.GRASP_BACK_SAFE["xy"]
            grasp_back["x"] += sign * self.GRASP_BACK["xy"]
        elif "y" in preliminary:
            grasp_back_safe["y"] += sign * self.GRASP_BACK_SAFE["xy"]
            grasp_back["y"] += sign * self.GRASP_BACK["xy"]

        gripper.command(1)
        whole_body.move_end_effector_pose(grasp_back_safe["x"], grasp_back_safe["y"], grasp_back_safe["z"], yaw, pitch, roll)
        whole_body.move_end_effector_pose( grasp_back["x"], grasp_back["y"], grasp_back["z"], yaw, pitch, roll)
        whole_body.move_end_effector_pose(
            grasp_pos["x"], grasp_pos["y"], grasp_pos["z"], yaw, pitch, roll)
        gripper.command(0)
        whole_body.move_end_effector_pose(grasp_back_safe["x"], grasp_back_safe["y"], grasp_back_safe["z"], yaw, pitch, roll)

    def get_most_likely_category(self, label):
        label_words = label.split('_')

        for ans_label in self.ORIENTATION_ITEM:
            for label_word in label_words:
                match = re.search(label_word, ans_label, re.IGNORECASE)
                if match:
                    pass
                else:
                    break
            else:
                return "ORIENTATION_ITEM"

        for ans_label in self.FOOD_ITEM:
            for label_word in label_words:
                match = re.search(label_word, ans_label, re.IGNORECASE)
                if match:
                    pass
                else:
                    break
            else:
                return "FOOD_ITEM"
            
        for ans_label in self.KITCHEN_ITEM:
            for label_word in label_words:
                match = re.search(label_word, ans_label, re.IGNORECASE)
                if match:
                    pass
                else:
                    break
            else:
                return "KITCHEN_ITEM"

        for ans_label in self.TOOL_ITEM:
            for label_word in label_words:
                match = re.search(label_word, ans_label, re.IGNORECASE)
                if match:
                    pass
                else:
                    break
            else:
                return "TOOL_ITEM"

        for ans_label in self.SHAPE_ITEM:
            for label_word in label_words:
                match = re.search(label_word, ans_label, re.IGNORECASE)
                if match:
                    pass
                else:
                    break
            else:
                return "SHAPE_ITEM"

        for ans_label in self.TASK_ITEM:
            for label_word in label_words:
                match = re.search(label_word, ans_label, re.IGNORECASE)
                if match:
                    pass
                else:
                    break
            else:
                return "TASK_ITEM"
        
        return "UNKNOWN"
    
    def grasp_from_left_side(self, grasp_pos):
        # オフセットを引く（マイナス側から近づくため）
        grasp_pos.x -= self.HAND_PALM_OFFSET 
        
        # preliminary を "-x" にする
        # yaw は右側と逆向き（例: 180）にする
        self.grasp_from_side(grasp_pos.x, grasp_pos.y, grasp_pos.z, 180, -100, 0, "-x")

    def grasp_from_right_side(self, grasp_pos):
        """
        横（X軸プラス方向）から把持を行う
        """
        # 【変更点1】 手のひらのオフセットを X軸 に対して適用する
        # プラス方向から近づくなら、目標より手前は「X座標が大きい方」なので "+" する
        grasp_pos.x += self.HAND_PALM_OFFSET 

        rospy.loginfo("grasp_from_right_side (%.2f, %.2f, %.2f)", grasp_pos.x, grasp_pos.y, grasp_pos.z)

        # 【変更点2 & 3】
        # yaw: 向きを90度変える（例: -90 -> 0）※ロボットの初期向きによります
        # preliminary: "+x" (X軸プラス側からアプローチ開始) に変更
        self.grasp_from_side(grasp_pos.x, grasp_pos.y, grasp_pos.z, 0, -100, 0, "+x")
    
    def grasp_from_front_side(self, grasp_pos):
        """
        正面把持を行う
        ややアームを下に向けている
        """
        grasp_pos.y -= self.HAND_PALM_OFFSET
        rospy.loginfo("grasp_from_front_side (%.2f, %.2f, %.2f)",grasp_pos.x, grasp_pos.y, grasp_pos.z)
        self.grasp_from_side(grasp_pos.x, grasp_pos.y, grasp_pos.z, -90, -100, 0, "-y")

    def grasp_from_upper_side(self, grasp_pos):
        """
        上面から把持を行う
        オブジェクトに寄るときは、y軸から近づく上面からは近づかない
        """
        grasp_pos.z += self.HAND_PALM_Z_OFFSET
        rospy.loginfo("grasp_from_upper_side (%.2f, %.2f, %.2f)",grasp_pos.x, grasp_pos.y, grasp_pos.z)
        self.grasp_from_side(grasp_pos.x, grasp_pos.y, grasp_pos.z, -90, -160, 0, "-y")

    def exec_graspable_method(self, grasp_pos, label=""):
        """
        task1専用:posの位置によって把持方法を判定し実行する。
        """
        method = None
        graspable_y = 1.85  # これ以上奥は把持できない
        desk_y = 1.5
        desk_z = 0.35

        # 把持禁止判定
        if graspable_y < grasp_pos.y and desk_z > grasp_pos.z:
            return False

        if label in ["cup", "frisbee", "bowl"]:
            # bowlの張り付き対策
            method = self.grasp_from_upper_side
        else:
            if desk_y < grasp_pos.y and desk_z > grasp_pos.z:
                # 机の下である場合
                method = self.grasp_from_front_side
            else:
                method = self.grasp_from_upper_side

        method(grasp_pos)

        def is_really_success():
            THRESHOLD = 0.02
            current_gap = gripper.get_current_gap()
            rospy.loginfo("Current Gripper Gap: %f", current_gap)

            if current_gap > THRESHOLD:
                rospy.logwarn("Grasp Success!")
                return True
            else:
                rospy.logwarn("Grasp Failed... (Gap is too small)")
                # 失敗したのでハンドを開くなどの処理が必要ならここに入れる
                gripper.command(1.0) 
                return False

        return is_really_success()

    def put_in_place(self, place, into_pose):
        # 指定場所に入れ、all_neutral姿勢を取る。
        self.change_pose("look_at_near_floor")
        a = "go_palce" # TODO 不要な変数
        self.goto_name(place)
        self.change_pose("all_neutral")
        self.change_pose(into_pose)
        gripper.command(1)
        rospy.sleep(4.0)
        self.change_pose("all_neutral")

    def pull_out_trofast(self, x, y, z, yaw, pitch, roll):
        # trofastの引き出しを引き出す
        self.goto_name("stair_like_drawer")
        self.change_pose("grasp_on_table")
        a = True  # TODO 不要な変数
        gripper.command(1)
        whole_body.move_end_effector_pose(x, y + self.TROFAST_Y_OFFSET, z, yaw, pitch, roll)
        whole_body.move_end_effector_pose(x, y, z, yaw, pitch, roll)
        gripper.command(0)
        whole_body.move_end_effector_pose(x, y + self.TROFAST_Y_OFFSET, z, yaw,  pitch, roll)
        gripper.command(1)
        self.change_pose("all_neutral")

    def push_in_trofast(self, pos_x, pos_y, pos_z, yaw, pitch, roll):
        """
        trofastの引き出しを戻す
        NOTE:サンプル
            self.push_in_trofast(0.178, -0.29, 0.75, -90, 100, 0)
        """
        self.goto_name("stair_like_drawer")
        self.change_pose("grasp_on_table")
        pos_y+=self.HAND_PALM_OFFSET

        # 予備動作-押し込む
        whole_body.move_end_effector_pose( pos_x, pos_y +    self.TROFAST_Y_OFFSET * 1.5, pos_z, yaw, pitch, roll)
        gripper.command(0)
        whole_body.move_end_effector_pose(  pos_x, pos_y + self.TROFAST_Y_OFFSET, pos_z, yaw, pitch, roll)
        whole_body.move_end_effector_pose(            pos_x, pos_y, pos_z, yaw, pitch, roll)

        self.change_pose("all_neutral")

    def deliver_to_target(self, target_obj, target_person):
        """
        棚で取得したものを人に渡す。
        """
        self.change_pose("look_at_near_floor")
        if(target_obj == "mustard_bottle" or target_obj == "apple"):
            self.goto_name("shelf_pos2")
        else:
            self.goto_name("shelf_pos1")
        self.change_pose("look_at_shelf")

        rospy.loginfo("target_obj: " + target_obj + "  target_person: " + target_person)
        # 物体検出結果から、把持するbboxを決定
        detected_objs = self.get_latest_detection()
        grasp_bbox = self.get_most_graspable_bboxes_by_label(detected_objs.bboxes, target_obj)
        if grasp_bbox is None:
            rospy.logwarn("Cannot find object to grasp. task2b is aborted.")
            return

        # BBoxの3次元座標を取得して、その座標で把持する
        grasp_pos = self.get_grasp_coordinate(grasp_bbox)
        self.change_pose("grasp_on_shelf")
        self.grasp_from_front_side(grasp_pos)
        self.change_pose("all_neutral")

        # target_personの前に持っていく
        self.change_pose("look_at_near_floor")

        if target_person == "right":
            self.goto_name("person_b")    # TODO: 配達先が固定されているので修正
        else:
            self.goto_name("person_a")
            
        self.change_pose("deliver_to_human")
        rospy.sleep(10.0)
        gripper.command(1)
        self.change_pose("all_neutral")

    def get_dist_point_to_segment(self, p, a, b):
        """
        点pと、線分abの最短距離を計算する
        p, a, b はそれぞれ [x, y] のリストまたはオブジェクト
        """
        # ベクトル ap, ab
        ap = [p[0] - a[0], p[1] - a[1]]
        ab = [b[0] - a[0], b[1] - a[1]]
        
        # abの長さの2乗
        len_ab_sq = ab[0]**2 + ab[1]**2
        if len_ab_sq == 0:
            # aとbが同じ場所にある場合
            return math.sqrt(ap[0]**2 + ap[1]**2)

        # 内積を使って、点pから線分への垂線の足の位置tを求める
        t = (ap[0]*ab[0] + ap[1]*ab[1]) / len_ab_sq
        
        # tを0~1の範囲に収める（線分の内側に限定）
        t = max(0, min(1, t))
        
        # 最短距離となる線分上の点 closest
        closest = [a[0] + t*ab[0], a[1] + t*ab[1]]
        
        # pとclosestの距離を返す
        return math.sqrt((p[0] - closest[0])**2 + (p[1] - closest[1])**2)

    def is_path_safe(self, start_pos, target_pos):
        """
        startからtargetへの移動ルートが安全か判定する
        """
        # ロボットの幅(半径) + 余裕。
        # 0.35mあれば、直径0.7mの筒が通れるかチェックすることになる
        safety_radius = 0.35 

        for obs in self.obstacle_memory:
            obs_pos = [obs.x, obs.y]
            dist = self.get_dist_point_to_segment(obs_pos, start_pos, target_pos)
            
            # 障害物がルートに近すぎるならNG
            if dist < safety_radius:
                return False
        return True

    def execute_avoid_blocks(self):
        # blockを避ける
        """
        2手先読み + 経路干渉チェックを行う高度な回避
        """
        rospy.loginfo("#### Start Advanced Lookahead Avoidance ####")

        # 移動目標地点のX座標定義
        # start -> step1 -> step2 -> step3(goal)
        x_steps = [1.7, 2.3, 2.9]
        
        # 探索するY座標の候補（0.1m刻みで細かく）
        # 2.0m 〜 3.5m の範囲
        candidate_ys = [y * 0.1 for y in range(20, 36)]

        # 現在地を取得
        current_pose = self.get_relative_coordinate("map", "base_footprint")
        current_x = current_pose.translation.x
        current_y = current_pose.translation.y

        for i, target_x in enumerate(x_steps):
            rospy.loginfo(f"--- Planning for Step {i+1} (Target X={target_x}) ---")

            look_angles = [-0.6, -0.35] 

            # 1. 2つの角度で見て記録 
            for angle in look_angles:
                # 首を動かす
                whole_body.move_to_joint_positions({"head_tilt_joint": angle, "head_pan_joint": 0.0})
                rospy.sleep(0.8) # ブレが収まるのを待つ

                # 認識実行
                detected_objs = self.get_latest_detection()
                
                # 記憶に追加
                for bbox in detected_objs.bboxes:
                    # スコアが低い誤検出は無視
                    if bbox.score < 0.2: continue

                    # 座標変換
                    pos = self.get_grasp_coordinate(bbox)
                    
                    # 重複チェック（既存の記憶と近すぎるなら追加しない）
                    is_known = False
                    for mem in self.obstacle_memory:
                        dist = math.sqrt((mem.x - pos.x)**2 + (mem.y - pos.y)**2)
                        # 20cm以内の誤差なら同じ物体とみなす
                        if dist < 0.1: 
                            is_known = True
                            break
                    
                    if not is_known:
                        self.obstacle_memory.append(pos)
                        rospy.loginfo(f"New Obstacle found at angle {angle}: ({pos.x:.2f}, {pos.y:.2f})")

            # 2. 【2手先読み】ルートプランニング
            # 「現在地 -> 次(Step1) -> その次(Step2)」が成立するルートを探す
            
            if i + 1 < len(x_steps):
                next_target_x = x_steps[i + 1]
            else:
                next_target_x = target_x + 0.5
            
            best_y = None
            min_cost = 999.0 # 移動コスト（少ないほうがいい）

            # 全ての「次のY候補」について調査
            for y1 in candidate_ys:
                # パス1: 現在地 -> 候補1 が安全か？
                if not self.is_path_safe([current_x, current_y], [target_x, y1]):
                    continue # ダメなら次の候補へ

                # もしこれが最後のステップなら、Step1に行けるだけでOK
                if i == len(x_steps) - 1:
                    cost = abs(y1 - current_y) # 横移動が少ないものを優先
                    if cost < min_cost:
                        min_cost = cost
                        best_y = y1
                    continue

                # まだ先がある場合、Step2へのルートがあるかチェック（詰み防止）
                can_go_further = False
                for y2 in candidate_ys:
                    # パス2: 候補1 -> 候補2 が安全か？
                    if self.is_path_safe([target_x, y1], [next_target_x, y2]):
                        can_go_further = True
                        break # 一つでも行ける未来があればOK
                
                if can_go_further:
                    # 未来があるルートの中で、最も移動量が少ないものを記録
                    cost = abs(y1 - current_y)
                    if cost < min_cost:
                        min_cost = cost
                        best_y = y1

            # 3. 移動実行
            if best_y is not None:
                rospy.loginfo(f"Valid Path Found! Moving to Y={best_y:.2f}")
                self.goto_pos([target_x, best_y, 90])
                # 現在地情報を更新（goto_posは誤差が出るのでtfで取り直しても良いが、ここでは目標値をセット）
                current_x = target_x
                current_y = best_y
            else:
                rospy.logerr("STUCK! No valid path found. Stopping task.")
                break
        """
        for i in range(3):
            # 毎回必ず床を見るようにする。
            self.change_pose("look_at_near_floor")
            rospy.sleep(0.5)

            detected_objs = self.get_latest_detection()
            bboxes = detected_objs.bboxes
            
            # スコアが低い(0.2未満など)誤検出は無視するフィルタリングを追加
            valid_bboxes = [bbox for bbox in bboxes if bbox.score > 0.2]

            # 座標変換を行う
            pos_bboxes = []
            for bbox in valid_bboxes:
                # 処理高速化: 明らかに遠くにあるものや関係ないものはtf変換しないなどの工夫も可能だが
                # ここでは安全重視で全て変換する
                pos_bboxes.append(self.get_grasp_coordinate(bbox))

            waypoint = self.select_next_waypoint(i, pos_bboxes)
            # TODO メッセージを確認するためコメントアウトを外す
            # rospy.loginfo(waypoint)
            self.goto_pos(waypoint)
        """

    def select_next_waypoint(self, current_stp, pos_bboxes):
        """
        waypoints から近い場所にあるものを除外し、最適なwaypointを返す。
        x座標を原点に近い方からxa,xb,xcに定義する。bboxを判断基準として移動先を決定する(デフォルトは0.45間隔)
        pos_bboxesは get_grasp_coordinate() 済みであること
        """
        interval = 0.45
        pos_xa = 1.7
        pos_xb = pos_xa + interval
        pos_xc = pos_xb + interval

        # xa配列はcurrent_stpに関係している
        waypoints = {"xa": [ [pos_xa, 2.5, 45],[pos_xa, 2.9, 45],[pos_xa, 3.3, 90] ], "xb": [ [pos_xb, 2.5, 90], [pos_xb, 2.9, 90], [pos_xb, 3.3, 90] ],
            "xc": [ [pos_xc, 2.5, 135],   [pos_xc, 2.9, 135],  [pos_xc, 3.3, 90 ]]
        }

        y_ranges = [[2.0, 2.8], [2.5, 3.2], [2.9, 3.6]]
        current_y_min, current_y_max = y_ranges[current_stp]

        # posがxa,xb,xcのラインに近い場合は候補から削除
        is_to_xa = True
        is_to_xb = True
        is_to_xc = True

        safety_margin = 0.28

        for bbox in pos_bboxes:
            pos_x = bbox.x
            pos_y = bbox.y

            # 今から通過しようとする物体以外は一旦無視する
            if not (current_y_min < pos_y < current_y_max):
                continue

            rospy.loginfo("Checking obstacle at (x=%.2f, y=%.2f) for step %d", pos_x, pos_y, current_stp)
            # TODO デバッグ時にコメントアウトを外す

            # NOTE Hint:ｙ座標次第で無視してよいオブジェクトもある。
            if (pos_xa - safety_margin < pos_x < pos_xa + safety_margin):
                is_to_xa = False
                rospy.loginfo("is_to_xa=False")
                continue
            if (pos_xb - safety_margin < pos_x < pos_xb + safety_margin):
                is_to_xb = False
                rospy.loginfo("is_to_xb=False")
                continue
            if (pos_xc - safety_margin < pos_x < pos_xc + safety_margin):
                is_to_xc = False
                rospy.loginfo("is_to_xc=False")
                continue

        x_line = None   # xa,xb,xcいずれかのリストが入る
        # NOTE 優先的にxcに移動する
        if is_to_xc:
            x_line = waypoints["xc"]
            rospy.loginfo("select next waypoint_xc")
        elif is_to_xb:
            x_line = waypoints["xb"]
            rospy.loginfo("select next waypoint_xb")
        elif is_to_xa:
            x_line = waypoints["xa"]
            rospy.loginfo("select next waypoint_xa")
        else:
            # a,b,cいずれにも移動できない場合
            x_line = waypoints["xb"]
            rospy.loginfo("select default waypoint")

        return x_line[current_stp]
    
    def open_all_drawers(self):
        """
        開始時にdrawerの3つの引き出しを全て開ける。
        ただし、各ハンドルの座標は調べる必要あり。
        """
        rospy.loginfo("#### Opening All Drawers ####")
        
        #引き出し前へ移動
        self.goto_name("stair_like_drawer")
        self.change_pose("look_at_near_floor")
        # --- 1. 左の引き出し(Drawer Left / Shape items) ---
        #以下のhandle_****達は書き換える必要あり
        handle_left_x = 0.44   # ロボットからの距離（奥行）
        handle_left_y = 0.20   # 左右（左がプラスの場合）
        handle_left_z = 0.40   # 高さ
        rospy.loginfo("Opening Drawer Left...")
        self.pull_out_trofast(handle_left_x, handle_left_y, handle_left_z, -90, 0, 0)

        # --- 2. 上の引き出し (Drawer Top / Tools) ---
        handle_top_x = 0.44
        handle_top_y = -0.10   
        handle_top_z = 0.80    # 高い位置
        rospy.loginfo("Opening Drawer Top...")
        self.pull_out_trofast(handle_top_x, handle_top_y, handle_top_z, -90, 0, 0)

        # --- 3. 下の引き出し (Drawer Bottom / Tools) ---
        handle_bottom_x = 0.44
        handle_bottom_y = -0.10
        handle_bottom_z = 0.50 # 低い位置
        rospy.loginfo("Opening Drawer Bottom...")
        self.pull_out_trofast(handle_bottom_x, handle_bottom_y, handle_bottom_z, -90, 0, 0)
        
        rospy.loginfo("All drawers are open.")


    def execute_task1(self):
        """
        task1を実行する
        """
        rospy.loginfo("#### start Task 1 ####")
        hsr_position = [
            #("near_tall_table_c", "look_at_near_floor"),
            ("tall_table", "look_at_tall_table"),
            ("near_logn_table_c", "look_at_near_floor"),
            #("near_long_table_l", "look_at_near_floor"),
            ("long_table_r", "look_at_tall_table"),
        ]

        food_cnt = 0
        tool_cnt = 0
        for plc, pose in hsr_position:
            # for _ in range(self.DETECT_CNT):
            ignore_labels_at_current_loc = []
            while True:
                # 移動と視線指示
                self.goto_name(plc)
                self.change_pose(pose)
                gripper.command(0)

                # 把持対象の有無チェック
                detected_objs = self.get_latest_detection()
                ##valid_bboxes = [
                ##    bbox for bbox in detected_objs.bboxes 
                ##    if bbox.label not in ignore_labels_at_current_loc
                ##]
                valid_bboxes = []
                for bbox in detected_objs.bboxes:
                    # 1. 無視リストに入っているか？
                    if bbox.label in ignore_labels_at_current_loc:
                        continue

                    # 2. サイズが小さすぎないか？ (幅 × 高さ で面積を計算)
                    bbox_area = bbox.w * bbox.h
                    '''
                    if bbox_area < self.MIN_BBOX_AREA:
                        rospy.logwarn("無視: [%s] は小さすぎます (Area: %d)", bbox.label, bbox_area)
                        continue
                    
                    if bbox_area > self.MAX_BBOX_AREA:
                        rospy.logwarn("無視: [%s] は大きすぎます (Area: %d)", bbox.label, bbox_area)
                        continue
                    '''
                    # 合格したものだけリストに入れる
                    valid_bboxes.append(bbox)
                graspable_obj = self.get_most_graspable_obj(valid_bboxes)
                ## graspable_obj = self.get_most_graspable_obj(detected_objs.bboxes)

                if graspable_obj is None:
                    rospy.logwarn("Cannot determine object to grasp. Grasping is aborted.")
                    #continue
                    break

                label = graspable_obj["label"]
                grasp_bbox = graspable_obj["bbox"]
                # TODO ラベル名を確認するためにコメントアウトを外す
                rospy.loginfo("grasp the " + label)

                # 把持対象がある場合は把持関数実施
                grasp_pos = self.get_grasp_coordinate(grasp_bbox)
                self.change_pose("grasp_on_table")
                is_success = self.exec_graspable_method(grasp_pos, label)
                # rospy.logwarn("is_success: [%s]", str(is_success))
                self.change_pose("all_neutral")

                if not is_success:
                    # ここで失敗した物体を除外する
                    rospy.logwarn("Failed to grasp [%s]", label)
                    if label not in self.IGNORE_LIST:
                        ignore_labels_at_current_loc.append(label)
                        #self.IGNORE_LIST.append(label)
                        rospy.sleep(1.0)
                    continue
                    #break

                most_likely_label = self.get_most_likely_category(label)
                rospy.logwarn("検出ラベル[%s]のもっともらしいカテゴリは [%s]", label, most_likely_label)
                # binに入れる
                if most_likely_label == "ORIENTATION_ITEM":
                    rospy.loginfo("カテゴリ [Orientation] -> Container_B")
                    self.put_in_place("container_B", "put_in_orientation_pose")
                elif most_likely_label == "FOOD_ITEM":
                    rospy.loginfo("カテゴリ [Food] -> Tray A / Tray B")
                    if food_cnt % 2 == 0:
                        self.put_in_place("tray_A", "put_in_tray_pose")
                    else:
                        self.put_in_place("tray_B", "put_in_tray_pose")
                    food_cnt += 1 # Food専用カウンターを増やす
                elif most_likely_label == "KITCHEN_ITEM":
                    # カテゴリ: Kitchen items -> Container_A 
                    rospy.loginfo("カテゴリ [Kitchen] -> Container A")
                    self.put_in_place("container_A", "put_in_container_pose")

                elif most_likely_label == "TOOL_ITEM":
                    # カテゴリ: Tools -> Drawer_top / Drawer_bottom 
                    rospy.loginfo("カテゴリ [Tools] -> Drawer Top / Bottom")
                    if tool_cnt % 2 == 0:
                        self.put_in_place("bin_a_place", "put_in_bin")
                        ## self.put_in_place("drawer_top", "put_in_drawer_pose")
                    else:
                        self.put_in_place("bin_b_place", "put_in_bin")
                        ## self.put_in_place("drawer_bottom", "put_in_drawer_pose")
                    tool_cnt += 1 # Tool専用カウンターを増やす

                elif most_likely_label == "SHAPE_ITEM":
                    # カテゴリ: Shape items -> Drawer_left 
                    rospy.loginfo("カテゴリ [Shape] -> Drawer left")
                    self.put_in_place("drawer_left", "put_in_drawer_pose")

                elif most_likely_label == "TASK_ITEM":
                    # カテゴリ: Task items -> Bin_A 
                    rospy.loginfo("カテゴリ [Task] -> Bin A")
                    self.put_in_place("bin_a_place", "put_in_bin") # 既存のbin_a_placeを使用

                else:
                    # カテゴリ: Unknown objects -> Bin_B 
                    rospy.logwarn("ラベル [%s] は分類外です。[Unknown] として Bin B に置きます。", label)
                    self.put_in_place("bin_b_place", "put_in_bin") # 既存のbin_b_placeを使用


    def execute_task2a(self):
        """
        task2aを実行する
        """
        rospy.loginfo("#### start Task 2a ####")
        self.change_pose("look_at_near_floor")
        gripper.command(0)
        self.change_pose("look_at_near_floor")
        self.goto_name("standby_2a")

        # 落ちているブロックを避けて移動
        self.execute_avoid_blocks()

        self.goto_name("go_throw_2a")
        whole_body.move_to_go()

    def execute_task2b(self):
        """
        task2bを実行する
        """
        rospy.loginfo("#### start Task 2b ####")

        # 命令文を取得
        if self.instruction_list:
            latest_instruction = self.instruction_list[-1]
            rospy.logwarn("recieved instruction: %s", latest_instruction)
        else:
            rospy.logwarn("instruction_list is None")
            return

        # 命令内容を解釈
        target_obj, target_person = self.extract_target_obj_and_person(latest_instruction)
        rospy.logwarn("target_obj: %s target_person: %s", target_obj, target_person)

        # 指定したオブジェクトを指定した配達先へ
        if target_obj and target_person:            self.deliver_to_target(target_obj, target_person)

    def run(self):
        """
        全てのタスクを実行する
        """
        self.change_pose("all_neutral")
        
        ##self.open_all_drawers() #テストのため一回無効

        self.execute_task1()
        self.execute_task2a() # task2bを実行するには必ずtask2aを実行しないといけないので注意
        self.execute_task2b()


def main():
    """
    WRS環境内でタスクを実行するためのメインノードを起動する
    """
    rospy.init_node('main_controller')
    try:
        ctrl = WrsMainController()
        rospy.loginfo("node initialized [%s]", rospy.get_name())

        # タスクの実行モードを確認する
        if rospy.get_param("~test_mode", default=False) is True:
            rospy.loginfo("#### start with TEST mode. ####")
            ctrl.check_positions()
        else:
            rospy.loginfo("#### start with NORMAL mode. ####")
            ctrl.run()

    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
