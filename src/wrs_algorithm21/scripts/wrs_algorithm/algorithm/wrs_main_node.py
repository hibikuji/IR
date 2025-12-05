#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WRS環境内でロボットを動作させるためのメインプログラム (ONNX対応版)
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
# 画像処理用に追加
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from detector_msgs.srv import (
    SetTransformFromBBox, SetTransformFromBBoxRequest,
    GetObjectDetection, GetObjectDetectionRequest)
from detector_msgs.msg import BBox
from wrs_algorithm.util import omni_base, whole_body, gripper
import math
import re


# ==========================================================
# 【追加】取っ手専用：YOLO(ONNX)をOpenCVだけで動かすクラス
# ==========================================================
class HandleDetectorONNX:
    """
    ONNXモデルを使用してハンドル検出を行うクラス。
    """
    def __init__(self, model_path, conf_thres=0.4, iou_thres=0.4):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.net = None

        # モデルファイルの存在確認とロード
        if os.path.exists(model_path):
            rospy.loginfo("【取っ手検出】ONNXモデルをロード中...: " + model_path)
            try:
                self.net = cv2.dnn.readNetFromONNX(model_path)
                # CPUで動作させる設定
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                rospy.loginfo("【取っ手検出】モデルロード完了！")
            except cv2.error as e:
                rospy.logerr(f"【取っ手検出】モデル読み込みエラー: {e}")
        else:
            rospy.logerr(f"【取っ手検出】エラー: {model_path} が見つかりません。")

    def detect(self, cv_image):
        """画像を受け取って取っ手のBBoxリストを返す"""
        if self.net is None:
            return []

        # 1. 画像の前処理 (YOLOv8標準の640x640にリサイズ)
        input_width, input_height = 640, 640
        blob = cv2.dnn.blobFromImage(
            cv_image, 1/255.0,
            (input_width, input_height), swapRB=True, crop=False)
        self.net.setInput(blob)
        # 2. 推論実行
        outputs = self.net.forward()
        # 3. 出力の整形
        # YOLOv8 output: [1, 5, 8400] (batch, xywh+conf, anchors) -> 転置して [8400, 5]
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []

        # 画像サイズの比率計算
        img_h, img_w = cv_image.shape[:2]
        x_factor = img_w / input_width
        y_factor = img_h / input_height

        for i in range(rows):
            # output[0][i] = [center_x, center_y, w, h, confidence]
            confidence = outputs[0][i][4]
            if confidence >= self.conf_threshold:
                box = outputs[0][i][0:4]
                x_center = box[0]
                y_center = box[1]
                w = box[2]
                h = box[3]
                # 座標を元の画像サイズに戻す
                left = int((x_center - w / 2) * x_factor)
                top = int((y_center - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                boxes.append([left, top, width, height])
                scores.append(float(confidence))

        # 4. NMS (重なりを除去)
        indices = cv2.dnn.NMSBoxes(
            boxes, scores, self.conf_threshold, self.iou_threshold)

        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                bbox = BBox()
                bbox.label = "handle"  # 学習させたクラス名
                bbox.score = scores[i]
                bbox.x = boxes[i][0]
                bbox.y = boxes[i][1]
                bbox.w = boxes[i][2]
                bbox.h = boxes[i][3]
                results.append(bbox)
        return results


class WrsMainController:
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
        # 変数の初期化
        self.instruction_list = []
        self.detection_list = []

        # configファイルの受信
        self.coordinates = self.load_json(self.get_path(["config", "coordinates.json"]))
        self.poses = self.load_json(self.get_path(["config", "poses.json"]))

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
            "skillet", "skillet_lid", "table_cloth", "hammer", "adjustable_wrench",
            "wood_block", "power_drill", "washers", "nails", "knife", "scissors",
            "padlock", "phillips_screwdriver", "flat_screwdriver", "clear_box",
            "box_lid", "footlocker"
        ]

        self.obstacle_memory = []

        # ROS通信関連の初期化
        tf_from_bbox_srv_name = "set_tf_from_bbox"
        rospy.wait_for_service(tf_from_bbox_srv_name)
        self.tf_from_bbox_clt = rospy.ServiceProxy(tf_from_bbox_srv_name, SetTransformFromBBox)

        # 既存の物体認識サービス（そのまま残す）
        obj_detection_name = "detection/get_object_detection"
        rospy.wait_for_service(obj_detection_name)
        self.detection_clt = rospy.ServiceProxy(obj_detection_name, GetObjectDetection)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.instruction_sub = rospy.Subscriber(
            "/message",    String, self.instruction_cb, queue_size=10)
        self.detection_sub = rospy.Subscriber(
            "/detect_msg", String, self.detection_cb,   queue_size=10)

        # ==========================================================
        # 【追加】自前で用意した取っ手認識器の初期化
        # ==========================================================
        self.bridge = CvBridge()
        self.latest_image = None
        # カメラ画像を常に受け取れるようにサブスクライバを追加
        rospy.Subscriber("/hsrb/head_rgbd_sensor/rgb/image_rect_color", Image, self.image_cb)
        # 同じフォルダにある best.onnx を読み込む
        onnx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.onnx")
        self.handle_detector = HandleDetectorONNX(onnx_path)

    def image_cb(self, msg):
        """【追加】カメラ画像を保存するコールバック"""
        self.latest_image = msg

    # ----------------------------------------------------------
    #  【追加】取っ手専用の認識関数 (既存の認識とは別物)
    # ----------------------------------------------------------
    def detect_handles_onnx(self):
        """
        best.onnxを使って取っ手を検出する
        """
        if self.latest_image is None:
            rospy.logwarn("カメラ画像がまだ届いていません")
            return []
        try:
            # ROS画像をOpenCV形式に変換
            cv_img = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
            # 自前のONNX検出器で認識
            return self.handle_detector.detect(cv_img)
        except Exception as e:
            rospy.logerr("画像変換または推論でエラー: " + str(e))
            return []

    # --- 以下、既存の関数 ---

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
            trans = self.tf_buffer.lookup_transform(parent, child,
                                                    rospy.Time.now(), rospy.Duration(4.0))
            return trans.transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            log_str = f"failed to get transform between [{parent}] and [{child}]\n"
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
        最新の認識結果が到着するまで待つ (既存の物体認識)
        """
        res = self.detection_clt(GetObjectDetectionRequest())
        return res.bboxes

    def get_grasp_coordinate(self, bbox):
        """
        BBox情報から把持座標を取得する
        """
        # BBox情報からtfを生成して、座標を取得
        self.tf_from_bbox_clt.call(SetTransformFromBBoxRequest(bbox=bbox,
                                                               frame=self.GRASP_TF_NAME))
        rospy.sleep(1.0)  # tfが安定するのを待つ
        return self.get_relative_coordinate("map", self.GRASP_TF_NAME).translation

    @classmethod
    def get_most_graspable_bbox(cls, obj_list):
        """
        最も把持が行えそうなbboxを一つ返す。
        """
        # objが一つもない場合は、Noneを返す
        obj = cls.get_most_graspable_obj(obj_list)
        if obj is None:
            return None
        return obj["bbox"]

    @classmethod
    def get_most_graspable_obj(cls, obj_list):
        """
        把持すべきscoreが最も高い物体を返す。
        """
        extracted = []
        extract_str = "detected object list\n"
        ignore_str = ""
        for obj in obj_list:
            info_str = (f"{obj.label:<15}({obj.score:.2%}, {obj.x:3d}, "
                        f"{obj.y:3d}, {obj.w:3d}, {obj.h:3d})\n")
            if obj.label in cls.IGNORE_LIST:
                ignore_str += "- ignored  : " + info_str
            else:
                score = cls.calc_score_bbox(obj)
                extracted.append({"bbox": obj, "score": score, "label": obj.label})
                extract_str += f"- extracted: {score:07.3f} {info_str}"

        rospy.loginfo(extract_str + ignore_str)

        # つかむべきかのscoreが一番高い物体を返す
        for obj_info in sorted(extracted, key=lambda x: x["score"], reverse=True):
            obj = obj_info["bbox"]
            info_str = (f"{obj.label} ({obj.score:.2%}, {obj.x:3d}, "
                        f"{obj.y:3d}, {obj.w:3d}, {obj.h:3d})\n")
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
        xy_diff = abs(320 - gravity_x) / 320 + abs(360 - gravity_y) / 240

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

        rospy.loginfo("[extract_target_obj_and_person] instruction:" + instruction)
        instruction_words = instruction.split()
        target_obj = instruction_words[0]
        target_person = instruction_words[3]

        return target_obj, target_person

    def grasp_from_side(self, pos_x, pos_y, pos_z, yaw, pitch, roll, preliminary="-y"):
        """
        把持の一連の動作を行う

        NOTE: tall_tableに対しての予備動作を生成するときはpreliminary="-y"と設定することになる。
        """
        if preliminary not in ["+y", "-y", "+x", "-x"]:
            raise RuntimeError(f"unnkown graps preliminary type [{preliminary}]")

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
        whole_body.move_end_effector_pose(
            grasp_back_safe["x"], grasp_back_safe["y"], grasp_back_safe["z"], yaw, pitch, roll)
        whole_body.move_end_effector_pose(
            grasp_back["x"], grasp_back["y"], grasp_back["z"], yaw, pitch, roll)
        whole_body.move_end_effector_pose(
            grasp_pos["x"], grasp_pos["y"], grasp_pos["z"], yaw, pitch, roll)
        gripper.command(0)
        whole_body.move_end_effector_pose(
            grasp_back_safe["x"], grasp_back_safe["y"], grasp_back_safe["z"], yaw, pitch, roll)

    def get_most_likely_category(self, label):
        """
        ラベル名から、その物体が属するカテゴリ名（FOOD_ITEMなど）を推定して返す
        """
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

    def grasp_from_front_side(self, grasp_pos):
        """
        正面把持を行う
        ややアームを下に向けている
        """
        grasp_pos.y -= self.HAND_PALM_OFFSET
        rospy.loginfo("grasp_from_front_side (%.2f, %.2f, %.2f)",
                      grasp_pos.x, grasp_pos.y, grasp_pos.z)
        self.grasp_from_side(grasp_pos.x, grasp_pos.y, grasp_pos.z, -90, -100, 0, "-y")

    def grasp_from_upper_side(self, grasp_pos):
        """
        上面から把持を行う
        オブジェクトに寄るときは、y軸から近づく上面からは近づかない
        """
        grasp_pos.z += self.HAND_PALM_Z_OFFSET
        rospy.loginfo("grasp_from_upper_side (%.2f, %.2f, %.2f)",
                      grasp_pos.x, grasp_pos.y, grasp_pos.z)
        self.grasp_from_side(grasp_pos.x, grasp_pos.y, grasp_pos.z, -90, -160, 0, "-y")

    def grasp_from_left_side(self, grasp_pos):
        """
        棚の右端にある物体（例：mustard）を、
        棚の前からロボットの右方向へ腕を伸ばして把持する。
        = アームを +x 方向に斜め差し込みする
        """

        rospy.loginfo("grasp_from_left_side (%.2f, %.2f, %.2f)",
                      grasp_pos.x, grasp_pos.y, grasp_pos.z)

        # --- 安全オフセット（棚との干渉回避） ---
        grasp_pos.y -= 0       # 手前に少し引く
        grasp_pos.z += 0       # 少しだけ上げる（層板との衝突防止）
        
        # --- 斜め差し込みに必要な姿勢 ---
        # yaw = 0°：手先をロボットの前ではなく右方向に向ける
        # pitch = -100°：先端をやや下に向ける
        # roll = 0°：水平維持
        yaw = -45        # 右方向
        pitch = -100
        roll = 0

        # preliminary="+x" → アームを右方向にバックさせて安全な軌道を確保
        self.grasp_from_side(grasp_pos.x, grasp_pos.y,
                             grasp_pos.z, yaw, pitch, roll, preliminary="-y")

    def grasp_from_right_side(self, grasp_pos):
        """
        棚の左端の物体を棚前から左方向へ差し込んで把持
        """
        rospy.loginfo("grasp_from_right_side (%.2f, %.2f, %.2f)",
                      grasp_pos.x, grasp_pos.y, grasp_pos.z)

        grasp_pos.y += 0
        grasp_pos.z += 0

        yaw = 45   # 左方向へ向ける
        pitch = -100
        roll = 0

        # 左方向へ preliminary="-x"
        self.grasp_from_side(grasp_pos.x, grasp_pos.y, grasp_pos.z,
                             yaw, pitch, roll, preliminary="+y")

    def grasp_from_front_side_with_body(self, grasp_pos, body_yaw_deg):
        """
        本体回転を考慮した正面把持（棚端の45度差し込み対応）
        """

        # 本体に対して正面へ差し込むには
        # 手先 yaw を body_yaw_deg と同じにするのが最も“真正面”
        yaw = body_yaw_deg      # ← ★ここが重要
        pitch = -100
        roll = 0

        rospy.loginfo(f"grasp_from_front_side_with_body yaw={yaw:.1f}")

        self.grasp_from_side(
            grasp_pos.x,
            grasp_pos.y - self.HAND_PALM_OFFSET,
            grasp_pos.z,
            yaw,
            pitch,
            roll,
            preliminary="-y"
        )

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
        return True

    def put_in_place(self, place, into_pose):
        """指定場所に入れ、all_neutral姿勢を取る。"""
        self.change_pose("look_at_near_floor")
        self.goto_name(place)
        self.change_pose("all_neutral")
        self.change_pose(into_pose)
        gripper.command(1)
        rospy.sleep(5.0)
        self.change_pose("all_neutral")

    def pull_out_trofast(self, x, y, z, yaw, pitch, roll):
        """
        棚を前に引く
        """
        # trofastの引き出しを引き出す (座標微調整)
        self.change_pose("grasp_on_table")
        gripper.command(1)
        # ちょっと手前からアプローチ (X-0.05)
        whole_body.move_end_effector_pose(x - 0.05, y, z, yaw, pitch, roll)
        # 突っ込む (X+0.02)
        whole_body.move_end_effector_pose(x + 0.02, y, z, yaw, pitch, roll)
        gripper.command(0)  # 掴む
        # 引く (手前にバック X-0.25)
        # Y軸方向のオフセット(TROFAST_Y_OFFSET)は状況によるが、基本は手前に引く
        whole_body.move_end_effector_pose(x - 0.25, y + self.TROFAST_Y_OFFSET, z, yaw, pitch, roll)
        gripper.command(1)  # 離す

        self.change_pose("all_neutral")

    def push_in_trofast(self, pos_x, pos_y, pos_z, yaw, pitch, roll):
        """
        trofastの引き出しを戻す
        NOTE:サンプル
            self.push_in_trofast(0.178, -0.29, 0.75, -90, 100, 0)
        """
        self.goto_name("stair_like_drawer")
        self.change_pose("grasp_on_table")
        pos_y += self.HAND_PALM_OFFSET

        # 予備動作-押し込む
        whole_body.move_end_effector_pose(
            pos_x, pos_y + self.TROFAST_Y_OFFSET * 1.5, pos_z, yaw, pitch, roll)
        gripper.command(0)
        whole_body.move_end_effector_pose(
            pos_x, pos_y + self.TROFAST_Y_OFFSET, pos_z, yaw, pitch, roll)
        whole_body.move_end_effector_pose(
            pos_x, pos_y, pos_z, yaw, pitch, roll)

        self.change_pose("all_neutral")

    """
    def deliver_to_target(self, target_obj, target_person):

        # -----------------------------
        # ① 棚へ移動
        # -----------------------------
        self.change_pose("look_at_near_floor")
        self.goto_name("shelf")
        self.change_pose("look_at_shelf")


        rospy.loginfo("target_obj: " + target_obj + "  target_person: " + target_person)
        # 物体検出結果から、把持するbboxを決定 (ここは既存の機能を使う)
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
            
        rospy.loginfo("target_obj: %s  target_person: %s",
                    target_obj, target_person)

        # -----------------------------
        # ② 物体検出
        # -----------------------------
        detected_objs = self.get_latest_detection()
        grasp_bbox = self.get_most_graspable_bboxes_by_label(
            detected_objs.bboxes, target_obj
        )
        if grasp_bbox is None:
            rospy.logwarn("Cannot find object. aborted.")
            return

        grasp_pos = self.get_grasp_coordinate(grasp_bbox)
        rospy.loginfo("Object position: x=%.3f y=%.3f z=%.3f",
                    grasp_pos.x, grasp_pos.y, grasp_pos.z)

        # -----------------------------
        # ③ 棚端なら x 方向を補正（前後移動）
        # -----------------------------
        # 棚中央の y の基準値
        CENTER_Y = 4.40
        # 端の判定
        LEFT_EDGE_Y  = 4.55
        RIGHT_EDGE_Y = 4.25

        # 本来の棚位置
        shelf_x, shelf_y, shelf_yaw = self.coordinates["positions"]["shelf"]

        # x補正量（前後調整）
        # 端であればx軸方向に5cm移動する
        x_offset = 0.00

        if grasp_pos.y > LEFT_EDGE_Y:     # 右端
            rospy.loginfo("RIGHT edge detected → move closer by 5cm")
            x_offset = +0.05    # 5cm 前進

        elif grasp_pos.y < RIGHT_EDGE_Y:  # 左端
            rospy.loginfo("LEFT edge detected → move closer by 5cm")
            x_offset = +0.05    # 5cm 前進

        else:
            rospy.loginfo("CENTER → no position adjustment")


        # 補正して棚の前に再度移動
        adjusted_x = shelf_x + x_offset
        omni_base.go_abs(adjusted_x, shelf_y, shelf_yaw)

        # grasp_pos 補正する 
        grasp_pos_corrected = type(grasp_pos)()
        grasp_pos_corrected.x = grasp_pos.x
        grasp_pos_corrected.y = grasp_pos.y
        grasp_pos_corrected.z = grasp_pos.z

        grasp_pos_corrected.x -= (x_offset-0.01)
        rospy.loginfo("Corrected grasp pos: x=%.3f y=%.3f z=%.3f",
                      grasp_pos_corrected.x, grasp_pos_corrected.y, grasp_pos_corrected.z)

        # 正面から把持（補正後）

        self.change_pose("grasp_on_shelf")
        self.grasp_from_front_side(grasp_pos_corrected)
        self.change_pose("all_neutral")

        # 人に届ける
        self.change_pose("look_at_near_floor")

        if target_person == "right":
            self.goto_name("person_b")
        else:
            self.goto_name("person_a")

        self.change_pose("deliver_to_human")
        rospy.sleep(10.0)
        gripper.command(1)
        self.change_pose("all_neutral")
    """

    def deliver_to_target(self, target_obj, target_person):
        """
        棚での物体認識&運搬
        """

        # 1. 棚へ移動
        self.change_pose("look_at_near_floor")
        self.goto_name("shelf")
        self.change_pose("look_at_shelf")

        rospy.loginfo("target_obj: %s  target_person: %s",
                      target_obj, target_person)

        # 2. 物体検出
        detected_objs = self.get_latest_detection()
        grasp_bbox = self.get_most_graspable_bboxes_by_label(
            detected_objs.bboxes, target_obj
        )
        if grasp_bbox is None:
            rospy.logwarn("Cannot find object. aborted.")
            return

        grasp_pos = self.get_grasp_coordinate(grasp_bbox)
        rospy.loginfo("Object pos 3D: x=%.3f y=%.3f z=%.3f",
                      grasp_pos.x, grasp_pos.y, grasp_pos.z)

        # 3. 棚端なら x方向（左右）を補正
        # 棚の中央（== coordinates.json の shelf.x）
        shelf_x, shelf_y, shelf_yaw = self.coordinates["positions"]["shelf"]

        CENTER_X = shelf_x              # 2.30
        LEFT_EDGE_X = CENTER_X - 0.15  # 2.15（棚の左端）
        RIGHT_EDGE_X = CENTER_X + 0.15  # 2.45（棚の右端）

        # ロボットが前後に近づくための x方向補正
        # → grasp_from_front_side が奥へ差し込むので、棚により近づく
        x_offset = 0.00

        if grasp_pos.x < LEFT_EDGE_X:       # 棚の左端
            rospy.loginfo("LEFT edge detected → move closer by 5cm")
            x_offset = +0.05

        elif grasp_pos.x > RIGHT_EDGE_X:    # 棚の右端
            rospy.loginfo("RIGHT edge detected → move closer by 5cm")
            x_offset = +0.05

        else:
            rospy.loginfo("CENTER area → No adjustment")

        # 4. 補正した棚前地点へ再移動

        adjusted_x = shelf_x + x_offset
        omni_base.go_abs(adjusted_x, shelf_y, shelf_yaw)

        # 5. grasp_pos もロボット基準で補正

        grasp_pos_corrected = type(grasp_pos)()
        grasp_pos_corrected.x = grasp_pos.x
        grasp_pos_corrected.y = grasp_pos.y  # アームが奥に行き過ぎる場合はここを調整
        grasp_pos_corrected.z = grasp_pos.z

        rospy.loginfo("Corrected pos: x=%.3f y=%.3f z=%.3f",
                      grasp_pos_corrected.x,
                      grasp_pos_corrected.y,
                      grasp_pos_corrected.z)

        # 6.正面から把持

        self.change_pose("grasp_on_shelf")
        self.grasp_from_front_side(grasp_pos_corrected)
        self.change_pose("all_neutral")

        # 7. 人へ届ける
        self.change_pose("look_at_near_floor")

        if target_person == "right":
            self.goto_name("person_b")
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

    def is_path_safety_margin(self, start_pos, target_pos):
        """
        startからtargetへの移動ルートにおいて、
        「最も近い障害物との距離」を返す。
        """
        min_dist_found = 999.0  # 十分大きな値で初期化

        # 記憶している全障害物との距離をチェック
        for obs in self.obstacle_memory:
            obs_pos = [obs.x, obs.y]

            # 線分（移動ルート）と障害物の距離を計算
            # ※ get_dist_point_to_segment は元のコードにある既存関数を利用
            dist = self.get_dist_point_to_segment(obs_pos, start_pos, target_pos)

            # 記録されている中で「最もルートに近い障害物」までの距離を更新
            min_dist_found = min(min_dist_found, dist)

        return min_dist_found

    def execute_avoid_blocks(self):
        """
        【修正・最終版】
        1. 自分フィルタ追加 (半径0.35m以内は無視) ← NEW!
        2. 止まらずに進む「突破型」ロジック
        3. Y軸ステップ細分化 & X軸探索範囲拡大
        """
        rospy.loginfo("#### Start Advanced Lookahead Avoidance (Self-Filter Enabled) ####")

        # 設定
        y_steps = [2.2, 2.5, 2.8, 3.1, 3.5]
        candidate_xs = [x * 0.02 for x in range(110, 151)]  # 2.20m ~ 3.02m

        # 現在地取得

        current_pose = self.get_relative_coordinate("map", "base_footprint")
        current_x = current_pose.translation.x
        current_y = current_pose.translation.y

        for i, target_y in enumerate(y_steps):
            rospy.loginfo(f"--- Planning for Step {i+1} (Target Y={target_y}) ---")

            # -------------------------------------------------
            # 1. 認識
            # -------------------------------------------------
            head_tilts = [-0.5, -1.1] 
            head_pans = [0.8, 0.0, -0.8] 

            for tilt in head_tilts:
                for pan in head_pans:
                    whole_body.move_to_joint_positions({
                        "head_tilt_joint": tilt, "head_pan_joint": pan})
                    rospy.sleep(0.6)

                    detected_objs = self.get_latest_detection()
                    for bbox in detected_objs.bboxes:
                        if bbox.score < 0.3:
                            continue
                        pos = self.get_grasp_coordinate(bbox)
                        if pos is None:
                            continue

                        # 【追加】自分フィルタ
                        # ロボット中心からの距離を計算
                        dist_from_self = math.sqrt((current_x - pos.x)**2 +
                                                   (current_y - pos.y)**2)
                        
                        # 半径35cm以内なら「自分」として無視
                        if dist_from_self < 0.35:
                            rospy.loginfo(f"Ignoring self/noise at ({pos.x:.2f}, {pos.y:.2f}), "
                                          f"dist={dist_from_self:.2f}")
                            continue

                        # 既存の重複チェック
                        is_known = False
                        for mem in self.obstacle_memory:
                            dist = math.sqrt((mem.x - pos.x)**2 + (mem.y - pos.y)**2)
                            if dist < 0.3:
                                is_known = True
                                break
                        if not is_known:
                            self.obstacle_memory.append(pos)
                            rospy.loginfo(f"Obstacle added: ({pos.x:.2f}, {pos.y:.2f})")

            # -------------------------------------------------
            # 2. ルートプランニング (絶対に止まらない)
            # -------------------------------------------------
            if i + 1 < len(y_steps):
                next_target_y = y_steps[i + 1]
                is_last_step = False
            else:
                next_target_y = target_y + 0.5
                is_last_step = True

            best_x1 = None
            max_route_score = -999.0

            for x1 in candidate_xs:
                margin1 = self.is_path_safety_margin([current_x, current_y], [x1, target_y])

                route_score_for_x1 = -1.0

                if is_last_step:
                    route_score_for_x1 = margin1
                else:
                    best_margin2 = -999.0
                    for x2 in candidate_xs:
                        margin2 = self.is_path_safety_margin([x1, target_y], [x2, next_target_y])
                        best_margin2 = max(best_margin2, margin2)

                    route_score_for_x1 = min(margin1, best_margin2)

                if route_score_for_x1 > max_route_score:
                    max_route_score = route_score_for_x1
                    best_x1 = x1

            # -------------------------------------------------
            # 3. 移動実行
            # -------------------------------------------------
            if best_x1 is not None:
                if max_route_score < 0.22:
                    rospy.logwarn(f"Narrow path: Margin={max_route_score:.3f}m")
                else:
                    rospy.loginfo(f"Safe path: Margin={max_route_score:.3f}m")

                self.goto_pos([best_x1, target_y, 90])
                rospy.sleep(1.0)
                current_x = best_x1
                current_y = target_y
            else:
                # 万が一候補がない場合（理論上ありえないが）
                rospy.logerr("CRITICAL: No candidates found. Staying.")

    def select_next_waypoint(self, current_stp, pos_bboxes):
        """
        waypoints から近い場所にあるものを除外し、最適なwaypointを返す。
        """
        interval = 0.45
        pos_xa = 1.7
        pos_xb = pos_xa + interval
        pos_xc = pos_xb + interval

        # xa配列はcurrent_stpに関係している
        waypoints = {"xa": [[pos_xa, 2.5, 45], [pos_xa, 2.9, 45], [pos_xa, 3.3, 90]],
                     "xb": [[pos_xb, 2.5, 90], [pos_xb, 2.9, 90], [pos_xb, 3.3, 90]],
                     "xc": [[pos_xc, 2.5, 135], [pos_xc, 2.9, 135], [pos_xc, 3.3, 90]]
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
            if not current_y_min < pos_y < current_y_max:
                continue

            rospy.loginfo(f"Checking obstacle at "
                          f"(x={pos_x:.2f}, y={pos_y:.2f}) for step {current_stp}")

            # NOTE Hint:ｙ座標次第で無視してよいオブジェクトもある。
            if pos_xa - safety_margin < pos_x < pos_xa + safety_margin:
                is_to_xa = False
                rospy.loginfo("is_to_xa=False")
                continue
            if pos_xb - safety_margin < pos_x < pos_xb + safety_margin:
                is_to_xb = False
                rospy.loginfo("is_to_xb=False")
                continue
            if pos_xc - safety_margin < pos_x < pos_xc + safety_margin:
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

    # ----------------------------------------------------------
    #  【改造】ここだけ書き換え：自前のONNX検出を使って引き出しを開ける
    # ----------------------------------------------------------
    def open_all_drawers(self):
        """
        [改造版] YOLO(ONNX)で検出した座標を使って、3つの引き出しを全て開ける
        """
        rospy.loginfo("#### Opening All Drawers (YOLO/ONNX Mode) ####")

        # 1. 棚の前へ移動
        self.goto_name("stair_like_drawer")
        self.change_pose("look_at_near_floor")
        rospy.sleep(2.0)  # 画像安定待ち

        # 2. 【重要】ここで既存の get_latest_detection ではなく、自作の detect_handles_onnx を使う
        handles = self.detect_handles_onnx()

        rospy.loginfo("取っ手を %d 個 発見しました！", len(handles))

        if not handles:
            rospy.logwarn("取っ手が見つかりませんでした。best.onnxモデルまたはカメラを確認してください。")
            return

        # 3. 見つかった取っ手の「3次元座標」を全部計算する
        # (既存の get_grasp_coordinate はそのまま再利用できる)
        handle_positions = []
        for bbox in handles:
            pos = self.get_grasp_coordinate(bbox)
            if pos:
                handle_positions.append(pos)
                rospy.loginfo("Handle Pos: x=%.2f, y=%.2f, z=%.2f", pos.x, pos.y, pos.z)

        # 4. 座標をもとに「左」「上」「下」を自動判定する
        drawer_left = None
        drawer_top = None
        drawer_bottom = None

        right_side_handles = []

        for pos in handle_positions:
            # ロボットから見て左(Y > 0.05) なら左の引き出し
            if pos.y > 0.05:
                drawer_left = pos
            else:
                right_side_handles.append(pos)

        # 右側にあるものを高さ(Z)が高い順に並べる
        right_side_handles.sort(key=lambda p: p.z, reverse=True)

        if len(right_side_handles) >= 1:
            drawer_top = right_side_handles[0]
        if len(right_side_handles) >= 2:
            drawer_bottom = right_side_handles[1]

        # 5. 判定できた場所を開けに行く
        if drawer_left:
            rospy.loginfo("【左】の引き出しを開けます")
            self.pull_out_trofast(drawer_left.x, drawer_left.y, drawer_left.z, -90, 0, 0)
        else:
            rospy.logwarn("【警告】左の引き出しが見つかりませんでした")

        if drawer_top:
            rospy.loginfo("【右上】の引き出しを開けます")
            self.pull_out_trofast(drawer_top.x, drawer_top.y, drawer_top.z, -90, 0, 0)
        else:
            rospy.logwarn("【警告】右上の引き出しが見つかりませんでした")

        if drawer_bottom:
            rospy.loginfo("【右下】の引き出しを開けます")
            self.pull_out_trofast(drawer_bottom.x, drawer_bottom.y, drawer_bottom.z, -90, 0, 0)
        else:
            rospy.logwarn("【警告】右下の引き出しが見つかりませんでした")

        rospy.loginfo("All drawers logic finished.")

    def execute_task1(self):
        """
        task1を実行する
        """
        rospy.loginfo("#### start Task 1 ####")
        hsr_position = [
            ("tall_table", "look_at_tall_table"),
            ("near_long_table_l", "look_at_near_floor"),
            ("long_table_r", "look_at_tall_table"),
        ]

        food_cnt = 0
        tool_cnt = 0
        for plc, pose in hsr_position:
            # for _ in range(self.DETECT_CNT):
            while True:
                # 移動と視線指示
                self.goto_name(plc)
                self.change_pose(pose)
                gripper.command(0)

                # 把持対象の有無チェック
                detected_objs = self.get_latest_detection()
                graspable_obj = self.get_most_graspable_obj(detected_objs.bboxes)

                if graspable_obj is None:
                    rospy.logwarn("Cannot determine object to grasp. Grasping is aborted.")
                    # continue
                    break

                label = graspable_obj["label"]
                grasp_bbox = graspable_obj["bbox"]
                rospy.loginfo("grasp the " + label)

                # 把持対象がある場合は把持関数実施
                grasp_pos = self.get_grasp_coordinate(grasp_bbox)
                self.change_pose("grasp_on_table")
                is_success = self.exec_graspable_method(grasp_pos, label)
                self.change_pose("all_neutral")

                if not is_success:
                    # ここで失敗した物体を除外する
                    rospy.logwarn("Failed to grasp [%s]", label)
                    if label not in self.IGNORE_LIST:
                        self.IGNORE_LIST.append(label)
                    break

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
                    food_cnt += 1  # Food専用カウンターを増やす
                elif most_likely_label == "KITCHEN_ITEM":
                    # カテゴリ: Kitchen items -> Container_A 
                    rospy.loginfo("カテゴリ [Kitchen] -> Container A")
                    self.put_in_place("container_A", "put_in_container_pose")

                elif most_likely_label == "TOOL_ITEM":
                    # カテゴリ: Tools -> Drawer_top / Drawer_bottom 
                    rospy.loginfo("カテゴリ [Tools] -> Drawer Top / Bottom")
                    if tool_cnt % 2 == 0:
                        self.put_in_place("drawer_top", "put_in_drawer_pose")
                    else:
                        self.put_in_place("drawer_bottom", "put_in_drawer_pose")
                    tool_cnt += 1  # Tool専用カウンターを増やす

                elif most_likely_label == "SHAPE_ITEM":
                    # カテゴリ: Shape items -> Drawer_left 
                    rospy.loginfo("カテゴリ [Shape] -> Drawer left")
                    self.put_in_place("drawer_left", "put_in_drawer_pose")

                elif most_likely_label == "TASK_ITEM":
                    # カテゴリ: Task items -> Bin_A 
                    rospy.loginfo("カテゴリ [Task] -> Bin A")
                    self.put_in_place("bin_a_place", "put_in_bin")  # 既存のbin_a_placeを使用

                else:
                    # カテゴリ: Unknown objects -> Bin_B 
                    rospy.logwarn("ラベル [%s] は分類外です。[Unknown] として Bin B に置きます。", label)
                    self.put_in_place("bin_b_place", "put_in_bin")  # 既存のbin_b_placeを使用

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
            rospy.loginfo("recieved instruction: %s", latest_instruction)
        else:
            rospy.logwarn("instruction_list is None")
            return

        # 命令内容を解釈
        target_obj, target_person = self.extract_target_obj_and_person(latest_instruction)

        # 指定したオブジェクトを指定した配達先へ
        if target_obj and target_person:
            self.deliver_to_target(target_obj, target_person)

    def run(self):
        """
        全てのタスクを実行する
        """
        self.change_pose("all_neutral")

        # 1. まず自前ONNXで引き出しを開ける
        self.open_all_drawers()

        # 2. その後、既存タスクを実行
        self.execute_task1()

        self.execute_task2a()
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