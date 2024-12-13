import time

from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import os
import logging
from threading import Lock

# Suppress TensorFlow and MediaPipe logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('mediapipe').setLevel(logging.ERROR)

app = Flask(__name__)

# Global variables
camera_lock = Lock()
camera = None
last_gesture = None
gesture_start_time = 0
GESTURE_DURATION = 1.5  # 手势需要保持的时间（秒）
last_stable_gesture = None  # 跟踪上一个稳定的手势
sequence_cooldown = 0       # 冷却时间控制
COOLDOWN_DURATION = 1.0    # 每个手势之间的冷却时间
last_hand_detected_time = time.time()  # 最后一次检测到手的时间
NO_HAND_TIMEOUT = 3.0  # 无手势超时时间（秒)
recognition_complete = False  # 用于标记是否完成识别
welcome_message = None  # 用于存储欢迎信息

# MediaPipe setup with improved parameters
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Gesture mappings
GESTURE_USER_MAPPING = {
    ("OPEN_HAND", "THUMB_UP", "OPEN_HAND"): {"name": "John Doe", "id": "12345"},

    ("FIST", "THUMB_UP", "FIST"): {"name": "Jane Smith", "id": "67890"},
    ("OPEN_HAND","OPEN_HAND","OPEN_HAND"):  {"name": "OP", "id": "56789"},
    ("FIST","FIST","FIST"):{"name": "FIST_john", "id": "85491"},
    ("THUMB_UP","THUMB_UP","THUMB_UP"):{"name": "THUMB_john", "id": "54982"}



}

entered_sequence = []


def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
    return camera


def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None


def is_fist(landmarks, frame_shape):
    """
    通过计算所有手指尖之间的平均距离来判断是否为拳头
    距离小意味着手指并拢/握拳
    """
    # 获取所有手指尖的坐标
    finger_tips = np.array([
        [landmarks.landmark[4].x, landmarks.landmark[4].y],  # THUMB_TIP
        [landmarks.landmark[8].x, landmarks.landmark[8].y],  # INDEX_TIP
        [landmarks.landmark[12].x, landmarks.landmark[12].y],  # MIDDLE_TIP
        [landmarks.landmark[16].x, landmarks.landmark[16].y],  # RING_TIP
        [landmarks.landmark[20].x, landmarks.landmark[20].y]  # PINKY_TIP
    ])

    # 转换为像素坐标
    finger_tips_px = finger_tips * frame_shape[1::-1]

    # 计算所有手指尖之间的距离
    n_fingers = len(finger_tips_px)
    distances = []
    for i in range(n_fingers):
        for j in range(i + 1, n_fingers):
            dist = np.linalg.norm(finger_tips_px[i] - finger_tips_px[j])
            distances.append(dist)

    avg_distance = np.mean(distances)
    threshold = frame_shape[1] * 0.13  # 基于图像宽度的自适应阈值

    return avg_distance < threshold


def is_thumb_up(landmarks, frame_shape):
    """
    判断是否为大拇指朝上：
    - 拇指需要远离其他手指
    - 其他手指需要并拢
    """
    # 获取所有手指尖的坐标
    thumb_tip = np.array([landmarks.landmark[4].x, landmarks.landmark[4].y])  # THUMB_TIP
    index_tip = np.array([landmarks.landmark[8].x, landmarks.landmark[8].y])  # INDEX_TIP
    middle_tip = np.array([landmarks.landmark[12].x, landmarks.landmark[12].y])  # MIDDLE_TIP
    ring_tip = np.array([landmarks.landmark[16].x, landmarks.landmark[16].y])  # RING_TIP
    pinky_tip = np.array([landmarks.landmark[20].x, landmarks.landmark[20].y])  # PINKY_TIP

    # 转换为像素坐标
    scale = frame_shape[1::-1]
    thumb_tip_px = thumb_tip * scale
    index_tip_px = index_tip * scale
    middle_tip_px = middle_tip * scale
    ring_tip_px = ring_tip * scale
    pinky_tip_px = pinky_tip * scale

    # 计算拇指到其他手指的距离
    thumb_distances = [
        np.linalg.norm(thumb_tip_px - index_tip_px),
        np.linalg.norm(thumb_tip_px - middle_tip_px),
        np.linalg.norm(thumb_tip_px - ring_tip_px),
        np.linalg.norm(thumb_tip_px - pinky_tip_px)
    ]

    # 检查拇指是否远离其�