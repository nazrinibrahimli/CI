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

    # 检查拇指是否远离其他手指
    distance_threshold = frame_shape[1] * 0.15  # 基于图像宽度的自适应阈值
    thumb_far_from_others = all(d > distance_threshold for d in thumb_distances)

    # 检查其他手指是否并拢
    other_fingers = np.array([index_tip_px, middle_tip_px, ring_tip_px, pinky_tip_px])
    n_others = len(other_fingers)
    other_distances = []
    for i in range(n_others):
        for j in range(i + 1, n_others):
            dist = np.linalg.norm(other_fingers[i] - other_fingers[j])
            other_distances.append(dist)

    other_fingers_close = np.mean(other_distances) < frame_shape[1] * 0.1

    # 检查拇指是否朝上
    thumb_mcp = np.array([landmarks.landmark[2].x, landmarks.landmark[2].y]) * scale
    thumb_vector = thumb_tip_px - thumb_mcp
    is_pointing_up = thumb_vector[1] < -frame_shape[0] * 0.1

    return thumb_far_from_others and other_fingers_close and is_pointing_up


def is_open_hand(landmarks, frame_shape):
    """
    通过计算所有手指尖之间的平均距离来判断是否为张开手掌
    距离大意味着手指张开
    """
    # 获取手指尖的坐标（不包括拇指）
    finger_tips = np.array([
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
    threshold = frame_shape[1] * 0.15  # 基于图像宽度的自适应阈值

    return avg_distance > threshold


def process_frame(frame):
    global last_gesture, gesture_start_time, entered_sequence, last_stable_gesture, sequence_cooldown
    global last_hand_detected_time, recognition_complete, welcome_message

    if frame is None:
        return frame, "None"

    # 如果已经完成识别，显示欢迎信息并返回
    if recognition_complete and welcome_message:
        # 创建纯色背景
        result_frame = np.zeros_like(frame)
        # 显示欢迎信息
        cv2.putText(result_frame, welcome_message,
                    (50, 240), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        return result_frame, welcome_message

    current_time = time.time()

    # 检查冷却时间
    if sequence_cooldown > 0 and current_time - sequence_cooldown >= COOLDOWN_DURATION:
        sequence_cooldown = 0

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_gesture = "None"
    stable_gesture = "None"
    display_gesture = "None"

    # 检查是否检测到手
    if results.multi_hand_landmarks:
        # 更新最后检测到手的时间
        last_hand_detected_time = current_time
    else:
        # 检查无手势持续时间
        if current_time - last_hand_detected_time > NO_HAND_TIMEOUT and entered_sequence:
            # 如果超过3秒没有检测到手势且序列不为空，清空序列
            entered_sequence.clear()
            last_stable_gesture = None
            last_gesture = None

            # 显示超时提示
            cv2.putText(frame, "Sequence cleared - No hand detected",
                        (10, 300), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

    # 如果序列不为空，显示倒计时
    if entered_sequence and not results.multi_hand_landmarks:
        time_left = max(0, NO_HAND_TIMEOUT - (current_time - last_hand_detected_time))
        if time_left > 0:
            cv2.putText(frame, f"Clearing sequence in: {time_left:.1f}s",
                        (10, 300), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

    # 显示当前序列状态
    if entered_sequence:
        sequence_text = " → ".join(entered_sequence)
        cv2.putText(frame, "Current Sequence:",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2)
        cv2.putText(frame, sequence_text,
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2)

        # 进度条显示
        max_sequence = 3
        progress_width = 300
        progress_height = 20
        segment_width = progress_width // max_sequence

        for i in range(max_sequence):
            start_x = 10 + i * segment_width
            cv2.rectangle(frame,
                          (start_x, 170),
                          (start_x + segment_width - 5, 170 + progress_height),
                          (100, 100, 100),
                          2)

        for i in range(len(entered_sequence)):
            start_x = 10 + i * segment_width
            cv2.rectangle(frame,
                          (start_x, 170),
                          (start_x + segment_width - 5, 170 + progress_height),
                          (0, 255, 0),
                          -1)
            gesture_text = entered_sequence[i][:4]
            cv2.putText(frame, gesture_text,
                        (start_x + 5, 185),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 0, 0), 1)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if is_thumb_up(hand_landmarks, frame.shape):
            current_gesture = "THUMB_UP"
            thumb_tip = hand_landmarks.landmark[4]
            x = int(thumb_tip.x * frame.shape[1])
            y = int(thumb_tip.y * frame.shape[0])
            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
        elif is_fist(hand_landmarks, frame.shape):
            current_gesture = "FIST"
        elif is_open_hand(hand_landmarks, frame.shape):
            current_gesture = "OPEN_HAND"

        if current_gesture != last_gesture:
            gesture_start_time = current_time
            last_gesture = current_gesture
        else:
            elapsed_time = current_time - gesture_start_time
            remaining_time = max(0, GESTURE_DURATION - elapsed_time)

            if remaining_time > 0 and current_gesture != "None":
                cv2.putText(frame, f"Hold for {remaining_time:.1f}s",
                            (10, 220), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 255), 2)

                progress = (GESTURE_DURATION - remaining_time) / GESTURE_DURATION
                bar_width = int(200 * progress)
                cv2.rectangle(frame, (10, 230), (210, 245), (0, 0, 255), 2)
                cv2.rectangle(frame, (10, 230), (10 + bar_width, 245),
                              (0, 255, 0), -1)

            if elapsed_time >= GESTURE_DURATION:
                stable_gesture = current_gesture

                # 只有当手势稳定且不是"None"时才添加到序列中
                if stable_gesture != last_stable_gesture and sequence_cooldown == 0 and stable_gesture != "None":
                    entered_sequence.append(stable_gesture)
                    last_stable_gesture = stable_gesture
                    sequence_cooldown = current_time

                    sequence_tuple = tuple(entered_sequence)
                    if sequence_tuple in GESTURE_USER_MAPPING:
                        user = GESTURE_USER_MAPPING[sequence_tuple]
                        welcome_message = f"Welcome, {user['name']} (ID: {user['id']})"
                        recognition_complete = True
                        # 关闭摄像头
                        release_camera()
                        # 创建结果帧
                        result_frame = np.zeros_like(frame)
                        cv2.putText(result_frame, welcome_message,
                                    (50, 240), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 255, 0), 2)
                        return result_frame, welcome_message
                    elif len(entered_sequence) > 3:
                        display_gesture = "Invalid sequence. Try again."
                        entered_sequence.clear()
                        last_stable_gesture = None
                    else:
                        display_gesture = stable_gesture
                else:
                    display_gesture = stable_gesture
    else:
        last_gesture = None
        gesture_start_time = current_time

    if display_gesture == "None" and current_gesture != "None":
        display_gesture = current_gesture

    cv2.putText(frame, f"Gesture: {display_gesture}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if sequence_cooldown > 0:
        cooldown_remaining = max(0, COOLDOWN_DURATION - (current_time - sequence_cooldown))
        if cooldown_remaining > 0:
            cv2.putText(frame, f"Next gesture in: {cooldown_remaining:.1f}s",
                        (10, 270), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)

    return frame, display_gesture


@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if recognition_complete:
                # 创建一个黑色背景的图像
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                # 显示欢迎信息
                cv2.putText(frame, welcome_message,
                           (50, 240), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 255, 0), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                break  # 结束视频流

            with camera_lock:
                camera = get_camera()
                if not camera.isOpened():
                    continue

                success, frame = camera.read()
                if not success:
                    continue

                frame, _ = process_frame(frame)

                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue

                frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/gesture_status')
def gesture_status():
    def generate():
        while True:
            if recognition_complete:
                yield f"data: {welcome_message}\n\n"
                break

            with camera_lock:
                camera = get_camera()
                if not camera.isOpened():
                    continue

                success, frame = camera.read()
                if not success:
                    continue

                _, gesture = process_frame(frame)

            yield f"data: {gesture}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route('/')
def index():
    return render_template('index.html')


@app.teardown_appcontext
def cleanup(exception=None):
    release_camera()


if __name__ == '__main__':
    app.run(debug=True, port=8000)