import cv2
import mediapipe as mp
import numpy as np
import socket

# 配置Socket通信
ubuntu_ip = "192.168.177.129"  # 替换为Ubuntu的IP地址 Replace with the IP address of Ubuntu
ubuntu_port = 12345            # 设置一个未被占用的端口号 Set a free port number
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 初始化 MediaPipe Initializing MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 初始化摄像头
cap = cv2.VideoCapture(0)  # 摄像头 ID，通常为 0

# 检查摄像头是否打开
if not cap.isOpened():
    print("Error: Cannot access the camera.")
    exit()

# 初始化标志变量
prev_index_x = None  # 记录食指指尖的上一次水平位置
index_up_mode = False  # 当前是否处于单独竖起食指模式

# 判断是否为单独竖起食指 Determine whether the index finger is raised alone
def is_index_up(finger_tips):
    index = finger_tips[1]
    other_tips = [finger_tips[0], finger_tips[2], finger_tips[3], finger_tips[4]]
    return all(np.linalg.norm(index - other) > 50 for other in other_tips)

# 判断是否为握拳 Determine whether it is a fist
def is_fist(finger_tips, threshold=50):
    n_fingers = len(finger_tips)
    distances = []
    for i in range(n_fingers):
        for j in range(i + 1, n_fingers):
            dist = np.linalg.norm(finger_tips[i] - finger_tips[j])
            distances.append(dist)
    avg_distance = np.mean(distances)
    return avg_distance < threshold

# 判断是否为张开手掌 Determine whether the palm is open
def is_open_hand(finger_tips, threshold=100):
    n_fingers = len(finger_tips)
    distances = []
    for i in range(n_fingers):
        for j in range(i + 1, n_fingers):
            dist = np.linalg.norm(finger_tips[i] - finger_tips[j])
            distances.append(dist)
    avg_distance = np.mean(distances)
    return avg_distance > threshold

# 判断是否为大拇指单独竖起 Determine whether the thumb is raised alone
def is_thumb_up(finger_tips):
    """
    判断是否为大拇指单独竖起：
    - 大拇指远离其他手指。
    - 其他手指呈并拢状态。
    """
    thumb, index, middle, ring, pinky = finger_tips
    thumb_far_from_others = (
        np.linalg.norm(thumb - index) > 50 and
        np.linalg.norm(thumb - middle) > 50 and
        np.linalg.norm(thumb - ring) > 50 and
        np.linalg.norm(thumb - pinky) > 50
    )
    other_fingers = [index, middle, ring, pinky]
    other_fingers_close = is_fist(other_fingers, threshold=50)  # 调用 is_fist 函数
    return thumb_far_from_others and other_fingers_close

# 实时检测
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # 转换为RGB并使用 MediaPipe 进行手部关键点检测
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 绘制手部关键点
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 提取指尖关键点
            finger_tips = np.array([
                [hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y],
                [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y],
                [hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y],
                [hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y],
                [hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y],
            ]) * frame.shape[1::-1]  # 转换为像素坐标

            # 检测单独竖起食指
            if is_index_up(finger_tips):
                if not index_up_mode:
                    index_up_mode = True
                    command = "INDEX_UP"
                    client_socket.sendto(command.encode(), (ubuntu_ip, ubuntu_port))
                    print("INDEX_UP - 开始检测方向")
                # 检测向左或向右（基于食指位置变化）
                index_tip_x = finger_tips[1][0]  # 获取食指指尖的水平坐标
                if prev_index_x is not None:
                    index_x_diff = index_tip_x - prev_index_x
                    if index_x_diff > 10:
                        command = "LEFT_SWIPE"
                        client_socket.sendto(command.encode(), (ubuntu_ip, ubuntu_port))
                        print("LEFT_SWIPE")
                    elif index_x_diff < -10:
                        command = "RIGHT_SWIPE"
                        client_socket.sendto(command.encode(), (ubuntu_ip, ubuntu_port))
                        print("RIGHT_SWIPE")
                prev_index_x = index_tip_x
                continue  # 跳过其他手势检测

            # 如果未单独竖起食指，退出方向检测模式
            index_up_mode = False

            # 检测握拳
            if is_fist(finger_tips):
                command = "FIST"
                client_socket.sendto(command.encode(), (ubuntu_ip, ubuntu_port))
                print("FIST - 前进")

            # 检测张开手掌
            elif is_open_hand(finger_tips):
                command = "OPEN_HAND"
                client_socket.sendto(command.encode(), (ubuntu_ip, ubuntu_port))
                print("OPEN_HAND - 停止")

            # 检测大拇指单独竖起
            elif is_thumb_up(finger_tips):
                command = "THUMB_UP"
                client_socket.sendto(command.encode(), (ubuntu_ip, ubuntu_port))
                print("THUMB_UP - 后退")

    # 显示检测结果
    cv2.imshow("Hand Gesture Detection", frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
client_socket.close()
