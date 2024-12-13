import cv2
from ultralytics import YOLO
import numpy as np
import socket

# 配置Socket通信
ubuntu_ip = "192.168.177.129"  # 替换为Ubuntu的IP地址
ubuntu_port = 12345            # 设置一个未被占用的端口号
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 加载训练好的模型 Load the trained model
model = YOLO("best.pt")

# 初始化摄像头
cap = cv2.VideoCapture(0)  # 摄像头 ID，通常为 0

# 检查摄像头是否打开
if not cap.isOpened():
    print("Error: Cannot access the camera.")
    exit()

# 初始化标志变量
prev_wrist_x = None
is_fist_detected = False  # 标记是否检测到握拳
is_thumb_up_detected = False  # 标记是否检测到竖大拇指
is_open_hand_detected = False  # 标记是否已经检测到张开手掌

# 判断是否为握拳 Determine whether it is a fist
#finger_tips: An array containing the coordinates of all fingertips
#threshold: Distance threshold (default value is 50),
# used to determine whether the fingertips are close to each other
def is_fist(finger_tips, threshold=50):
    n_fingers = len(finger_tips)
    distances = []
    for i in range(n_fingers):
        for j in range(i + 1, n_fingers):
            dist = np.linalg.norm(finger_tips[i] - finger_tips[j])
            distances.append(dist)
    avg_distance = np.mean(distances)
    return avg_distance < threshold

# 判断是否为竖大拇指手势 Determine whether it is a thumbs-up gesture
def is_thumb_up(finger_tips, wrist_coords):
    thumb, index, middle, ring, pinky = finger_tips
    thumb_far_from_others = (
        np.linalg.norm(thumb - index) > 50 and
        np.linalg.norm(thumb - middle) > 50 and
        np.linalg.norm(thumb - ring) > 50 and
        np.linalg.norm(thumb - pinky) > 50
    )
    other_fingers = [index, middle, ring, pinky]
    other_fingers_close = is_fist(other_fingers, threshold=50)
    return thumb_far_from_others and other_fingers_close

# 判断是否为张开手掌 Determine whether the palm is open
#如果五指指尖的平均距离大于阈值，表示手势为张开手掌。
# If the average distance between the five fingertips is greater than the threshold,
# it indicates that the gesture is an open palm.
#如果五指指尖的平均距离小于或等于阈值，表示不是张开手掌
#If the average distance between the five fingertips is less than or equal to the threshold,
# it means that the palm is not open.
def is_open_hand(finger_tips, threshold=100):
    n_fingers = len(finger_tips)
    distances = []
    for i in range(n_fingers):
        for j in range(i + 1, n_fingers):
            dist = np.linalg.norm(finger_tips[i] - finger_tips[j])
            distances.append(dist)
    avg_distance = np.mean(distances)
    return avg_distance > threshold

# 实时检测
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # 使用模型进行推理
    results = model.predict(source=frame, show=False, conf=0.7, verbose=False)

    # 检查是否有检测结果
    if results[0].keypoints is not None:
        keypoints = results[0].keypoints.xy
        if keypoints.shape[1] > 0:
            keypoints = keypoints.cpu().numpy()
            wrist_coords = keypoints[0][0]
            wrist_x, wrist_y = wrist_coords[0], wrist_coords[1]

            # 获取指尖关键点
            finger_tip_indices = [4, 8, 12, 16, 20]
            finger_tips = keypoints[0][finger_tip_indices]

            # 检测张开手掌
            if is_open_hand(finger_tips, threshold=100):
                if not is_open_hand_detected:
                    command = "OPEN_HAND"
                    client_socket.sendto(command.encode(), (ubuntu_ip, ubuntu_port))
                    print("OPEN_HAND - 停止")
                    is_open_hand_detected = True  # 标记张开手掌已检测到
                else:
                    # 检测向左或向右挥动
                    if prev_wrist_x is not None:
                        wrist_x_diff = wrist_x - prev_wrist_x
                        if wrist_x_diff > 5:
                            command = "LEFT_SWIPE"
                            client_socket.sendto(command.encode(), (ubuntu_ip, ubuntu_port))
                            print("LEFT_SWIPE")
                        elif wrist_x_diff < -5:
                            command = "RIGHT_SWIPE"
                            client_socket.sendto(command.encode(), (ubuntu_ip, ubuntu_port))
                            print("RIGHT_SWIPE")
                    prev_wrist_x = wrist_x

            # 检测握拳
            elif is_fist(finger_tips, threshold=50):
                command = "FIST"
                client_socket.sendto(command.encode(), (ubuntu_ip, ubuntu_port))
                print("FIST - 前进")
                is_fist_detected = True
                is_thumb_up_detected = False
                is_open_hand_detected = False  # 重置张开手掌标志

            # 检测竖大拇指
            elif is_thumb_up(finger_tips, wrist_coords):
                command = "THUMB_UP"
                client_socket.sendto(command.encode(), (ubuntu_ip, ubuntu_port))
                print("THUMB_UP - 后退")
                is_thumb_up_detected = True
                is_fist_detected = False
                is_open_hand_detected = False  # 重置张开手掌标志
            else:
                is_fist_detected = False
                is_thumb_up_detected = False

    # 显示检测结果
    annotated_frame = results[0].plot()
    cv2.imshow("Hand Gesture Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
client_socket.close()
