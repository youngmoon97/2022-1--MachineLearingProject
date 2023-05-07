# openCV, mediapipe, numpy 모듈 사용

import cv2
import mediapipe as mp
import numpy as np
import time, os

# 총 8가지의 액션을 만든다.
# 차례대로 0, 1, 2, 3 순서로

actions = ['good', 'bad', 'ok', 'love', 'call', 'hello', 'no', 'come']
seq_length = 40 # 윈도우 사이즈 40
secs_for_action = 5 # 액션을 녹화하는 시간 40초 (데이터를 많이 수집하여 정확도를 높이기 위해 시간을 좀 더 늘렸다.)

# MediaPipe hands model를 생성하는 구문
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# OpenCV. 웹캠을 생성
cap = cv2.VideoCapture(0)

created_time = int(time.time())
os.makedirs('dataset', exist_ok=True) # 데이터셋을 저장할 폴더를 만든다.

# 제스처마다 녹화를 하도록 한다.
while cap.isOpened():
    for idx, action in enumerate(actions):
        data = []

        # 첫 번째 이미지를 읽어 flip 시킨다.
        ret, img = cap.read()

        img = cv2.flip(img, 1)

        # 어떤 제스처를 취해야 하는지 3초동안 보여준다.
        cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.imshow('img', img)
        cv2.waitKey(3000)

        start_time = time.time()

        # 40초동안 반복한다.
        while time.time() - start_time < secs_for_action:
            ret, img = cap.read()

            img = cv2.flip(img, 1) # 하나씩 프레임을 읽는다.
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img) # 결과를 MediaPipe에 넣는다.
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # 각도를 뽑아내는 구문. x, y, z, visibility의 각도를 뽑아낸다.
            # visibility : 손가락 랜드마크가 이미지 상에서 보이는지 안 보이는지 판단
            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    # Compute angles between joints
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                    v = v2 - v1 # [20, 3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                    angle = np.degrees(angle) # Convert radian to degree

                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label = np.append(angle_label, idx)   # label을 넣어주는 구문. 0, 1, 2 순서대로 label을 넣어준다.

                    d = np.concatenate([joint.flatten(), angle_label])  # x, y, z, visibility을 펼쳐서 concatenate(여러개의 텍스트를 하나로 합치는) 시킨다.  

                    data.append(d) # data에 전부 첨부 시킨다.

                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break

        # data를 40초간 전부 모으면 np.array 형태로 변환시킨다. 
        data = np.array(data)
        print(action, data.shape)
        np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), data)

        # 시퀀스 데이터 만들기
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data)
    break
