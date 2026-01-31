import cv2
import numpy as np
import mediapipe as mp
import time
import json
from tensorflow.keras.models import load_model

MAX_FRAMES = 90

def show(frame):
    frame = cv2.resize(frame, (1000, 600))
    cv2.imshow("LIVE", frame)


# load model + labels
model = load_model("latest_model_lstm_2.h5")
with open("latest_dataset_1/label_map.json","r",encoding="utf-8") as f:
    label_map = json.load(f)
idx2label = {v:k for k,v in label_map.items()}

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

### ---- SAME FUNCTIONS YOU HAD ---- ###
def calculate_additional_features(hand_data, handedness):
    features = []
    for hand_idx in range(2):
        hand = hand_data[hand_idx]
        if np.all(hand == 0):
            features.extend([0.0]*5)
            continue
        com = hand.mean(axis=0)
        features.extend(com)
        wrist = hand[0]
        middle_tip = hand[12]
        span = np.linalg.norm(wrist-middle_tip)
        features.append(span)
        fingertips = hand[[8,12,16,20]]
        center = hand[0]
        avg_f = np.mean([np.linalg.norm(ft-center) for ft in fingertips])
        features.append(avg_f)
    while len(features)<10:
        features.append(0.0)
    return np.array(features[:10],dtype=np.float32)

def extract_landmarks(frame, res):
    frame_arr = np.zeros((2,21,3),dtype=np.float32)
    handedness = np.zeros(2,dtype=np.float32)

    if res.multi_hand_landmarks:
        if res.multi_handedness:
            for i,hinfo in enumerate(res.multi_handedness[:2]):
                handedness[i] = 1.0 if hinfo.classification[0].label=="Right" else 0.0

        for i,lm in enumerate(res.multi_hand_landmarks[:2]):
            for j,p in enumerate(lm.landmark):
                frame_arr[i,j] = [p.x,p.y,p.z]

    extra = calculate_additional_features(frame_arr,handedness)

    norm = []
    for h in frame_arr:
        if np.all(h==0):
            norm.extend(h.reshape(-1))
            continue
        wrist = h[0]
        nh = h-wrist
        hand_size = np.linalg.norm(nh[5])
        if hand_size>0:
            nh = nh/hand_size
        norm.extend(nh.reshape(-1))

    combined = np.concatenate([np.array(norm), handedness, extra])
    return combined
### --------------------------------- ###

cap = cv2.VideoCapture(0)
prediction_text = ""

print("READY. Press ENTER to capture 90 frames and predict.")

while True:
    ret, frame = cap.read()
    if not ret: break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands_detector.process(rgb)

    # draw landmarks live
    if res.multi_hand_landmarks:
        for handLms in res.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # draw current prediction text
    if prediction_text != "":
        cv2.putText(frame, prediction_text, (10,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

    show(frame)


    key = cv2.waitKey(1)
    if key == 13:   # ENTER
        print("wait 2 seconds... get your pose")
        time.sleep(2)
        seq=[]
        print("Capturing frames...")

        for i in range(MAX_FRAMES):
            ret, frame = cap.read()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands_detector.process(rgb)

            lm = extract_landmarks(frame, res)
            seq.append(lm)

            # draw while capturing also
            if res.multi_hand_landmarks:
                for handLms in res.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            show(frame)
            cv2.waitKey(1)

        seq = np.array(seq,dtype=np.float32)
        seq = np.expand_dims(seq,axis=0)

        pred = model.predict(seq)[0]
        cls_id = int(np.argmax(pred))
        cls = idx2label[cls_id]
        conf = float(pred[cls_id])*100

        prediction_text = f"{cls.upper()}  {conf:.1f}%"
        print("PREDICTION:", prediction_text)

    elif key==27:
        break

cap.release()
cv2.destroyAllWindows()