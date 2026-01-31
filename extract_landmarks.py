# extract_landmarks.py
import os
import cv2
import numpy as np
import mediapipe as mp
import json
from tqdm import tqdm

DATA_DIR = "data"
OUTPUT_DIR = "latest_dataset_1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_FRAMES = 90
SAMPLING = True

# Enhanced MediaPipe configuration
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose  # Add pose detection for body context
mp_face = mp.solutions.face_mesh  # Add face detection for expressions

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.3,  # Lowered to detect more subtle hand movements
    min_tracking_confidence=0.3
)

# Optional: Add pose detector for body context
pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_frames_from_video(path, target_fps=30):
    """Extract frames with FPS control and validation"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Warning: Could not open video {path}")
        return []
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Handle FPS sampling if needed
        if original_fps > target_fps:
            # Skip frames to match target FPS
            skip_factor = int(original_fps / target_fps)
            if frame_count % skip_factor == 0:
                frames.append(frame)
        else:
            frames.append(frame)
            
        frame_count += 1
    
    cap.release()
    
    if len(frames) == 0:
        print(f"Warning: No frames extracted from {path}")
    
    return frames

def extract_enhanced_landmarks_for_frames(frames, max_frames=MAX_FRAMES):
    """
    Extract enhanced landmarks including:
    - Hand landmarks (both hands)
    - Handedness information
    - Pose landmarks (optional)
    - Relative positioning
    """
    seq = []
    hand_presence = []  # Track hand presence per frame
    
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        
        # Process hands
        hand_results = hands_detector.process(rgb)
        
        # Initialize frame data
        frame_data = np.zeros((2, 21, 3), dtype=np.float32)  # Two hands
        handedness = np.zeros(2, dtype=np.float32)  # 0=left, 1=right, 0.5=unknown
        
        if hand_results.multi_hand_landmarks:
            # Get handedness information
            if hand_results.multi_handedness:
                for i, hand_info in enumerate(hand_results.multi_handedness[:2]):
                    handedness[i] = 1.0 if hand_info.classification[0].label == "Right" else 0.0
            
            # Extract landmarks
            for i, lm in enumerate(hand_results.multi_hand_landmarks[:2]):
                arr = np.array([[p.x, p.y, getattr(p, "z", 0.0)] for p in lm.landmark], dtype=np.float32)
                frame_data[i] = arr
        
        # Calculate additional features
        features = calculate_additional_features(frame_data, handedness)
        
        # Combine all features
        combined_features = np.concatenate([
            frame_data.reshape(-1),  # 126 features (2*21*3)
            handedness,              # 2 features
            features                 # Additional calculated features
        ])
        
        seq.append(combined_features)
        hand_presence.append(1.0 if hand_results.multi_hand_landmarks else 0.0)
    
    # Handle sequence length normalization
    if len(seq) == 0:
        # Create zero-padded sequence
        feature_dim = seq[0].shape[0] if seq else 126 + 2 + 10  # Default dimension
        seq = [np.zeros(feature_dim, dtype=np.float32)] * max_frames
        hand_presence = [0.0] * max_frames
    else:
        if len(seq) > max_frames:
            if SAMPLING:
                idx = np.linspace(0, len(seq) - 1, max_frames, dtype=int)
                seq = [seq[i] for i in idx]
                hand_presence = [hand_presence[i] for i in idx]
            else:
                seq = seq[:max_frames]
                hand_presence = hand_presence[:max_frames]
        elif len(seq) < max_frames:
            feature_dim = seq[0].shape[0]
            pad = [np.zeros(feature_dim, dtype=np.float32)] * (max_frames - len(seq))
            seq.extend(pad)
            hand_presence.extend([0.0] * (max_frames - len(seq)))
    
    # Add hand presence as additional channel
    final_seq = np.stack(seq).astype(np.float32)
    hand_presence = np.array(hand_presence).astype(np.float32)
    
    return final_seq, hand_presence

def calculate_additional_features(hand_data, handedness):
    """Calculate additional spatial and temporal features"""
    features = []
    
    # For each detected hand
    for hand_idx in range(2):
        hand = hand_data[hand_idx]
        
        if np.all(hand == 0):  # No hand detected
            features.extend([0.0] * 5)
            continue
        
        # 1. Hand center of mass
        com = hand.mean(axis=0)
        features.extend(com)
        
        # 2. Hand span (distance between wrist and middle finger tip)
        wrist = hand[0]
        middle_tip = hand[12]
        span = np.linalg.norm(wrist - middle_tip)
        features.append(span)
        
        # 3. Hand openness (average distance from center to fingertips)
        fingertips = hand[[8, 12, 16, 20]]  # thumb, index, middle, pinky tips
        center = hand[0]  # wrist as reference
        avg_finger_dist = np.mean([np.linalg.norm(ft - center) for ft in fingertips])
        features.append(avg_finger_dist)
    
    # Ensure fixed number of features
    while len(features) < 10:  # 5 features per hand * 2 hands
        features.append(0.0)
    
    return np.array(features[:10], dtype=np.float32)

def normalize_landmarks(sequence):
    """Normalize landmarks to be invariant to position and scale"""
    normalized_seq = []
    
    for frame in sequence:
        # Reshape to (2, 21, 3)
        frame_reshaped = frame[:126].reshape(2, 21, 3)
        normalized_frame = []
        
        for hand in frame_reshaped:
            if np.all(hand == 0):
                normalized_frame.extend(hand.reshape(-1))
                continue
            
            # Normalize relative to wrist
            wrist = hand[0]
            normalized_hand = hand - wrist
            
            # Scale normalization (based on hand size)
            hand_size = np.linalg.norm(normalized_hand[5])  # Use index finger base as reference
            if hand_size > 0:
                normalized_hand = normalized_hand / hand_size
            
            normalized_frame.extend(normalized_hand.reshape(-1))
        
        # Combine with remaining features (handedness, additional features)
        remaining_features = frame[126:]
        normalized_frame = np.concatenate([np.array(normalized_frame), remaining_features])
        normalized_seq.append(normalized_frame)
    
    return np.array(normalized_seq)

def build_enhanced_dataset(data_dir=DATA_DIR):
    X, Y, FNS, CLASSES, HAND_PRESENCE = [], [], [], [], []
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    classes = [c for c in classes if not c.startswith(".")]
    #classes= [c for c in classes if c in ["by by","great","how","name","so so","you"]]  # Filter specific classes if needed
    if len(classes) == 0:
        raise RuntimeError(f"No class folders found in {data_dir}")
    
    print("Found classes:", classes)
    class2idx = {c: i for i, c in enumerate(classes)}
    
    # Collect statistics
    class_stats = {cls: {"total": 0, "failed": 0} for cls in classes}

    for cls in classes:
        folder = os.path.join(data_dir, cls)
        files = sorted([f for f in os.listdir(folder) if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))])
        print(f"Processing class={cls}  files={len(files)}")
        
        for fn in tqdm(files):
            path = os.path.join(folder, fn)
            class_stats[cls]["total"] += 1
            
            try:
                frames = extract_frames_from_video(path)
                if len(frames) == 0:
                    print(f"Warning: No frames in {path}")
                    class_stats[cls]["failed"] += 1
                    continue
                
                seq, hand_presence = extract_enhanced_landmarks_for_frames(frames, max_frames=MAX_FRAMES)
                
                # Skip sequences with no hand detection
                if np.mean(hand_presence) < 0.2:  # Less than 20% hand presence
                    print(f"Warning: Low hand detection in {path}")
                    class_stats[cls]["failed"] += 1
                    continue
                
                # Normalize landmarks
                seq_normalized = normalize_landmarks(seq)
                
                X.append(seq_normalized)
                Y.append(class2idx[cls])
                FNS.append(path)
                HAND_PRESENCE.append(hand_presence)
                
            except Exception as e:
                print(f"Error processing {path}: {e}")
                class_stats[cls]["failed"] += 1

    # Print statistics
    print("\nProcessing Statistics:")
    for cls in classes:
        stats = class_stats[cls]
        success_rate = (stats["total"] - stats["failed"]) / stats["total"] * 100
        print(f"  {cls}: {stats['total'] - stats['failed']}/{stats['total']} ({success_rate:.1f}%)")

    if len(X) == 0:
        raise RuntimeError("No valid samples processed!")

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int32)
    FNS = np.array(FNS)
    HAND_PRESENCE = np.array(HAND_PRESENCE, dtype=np.float32)
    
    return X, Y, FNS, HAND_PRESENCE, class2idx

if __name__ == "__main__":
    X, Y, FNS, HAND_PRESENCE, class2idx = build_enhanced_dataset(DATA_DIR)
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    print("Hand presence shape:", HAND_PRESENCE.shape)
    
    out_file = os.path.join(OUTPUT_DIR, "enhanced_signs_landmarks.npz")
    np.savez_compressed(out_file, X=X, Y=Y, FNS=FNS, HAND_PRESENCE=HAND_PRESENCE)
    
    # Save label map and configuration
    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w", encoding="utf-8") as fh:
        json.dump(class2idx, fh, indent=2, ensure_ascii=False)
    
    # Save feature configuration
    feature_info = {
        "feature_dim": X.shape[-1],
        "max_frames": MAX_FRAMES,
        "feature_breakdown": {
            "hand_landmarks": 126,
            "handedness": 2,
            "additional_features": 10
        }
    }
    
    with open(os.path.join(OUTPUT_DIR, "feature_config.json"), "w", encoding="utf-8") as fh:
        json.dump(feature_info, fh, indent=2, ensure_ascii=False)
    
    print("Saved enhanced dataset to", out_file)
    print("Label map:", class2idx)
    print("Feature dimension:", X.shape[-1])