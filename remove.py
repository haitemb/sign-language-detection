import numpy as np
import json
import os

OUTPUT_DIR = "latest_dataset_1"
NPZ_PATH = os.path.join(OUTPUT_DIR, "enhanced_signs_landmarks.npz")
LABEL_MAP_PATH = os.path.join(OUTPUT_DIR, "label_map.json")

# ---------------------------------------------
# EDIT THIS: list of glosses to remove
REMOVE_CLASSES = ["you"]   # example
# ---------------------------------------------

# Load NPZ
data = np.load(NPZ_PATH, allow_pickle=True)
keys = list(data.keys())
print("Found keys in NPZ:", keys)

X = data["X"]
Y = data["Y"]
FNS = data["FNS"]
HP = data["HAND_PRESENCE"]

print("[INFO] Shapes:")
print("  X:", X.shape)
print("  Y:", Y.shape)
print("  FNS:", FNS.shape)
print("  HP:", HP.shape)

# Load original label map
with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    class2idx = json.load(f)

idx2class = {v: k for k, v in class2idx.items()}

print("[INFO] Original classes:", len(class2idx))

# Determine which samples to keep
keep_indices = [
    i for i, y in enumerate(Y)
    if idx2class[y] not in REMOVE_CLASSES
]

# Filter dataset
X_new = X[keep_indices]
Y_new = Y[keep_indices]
FNS_new = FNS[keep_indices]
HP_new = HP[keep_indices]

# Build new label map
remaining_classes = sorted({idx2class[y] for y in Y_new})
new_class2idx = {cls: i for i, cls in enumerate(remaining_classes)}

# Remap Y
Y_new = np.array([new_class2idx[idx2class[y]] for y in Y_new], dtype=np.int64)

print("[INFO] New shapes after removal:")
print("  X:", X_new.shape)
print("  Y:", Y_new.shape)
print("  FNS:", FNS_new.shape)
print("  HP:", HP_new.shape)

print("[INFO] New label map:", new_class2idx)

# Save updated NPZ
np.savez_compressed(
    NPZ_PATH,
    X=X_new,
    Y=Y_new,
    FNS=FNS_new,
    HAND_PRESENCE=HP_new,
)

# Update label_map.json
with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
    json.dump(new_class2idx, f, indent=2)

print("\n[✔] DONE — unwanted classes removed successfully!")
