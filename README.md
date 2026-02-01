# Sign Language Detection System

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/haitemb/sign-language-detection)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-blue)](https://mediapipe.dev/)

A real-time sign language recognition system that uses **MediaPipe** for hand landmark extraction and **LSTM neural networks** for sequence classification. This project allows you to create your own custom sign language dataset and train a model to recognize specific signs in real-time.

## 📹 Demo Video
[![Demo Video](https://img.shields.io/badge/Watch-Demo_Video-red)](https://github.com/haitemb/sign-language-detection/raw/main/test_video.mp4)

*Include your test video file named `test_video.mp4` in the repository root*

## ✨ Features

- **🎥 Custom Dataset Recording**: Interactive script to record your own sign language videos
- **✋ Automatic Landmark Extraction**: Uses MediaPipe to extract 3D hand landmarks from videos
- **🧠 LSTM Sequence Classification**: Deep learning model that understands temporal patterns in signs
- **🔮 Real-time Prediction**: Live webcam recognition with visual feedback
- **🔄 Dataset Management**: Tools to add/remove signs and manage your dataset
- **📊 Custom Vocabulary**: Train on any signs you want to recognize

## 🛠️ How It Was Built

### The Journey & Challenges

Building this system involved several key challenges:

1. **Dataset Creation**: 
   - Recording consistent sign language videos is difficult
   - Different lighting conditions, backgrounds, and hand positions affect accuracy
   - I recorded **18 distinct signs** myself, with 30-50 videos per sign

2. **Model Training Difficulties**:
   - Similar signs (like "i" and "you") often confused the model
   - Solution: Carefully selected **distinct signs** and trained multiple model versions
   - Created **7 different model iterations** before achieving good accuracy
   - The included model is version 7 (`latest_model_lstm_1.h5`) - the best performing one

3. **Technical Approach**:
   - Used MediaPipe for robust hand tracking in various conditions
   - Implemented LSTM to capture the temporal nature of sign language
   - Added data augmentation to improve generalization

## 📁 Project Structure
sign-language-detection/

├── 📄 record.py # Record your own sign language videos

├── 📄 extract_landmarks.py # Extract hand landmarks from videos

├── 📄 train_lstm.py # Train LSTM model on your dataset

├── 📄 live_test.py # Real-time sign language recognition

├── 📄 remove.py # Manage dataset (add/remove classes)

├── 📄 sign_language_model.h5 # Pre-trained model (18 signs)

├── 📄 requirements.txt # Python dependencies

├── 📄 README.md # This file

└── 📁 data/ # i did not put my dataset because i dont want to put my images in the internet sorry for that but you can record your own with record.py (empty)



## 🚀 Quick Start (Test Pre-trained Model)

### 1. Clone & Setup
```bash

git clone https://github.com/haitemb/sign-language-detection.git
cd sign-language-detection

# Create virtual environment (recommended)
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

```
Test with Webcam
```bash

python live_test.py

```

Instructions:

Press Enter to start recording (2-second delay)

Perform the sign clearly

See prediction displayed in green text

Press q to quit

3. Supported Signs (Pre-trained Model)
The included model recognizes these 18 signs:

Sign	Meaning	Sign	Meaning

👋	bye bye	👉	come

👨	father	👍	great

👋	hello	❓	how

👆	i	🤟	i love you

➕	more	👩	mother

📛	name	❌	no

🙏	please	👋	see you later

✋	stop	🙏	thanks

✅	yes	👉	you

note: the emojis are just emojis if you want to know how the actual word looks like just go ao youtube and search for that word in ASL and watch a video 

### 🎬 Create Your Own Dataset
Step 1: Record Videos
```bash

python record.py

```
Process:

Enter the sign name (e.g., "hello")

Choose number of videos (30-50 recommended)

Position yourself in frame

Press Enter to start recording (3 seconds per video)

Videos save to data/<sign_name>/

Tips:

Use consistent lighting

Keep background simple

Maintain same distance from camera

Perform sign clearly and consistently
Step 2: Extract Landmarks
```bash

python extract_landmarks.py

```

This processes all videos in data/ and:

Extracts 21 hand landmarks per frame using MediaPipe

Saves landmarks as numpy arrays

Creates label_map.json for class mapping

Step 3: Train Your Model
```bash

python train_lstm.py

```
Training Process:

80/20 train-test split

50 epochs (adjustable)

Saves best model as sign_language_model.h5

Training history plot saved as training_history.png

### 🛠️ Advanced: Dataset Management
Using remove.py
This utility helps manage your dataset:

```bash

python remove.py

```
What it does:

Remove problematic sign classes from your dataset

Update label mappings automatically

Clean up before retraining

To use:

Edit remove.py and set REMOVE_CLASSES = ["class_to_remove"]

Run the script

Retrain your model

Adding New Signs
Record new videos: python record.py

Extract landmarks: python extract_landmarks.py

Retrain model: python train_lstm.py
### 📊 Model Architecture
```bash

LSTM-based Sequence Classifier:
Input: (frames, 63)  # 21 landmarks × 3 coordinates
↓
LSTM (128 units) with Dropout (0.5)
↓
LSTM (64 units) with Dropout (0.5)
↓
Dense (64 units, ReLU)
↓
Dense (num_classes, Softmax)

```

Key Specifications:

Sequence length: 30 frames (1 second at 30 FPS)

Input features: 63 (21 landmarks × 3 coordinates)

Optimizer: Adam with learning rate 0.001

Loss: Categorical Crossentropy

### 🔧 Customization
Adjust Training Parameters
Edit train_lstm.py:
```bash

EPOCHS = 50           # Increase for better accuracy
BATCH_SIZE = 32       # Adjust based on GPU memory
SEQ_LENGTH = 30       # Frames per sequence

```
Add Data Augmentation
In train_lstm.py, uncomment augmentation section:
```bash

# Add noise to landmarks
noise = np.random.normal(0, 0.01, X_train.shape)
X_train = X_train + noise

```
## 🚨 Common Issues & Solutions
1. Model Confuses Similar Signs
Problem: Signs like "i" and "you" look similar

Solution:

Record more distinctive examples

Use remove.py to remove confusing classes

Increase training data for similar pairs

2. Poor Real-time Detection
Problem: Model doesn't recognize signs in real-time

Solution:

Ensure consistent lighting during recording

Position hands similarly in training and testing

Increase sequence length to capture full sign
3. MediaPipe Detection Fails
Problem: Hand landmarks not detected

Solution:

Improve lighting conditions

Move closer to camera

Use plain background

Adjust min_detection_confidence in scripts

## 📈 Performance Tips
Dataset Quality > Quantity: 30 well-recorded videos beat 100 poor ones

Lighting Matters: Consistent lighting improves accuracy by 30%

Background: Plain backgrounds work best

Sign Clarity: Exaggerate signs slightly for better detection

Training: More epochs = better accuracy (but watch for overfitting)

## 🤝 Contributing
Found a bug or want to improve the project?

Fork the repository

Create a feature branch (git checkout -b feature/improvement)

Commit changes (git commit -m 'Add some feature')

Push to branch (git push origin feature/improvement)

Open a Pull Request

## 📝 License
This project is open source and available under the MIT License.
## 🙏 Acknowledgments
MediaPipe for hand tracking

TensorFlow for deep learning framework

All open-source contributors

## 📧 Contact
Have questions or suggestions?

GitHub Issues: Create an issue

Project Link: https://github.com/haitemb/sign-language-detection



### ⭐ If you find this project useful, please give it a star on GitHub! ⭐

