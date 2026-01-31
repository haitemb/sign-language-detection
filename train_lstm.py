import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# load dataset
data = np.load("latest_dataset_1/enhanced_signs_landmarks.npz")
X = data['X']   # (samples, frames, features)
Y = data['Y']

num_classes = len(np.unique(Y))
input_shape = (X.shape[1], X.shape[2])

model = Sequential([
    LSTM(256, return_sequences=True, input_shape=input_shape),
    Dropout(0.3),
    LSTM(128),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint("latest_model_lstm_2.h5", save_best_only=True, monitor="val_accuracy", mode="max")

model.fit(X, Y, epochs=40, batch_size=8, validation_split=0.2, callbacks=[checkpoint])

print("training finished. best model saved as latest_model_lstm_2.h5")