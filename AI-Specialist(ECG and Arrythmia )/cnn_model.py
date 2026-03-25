import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten

# Dummy data (replace with real ECG later)
X = np.random.rand(100,100,1)
y = np.random.randint(0,2,100)

model = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(100,1)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X, y, epochs=5)

model.save("models/cnn_model.h5")

print("CNN model saved!")