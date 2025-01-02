import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pickle

# Load Data
X_train = np.load('./data/X_train.npy')
X_test = np.load('./data/X_test.npy')
y_train = np.load('./data/y_train.npy')
y_test = np.load('./data/y_test.npy')

# Define the Model
model = Sequential([
    LSTM(64, activation='tanh', return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the Model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the Model
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

# Make Predictions
predictions = model.predict(X_test)
predicted_values = scaler.inverse_transform(predictions)
actual_values = scaler.inverse_transform(y_test)

# Plot Results
import matplotlib.pyplot as plt

plt.plot(actual_values, label='Actual')
plt.plot(predicted_values, label='Predicted')
plt.legend()
plt.show()

model.save('../models/rnn_model.h5')
