import numpy as np
import os
import pandas as pd
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import keras
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load Data
print("Loading Rraining and Testing Data")

X_train = np.load('./data/X_train.npy')
X_test = np.load('./data/X_test.npy')
y_train = np.load('./data/y_train.npy')
y_test = np.load('./data/y_test.npy')

# Reshape the data because the RNN expects 3D Input
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))  # Add feature dimension
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))      # Do the same for test data

print("Data Loaded")

# Define the Model
model = Sequential()
model.add(keras.Input(shape=(X_train.shape[1], 1)))
model.add(LSTM(units=16, activation='relu'))
model.add(Dense(1))

print("Compiling the Model")
model.compile(optimizer='RMSprop', loss='mse', metrics=['mae'])
print("Model Compiled")

# Train the Model
print("Training the Model")
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the Model
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

# Make Predictions

predictions = model.predict(X_test)
predicted_values = (predictions)
actual_values = (y_test)

# Plot Results
plt.scatter(actual_values, predicted_values, label='Actual')
p1 = max(max(predicted_values), max(actual_values))
p2 = min(min(predicted_values), min(actual_values))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('Actual Values')
plt.ylabel("Predicted Values")
plt.show()

print("Saving model")
model.save('../models/rnn_model.h5')
