import numpy as np
import os
import pandas as pd
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import keras
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from keras import regularizers
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from time import time
from datetime import datetime
import experiment_logging

# Load Data
print("Loading Training and Testing Data")

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
model.add(LSTM(units=16, activation='tanh'))
model.add(Dropout(rate=0.2))
#model.add(LSTM(units=32, activation='tanh', activity_regularizer=regularizers.L2(1e-5)))
#model.add(Dropout(rate=0.2))
model.add(Dense(1))

print("Compiling the Model")
model.compile(optimizer='RMSprop', loss='mse', metrics=['mae'])
print("Model Compiled")

# Record the start time
start_time = time()

# Train the Model
print("Training the Model")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=2)

# Record the end time
end_time = time()
training_time = f'{(end_time - start_time) // 60}m{(end_time - start_time) % 60:.0f}s'

# Evaluate the Model
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")


# Make Predictions
predictions = model.predict(X_test)
predicted_values = (predictions)
actual_values = (y_test)

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()


# Plot Results
#print(f"Actual Vals: {actual_values.shape}")
#print(f"Predicted Vals: {predicted_values.shape}")
plt.scatter(actual_values, predicted_values, label='Actual')
#p1 = max(max(predicted_values), max(actual_values))
#p2 = min(min(predicted_values), min(actual_values))
#plt.plot([0, 0], [max(predicted_values), max(actual_values)], 'b-')
plt.xlabel('Actual Values')
plt.ylabel("Predicted Values")
plt.show()

experiment_data = {
    'Experiment ID': 'LSTM_001',
    'Model Details': '1 LSTM layers, 16 units, tanh activation',
    'Hyperparameters': 'batch=32',
    'Metrics': f"Train Loss={history.history['loss'][-1]:.4f}, Val Loss={history.history['val_loss'][-1]:.4f}, Val MAE={history.history['val_mae'][-1]:.4f}",
    'Training Time': training_time,
    'Notes': 'Baseline Model.',
    'Timestamp': datetime.now()
}

# Log the experiment
experiment_logging.log_experiment_pandas(experiment_data)

print("Saving model")
model.save('./models/LSTM_001.keras')
