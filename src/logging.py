import pandas as pd
from datetime import datetime

def log_experiment_pandas(experiment_data, log_file='./docs/experiment_log.csv'):
    try:
        log_df = pd.read_csv(log_file)
    except FileNotFoundError:
        log_df = pd.DataFrame(columns=['Experiment ID', 'Model Details', 'Hyperparameters', 'Metrics', 'Training Time', 'Notes', 'Timestamp'])

    log_df = log_df.append(experiment_data, ignore_index=True)
    log_df.to_csv(log_file, index=False)

# Example usage
experiment_data = {
    'Experiment ID': 'LSTM_001',
    'Model Details': '1 LSTM layer, 16 units, tanh activation',
    'Hyperparameters': 'LR=0.001, batch=32, dropout=0.2',
    'Metrics': 'Train Loss=0.015, Val Loss=0.025, Val MAE=0.01',
    'Training Time': '5m10s',
    'Notes': 'Baseline model.',
    'Timestamp': 'datetime.now()'
}

log_experiment_pandas(experiment_data)