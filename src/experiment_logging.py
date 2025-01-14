import pandas as pd
from datetime import datetime

def log_experiment_pandas(experiment_data, log_file='./docs/experiment_log.csv'):
    try:
        log_df = pd.read_csv(log_file)
    except FileNotFoundError:
        log_df = pd.DataFrame(columns=['Experiment ID', 'Model Details', 'Hyperparameters', 'Metrics', 'Training Time', 'Notes', 'Timestamp'])

    
    experiment_df = pd.DataFrame(experiment_data, index=[0])
    print(experiment_df)
    print(log_df)
    log_df = pd.concat([log_df, experiment_df])
    log_df.to_csv(log_file, index=False)
