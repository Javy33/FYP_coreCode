import numpy as np
import pandas as pd

RAW_PATH = 'sensor_data.csv'
OUT_PATH = 'processed_weather_data.csv'
PERIOD_MIN = 24 * 60

def preprocess(raw_path=RAW_PATH, out_path=OUT_PATH):
    df = pd.read_csv(raw_path, parse_dates=['Timestamp']).dropna().reset_index(drop=True)

    t_min = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds() / 60.0

    T = df['Temperature'].astype(float).values
    H = df['Humidity'].astype(float).values
    dT = np.concatenate([[0.0], np.diff(T)])
    dH = np.concatenate([[0.0], np.diff(H)])
    time_sin = np.sin(2 * np.pi * t_min / PERIOD_MIN)
    time_cos = np.cos(2 * np.pi * t_min / PERIOD_MIN)

    out = pd.DataFrame({
        'T': T, 'H': H,
        'dT': dT, 'dH': dH,
        'Time_sin': time_sin, 'Time_cos': time_cos,
    })
    out.to_csv(out_path, index=False)
    print(f'Saved {len(out)} rows to {out_path}')

if __name__ == '__main__':
    preprocess()
