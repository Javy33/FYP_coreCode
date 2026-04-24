import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from GRU import MIMOGRU, LOOKBACK

# Phase I steady-state window (Algorithm 2, line 2)
CAL_START, CAL_END = 3300, 3500
# Ensemble weights (Eq. 12)
W1, W2, W3 = 0.5, 0.3, 0.2


def inject_anomaly(data, start_idx, scenario, magnitude, duration):
    out = np.copy(data)
    if scenario == 'spike':
        end = min(start_idx + 2, len(out))
        out[start_idx:end, 0] += magnitude
    elif scenario == 'ramp':
        end = min(start_idx + duration, len(out))
        k = end - start_idx
        if k > 0:
            out[start_idx:end, 0] += np.linspace(0.1, magnitude, k)
    return out


def spc_limits(residuals):
    mu = np.mean(residuals)
    sigma = np.std(residuals)
    return mu, mu + 3 * sigma, mu - 3 * sigma


def ensemble_predict(model, scaler, data, t, device):
    """Weighted ensemble residual targeting time t (Eq. 12).

    Windows X1/X2/X3 end at t-1, t-2, t-3 respectively; the 1/2/3-step-ahead
    prediction for time t is taken from each window's output head.
    Returns (T_hat, H_hat) in physical units.
    """
    T_min, T_range = float(scaler.data_min_[0]), float(scaler.data_range_[0])
    H_min, H_range = float(scaler.data_min_[1]), float(scaler.data_range_[1])

    preds_T, preds_H = [], []
    for k in range(3):
        x_raw = data[t - LOOKBACK - k : t - k]
        x = torch.tensor(scaler.transform(x_raw), dtype=torch.float32).unsqueeze(0).to(device)
        p = model(x).cpu().numpy()[0]
        preds_T.append(p[2 * k]     * T_range + T_min)
        preds_H.append(p[2 * k + 1] * H_range + H_min)

    T_hat = W1 * preds_T[0] + W2 * preds_T[1] + W3 * preds_T[2]
    H_hat = W1 * preds_H[0] + W2 * preds_H[1] + W3 * preds_H[2]
    return T_hat, H_hat


def residual_series(model, scaler, data, start, end, device):
    T_res, H_res, T_hat_list, H_hat_list = [], [], [], []
    with torch.no_grad():
        for t in range(start, end + 1):
            T_hat, H_hat = ensemble_predict(model, scaler, data, t, device)
            T_res.append(data[t, 0] - T_hat)
            H_res.append(data[t, 1] - H_hat)
            T_hat_list.append(T_hat)
            H_hat_list.append(H_hat)
    return (np.array(T_res), np.array(H_res),
            np.array(T_hat_list), np.array(H_hat_list))


def evaluate_and_visualize(N=3600, scenario='none', magnitude=15.0, duration=40,
                           anomaly_start=None):
    df = pd.read_csv('processed_weather_data.csv')
    clean = df.values
    data = np.copy(clean)

    scaler = MinMaxScaler()
    scaler.fit(clean[:int(len(clean) * 0.7)])

    if scenario in ('spike', 'ramp'):
        if anomaly_start is None:
            anomaly_start = N - 20 if scenario == 'spike' else N - 40
        data = inject_anomaly(data, anomaly_start, scenario, magnitude, duration)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MIMOGRU().to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()

    # Phase I: compute SPC limits from anomaly-free steady-state window
    cal_T, cal_H, _, _ = residual_series(model, scaler, clean, CAL_START, CAL_END, device)
    muT, uclT, lclT = spc_limits(cal_T)
    muH, uclH, lclH = spc_limits(cal_H)

    # Phase II: residuals over plot window
    plot_start = N - 60
    T_res, H_res, T_hat, H_hat = residual_series(model, scaler, data, plot_start, N, device)
    time_axis = np.arange(plot_start, N + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    axes[0, 0].plot(time_axis, data[plot_start:N + 1, 0], 'b.-', label='Actual T')
    axes[0, 0].plot(time_axis, T_hat, 'rx', label='Predicted T')
    axes[0, 0].set_title('Temperature prediction')
    axes[0, 0].set_ylabel('T (C)')
    axes[0, 0].legend(); axes[0, 0].grid(True)

    axes[0, 1].plot(time_axis, data[plot_start:N + 1, 1], 'b.-', label='Actual H')
    axes[0, 1].plot(time_axis, H_hat, 'rx', label='Predicted H')
    axes[0, 1].set_title('Humidity prediction')
    axes[0, 1].set_ylabel('H (%)')
    axes[0, 1].legend(); axes[0, 1].grid(True)

    for ax, res, mu, ucl, lcl, name in (
        (axes[1, 0], T_res, muT, uclT, lclT, 'Temperature residual'),
        (axes[1, 1], H_res, muH, uclH, lclH, 'Humidity residual'),
    ):
        ax.plot(time_axis, res, 'bo-', label='Residual')
        ax.axhline(mu,  color='g', linestyle='-',  label='Mean')
        ax.axhline(ucl, color='r', linestyle='--', label='+3 sigma')
        ax.axhline(lcl, color='r', linestyle='--', label='-3 sigma')
        ax.set_title(name + ' X-chart')
        ax.set_xlabel('Time step (t)')
        ax.legend(); ax.grid(True)

    if scenario in ('spike', 'ramp'):
        shade_len = 2 if scenario == 'spike' else duration
        for ax in axes.flat:
            ax.axvspan(anomaly_start, min(N, anomaly_start + shade_len),
                       color='red', alpha=0.15)

    plt.tight_layout()
    plt.savefig(f'residual_chart_{scenario}.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    # Scenario 1: spike anomaly at t = 3580, +9 C
    evaluate_and_visualize(N=3600, scenario='spike', magnitude=9.0, anomaly_start=3580)
    # Scenario 2: ramp anomaly starting at t = 3560, +15 C over 40 steps
    evaluate_and_visualize(N=3600, scenario='ramp', magnitude=15.0, duration=40,
                           anomaly_start=3560)
