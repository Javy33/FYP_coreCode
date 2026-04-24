import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

LOOKBACK = 120
STEPS_AHEAD = 3
N_FEATURES = 6

class WeatherDataset(Dataset):
    def __init__(self, data, lookback=LOOKBACK, steps_ahead=STEPS_AHEAD):
        self.data = data
        self.lookback = lookback
        self.steps_ahead = steps_ahead

    def __len__(self):
        return len(self.data) - self.lookback - self.steps_ahead + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.lookback]
        y = []
        for i in range(self.steps_ahead):
            y.extend([self.data[idx + self.lookback + i, 0],
                      self.data[idx + self.lookback + i, 1]])
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class MIMOGRU(nn.Module):
    def __init__(self, input_size=N_FEATURES, hidden_size=64, dense_size=32,
                 output_size=2 * STEPS_AHEAD, num_layers=1, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, dense_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        return self.fc2(out)

def train_model():
    df = pd.read_csv('processed_weather_data.csv')
    data = df.values
    n = len(data)
    train_raw = data[:int(n * 0.7)]
    val_raw   = data[int(n * 0.7):int(n * 0.85)]
    test_raw  = data[int(n * 0.85):]

    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_raw)
    val_data   = scaler.transform(val_raw)
    test_data  = scaler.transform(test_raw)

    batch_size, epochs, lr = 64, 50, 1e-3

    train_loader = DataLoader(WeatherDataset(train_data), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(WeatherDataset(val_data),   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(WeatherDataset(test_data),  batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MIMOGRU().to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    patience, best_val, counter = 5, float('inf'), 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        tr = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            tr += loss.item() * x.size(0)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                vl += criterion(model(x), y).item() * x.size(0)

        train_losses.append(tr / len(train_loader.dataset))
        val_losses.append(vl / len(val_loader.dataset))
        print(f'Epoch {epoch+1}: train {train_losses[-1]:.4f}, val {val_losses[-1]:.4f}')

        if val_losses[-1] < best_val:
            best_val = val_losses[-1]
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                break

    # Figure 4.1: loss curves
    plt.figure(figsize=(7, 4))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Smooth L1 loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=150)

    # Test set evaluation in physical units
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    T_min, T_range = float(scaler.data_min_[0]), float(scaler.data_range_[0])
    H_min, H_range = float(scaler.data_min_[1]), float(scaler.data_range_[1])

    mae, mse = 0.0, 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            for i in range(STEPS_AHEAD):
                pred[:, 2*i]   = pred[:, 2*i]   * T_range + T_min
                pred[:, 2*i+1] = pred[:, 2*i+1] * H_range + H_min
                y[:, 2*i]      = y[:, 2*i]      * T_range + T_min
                y[:, 2*i+1]    = y[:, 2*i+1]    * H_range + H_min
            mae += nn.L1Loss()(pred, y).item()  * x.size(0)
            mse += nn.MSELoss()(pred, y).item() * x.size(0)

    n_test = len(test_loader.dataset)
    print(f'Test MAE: {mae / n_test:.4f}')
    print(f'Test RMSE: {np.sqrt(mse / n_test):.4f}')

if __name__ == '__main__':
    train_model()
