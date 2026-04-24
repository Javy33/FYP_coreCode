# Edge-Side Collaborative Anomaly Monitoring System

Source code for the Final Year Project report *"Edge-Side Collaborative Anomaly Monitoring System for Non-Stationary IoT Environments"* (University of Glasgow / UESTC, 2025–26).

The system combines an embedded edge node (STM32 Nucleo-L432KC + SHT3x + ESP8266) with a PC-side analytical pipeline that monitors **decoupled residuals** from a MIMO-GRU predictive model instead of raw sensor values, eliminating the false alarms that traditional SPC suffers from under diurnal environmental drift.

---

## Repository Structure

| File | Report Section | Purpose |
|---|---|---|
| `main.cpp` | §3.1, §3.2.1 | Mbed OS firmware: SHT3x I2C acquisition + ESP8266 TCP transmission |
| `tcp_server.py` | §3.2.2 | Python TCP server, receives sensor stream and writes `sensor_data.csv` |
| `preprocess.py` | §2.4.2, §2.4.4 | Feature engineering: cyclical time encoding + first-order differences |
| `GRU.py` | §2.5, §3.3.2 | MIMO-GRU model definition, training loop with early stopping, test-set evaluation |
| `visualize.py` | Algorithm 2, §4.3.3 | Weighted-ensemble residual computation, SPC X-charts, synthetic anomaly injection |
| `version1.m` | Algorithm 1, §3.3.1 | MATLAB baseline: static X-chart + sliding-window polynomial regression |

---

## Requirements

**Edge (firmware):**
- STM32 Nucleo-L432KC
- SHT3x temperature/humidity sensor (I2C, pins D4/D5)
- ESP8266 Wi-Fi module (UART, pins D0/D1)
- Mbed OS 6

**PC (analysis):**
- Python 3.9+
- `numpy`, `pandas`, `torch`, `scikit-learn`, `matplotlib`

**Baseline:**
- MATLAB R2018b or later (requires `yline`)

---

## How to Reproduce

1. **Flash firmware.** Edit the SSID, password, and PC IP in `main.cpp`, build with Mbed Studio, flash to the Nucleo board.
2. **Start the TCP server** on the PC:
   ```bash
   python tcp_server.py
   ```
   Sensor readings are appended to `analysis/sensor_data.csv` at 1-minute intervals.
3. **Preprocess** into the 6-feature matrix:
   ```bash
   python preprocess.py
   ```
4. **Train the MIMO-GRU model:**
   ```bash
   python GRU.py
   ```
   Saves `best_model.pth` and `loss_curve.png` (Figure 4.1).
5. **Evaluate under injected anomalies:**
   ```bash
   python visualize.py
   ```
   Produces residual X-charts for the spike and ramp scenarios (Figures 4.3, 4.4).
6. **(Optional)** Run the polynomial baseline for comparison:
   ```matlab
   >> version1
   ```

---

## Notes

- Normalization uses `MinMaxScaler` fit on the training split only (70 / 15 / 15 chronological split) to prevent leakage.
- SPC control limits are computed on the anomaly-free steady-state window `[3300, 3500]` as specified in Algorithm 2.
- The ensemble weighting (0.5, 0.3, 0.2) applied to 1-, 2-, and 3-step-ahead predictions implements Eq. (12) in the report.

---

## Author

Liu Jiawei, 2025–26 Final Year Project.# FYP_coreCode
You can get all codes in FYP paper there.
