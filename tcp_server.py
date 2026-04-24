import os
import socket
from datetime import datetime

HOST, PORT = '0.0.0.0', 8080

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'analysis')
os.makedirs(OUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUT_DIR, 'sensor_data.csv')

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, 'w') as f:
        f.write('Timestamp,Temperature,Humidity\n')

print(f'Listening on {HOST}:{PORT}')
print(f'Saving to {CSV_PATH}')

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print(f'Connected by {addr}')
        while True:
            data = conn.recv(1024)
            if not data:
                break
            line = data.decode('utf-8', errors='ignore').strip()
            print(f'Received: {line}')
            try:
                if 'T:' in line and 'H:' in line:
                    parts = line.split(',')
                    temp = parts[0].split(':')[1]
                    hum = parts[1].split(':')[1]
                    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    with open(CSV_PATH, 'a') as f:
                        f.write(f'{ts},{temp},{hum}\n')
            except Exception as e:
                print(f'Parse error: {e}')
