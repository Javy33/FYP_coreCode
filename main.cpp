#include "mbed.h"

// SHT3x via I2C
I2C i2c(D4, D5);
const int SHT3X_ADDR = 0x44 << 1;

// ESP8266 via UART
UnbufferedSerial esp(D1, D0, 115200);

void send_at(const char* cmd, int delay_ms = 1000) {
    char buf[128];
    int len = sprintf(buf, "%s\r\n", cmd);
    esp.write(buf, len);
    ThisThread::sleep_for(std::chrono::milliseconds(delay_ms));
}

void setup_wifi() {
    send_at("AT+RST", 2000);
    send_at("AT+CWMODE=1", 1000);
    send_at("AT+CWJAP=\"SSID\",\"Password\"", 5000);
    send_at("AT+CIPSTART=\"TCP\",\"PC_IP\",8080", 2000);
}

void read_sht3x(float &temp, float &hum) {
    char cmd[2] = {0x2C, 0x06};
    i2c.write(SHT3X_ADDR, cmd, 2);
    ThisThread::sleep_for(20ms);

    char data[6];
    if (i2c.read(SHT3X_ADDR, data, 6) == 0) {
        uint16_t st  = (data[0] << 8) | data[1];
        uint16_t srh = (data[3] << 8) | data[4];
        temp = -45.0f + 175.0f * ((float)st / 65535.0f);
        hum  = 100.0f * ((float)srh / 65535.0f);
    } else {
        printf("I2C read error\n");
    }
}

void send_data_wifi(float temp, float hum) {
    char payload[64];
    int len = sprintf(payload, "T:%.2f,H:%.2f\n", temp, hum);

    char cmd[32];
    sprintf(cmd, "AT+CIPSEND=%d", len);
    send_at(cmd, 500);
    esp.write(payload, len);
    ThisThread::sleep_for(500ms);
}

int main() {
    setup_wifi();
    while (true) {
        float temp = 0.0f, hum = 0.0f;
        read_sht3x(temp, hum);
        printf("T = %.2f C, RH = %.2f %%\n", temp, hum);
        send_data_wifi(temp, hum);
        ThisThread::sleep_for(60s);
    }
}
