import time
import csv
import os
from datetime import datetime
import RPi.GPIO as GPIO
from config import MOISTURE_SENSOR_PIN_AO, COLLECTION_INTERVAL, COLLECTION_DURATION, RAW_DATA_PATH

def init_gpio():
    """初始化树莓派GPIO引脚"""
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(MOISTURE_SENSOR_PIN_AO, GPIO.IN)  # AO引脚设为输入模式

def read_rpi_temperature():
    """读取树莓派内置温度传感器数据（°C）"""
    with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
        temp = float(f.read()) / 1000.0
    return round(temp, 1)

def read_moisture():
    """读取FC-28传感器模拟湿度值（0-1023），值越大湿度越高"""
    value = 0
    # 10次采样取平均，降低噪声
    for _ in range(10):
        value += GPIO.input(MOISTURE_SENSOR_PIN_AO)
        time.sleep(0.01)
    avg_value = (value / 10) * 1023  # 缩放至0-1023范围
    return round(avg_value, 0)

def init_data_file():
    """初始化数据文件，写入表头"""
    if not os.path.exists(RAW_DATA_PATH):
        with open(RAW_DATA_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Moisture_Value(0-1023)", "Temperature(°C)"])

def main():
    try:
        init_gpio()
        init_data_file()
        
        start_time = time.time()
        # 循环采集数据，直到达到设定时长
        while time.time() - start_time < COLLECTION_DURATION:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            moisture = read_moisture()
            temperature = read_rpi_temperature()
            
            # 追加写入数据到CSV
            with open(RAW_DATA_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, moisture, temperature])
            
            print(f"[{timestamp}] Moisture: {moisture} | Temp: {temperature}°C")
            time.sleep(COLLECTION_INTERVAL)
    
    except KeyboardInterrupt:
        print("\nData collection interrupted by user.")
    finally:
        GPIO.cleanup()  # 释放GPIO资源

if __name__ == "__main__":
    main()
