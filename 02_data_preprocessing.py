import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config import RAW_DATA_PATH, PREPROCESSED_DATA_PATH, DENOISING_WINDOW, NORMALIZE_RANGE

def moving_average_denoising(data):
    """移动平均滤波去噪，填充首尾缺失值"""
    denoised_data = data.rolling(window=DENOISING_WINDOW, center=True).mean()
    denoised_data = denoised_data.fillna(method='bfill').fillna(method='ffill')
    return denoised_data

def normalize_data(data):
    """Min-Max归一化，缩放至[NORMALIZE_RANGE]"""
    scaler = MinMaxScaler(feature_range=NORMALIZE_RANGE)
    normalized_data = scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
    return normalized_data

def load_data(file_path):
    """加载CSV数据，返回DataFrame"""
    return pd.read_csv(file_path, parse_dates=["Timestamp"], index_col="Timestamp")

def main():
    # 加载原始数据
    raw_df = load_data(RAW_DATA_PATH)
    moisture_raw = raw_df["Moisture_Value(0-1023)"]
    temperature_raw = raw_df["Temperature(°C)"]
    
    # 1. 去噪处理
    moisture_denoised = moving_average_denoising(moisture_raw)
    temperature_denoised = moving_average_denoising(temperature_raw)
    
    # 2. 归一化处理
    moisture_normalized = normalize_data(moisture_denoised)
    temperature_normalized = normalize_data(temperature_denoised)
    
    # 构建预处理后的数据框并保存
    preprocessed_df = pd.DataFrame({
        "Timestamp": raw_df.index,
        "Moisture_Raw": moisture_raw.values,
        "Moisture_Denoised": moisture_denoised.values,
        "Moisture_Normalized": moisture_normalized,
        "Temperature_Raw": temperature_raw.values,
        "Temperature_Denoised": temperature_denoised.values,
        "Temperature_Normalized": temperature_normalized
    }).set_index("Timestamp")
    
    preprocessed_df.to_csv(PREPROCESSED_DATA_PATH, encoding="utf-8")
    print(f"Preprocessed data saved to: {PREPROCESSED_DATA_PATH}")

if __name__ == "__main__":
    main()
