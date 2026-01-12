import matplotlib.pyplot as plt
import pandas as pd
from config import PREPROCESSED_DATA_PATH, TIME_SERIES_PLOT_PATH

def load_data(file_path):
    """加载预处理数据"""
    return pd.read_csv(file_path, parse_dates=["Timestamp"], index_col="Timestamp")

def plot_time_series_trends(preprocessed_df):
    """绘制湿度、温度的原始值vs去噪值趋势图"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # 湿度趋势
    ax1.plot(preprocessed_df.index, preprocessed_df["Moisture_Raw"], label="Raw Moisture", color="#ff7f0e", alpha=0.6, linewidth=1)
    ax1.plot(preprocessed_df.index, preprocessed_df["Moisture_Denoised"], label="Denoised Moisture", color="#1f77b4", linewidth=2)
    ax1.set_ylabel("Moisture Value (0-1023)")
    ax1.set_title("Soil Moisture Trend (Raw vs Denoised)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 温度趋势
    ax2.plot(preprocessed_df.index, preprocessed_df["Temperature_Raw"], label="Raw Temperature", color="#2ca02c", alpha=0.6, linewidth=1)
    ax2.plot(preprocessed_df.index, preprocessed_df["Temperature_Denoised"], label="Denoised Temperature", color="#d62728", linewidth=2)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Temperature (°C)")
    ax2.set_title("Temperature Trend (Raw vs Denoised)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    fig.savefig(f"{TIME_SERIES_PLOT_PATH}/moisture_temperature_trends.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_normalized_distribution(preprocessed_df):
    """绘制归一化后的数据分布直方图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.hist(preprocessed_df["Moisture_Normalized"], bins=20, color="#1f77b4", alpha=0.7)
    ax1.set_xlabel("Normalized Moisture")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Normalized Moisture Distribution")
    ax1.grid(True, alpha=0.3)
    
    ax2.hist(preprocessed_df["Temperature_Normalized"], bins=20, color="#d62728", alpha=0.7)
    ax2.set_xlabel("Normalized Temperature")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Normalized Temperature Distribution")
    ax2.grid(True, alpha=0.3)
    
    fig.savefig(f"{TIME_SERIES_PLOT_PATH}/normalized_data_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 加载数据
    preprocessed_df = load_data(PREPROCESSED_DATA_PATH)
    
    # 输出描述性统计
    print("Descriptive Statistics (Denoised Data):")
    print(preprocessed_df[["Moisture_Denoised", "Temperature_Denoised"]].describe().round(3))
    
    # 绘制图表
    plot_time_series_trends(preprocessed_df)
    plot_normalized_distribution(preprocessed_df)
    print("Plots saved to:", TIME_SERIES_PLOT_PATH)

if __name__ == "__main__":
    main()
