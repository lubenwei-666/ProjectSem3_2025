import os

# 硬件配置：传感器引脚定义（树莓派GPIO编号）
MOISTURE_SENSOR_PIN_AO = 2  # 传感器模拟输出引脚
MOISTURE_SENSOR_PIN_DO = 3  # 传感器数字输出引脚
RPI_TEMPERATURE_PATH = "/sys/class/thermal/thermal_zone0/temp"  # 树莓派内置温度文件路径

# 数据采集配置
COLLECTION_INTERVAL = 1800  # 采集间隔（秒）→ 30分钟
COLLECTION_DURATION = 86400 * 2  # 采集时长（秒）→ 48小时
RAW_DATA_PATH = "../data/raw_data.csv"  # 原始数据保存路径

# 数据预处理配置
DENOISING_WINDOW = 5  # 移动平均滤波窗口大小
NORMALIZE_RANGE = (0, 1)  # 归一化范围
PREPROCESSED_DATA_PATH = "../data/preprocessed_data.csv"  # 预处理数据保存路径

# 模型配置
SEQ_LENGTH = 6  # 时间序列输入长度（用前6个数据点预测）
PRED_LENGTH = 2  # 预测长度（预测后2个数据点→12小时）
TEST_SIZE = 0.2  # 测试集占比
LSTM_EPOCHS = 50  # LSTM训练轮数
LSTM_BATCH_SIZE = 8  # LSTM批次大小

# 结果保存配置
RESULTS_BASE_PATH = "../results/"
TIME_SERIES_PLOT_PATH = os.path.join(RESULTS_BASE_PATH, "time_series_plots/")
MODEL_PREDICTION_PATH = os.path.join(RESULTS_BASE_PATH, "model_predictions/")

# 自动创建输出目录
for path in [RAW_DATA_PATH, PREPROCESSED_DATA_PATH, TIME_SERIES_PLOT_PATH, MODEL_PREDICTION_PATH]:
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
