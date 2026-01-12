import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from config import PREPROCESSED_DATA_PATH, SEQ_LENGTH, PRED_LENGTH, TEST_SIZE, MODEL_PREDICTION_PATH

def load_data(file_path):
    """加载预处理数据"""
    return pd.read_csv(file_path, parse_dates=["Timestamp"], index_col="Timestamp")

def create_time_series_dataset(data):
    """创建时间序列数据集：X(前SEQ_LENGTH个数据) → y(后PRED_LENGTH个数据)"""
    X, y = [], []
    for i in range(len(data) - SEQ_LENGTH - PRED_LENGTH + 1):
        X.append(data[i:i+SEQ_LENGTH])
        y.append(data[i+SEQ_LENGTH:i+SEQ_LENGTH+PRED_LENGTH])
    return np.array(X), np.array(y)

def main():
    # 加载数据（使用归一化后的湿度数据）
    preprocessed_df = load_data(PREPROCESSED_DATA_PATH)
    moisture_normalized = preprocessed_df["Moisture_Normalized"].values
    
    # 创建数据集
    X, y = create_time_series_dataset(moisture_normalized)
    
    # 划分训练集/测试集（时间序列不打乱）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)
    
    # 调整数据形状（线性回归需要2D输入）
    X_train_flat = X_train.reshape(-1, SEQ_LENGTH)
    X_test_flat = X_test.reshape(-1, SEQ_LENGTH)
    y_train_flat = y_train.reshape(-1, PRED_LENGTH)
    y_test_flat = y_test.reshape(-1, PRED_LENGTH)
    
    # 训练模型
    lr_model = LinearRegression()
    lr_model.fit(X_train_flat, y_train_flat)
    
    # 预测
    y_test_pred = lr_model.predict(X_test_flat)
    
    # 模型评估
    test_mse = mean_squared_error(y_test_flat, y_test_pred)
    test_r2 = r2_score(y_test_flat, y_test_pred)
    print(f"Linear Regression Evaluation - MSE: {test_mse:.4f} | R²: {test_r2:.4f}")
    
    # 可视化预测结果（取前20个测试样本）
    fig, ax = plt.subplots(figsize=(12, 6))
    sample_size = min(20, len(y_test_flat))
    x_axis = np.arange(sample_size)
    
    ax.scatter(x_axis, y_test_flat[:sample_size, 0], label="True (1st Point)", color="#1f77b4", s=50)
    ax.scatter(x_axis, y_test_pred[:sample_size, 0], label="Pred (1st Point)", color="#ff7f0e", s=50, marker='^')
    ax.scatter(x_axis, y_test_flat[:sample_size, 1], label="True (2nd Point)", color="#2ca02c", s=50)
    ax.scatter(x_axis, y_test_pred[:sample_size, 1], label="Pred (2nd Point)", color="#d62728", s=50, marker='^')
    
    ax.set_xlabel("Test Sample Index")
    ax.set_ylabel("Normalized Moisture")
    ax.set_title(f"Linear Regression Predictions (MSE={test_mse:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.savefig(f"{MODEL_PREDICTION_PATH}/linear_regression_predictions.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Prediction plot saved to:", MODEL_PREDICTION_PATH)

if __name__ == "__main__":
    main()
