import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from config import PREPROCESSED_DATA_PATH, SEQ_LENGTH, PRED_LENGTH, TEST_SIZE, LSTM_EPOCHS, LSTM_BATCH_SIZE, MODEL_PREDICTION_PATH

def load_data(file_path):
    """加载预处理数据"""
    return pd.read_csv(file_path, parse_dates=["Timestamp"], index_col="Timestamp")

def create_time_series_dataset(data):
    """创建LSTM输入数据集（3D格式：[样本数, 时间步, 特征数]）"""
    X, y = [], []
    for i in range(len(data) - SEQ_LENGTH - PRED_LENGTH + 1):
        # reshape为(SEQ_LENGTH, 1)，1表示单特征（湿度）
        X.append(data[i:i+SEQ_LENGTH].reshape(-1, 1))
        y.append(data[i+SEQ_LENGTH:i+SEQ_LENGTH+PRED_LENGTH])
    return np.array(X), np.array(y)

def plot_training_history(history):
    """绘制训练损失曲线"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history["loss"], label="Training Loss", color="#1f77b4")
    ax.plot(history.history["val_loss"], label="Validation Loss", color="#ff7f0e")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("LSTM Model Training History")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(f"{MODEL_PREDICTION_PATH}/lstm_training_history.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 加载数据
    preprocessed_df = load_data(PREPROCESSED_DATA_PATH)
    moisture_normalized = preprocessed_df["Moisture_Normalized"].values
    
    # 创建3D数据集
    X, y = create_time_series_dataset(moisture_normalized)
    
    # 划分训练集/测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)
    
    # 构建轻量级LSTM模型
    model = Sequential([
        LSTM(32, input_shape=(SEQ_LENGTH, 1), return_sequences=False),  # 32个LSTM单元
        Dropout(0.2),  # 防止过拟合
        Dense(PRED_LENGTH)  # 输出层：预测PRED_LENGTH个值
    ])
    
    model.compile(optimizer="adam", loss="mse")
    
    # 早停回调：验证损失10轮不下降则停止，恢复最优权重
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    
    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        validation_split=0.1,  # 10%训练集作为验证集
        callbacks=[early_stopping],
        verbose=1
    )
    
    # 预测
    y_test_pred = model.predict(X_test, verbose=0)
    
    # 模型评估
    test_mse = mean_squared_error(y_test.reshape(-1, PRED_LENGTH), y_test_pred)
    test_r2 = r2_score(y_test.reshape(-1, PRED_LENGTH), y_test_pred)
    print(f"LSTM Model Evaluation - MSE: {test_mse:.4f} | R²: {test_r2:.4f}")
    
    # 可视化训练历史和预测结果
    plot_training_history(history)
    
    # 预测结果可视化
    fig, ax = plt.subplots(figsize=(12, 6))
    sample_size = min(20, len(y_test))
    x_axis = np.arange(sample_size)
    
    ax.scatter(x_axis, y_test[:sample_size, 0], label="True (1st Point)", color="#1f77b4", s=50)
    ax.scatter(x_axis, y_test_pred[:sample_size, 0], label="Pred (1st Point)", color="#ff7f0e", s=50, marker='^')
    ax.scatter(x_axis, y_test[:sample_size, 1], label="True (2nd Point)", color="#2ca02c", s=50)
    ax.scatter(x_axis, y_test_pred[:sample_size, 1], label="Pred (2nd Point)", color="#d62728", s=50, marker='^')
    
    ax.set_xlabel("Test Sample Index")
    ax.set_ylabel("Normalized Moisture")
    ax.set_title(f"LSTM Predictions (MSE={test_mse:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.savefig(f"{MODEL_PREDICTION_PATH}/lstm_predictions.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("LSTM results saved to:", MODEL_PREDICTION_PATH)

if __name__ == "__main__":
    main()
