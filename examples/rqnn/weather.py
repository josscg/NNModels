import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt

data_df = pd.read_csv(r"PRSA_Data_Aotizhongxin_20130301-20170228.csv")

data_df['datetime'] = pd.to_datetime(data_df[['year', 'month', 'day', 'hour']])

# Apply sin/cos encoding for cyclic features
data_df['hour_sin'] = np.sin(2 * np.pi * data_df['hour'] / 24)
data_df['hour_cos'] = np.cos(2 * np.pi * data_df['hour'] / 24)

data_df['dayofweek_sin'] = np.sin(2 * np.pi * data_df['day'] / 7)
data_df['dayofweek_cos'] = np.cos(2 * np.pi * data_df['day'] / 7)

data_df['month_sin'] = np.sin(2 * np.pi * data_df['month'] / 12)
data_df['month_cos'] = np.cos(2 * np.pi * data_df['month'] / 12)

data_df = data_df.dropna()

datetime = data_df['datetime']

data_df = data_df.drop(columns=['No', 'year', 'month', 'day', 'hour', 'station', 'datetime'])

training_split = int(0.8 * len(data_df))

df_train, df_test = data_df[:training_split], data_df[training_split:]

from dataprocessing import *

target_col = 'PM2.5'


# Preprocess
X_train, y_train, scaler, encoder = fit_transform_train(df_train, target_col, quantum=True)
X_test, y_test = transform_test(df_test, target_col, scaler, encoder, quantum=True)

window_size = 5

# Create RNN windows
X_train_rnn, y_train_rnn = create_windows_rnn(X_train, y_train, window_size)
X_test_rnn, y_test_rnn = create_windows_rnn(X_test, y_test, window_size)


from quantum_models.HQRNNModel import *

architecture = {
    "pre_quantum": True,
    "quantum": {
        "read_in": {"two_features_per_qbit": True},
        "hidden_state": {"units": 5, "encoding": "angle"},
        "return_sequences": False,
        "ansatz": "aspen"},
    "post_quantum": True}

qrnn_model = QRNNModel(X_train_rnn.shape[-1], architecture)

qrnn_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate= 0.001),
                  loss = "mse", 
                  metrics = ['mae']
)
        
qrnn_model.fit(X_train_rnn, y_train_rnn, 
              validation_split=0.2, epochs=10, batch_size=128, 
              shuffle=False
)

y_pred_train_rnn = qrnn_model.predict(X_train_rnn)
y_pred_test_rnn = qrnn_model.predict(X_test_rnn)

"QRNN"
plt.figure(figsize=(12,6))
plt.plot(datetime, data_df[target_col], label="Actual")
plt.plot(datetime[window_size:window_size+len(y_pred_train_rnn.flatten())], 
         y_pred_train_rnn.flatten(), label='Train Predictions', color='green')
plt.plot(datetime[window_size+len(y_pred_train_rnn):window_size+len(y_pred_train_rnn)+len(y_pred_test_rnn.flatten())], 
         y_pred_test_rnn.flatten(), label='Test Predictions', color='red')
plt.title("QRNN: True vs Predicted Values")
plt.xlabel("Test sample index (day)")
plt.ylabel("Next day log return")
plt.legend()
plt.grid(True)
plt.show()
"QRNN"
plt.figure(figsize=(12,6))
plt.plot(datetime[training_split+window_size:], 
         data_df[target_col].iloc[training_split+window_size:], 
         label="Actual", color="blue")
plt.plot(datetime[training_split+window_size:], 
         y_pred_test_rnn.flatten(), 
         label="Test Predictions", color="red")
plt.title("QRNN Zoom: Test Predictions vs Actual")
plt.xlabel("Date")
plt.ylabel("Next day log return")
plt.legend()
plt.grid(True)
plt.show()