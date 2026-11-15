# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 20:46:11 2025

@author: lm10_
"""

import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt

data_df = pd.read_csv(r"C:\Users\lm10_\OneDrive\Escritorio\PRSA_Data_20130301-20170228\PRSA_Data_Aotizhongxin_20130301-20170228.csv")

data_df['datetime'] = pd.to_datetime(data_df[['year', 'month', 'day', 'hour']])


#%%

# Apply sin/cos encoding for cyclic features
data_df['hour_sin'] = np.sin(2 * np.pi * data_df['hour'] / 24)
data_df['hour_cos'] = np.cos(2 * np.pi * data_df['hour'] / 24)

data_df['dayofweek_sin'] = np.sin(2 * np.pi * data_df['day'] / 7)
data_df['dayofweek_cos'] = np.cos(2 * np.pi * data_df['day'] / 7)

data_df['month_sin'] = np.sin(2 * np.pi * data_df['month'] / 12)
data_df['month_cos'] = np.cos(2 * np.pi * data_df['month'] / 12)


data_df = data_df.drop(columns=['No', 'year', 'month', 'day', 'hour', 'station'])


data_df = data_df.dropna()

#%%
for feat in data_df:
    if feat == "wd":
        pass
    else:
        plt.plot(data_df["datetime"],data_df[feat], label=feat)
        plt.xlabel("Time")
        plt.ylabel(feat)
        plt.show()


#%%

training_split = int(0.8 * len(data_df))

df_train, df_test = data_df[:training_split], data_df[training_split:]

from DataPreprocessing import *

target_col = 'PM2.5'

# Preprocess
X_train, y_train, scaler, encoder = fit_transform_train(df_train, target_col)
X_test, y_test = transform_test(df_test, target_col, scaler, encoder)


#%%

# WINDOW GENERATORS 

window_size = 20

# Create RNN windows
X_train_rnn, y_train_rnn = create_windows_rnn(X_train, y_train, window_size)
X_test_rnn, y_test_rnn = create_windows_rnn(X_test, y_test, window_size)


# Create DNN windows
# X_train_dnn, y_train_dnn = create_windows_dnn(X_train, y_train, window_size, use_moving_stats=False)
# X_test_dnn, y_test_dnn = create_windows_dnn(X_test, y_test, window_size, use_moving_stats=False)

#%%

from RNNModel import *

rnn_model = RNNModel([{"units":128, "activation":"tanh", "return_sequences":False, "dropout":0.1, "recurrent_dropout":0.1}],
                      output_dim=1,
                      output_act=None)

rnn_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate= 0.001),
                  loss = "mse", 
                  metrics = ['mae'])
        
rnn_model.fit(X_train_rnn, y_train_rnn, 
              validation_split=0.2, epochs=100, batch_size=128, 
              shuffle=False)




# lstm_model = LSTMModel([{"units":128, "activation":"tanh", "return_sequences":False, "dropout":0.1, "recurrent_dropout":0.1}],
#                      output_dim=1,
#                      output_act=None)

# lstm_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate= 0.001),
#                   loss = "mse", 
#                   metrics = ['mae'])
        
# lstm_model.fit(X_train_rnn, y_train_rnn, 
#               validation_split=0.2, epochs=100, batch_size=128, 
#               shuffle=False)




# dnn_model = DNNModel([{"units":128, "activation":"relu", "batchnorm":False, "dropout":0.5},
#                       {"units":128, "activation":"relu", "batchnorm":False, "dropout":0.5},
#                       {"units":32, "activation":"tanh", "batchnorm":False, "dropout":0.5}], 
#                      1, 
#                      None)

# dnn_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate= 0.001),
#                   loss = "mse", 
#                   metrics = ['mae'])

# dnn_model.fit(X_train_dnn, y_train_dnn, 
#               validation_split=0.2, epochs=100, batch_size=128, 
#               shuffle=False)

#%%

from RCModel import *

rc_model = ReservoirModel(reservoir_type="stochastic", n_inputs=X_train_rnn.shape[-1])


rc_model.fit(X_train_rnn, y_train_rnn)





#%%

y_pred_train_rnn = rnn_model.predict(X_train_rnn)
y_pred_test_rnn = rnn_model.predict(X_test_rnn)

# y_pred_train_lstm = lstm_model.predict(X_train_rnn)
# y_pred_test_lstm = lstm_model.predict(X_test_rnn)

# y_pred_train_dnn = dnn_model.predict(X_train_dnn)
# y_pred_test_dnn = dnn_model.predict(X_test_dnn)


y_pred_train_rc = rc_model.predict(X_train_rnn)
y_pred_test_rc = rc_model.predict(X_test_rnn)


#%%

"RNN"
plt.figure(figsize=(12,6))
plt.plot(data_df["datetime"], data_df[target_col], label="Actual")
plt.plot(data_df["datetime"][window_size:window_size+len(y_pred_train_rnn.flatten())], 
         y_pred_train_rnn.flatten(), label='Train Predictions', color='green')
plt.plot(data_df["datetime"][window_size+len(y_pred_train_rnn):window_size+len(y_pred_train_rnn)+len(y_pred_test_rnn.flatten())], 
         y_pred_test_rnn.flatten(), label='Test Predictions', color='red')
plt.title("RNN: True vs Predicted Values")
plt.xlabel("Test sample index (day)")
plt.ylabel("Next day log return")
plt.legend()
plt.grid(True)
plt.show()
"RNN"
plt.figure(figsize=(12,6))
plt.plot(data_df["datetime"][training_split+window_size:], 
         data_df[target_col].iloc[training_split+window_size:], 
         label="Actual", color="blue")
plt.plot(data_df["datetime"][training_split+window_size:], 
         y_pred_test_rnn.flatten(), 
         label="Test Predictions", color="red")
plt.title("RNN Zoom: Test Predictions vs Actual")
plt.xlabel("Date")
plt.ylabel("Next day log return")
plt.legend()
plt.grid(True)
plt.show()


"RC"
plt.figure(figsize=(12,6))
plt.plot(data_df["datetime"], data_df[target_col], label="Actual")
plt.plot(data_df["datetime"][window_size:window_size+len(y_pred_train_rc.flatten())], 
         y_pred_train_rc.flatten(), label='Train Predictions', color='green')
plt.plot(data_df["datetime"][window_size+len(y_pred_train_rc):window_size+len(y_pred_train_rc)+len(y_pred_test_rc.flatten())], 
         y_pred_test_rc.flatten(), label='Test Predictions', color='red')
plt.title("RC: True vs Predicted Values")
plt.xlabel("Test sample index (day)")
plt.ylabel("Next day log return")
plt.legend()
plt.grid(True)
plt.show()
"RC"
plt.figure(figsize=(12,6))
plt.plot(data_df["datetime"][training_split+window_size:], 
         data_df[target_col].iloc[training_split+window_size:], 
         label="Actual", color="blue")
plt.plot(data_df["datetime"][training_split+window_size:], 
         y_pred_test_rc.flatten(), 
         label="Test Predictions", color="red")
plt.title("RC Zoom: Test Predictions vs Actual")
plt.xlabel("Date")
plt.ylabel("Next day log return")
plt.legend()
plt.grid(True)
plt.show()

#%%
""" -----------------------
Plotting the entire history
----------------------- """ 

"RNN"
plt.figure(figsize=(12,6))
plt.plot(data_df["datetime"], data_df[target_col], label="Actual")
plt.plot(data_df["datetime"][window_size:window_size+len(y_pred_train_rnn.flatten())], 
         y_pred_train_rnn.flatten(), label='Train Predictions', color='green')
plt.plot(data_df["datetime"][window_size+len(y_pred_train_rnn):window_size+len(y_pred_train_rnn)+len(y_pred_test_rnn.flatten())], 
         y_pred_test_rnn.flatten(), label='Test Predictions', color='red')
plt.title("RNN: True vs Predicted Values")
plt.xlabel("Test sample index (day)")
plt.ylabel("Next day log return")
plt.legend()
plt.grid(True)
plt.show()

"LSTM"
plt.figure(figsize=(12,6))
plt.plot(data_df["datetime"], data_df[target_col], label="Actual")
plt.plot(data_df["datetime"][window_size:window_size+len(y_pred_train_lstm.flatten())], 
         y_pred_train_lstm.flatten(), label='Train Predictions', color='green')
plt.plot(data_df["datetime"][window_size+len(y_pred_train_lstm):window_size+len(y_pred_train_rnn)+len(y_pred_test_rnn.flatten())], 
         y_pred_test_lstm.flatten(), label='Test Predictions', color='red')
plt.title("LSTM: True vs Predicted Values")
plt.xlabel("Test sample index (day)")
plt.ylabel("Next day log return")
plt.legend()
plt.grid(True)
plt.show()

"DNN"
plt.figure(figsize=(12,6))
plt.plot(data_df["datetime"], data_df[target_col], label="Actual")
plt.plot(data_df["datetime"][window_size:window_size+len(y_pred_train_dnn.flatten())], 
         y_pred_train_dnn.flatten(), label='Train Predictions', color='green')
plt.plot(data_df["datetime"][window_size+len(y_pred_train_dnn):window_size+len(y_pred_train_dnn)+len(y_pred_test_dnn.flatten())], 
         y_pred_test_dnn.flatten(), label='Test Predictions', color='red')
plt.title("DNN: True vs Predicted Values")
plt.xlabel("Test sample index (day)")
plt.ylabel("Next day log return")
plt.legend()
plt.grid(True)
plt.show()

"RC"
plt.figure(figsize=(12,6))
plt.plot(data_df["datetime"], data_df[target_col], label="Actual")
plt.plot(data_df["datetime"][window_size:window_size+len(y_pred_train_rc.flatten())], 
         y_pred_train_rc.flatten(), label='Train Predictions', color='green')
plt.plot(data_df["datetime"][window_size+len(y_pred_train_rc):window_size+len(y_pred_train_rc)+len(y_pred_test_rc.flatten())], 
         y_pred_test_rc.flatten(), label='Test Predictions', color='red')
plt.title("RC: True vs Predicted Values")
plt.xlabel("Test sample index (day)")
plt.ylabel("Next day log return")
plt.legend()
plt.grid(True)
plt.show()


""" -------------------------
Plotting the predicted values
------------------------- """

"RNN"
plt.figure(figsize=(12,6))
plt.plot(data_df["datetime"][training_split+window_size:], 
         data_df[target_col].iloc[training_split+window_size:], 
         label="Actual", color="blue")
plt.plot(data_df["datetime"][training_split+window_size:], 
         y_pred_test_rnn.flatten(), 
         label="Test Predictions", color="red")
plt.title("RNN Zoom: Test Predictions vs Actual")
plt.xlabel("Date")
plt.ylabel("Next day log return")
plt.legend()
plt.grid(True)
plt.show()
        
"LSTM"
plt.figure(figsize=(12,6))
plt.plot(data_df["datetime"][training_split+window_size:], 
         data_df[target_col].iloc[training_split+window_size:], 
         label="Actual", color="blue")
plt.plot(data_df["datetime"][training_split+window_size:], 
         y_pred_test_lstm.flatten(), 
         label="Test Predictions", color="red")
plt.title("LSTM Zoom: Test Predictions vs Actual")
plt.xlabel("Date")
plt.ylabel("Next day log return")
plt.legend()
plt.grid(True)
plt.show()

"DNN"
plt.figure(figsize=(12,6))
plt.plot(data_df["datetime"][training_split+window_size:], 
         data_df[target_col].iloc[training_split+window_size:], 
         label="Actual", color="blue")
plt.plot(data_df["datetime"][training_split+window_size:], 
         y_pred_test_dnn.flatten(), 
         label="Test Predictions", color="red")
plt.title("DNN Zoom: Test Predictions vs Actual")
plt.xlabel("Date")
plt.ylabel("Next day log return")
plt.legend()
plt.grid(True)
plt.show()

"RC"
plt.figure(figsize=(12,6))
plt.plot(data_df["datetime"][training_split+window_size:], 
         data_df[target_col].iloc[training_split+window_size:], 
         label="Actual", color="blue")
plt.plot(data_df["datetime"][training_split+window_size:], 
         y_pred_test_rc.flatten(), 
         label="Test Predictions", color="red")
plt.title("RNN Zoom: Test Predictions vs Actual")
plt.xlabel("Date")
plt.ylabel("Next day log return")
plt.legend()
plt.grid(True)
plt.show()
