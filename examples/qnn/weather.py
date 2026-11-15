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

# Create DNN windows
X_train_dnn, y_train_dnn = create_windows_dnn(X_train, y_train, window_size, use_moving_stats=True)
X_test_dnn, y_test_dnn = create_windows_dnn(X_test, y_test, window_size, use_moving_stats=True)


from quantum_models.QNNModel_PL import *

num_features = X_train_dnn.shape[1]
    
architecture = {
                "pre_quantum": True,                    
                "quantum": {
                            "read_in": {"two_features_per_qbit": False},
                            "anzats": "aspen"},
                "post_quantum": True
                }

qnn_model = QNNModel(num_features, architecture)

qnn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss = "mse",
    metrics=['mae']
)

qnn_model.fit(X_train_dnn, y_train_dnn,
              epochs=10, batch_size=5, validation_split=0.2,
              shuffle=False
)

y_pred_train_dnn = qnn_model.predict(X_train_dnn)
y_pred_test_dnn = qnn_model.predict(X_test_dnn)