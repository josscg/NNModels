# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 14:07:48 2025

@author: lm10_
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, LSTM, Lambda

# NN with noisy dataset: https://arxiv.org/html/2411.01670v1
# NN in time series: chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/1401.1333


class DNNModel(tf.keras.Model):
    
    """
    Simple Depp Neural Network model. Handles multiple layers configurations, 
    batch normalization and dropout rates.
    
    Example:
        [{"units":128, "activation":"relu", "batchnorm":False, "dropout":0.25},
         {"units":32, "activation":"tanh", "batchnorm":False, "dropout":0.5}], 
         1, 
        None
    """
    
    def __init__(self, layer_configs, output_dim=1, output_act=None):
        
        super(DNNModel, self).__init__()
        
        self.hidden_layers = []
        for cfg in layer_configs:
            self.hidden_layers.append(
                Dense(cfg["units"],
                      activation=cfg.get("activation", None),
                      kernel_regularizer=cfg.get("kernel_reg", None),
                      kernel_initializer=cfg.get("kernel_init", None))
            )
            if cfg.get("batchnorm", False):
                self.hidden_layers.append(BatchNormalization())
            if cfg.get("dropout", 0) > 0:
                self.hidden_layers.append(Dropout(cfg["dropout"]))
            
        self.output_layer = Dense(output_dim, activation=output_act)
        
    def call(self, inputs, training=False):
        
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        return self.output_layer(x)



class FlexibleFFNN(tf.keras.Model):
    def __init__(self, layer_configs, output_dim=1, output_act=None):
        """
        Example:
        [
            {"type": "dense", "units": 64, "activation": "relu", "dropout":0.2, "batchnorm":True},
            {"type": "dense", "units": 64, "activation": "relu", "residual": True},
            {"type": "lambda", "function": tf.math.sin},   # apply sin element-wise
            {"type": "conv1d", "filters": 32, "kernel_size": 3, "activation": "relu"},
            {"type": "rnn", "units": 32, "activation": "tanh"}
        ]
        """
        super(FlexibleFFNN, self).__init__()
        self.blocks = []
        
        for cfg in layer_configs:
            if cfg["type"] == "dense":
                layers = [Dense(cfg["units"], 
                                activation=cfg.get("activation", None),
                                kernel_regularizer=cfg.get("kernel_reg", None))]
                if cfg.get("batchnorm", False):
                    layers.append(BatchNormalization())
                if cfg.get("dropout", 0) > 0:
                    layers.append(Dropout(cfg["dropout"]))
                self.blocks.append({"layers": layers, "residual": cfg.get("residual", False)})
            
            elif cfg["type"] == "conv1d":
                self.blocks.append({
                    "layers": [Conv1D(cfg["filters"], 
                                      kernel_size=cfg["kernel_size"], 
                                      activation=cfg.get("activation", None))],
                    "residual": False
                })
            
            elif cfg["type"] == "rnn":
                self.blocks.append({
                    "layers": [LSTM(cfg["units"], activation=cfg.get("activation", None), return_sequences=False)],
                    "residual": False
                })
            
            elif cfg["type"] == "lambda":
                self.blocks.append({
                    "layers": [Lambda(cfg["function"])],
                    "residual": False
                })
            
            else:
                raise ValueError(f"Unknown layer type: {cfg['type']}")
        
        self.out = Dense(output_dim, activation=output_act)
    
    def call(self, inputs, training=False):
        x = inputs
        for block in self.blocks:
            if isinstance(block["layers"][0], LSTM) and len(x.shape) == 2:
                x = tf.expand_dims(x, axis=1)  # [batch, 1, features]
            elif isinstance(block["layers"][0], Dense) and len(x.shape) == 3:
                x = tf.reshape(x, [tf.shape(x)[0], -1])  # flatten
        
            shortcut = x
            for layer in block["layers"]:
                x = layer(x, training=training)
            if block.get("residual", False):
                x = tf.keras.layers.add([x, shortcut])
        return self.out(x)


# ===================================================================
# ==================== GRAPH NEURAL NETWORK MODEL ===================
# ===================================================================

#chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/2307.03759


# ===================================================================
# ================== REINFORCEMENT LEARNING MODEL ===================
# ===================================================================

# ===================================================================
# ========================= PINN MODEL ==============================
# ===================================================================

# chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/1708.07469





