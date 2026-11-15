# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 15:23:24 2025

@author: lm10_
"""
# BIBLIOGRAPHY
# chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/2204.11115


import tensorflow as tf
from tensorflow.keras.layers import Dense, SimpleRNN, LayerNormalization, GRU, LSTM

class BaseRecurrentModel(tf.keras.Model):
    """
    Base class for recurrent models (RNN, LSTM, GRU)
    
    layer_class: the recurrent layer type (SimpleRNN, LSTM, GRU)
    layer_configs: list of dicts describing each recurrent layer
    output_dim: output layer size
    output_act: output layer activation
    """
    def __init__(self, layer_class, layer_configs, output_dim=1, output_act=None):
        super(BaseRecurrentModel, self).__init__()
        
        self.recurrent_layers = []
        for cfg in layer_configs:
            self.recurrent_layers.append(
                layer_class(
                    cfg["units"],
                    activation=cfg.get("activation", None),
                    return_sequences=cfg.get("return_sequences", False),
                    dropout=cfg.get("dropout", 0.0),               # input dropout
                    recurrent_dropout=cfg.get("recurrent_dropout", 0.0),  # hidden-state dropout
                    kernel_regularizer=cfg.get("kernel_reg", None),
                    kernel_initializer=cfg.get("kernel_init", None)
                )
            )
            if cfg.get("layernorm", False):
                self.recurrent_layers.append(LayerNormalization())
        
        self.output_layer = Dense(output_dim, activation=output_act)
    
    def call(self, inputs, training=False):
        x = inputs
        for layer in self.recurrent_layers:
            x = layer(x, training=training)
        return self.output_layer(x)


# ----------------------------------------------------------
# RNN 
# ----------------------------------------------------------
class RNNModel(BaseRecurrentModel):
    """
    Example:
        layer_configs = [
            {"units":128, "activation":"tanh", "return_sequences":True, "dropout":0.1, "recurrent_dropout":0.1},
            {"units":64, "activation":"relu", "return_sequences":False}
        ]
        model = RNNModel(layer_configs, output_dim=1)
    """
    def __init__(self, layer_configs, output_dim=1, output_act=None):
        super().__init__(SimpleRNN, layer_configs, output_dim, output_act)


# ----------------------------------------------------------
# LSTM
# ----------------------------------------------------------
class LSTMModel(BaseRecurrentModel):
    """
    Example:
        layer_configs = [
            {"units":128, "activation":"tanh", "return_sequences":True, "dropout":0.2},
            {"units":64, "activation":"tanh", "return_sequences":False}
        ]
        model = LSTMModel(layer_configs, output_dim=1)
    """
    def __init__(self, layer_configs, output_dim=1, output_act=None):
        super().__init__(LSTM, layer_configs, output_dim, output_act)


# ----------------------------------------------------------
# GRU
# ---------------------------------------------------------- 
class GRUModel(BaseRecurrentModel):
    """
    Example:
        layer_configs = [
            {"units":64, "activation":"tanh", "return_sequences":True, "dropout":0.2},
            {"units":32, "activation":"relu", "return_sequences":False}
        ]
        model = GRUModel(layer_configs, output_dim=1)
    """
    def __init__(self, layer_configs, output_dim=1, output_act=None):
        super().__init__(GRU, layer_configs, output_dim, output_act)
