# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 15:52:19 2025

@author: lm10_
"""

# RESOURCES
# chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/2302.03244 
# HYBRID ARCHITECTURE

import tensorflow as tf
import pennylane as qml
import numpy as np
from matplotlib import pyplot as plt
from quantum_models.QNNModel_PL import ReadIn, Anzats


class HiddenState:
    def __init__(self, arch, wire_offset):
        self.n_qubits = arch.get("units", 5)
        self.encoding = arch.get("encoding", "angle")
        self.wires = list(range(wire_offset, wire_offset + self.n_qubits))

    def circuit(self, h):
        for i, w in enumerate(self.wires):
            qml.RY(h[i], wires=w)


class QRNNCell(tf.keras.layers.Layer):
    def __init__(self,  n_feat, quantum_arch):
        super().__init__()
        self.n_feat = n_feat
        self.arch = quantum_arch
        
        # Build subcomponents
        read_in_arch = self.arch.get("read_in", {"two_features_per_qbit": True})
        self.read_in = ReadIn(self.n_feat, read_in_arch)
        self.n_qbits_data = self.read_in.n_qbits
        hidden_arch = self.arch.get("hidden_state", {"units": 5, "encoding":"angle"})
        self.hidden = HiddenState(hidden_arch, wire_offset=self.n_qbits_data)
        self.n_qbits_hidden = self.hidden.n_qubits
        anzats_arch = self.arch.get("ansatz", "aspen")
        self.anzats = Anzats(self.n_qbits_data + self.n_qbits_hidden, anzats_arch)
        self.n_weights = self.anzats.n_weights
        self.n_qbits = self.n_qbits_data + self.n_qbits_hidden

        
        self.quantum_weights = self.add_weight(
            name = "quantum_weights",
            shape = (self.n_weights,),
            initializer = "random_normal",
            trainable = True,
            dtype = tf.float32,
        )
        
        self.dev = None
        self.quantum_circuit = None
    

    def build(self, input_shape):
        self.dev = qml.device(
            "default.qubit",
            wires=self.n_qbits,
            shots=None)
        self.quantum_circuit = self._build_qnode()
        super().build(input_shape)
    
    def _build_qnode(self):
        qbits_measured =  self.hidden.wires
        @qml.qnode(self.dev, interface="tf", diff_method="backprop")
        def circuit(x, h_prev, weights):
            self.read_in.circuit(x)
            self.hidden.circuit(h_prev)    
            self.anzats.circuit(weights)       
            return [qml.expval(qml.PauliZ(w)) for w in qbits_measured]

        return circuit
    
    def _draw_circuit(self):
        dummy_input = np.zeros(self.n_feat, dtype=np.float32)
        dummy_weights = np.zeros(self.n_weights, dtype=np.float32)
        fig, ax = qml.draw_mpl(self.quantum_circuit)(dummy_input, dummy_weights)
        plt.show()

    @property
    def state_size(self):
        return self.n_qbits_hidden

    @property
    def output_size(self):
        return self.n_qbits_hidden

    def call(self, inputs, states):
        h_prev = states[0]
        
        def step(args):
            x, h = args
            return tf.stack(self.quantum_circuit(x, h, self.quantum_weights))

        q_out = tf.vectorized_map(step, (inputs, h_prev))
        q_out = tf.cast(q_out, tf.float32)
        h = tf.tanh(q_out)
        
        return h, [h]
    
    
class QRNNModel(tf.keras.Model):
    def __init__(self, n_feat, architecture):
        super().__init__()

        self.layers_stack = []

        if n_feat > 10:
            tf.print(
                f"[Warning] Too many features ({n_feat}). "
                "Reducing to 10."
            )
            n_feat = 10

        pre_quantum = architecture.get("pre_quantum", None)
        if pre_quantum:
            self.layers_stack.append(tf.keras.layers.Dense(n_feat, activation=None))#, dtype=tf.float32)

        quantum_arch = architecture.get("quantum", {})
        cell = QRNNCell(n_feat, quantum_arch)
        self.layers_stack.append(
                tf.keras.layers.RNN(cell, return_sequences=quantum_arch.get("return_sequences", False), return_state=False)
                )


        post_quantum = architecture.get("post_quantum", None)
        if post_quantum:
            self.layers_stack.append(tf.keras.layers.Dense(1, activation=None))

    def call(self, inputs):
        x = inputs
        for layer in self.layers_stack:
            x = layer(x)
        return x
