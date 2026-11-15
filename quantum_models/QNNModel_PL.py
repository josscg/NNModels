# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 16:31:26 2025

@author: lm10_
"""


# BOOK/PAPER: Quantum Machine Learning A Hands-on Tutorial for Machine Learning Practitioners and Researchers
# chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/2502.01146
# BOOK: Quantum neural networks (200 pages)
# chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/2205.08154
# PAPER: Quantum computing with Qiskit (very high-level)
# chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/2405.08810
# https://arxiv.org/html/2401.10683v1/#S2


import pennylane as qml
from pennylane.transforms.optimization import single_qubit_fusion
from functools import partial
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Configure TensorFlow to use float32 consistently
# tf.keras.backend.set_floatx('float64')


class ReadIn():
    """
    Read-in layer of the quantum circuit.
    It embeds the classical data into quantum interpretable objects.
    """
    def __init__(self, n_feat, read_in_arch):
        self.two_features_per_qbit = read_in_arch.get("two_features_per_qbit", False)
        self.n_feat = n_feat
        self.n_qbits = int(np.ceil(n_feat/2)) if self.two_features_per_qbit else self.n_feat

        # Precompute indexes
        if self.two_features_per_qbit:
            self.indexes = [(2*i, 2*i + 1) for i in range(self.n_qbits)]
        else:
            self.indexes = [i for i in range(self.n_qbits)]

    def circuit(self, inputs):
        
        # Data encoding
        if self.two_features_per_qbit:
            for i, (idx1, idx2) in enumerate(self.indexes):
                qml.RY(inputs[idx1], wires=i)
                if idx2 < self.n_feat:
                    qml.RZ(inputs[idx2], wires=i)
        else:
            for i, idx in enumerate(self.indexes):
                qml.RY(inputs[idx], wires=i)

            

class Anzats():
    """
    
    Anzats layer describing the variational layer of the quantum circuit
    We must describe this structure by a dictionary  
    Example:
        architecture = {
                        "pre_quantum": True,                    
                        "quantum": {
                                    "read_in": {"two_features_per_qbit": True},
                                    "anzats": [{"RZ": list(range(4))},
                                                {"CNOT":[[i, i+1] for i in range(0, 3, 2)]},
                                                {"RZ": [i for i in range(4) if i%2!=0]},
                                                {"CNOT":[[i, i+2] for i in range(1, 2)]},
                                                {"RZ": [3]}],
                                    "measure": [1, 5, 8]},
                        "post_quantum": True
                        }
    
    """
    def __init__(self, n_qbits, anzats_arch):
        
        self.gates = {"RX": qml.RX, "RY": qml.RY, "RZ": qml.RZ}
        self.control_gates = {"CNOT": qml.CNOT, "CY": qml.CY, "CZ": qml.CZ}
        
        self.n_qbits = n_qbits
        self.arch = anzats_arch
        
        if self.arch == "default":
            self.n_weights = n_qbits
            
        elif self.arch == "aspen":
            self.n_weights = 0 
            step = 1 
            n = self.n_qbits 
            while n//step >= 1: 
                if n==step:
                    break
                self.n_weights += n//step 
                step *= 2
            print("Number of weights:", self.n_weights)
        # 16            
        else:
            self.n_weights = 0
            for layer in self.arch:
                for op, qbits in layer.items():
                    if op in self.gates:
                        self.n_weights += len(qbits)
                    
    def circuit(self, weights):
        if self.arch == "default":
            for i in range(self.n_qbits):
                qml.RZ(weights[i], wires=i)
                
            for i in range(self.n_qbits-1):
                qml.CNOT(wires=[i, i+1])
                
        elif self.arch == "aspen":
            step = 1
            weight_idx = 0   # keep track of which weights are consumed
            while step < self.n_qbits:
                for q in range(step - 1, self.n_qbits, step):
                    qml.RZ(weights[weight_idx], wires=q)
                    weight_idx += 1

                stride = 2 * step
                for i in range(step-1, self.n_qbits - step, stride):
                    qml.CNOT(wires=[i, i + step])
                step *= 2
        
        else:
            weight_idx = 0
            for layer in self.arch:
                for op, qbits in layer.items():
                    if op in self.gates:
                        for qbit in qbits:
                            self.gates[op](weights[weight_idx], wires=qbit)
                            weight_idx += 1
                    elif op in self.control_gates:
                        for control, qbit in qbits:
                            self.control_gates[op](wires=[control, qbit])
                    else:
                        raise ValueError(f"Operator {op} not recognized in anzats architecture.")



class QuantumLayer(tf.keras.layers.Layer):
    def __init__(self, n_feat, quantum_arch):
        super().__init__()
        self.n_feat = n_feat
        self.arch = quantum_arch

        # Build subcomponents
        read_in_arch = self.arch.get("read_in", {"two_features_per_qbit": True})
        self.read_in = ReadIn(self.n_feat, read_in_arch)
        self.n_qbits = self.read_in.n_qbits
        anzats_arch = self.arch.get("ansatz", "aspen")
        self.anzats = Anzats(self.n_qbits, anzats_arch)
        self.n_weights = self.anzats.n_weights
        
        self.quantum_weights = self.add_weight(
            name = "quantum_weights",
            shape = (self.n_weights,),
            initializer = "random_normal",
            trainable = True,
            dtype = tf.float32,
        )

        # Placeholders
        self.dev = None
        self.quantum_circuit = None

    def build(self, input_shape):
        # Runtime-safe initialization
        self.dev = qml.device(
            "default.qubit",
            wires=self.n_qbits,
            shots=None
        )
        self.quantum_circuit = self._build_qnode()
        super().build(input_shape)
        
    def _build_qnode(self):
        #@partial(qml.batch_input, argnum=0)
        qbits_measured = self.arch.get("measure", range(self.n_qbits))
        @qml.qnode(self.dev, interface="tf", diff_method="backprop")
        def circuit(inputs, weights):
            self.read_in.circuit(inputs)
            self.anzats.circuit(weights)
            return [qml.expval(qml.PauliZ(i)) for i in qbits_measured]

        return circuit

    def _draw_circuit(self):
        dummy_input = np.zeros(self.n_feat, dtype=np.float32)
        dummy_weights = np.zeros(self.n_weights, dtype=np.float32)
        fig, ax = qml.draw_mpl(self.quantum_circuit)(dummy_input, dummy_weights)
        plt.show()

    def call(self, inputs):
        #print("QL inputs:", inputs)
        def _process_sample(sample):
            result = self.quantum_circuit(sample, self.quantum_weights)
            return tf.stack(result)
        return tf.vectorized_map(_process_sample, inputs)



class QNNModel(tf.keras.Model):
    def __init__(self, n_feat, architecture):
        super().__init__()
        
        self.hidden_layers = []
        
        if n_feat > 12:
            print(f"The number of features is too big for a quantum circuit: {n_feat}")
            print("Feature reduction to 12 total features")
            n_feat = 12
        
        # Classical pre-quantum layers
        pre_quantum = architecture.get("pre_quantum", None)
        if pre_quantum:    
            self.hidden_layers.append(tf.keras.layers.Dense(n_feat, activation=None))#, dtype=tf.float32)

        
        # Quantum layer
        quantum_arch = architecture.get("quantum", None)
        self.hidden_layers.append(QuantumLayer(n_feat, quantum_arch))
        
        # Classical post-quantum layers
        post_quantum = architecture.get("post_quantum", None)
        if post_quantum:
            #self.hidden_layers.append(tf.keras.layers.Dense(4, activation='relu'))#, dtype=tf.float32)
            self.hidden_layers.append(tf.keras.layers.Dense(1, activation=None))#, dtype=tf.float32)
    
    def call(self, inputs):
        #print("Model inputs:", inputs)
        #inputs = tf.cast(inputs, tf.float32)        
        x = inputs 
        for layer in self.hidden_layers:
            x = layer(x)
        return x

