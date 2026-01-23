# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 11:00:09 2025

@author: lm10_
"""

import numpy as np
import pennylane as qml
import tensorflow as tf
from QNNModel_PL import ReadIn, Anzats

# RESOURCES
# From quantum feature maps to quantum reservoir computing: perspectives and applications: chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/2510.01797

class Readout:
    def __init__(self, ridge=1e-6, use_sklearn=False):
        self.ridge = float(ridge)
        self.W_out = None
        self.use_sklearn = use_sklearn
        if self.use_sklearn:
            try:
                from sklearn.linear_model import Ridge
                self._Ridge = Ridge
            except Exception:
                self.use_sklearn = False

    def fit(self, states, inputs, targets):
        X_ext = np.hstack([states, inputs, np.ones((states.shape[0], 1))])
        if self.use_sklearn:
            model = self._Ridge(alpha=self.ridge, fit_intercept=False)
            model.fit(X_ext, targets)
            self.W_out = model.coef_.T
        else:
            I = np.eye(X_ext.shape[1], dtype=float)
            A = X_ext.T @ X_ext + self.ridge * I
            B = X_ext.T @ targets
            self.W_out = np.linalg.solve(A, B)

    def predict(self, states, inputs):
        X_ext = np.hstack([states, inputs, np.ones((states.shape[0], 1))])
        return X_ext @ self.W_out


# --------------
# ReservoirBase
# --------------
class ReservoirBase:
    def __init__(self, n_feat, n_reservoir=8, leaking_rate=1.0, random_state=None):
        self.n_inputs = n_feat
        self.n_reservoir = n_reservoir
        self.leaking_rate = leaking_rate
        self.random_state = np.random.default_rng(random_state)
        self.state = np.zeros(self.n_reservoir)

    def reset_state(self):
        self.state = np.zeros(self.n_reservoir)

    def step(self, u):
        raise NotImplementedError

    def run(self, X):
        T = X.shape[0]
        states = np.zeros((T, self.n_reservoir), dtype=float)
        self.reset_state()
        for t in range(T):
            states[t] = self.step(X[t])
        return states


# -------------------------
# QuantumReservoir (Option A)
# -------------------------
class QuantumReservoir(ReservoirBase):

    def __init__(self, n_inputs, n_reservoir=8, read_in_arch=None,
        anzats_arch="default", leaking_rate=1.0, seed=None,):
        
        super().__init__(n_inputs, n_reservoir=n_reservoir, leaking_rate=leaking_rate, random_state=seed)
        read_in_arch = {} if read_in_arch is None else read_in_arch
        # We will form a combined input vector made of [external_input, previous_state]
        # Choose combined feature size:
        self.n_external = n_inputs
        self.n_reservoir = n_reservoir
        self.n_feat = self.n_external + self.n_reservoir

        self.read_in = ReadIn(self.n_feat, {"two_features_per_qbit": read_in_arch.get("two_features_per_qbit", True)})
        self.n_qbits = self.read_in.n_qbits
        self.anzats = Anzats(self.n_qbits, anzats_arch)
        self.n_weights = self.anzats.n_weights

        self.dev = qml.device("default.qubit", wires=self.n_qbits)
        measure_wires = list(range(self.n_qbits))
        @qml.qnode(self.dev, interface="autograd")
        def _circuit(inputs, weights):
            self.read_in.circuit(inputs)
            self.anzats.circuit(weights)
            return [qml.expval(qml.PauliZ(i)) for i in measure_wires]

        self._circuit = _circuit

        # Initialize fixed random quantum weights (NOT trainable)
        rng = np.random.default_rng(seed)
        self.quantum_weights = rng.standard_normal(self.n_weights).astype(float) * 0.5

        # Initialize previous state (classical expectations)
        self.state = np.zeros(self.n_reservoir, dtype=float)

    def step(self, u):
        """
        u: array of length n_external
        Returns new classical reservoir state (n_reservoir)
        """
        # prepare combined input: [u, prev_state]
        prev = self.state
        combined = np.zeros(self.n_feat, dtype=float)
        # Insert external input first
        combined[: self.n_external] = u
        # Insert previous state next
        combined[self.n_external : self.n_external + len(prev)] = prev

        q_out = np.array(self._circuit(combined, self.quantum_weights), dtype=float)

        if q_out.size == self.n_reservoir:
            meas = q_out
        elif q_out.size > self.n_reservoir:
            meas = q_out[: self.n_reservoir]
        else:
            # tile to produce required length
            reps = int(np.ceil(self.n_reservoir / q_out.size))
            meas = np.tile(q_out, reps)[: self.n_reservoir]

        # Apply leaking_rate mixing (loptionally)
        new_state = (1 - self.leaking_rate) * self.state + self.leaking_rate * meas

        self.state = new_state
        return self.state




def create_data(T=500, input_dim=1, window_size=10, max_delay=5,
    alpha_range=(0.6, 0.95), noise_std=0.05, seed=42):

    rng = np.random.default_rng(seed)

    u = np.zeros((T, input_dim))
    phase = rng.uniform(0, 2 * np.pi, size=input_dim)
    freq = rng.uniform(0.05, 0.5, size=input_dim)

    for t in range(T):
        freq += 0.01 * rng.normal(size=input_dim)

        u[t] = (
            np.sin(freq * t + phase)
            + 0.3 * np.sin(3 * freq * t)
            + 0.2 * rng.normal(size=input_dim)
        )

        if rng.random() < 0.02:
            u[t] += rng.normal(scale=2.0, size=input_dim)

    u /= np.std(u, axis=0, keepdims=True)

    y = np.zeros(T)
    alpha = rng.uniform(*alpha_range)
    w_in = rng.normal(size=input_dim)
    w_delay = rng.normal(size=max_delay)

    for t in range(max_delay, T):
        d = rng.integers(1, max_delay + 1)

        in_term = np.tanh(u[t] @ w_in)
        fb_term = np.sin(y[t - d]) * w_delay[d - 1]
        mult_noise = (1 + 0.3 * y[t - 1]) * rng.normal(scale=noise_std)

        y[t] = alpha * y[t - 1] + in_term + fb_term + mult_noise

        if rng.random() < 0.01:
            alpha = rng.uniform(*alpha_range)

    Xw, yw = [], []
    for t in range(T - window_size - 1):
        window = u[t:t + window_size].reshape(-1)
        Xw.append(window)
        yw.append(y[t + window_size])

    Xw = np.array(Xw, dtype=np.float32)
    yw = np.array(yw, dtype=np.float32)[:, None]

    return Xw, yw

X, y = create_data(T=400, window_size=12, input_dim=1)

#%%
split = int(0.7 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

qr = QuantumReservoir(n_inputs=12, n_reservoir=15, read_in_arch={"two_features_per_qbit": True}, anzats_arch="default", leaking_rate=0.7, seed=42)

states_train = qr.run(X_train)   
states_test = qr.run(X_test)     

ro = Readout(ridge=1e-6)
ro.fit(states_train, X_train, y_train)

y_pred_train = ro.predict(states_train, X_train)
y_pred_test = ro.predict(states_test, X_test)

def mse(a, b): return np.mean((a - b) ** 2)
print("Train MSE:", mse(y_pred_train, y_train))
print("Test MSE:", mse(y_pred_test, y_test))

import matplotlib.pyplot as plt
plt.plot(y_test[:100], label="Actual")
plt.plot(y_pred_test[:100], label="Train predictions")
plt.legend()
plt.title("QRC: True vs Predicted Values")
plt.show()
