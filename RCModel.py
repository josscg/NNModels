# # -*- coding: utf-8 -*-
# """
# Created on Mon Oct 13 14:48:23 2025

# @author: lm10_
# """

# # https://www.nature.com/articles/s41467-021-25801-2
# # https://arxiv.org/html/2412.13212v1#Ch0.S5

# import numpy as np

# class Reservoir:
#     """
#     Basic Echo State Network (Reservoir) module.
#     Internal weights and states are fixed after initialization.
#     """
#     def __init__(self, n_inputs, n_reservoir=100, spectral_radius=0.9, 
#                  density=0.1, leaking_rate=1.0, random_state=None):
#         rng = np.random.default_rng(random_state)
        
#         self.n_inputs = n_inputs
#         self.n_reservoir = n_reservoir
#         self.spectral_radius = spectral_radius
#         self.density = density
#         self.leaking_rate = leaking_rate

#         # Input weights (random)
#         self.W_in = rng.uniform(-1, 1, (n_reservoir, n_inputs))

#         # Recurrent weights (sparse random)
#         W = rng.uniform(-1, 1, (n_reservoir, n_reservoir))
#         mask = rng.random((n_reservoir, n_reservoir)) < density
#         W *= mask

#         # Normalize to desired spectral radius
#         eigvals = np.linalg.eigvals(W)
#         W /= np.max(np.abs(eigvals))
#         self.W = W * spectral_radius

#         # Initialize reservoir state
#         self.state = np.zeros(self.n_reservoir)

#     def reset_state(self):
#         self.state = np.zeros(self.n_reservoir)

#     def step(self, u):
#         """
#         One update step of the reservoir.
#         u: input vector at current timestep (shape: n_inputs,)
#         """
#         pre_activation = np.dot(self.W_in, u) + np.dot(self.W, self.state)
#         new_state = (1 - self.leaking_rate) * self.state + \
#                     self.leaking_rate * np.tanh(pre_activation)
#         self.state = new_state
#         return self.state


#     def run(self, X):
#         """
#         Run the reservoir over a full sequence.
#         X: input sequence of shape (timesteps, n_inputs)
#         Returns: states of shape (timesteps, n_reservoir)
#         """
#         self.reset_state()
#         states = np.zeros((X.shape[0], self.n_reservoir))
#         for t in range(X.shape[0]):
#             states[t] = self.step(X[t])
#         return states
    


# class Readout:
#     """
#     Linear readout: y = W_out * [state; input; 1]
#     Trained via ridge regression (L2 regularization).
#     """
#     def __init__(self, ridge=1e-6):
#         self.ridge = ridge
#         self.W_out = None

#     def fit(self, states, inputs, targets):
#         # Concatenate states, inputs, and bias
#         X_ext = np.hstack([states, inputs, np.ones((states.shape[0], 1))])
#         # Closed-form ridge regression solution
#         I = np.eye(X_ext.shape[1])
#         self.W_out = np.linalg.solve(
#             X_ext.T @ X_ext + self.ridge * I,
#             X_ext.T @ targets)

#     def predict(self, states, inputs):
#         X_ext = np.hstack([states, inputs, np.ones((states.shape[0], 1))])
#         return X_ext @ self.W_out


# class ReservoirModel:
#     """
#     Full Reservoir Computing model (ESN-style).
#     Combines a Reservoir + Linear Readout.
#     """
#     def __init__(self, n_inputs, n_reservoir=100, n_outputs=1,
#                  spectral_radius=0.9, density=0.1, leaking_rate=1.0,
#                  ridge=1e-6, random_state=None):
        
#         self.reservoir = Reservoir(
#             n_inputs=n_inputs,
#             n_reservoir=n_reservoir,
#             spectral_radius=spectral_radius,
#             density=density,
#             leaking_rate=leaking_rate,
#             random_state=random_state)
        
#         self.readout = Readout(ridge=ridge)
#         self.n_outputs = n_outputs

#     def fit(self, X, y, washout=0):
#         """
#         Train the readout using sequences (X -> Y).
#         X: shape (n_samples, timesteps, n_inputs)
#         Y: shape (n_samples, timesteps, n_outputs)
#         """

#         last_states, last_inputs, targets = [], [], []
#         for seq, target in zip(X, y):
#             states = self.reservoir.run(seq)
#             last_states.append(states[-1])
#             last_inputs.append(seq[-1])
#             targets.append(target)
        
        
#         self.readout.fit(np.array(last_states), np.array(last_inputs), 
#                          np.array(targets))
        

#     def predict(self, X):
#         states = np.array([self.reservoir.run(seq)[-1] for seq in X])
#         inputs = np.array([seq[-1] for seq in X])
#         return self.readout.predict(states, inputs)
        






"""
Optimized, modular Reservoir Computing implementation (NumPy).
- Multiple reservoir models: ESN (default), Stochastic, Continuous-time ESN
- Power-iteration spectral radius normalization (faster, stable)
- Optional Numba-accelerated time stepping (if numba installed)
- Readout: closed-form ridge or sklearn.Ridge backend
- Keep API compatible with your previous ReservoirModel.fit/predict
"""

import numpy as np

# -------------------------
# Utility: power iteration for spectral radius 
# -------------------------
def spectral_radius_power_iteration(A, num_iters=100):
    """
    Estimate spectral radius (largest absolute eigenvalue) using power iteration.
    Faster and more stable for large sparse matrices than full eigvals.
    """
    n = A.shape[0]
    v = np.random.randn(n)
    v /= np.linalg.norm(v) + 1e-12
    for _ in range(num_iters):
        v = A @ v
        norm = np.linalg.norm(v)
        if norm == 0:
            return 0.0
        v /= norm
    # Rayleigh quotient approx
    Av = A @ v
    return float(np.dot(v, Av))

# -------------------------
# ReservoirBase and models
# -------------------------
class ReservoirBase:
    """
    Base reservoir interface. Specific dynamics implemented in subclasses.
    """

    def __init__(self, n_inputs, n_reservoir=100, spectral_radius=0.9, density=0.1,
                  leaking_rate=1.0, random_state=None, init_scale=1.0):
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.density = density
        self.leaking_rate = leaking_rate
        self.random_state = np.random.default_rng(random_state)

        # Initialize Win shape (n_reservoir, n_inputs)
        self.W_in = self.random_state.uniform(-init_scale, init_scale, (n_reservoir, n_inputs)) 

        # Initialize sparse recurrent matrix W (n_reservoir, n_reservoir)
        W = self.random_state.uniform(-init_scale, init_scale, (n_reservoir, n_reservoir))
        mask = self.random_state.random((n_reservoir, n_reservoir)) < density
        W *= mask

        # ---- Optimization: compute spectral radius with power iteration (faster than eigvals for big N) ----
        max_ev = spectral_radius_power_iteration(W, num_iters=200)
        if max_ev == 0:
            # fallback safe small random perturbation
            max_ev = 1.0
        W = W/max_ev * spectral_radius
        
        self.W = W
        self.state = np.zeros(self.n_reservoir)


    def reset_state(self):
        self.state = np.zeros(self.n_reservoir)

    # Subclasses must implement step() and optionally run_fast()
    def step(self, u):
        """
        One update step given an input vector `u`.
        This method MUST be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must override step().")

    def run(self, X):
        """
        Run the reservoir on a single sequence X of shape (timesteps, n_inputs)
        Returns states: (timesteps, n_reservoir)
        """
        T = X.shape[0]
        states = np.zeros((T, self.n_reservoir), dtype=float)
        self.reset_state()
        for t in range(T):
            states[t] = self.step(X[t])
        return states



# -------------------------
# ESN (standard) model 
# -------------------------
class ESNReservoir(ReservoirBase):
    """
    Standard Echo State Network reservoir.
    Discrete-time update:
        x(t+1) = (1 - α) * x(t) + α * tanh(W_in * u + W * x)
    """
    
    def __init__(self, *args, activation=np.tanh, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation = activation


    def step(self, u):
        # u shape (n_inputs,)
        pre_activation = self.W_in @ u + self.W @ self.state
        new_state = (1 - self.leaking_rate) * self.state + self.leaking_rate * np.tanh(pre_activation)
        self.state = new_state
        return self.state

# -------------------------
# Stochastic reservoir: ESN + additive noise each step
# -------------------------
class StochasticReservoir(ReservoirBase):   
    """
    ESN-like reservoir but with additive Gaussian noise on the state.
    Adds stochasticity to the internal dynamics.
    """
    
    def __init__(self, noise_std=0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_std = noise_std


    def step(self, u):
        pre_activation = self.W_in @ u + self.W @ self.state
        noise = self.random_state.normal(0.0, self.noise_std, size=self.n_reservoir)
        new_state = (1 - self.leaking_rate) * self.state + self.leaking_rate * np.tanh(pre_activation + noise)
        self.state = new_state
        return self.state

# -------------------------
# Continuous-time ESN: Euler discretization of an ODE
# x_dot = -gamma*x + f(Wx + Win u)
# step uses dt as 1.0 by default; leaking_rate used as gamma
# -------------------------
class ContinuousTimeESN(ReservoirBase):  
    """
    Continuous-time reservoir (Euler integration of ODE-like dynamics):
        dx/dt = -x + tanh(W_in * u + W * x)
    """
    
    def __init__(self, dt=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dt = dt  # integration step
       
    def step(self, u):
        # simple Euler integration for continuous dynamics
        pre_activation = self.W_in @ u + self.W @ self.state
        x_dot = -self.leaking_rate * self.state + np.tanh(pre_activation)
        new_state = self.state + self.dt * x_dot
        self.state = new_state
        return self.state

# -------------------------
# Readout: support closed-form ridge AND sklearn.Ridge backend
# -------------------------
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
        """
        states: (N_samples, n_reservoir)
        inputs: (N_samples, n_inputs)
        targets: (N_samples, n_outputs)
        """
        X_ext = np.hstack([states, inputs, np.ones((states.shape[0], 1))])
        if self.use_sklearn:
            # sklearn expects shape (n_samples, n_features)
            model = self._Ridge(alpha=self.ridge, fit_intercept=False)
            model.fit(X_ext, targets)
            # sklearn stores coef_ shape (n_targets, n_features), transpose to (n_features, n_targets)
            self.W_out = model.coef_.T
        else:
            # stable linear solve
            I = np.eye(X_ext.shape[1], dtype=float)
            A = X_ext.T @ X_ext + self.ridge * I
            B = X_ext.T @ targets
            self.W_out = np.linalg.solve(A, B)  # shape (n_features, n_outputs)


    def predict(self, states, inputs):
        X_ext = np.hstack([states, inputs, np.ones((states.shape[0], 1))])
        return X_ext @ self.W_out





# -------------------------
# Model orchestration 
# -------------------------
class ReservoirModel:
    def __init__(self, reservoir_type, n_inputs, n_outputs=1, n_reservoir=100,
                  spectral_radius=0.9, density=0.1, leaking_rate=1.0,
                  ridge=1e-6, random_state=None,
                  use_sklearn_readout=False, reservoir_kwargs=None):
        
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.ridge = ridge
        self.random_state = random_state
        self.use_sklearn_readout = use_sklearn_readout

        reservoir_kwargs = reservoir_kwargs or {}
        # instantiate chosen reservoir model
        if reservoir_type.lower() in ("esn", "standard", "default"):
            self.reservoir = ESNReservoir(n_inputs=n_inputs, n_reservoir=n_reservoir,
                                          spectral_radius=spectral_radius, density=density,
                                          leaking_rate=leaking_rate, random_state=random_state,
                                          **reservoir_kwargs)
        elif reservoir_type.lower() in ("stochastic", "stoch"):
            self.reservoir = StochasticReservoir(n_inputs=n_inputs, n_reservoir=n_reservoir,
                                                  spectral_radius=spectral_radius, density=density,
                                                  leaking_rate=leaking_rate, random_state=random_state,
                                                  **reservoir_kwargs)
        elif reservoir_type.lower() in ("ctesn", "continuous", "continuousesn"):
            self.reservoir = ContinuousTimeESN(n_inputs=n_inputs, n_reservoir=n_reservoir,
                                                spectral_radius=spectral_radius, density=density,
                                                leaking_rate=leaking_rate, random_state=random_state,
                                                **reservoir_kwargs)
        else:
            raise ValueError("Unknown reservoir_type: " + str(reservoir_type))

        self.readout = Readout(ridge=ridge, use_sklearn=use_sklearn_readout)

    def fit(self, X, y, washout=0, summary_fn="last"):
        """
        Train the readout for sequence->one task.
        X: (n_samples, timesteps, n_inputs)
        Y: (n_samples, n_outputs)  # one output per sequence
        - washout: number of initial timesteps to discard
        - summary_fn: how to map post-washout states to a single vector per sequence:
            'last' (default) -> use last state
            'mean' -> mean of post-washout states
            'max' -> max pooling
        """
        n_samples = X.shape[0]
        all_repr = np.zeros((n_samples, self.n_reservoir), dtype=float)

        for i in range(n_samples):
            seq = X[i]
            states = self.reservoir.run(seq)
            states = states[washout:] if washout > 0 else states
            if states.shape[0] == 0:
                # nothing left after washout -> fallback
                repr_vec = np.zeros(self.n_reservoir)
            else:
                if summary_fn == "last":
                    repr_vec = states[-1]
                elif summary_fn == "mean":
                    repr_vec = states.mean(axis=0)
                elif summary_fn == "max":
                    repr_vec = states.max(axis=0)
                else:
                    raise ValueError("Unknown summary_fn")
            all_repr[i] = repr_vec

        # optionally include last input as extra features (here we will include it)
        last_inputs = X[:, -1, :]  # (n_samples, n_inputs)
        self.readout.fit(all_repr, last_inputs, y)
        
        
        
    def old_fit(self, X, y, washout=0):
        """
        Train the readout using sequences (X -> Y).
        X: shape (n_samples, timesteps, n_inputs)
        Y: shape (n_samples, timesteps, n_outputs)
        """
        last_states, last_inputs, targets = [], [], []
        for seq, target in zip(X, y):
            states = self.reservoir.run(seq)
            last_states.append(states[-1])
            last_inputs.append(seq[-1])
            targets.append(target)
        
        
        self.readout.fit(np.array(last_states), np.array(last_inputs), 
                             np.array(targets))
            
          
    def predict(self, X, return_sequences=False):
        # The old predict method would work
        """
        For sequence -> we return shape (n_samples, n_outputs).
        If return_sequences=True, we return full sequence predictions.
        """
        n_samples = X.shape[0]
        preds = np.zeros((n_samples, self.n_outputs))
        for i in range(n_samples):
            seq = X[i]
            states = self.reservoir.run(seq)
            y_seq = self.readout.predict(states, np.broadcast_to(seq, (states.shape[0], seq.shape[1])))
            # y_seq shape: (T, n_outputs)
            preds[i] = y_seq[-1]  # last output
        return preds








# # reservoir_tf.py
# import tensorflow as tf
# import numpy as np

# # ---------------------------
# # Reservoir layer (TF)
# # ---------------------------
# class ReservoirLayer(tf.keras.layers.Layer):
#     """
#     TensorFlow Reservoir layer (non-trainable internal weights).
#     - Inputs: (batch, timesteps, n_inputs)
#     - Output: full states (batch, timesteps, n_reservoir)
#     Use add_weight(..., trainable=False) so weights are accessible in graph mode.
#     """

#     def __init__(self,
#                  n_reservoir,
#                  spectral_radius=0.9,
#                  input_scaling=1.0,
#                  leaking_rate=1.0,
#                  density=0.1,
#                  seed=None,
#                  return_sequences=True,
#                  **kwargs):
#         super().__init__(**kwargs)
#         self.n_reservoir = int(n_reservoir)
#         self.spectral_radius = float(spectral_radius)
#         self.input_scaling = float(input_scaling)
#         self.leaking_rate = float(leaking_rate)
#         self.density = float(density)
#         self.seed = seed
#         self.return_sequences = return_sequences

#     def build(self, input_shape):
#         """
#         input_shape: (batch, timesteps, n_inputs)
#         Build is called once and gives access to n_inputs -> we can create Win accordingly.
#         """
#         n_inputs = int(input_shape[-1])
#         rng = np.random.default_rng(self.seed)

#         # Create numpy initializers (so we can control sparsity and spectral normalization)
#         Win_np = rng.uniform(-1.0, 1.0, size=(self.n_reservoir, n_inputs)) * self.input_scaling

#         W_np = rng.uniform(-1.0, 1.0, size=(self.n_reservoir, self.n_reservoir))
#         mask = rng.random((self.n_reservoir, self.n_reservoir)) < self.density
#         W_np *= mask

#         # spectral radius normalization (numpy)
#         eigs = np.linalg.eigvals(W_np)
#         max_ev = np.max(np.abs(eigs))
#         if max_ev == 0:
#             max_ev = 1.0
#         W_np = (W_np / max_ev) * self.spectral_radius

#         bias_np = rng.uniform(-0.1, 0.1, size=(self.n_reservoir,))

#         # Register as layer "weights" (IMPORTANT: trainable=False => not updated by optimizers)
#         # Using add_weight ensures TF graph scopes can access the tensors.
#         self.Win = self.add_weight(
#             name="Win",
#             shape=Win_np.shape,
#             dtype=tf.float32,
#             initializer=tf.constant_initializer(Win_np),
#             trainable=False
#         )
#         self.W_res = self.add_weight(
#             name="W_res",
#             shape=W_np.shape,
#             dtype=tf.float32,
#             initializer=tf.constant_initializer(W_np),
#             trainable=False
#         )
#         self.bias = self.add_weight(
#             name="bias",
#             shape=bias_np.shape,
#             dtype=tf.float32,
#             initializer=tf.constant_initializer(bias_np),
#             trainable=False
#         )

#         super().build(input_shape)

#     def call(self, inputs, training=False):
#         """
#         Forward pass.
#         inputs: (batch, timesteps, n_inputs)
#         returns: (batch, timesteps, n_reservoir)  OR (batch, n_reservoir) if return_sequences=False
#         """
#         # 1) ensure dtype consistency (TF is strict about dtypes in matmuls)
#         inputs = tf.cast(inputs, tf.float32)

#         # 2) prepare time-major elems for tf.scan: shape -> (timesteps, batch, n_inputs)
#         inputs_time_major = tf.transpose(inputs, perm=[1, 0, 2])

#         batch_size = tf.shape(inputs)[0]
#         # initial state: zeros (batch, n_reservoir)
#         x0 = tf.zeros((batch_size, self.n_reservoir), dtype=tf.float32)

#         # step function for tf.scan
#         def step(prev_state, u_t):
#             # prev_state: (batch, n_reservoir)
#             # u_t: (batch, n_inputs)
#             # preact = prev_state @ W_res^T + u_t @ Win^T + bias
#             preact_rec = tf.matmul(prev_state, self.W_res, transpose_b=True)    # (batch, n_reservoir)
#             preact_in = tf.matmul(u_t, self.Win, transpose_b=True)              # (batch, n_reservoir)
#             preact = preact_rec + preact_in + self.bias                        # broadcasting bias across batch
#             new_state = (1.0 - self.leaking_rate) * prev_state + self.leaking_rate * tf.tanh(preact)
#             return new_state

#         # tf.scan iterates over the first dimension of elems (timesteps)
#         states_time_major = tf.scan(fn=step, elems=inputs_time_major, initializer=x0)
#         # states_time_major: (timesteps, batch, n_reservoir)
#         states = tf.transpose(states_time_major, perm=[1, 0, 2])  # -> (batch, timesteps, n_reservoir)

#         if self.return_sequences:
#             return states
#         else:
#             # return last state only (batch, n_reservoir)
#             return states[:, -1, :]

# # ---------------------------
# # ReservoirModel (Keras Model wrapper)
# # ---------------------------
# class ReservoirModel(tf.keras.Model):
#     """
#     Keras-style wrapper that contains a ReservoirLayer and handles closed-form readout training in TF.
#     Usage:
#         model = ReservoirModel(n_inputs=..., n_reservoir=..., n_outputs=..., ...)
#         model.train_readout(X_train, Y_train, summary="last", ridge=1e-6)
#         Y_pred = model.predict(X_test)  # or call(X_test)
#     """

#     def __init__(self,
#                  n_inputs,
#                  n_reservoir=100,
#                  n_outputs=1,
#                  spectral_radius=0.9,
#                  input_scaling=1.0,
#                  leaking_rate=1.0,
#                  density=0.1,
#                  seed=None,
#                  reservoir_return_sequences=True,
#                  **kwargs):
#         super().__init__(**kwargs)
#         self.reservoir = ReservoirLayer(
#             n_reservoir=n_reservoir,
#             spectral_radius=spectral_radius,
#             input_scaling=input_scaling,
#             leaking_rate=leaking_rate,
#             density=density,
#             seed=seed,
#             return_sequences=reservoir_return_sequences
#         )
#         self.n_outputs = n_outputs
#         self.W_out = None  # will be a tf.Tensor or tf.Variable after training

#     @tf.function(input_signature=[tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
#                                   tf.TensorSpec(shape=(None, None), dtype=tf.float32)])  # optional
#     def _reservoir_states(self, X, dummy):
#         # Helper to run reservoir inside graph; dummy is unused but gives signature
#         return self.reservoir(X, training=False)

#     def train_readout(self, X, Y, summary="last", include_last_input=True, ridge=1e-6):
#         """
#         Train the linear readout using closed-form ridge regression.
#         X: np.ndarray or tf.Tensor shape (batch, timesteps, n_inputs)
#         Y: np.ndarray or tf.Tensor shape (batch, n_outputs)  # sequence->one mode
#         summary: 'last'|'mean'|'max' -> how to reduce reservoir time axis to single vector
#         include_last_input: whether to append last input as features
#         ridge: regularization (lambda)
#         Stores W_out as tf.Variable(trainable=False)
#         """

#         # Ensure TF tensors, float32
#         X = tf.convert_to_tensor(X, dtype=tf.float32)
#         Y = tf.convert_to_tensor(Y, dtype=tf.float32)

#         # 1) run reservoir -> states shape (batch, timesteps, n_reservoir)
#         states = self.reservoir(X, training=False)
#         # If reservoir layer returns (batch, n_reservoir) for return_sequences=False,
#         # wrap to (batch, 1, n_reservoir) for uniformity
#         if len(states.shape) == 2:
#             states = tf.expand_dims(states, axis=1)

#         # 2) summarize along time axis to produce one vector per sequence
#         if summary == "last":
#             repr_vec = states[:, -1, :]                         # (batch, n_reservoir)
#         elif summary == "mean":
#             repr_vec = tf.reduce_mean(states, axis=1)
#         elif summary == "max":
#             repr_vec = tf.reduce_max(states, axis=1)
#         else:
#             raise ValueError("Unknown summary: " + str(summary))

#         # 3) optionally append last input
#         if include_last_input:
#             last_input = X[:, -1, :]                            # (batch, n_inputs)
#             features = tf.concat([repr_vec, last_input], axis=1)  # (batch, n_reservoir + n_inputs)
#         else:
#             features = repr_vec

#         # 4) add bias column
#         ones = tf.ones((tf.shape(features)[0], 1), dtype=tf.float32)
#         Z = tf.concat([features, ones], axis=1)                # (batch, F)

#         # 5) closed-form ridge: W = (Z^T Z + lambda I)^{-1} Z^T Y
#         #    Use tf.linalg.solve for numerical stability
#         F = tf.shape(Z)[1]
#         lambda_reg = tf.cast(ridge, tf.float32)
#         I = tf.eye(F, dtype=tf.float32)
#         # compute (F,F) matrix and (F,outputs) rhs
#         ZTZ = tf.matmul(Z, Z, transpose_a=True)                # (F, F)
#         A = ZTZ + lambda_reg * I
#         ZTY = tf.matmul(Z, Y, transpose_a=True)                # (F, outputs)

#         W_out = tf.linalg.solve(A, ZTY)                        # (F, outputs)
#         # store as non-trainable variable so it travels with the model
#         if self.W_out is None:
#             self.W_out = tf.Variable(W_out, trainable=False, dtype=tf.float32, name="W_out")
#         else:
#             # assign new value
#             self.W_out.assign(W_out)

#         # Return W_out for convenience
#         return self.W_out

#     def call(self, X, training=False, summary="last", include_last_input=True):
#         """
#         Run reservoir (driven) and apply trained readout.
#         Returns:
#             - sequence-to-one: (batch, n_outputs)
#             - If reservoir.return_sequences=True and you want full-sequence readout,
#               you could modify to return (batch, timesteps, n_outputs). Here we use last by default.
#         """
#         X = tf.cast(X, tf.float32)
#         states = self.reservoir(X, training=training)        # (batch, timesteps, n_reservoir) or (batch, n_reservoir)

#         # ensure time dimension exists
#         if len(states.shape) == 2:
#             states = tf.expand_dims(states, axis=1)         # (batch, 1, n_reservoir)

#         if summary == "last":
#             repr_vec = states[:, -1, :]
#         elif summary == "mean":
#             repr_vec = tf.reduce_mean(states, axis=1)
#         elif summary == "max":
#             repr_vec = tf.reduce_max(states, axis=1)
#         else:
#             raise ValueError("Unknown summary: " + str(summary))

#         if include_last_input:
#             last_input = X[:, -1, :]
#             features = tf.concat([repr_vec, last_input], axis=1)
#         else:
#             features = repr_vec

#         ones = tf.ones((tf.shape(features)[0], 1), dtype=tf.float32)
#         Z = tf.concat([features, ones], axis=1)   # (batch, F)

#         if self.W_out is None:
#             raise ValueError("Readout not trained. Call train_readout(...) first.")

#         Y_pred = tf.matmul(Z, self.W_out)         # (batch, n_outputs)
#         return Y_pred

#     def predict(self, X, summary="last", include_last_input=True):
#         # wrapper for numpy-like predict: accepts numpy arrays as input
#         X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
#         Y_tf = self.call(X_tf, training=False, summary=summary, include_last_input=include_last_input)
#         return Y_tf.numpy()

# # ---------------------------
# # Example usage
# # ---------------------------
# if __name__ == "__main__":
#     # toy data (sequence->one)
#     n_samples, T, n_inputs = 128, 20, 3
#     X = np.random.randn(n_samples, T, n_inputs).astype(np.float32)
#     # target: sum of last two timesteps features (toy)
#     Y = np.sum(X[:, -2:, :], axis=(1, 2), keepdims=True).astype(np.float32)  # shape (n_samples, 1)

#     model = ReservoirModel(n_inputs=n_inputs, n_reservoir=200, n_outputs=1,
#                            spectral_radius=0.95, leaking_rate=0.8, density=0.1, seed=42,
#                            reservoir_return_sequences=True)
#     # Train readout (closed-form ridge)
#     model.train_readout(X, Y, summary="last", include_last_input=True, ridge=1e-6)

#     # Predict
#     Y_pred = model.predict(X)
#     print("Y shape:", Y.shape, "Y_pred shape:", Y_pred.shape)
#     print("MSE:", np.mean((Y[:, 0] - Y_pred[:, 0])**2))
