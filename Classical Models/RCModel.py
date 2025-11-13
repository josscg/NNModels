# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 14:48:23 2025

@author: lm10_
"""

# BIBLIOGRAPHY
# https://www.nature.com/articles/s41467-021-25801-2
# https://arxiv.org/html/2412.13212v1#Ch0.S5


"""
Modular Reservoir Computing implementation (NumPy).
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

        # OPTIONAL
        # Compute spectral radius with power iteration (faster than eigvals for big N)
        max_ev = spectral_radius_power_iteration(W, num_iters=200)
        if max_ev == 0:
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
            # Standard scikit-learn implementation
            model = self._Ridge(alpha=self.ridge, fit_intercept=False)
            model.fit(X_ext, targets)
            self.W_out = model.coef_.T
        else:
            # Analytical Solution
            I = np.eye(X_ext.shape[1], dtype=float)
            A = X_ext.T @ X_ext + self.ridge * I
            B = X_ext.T @ targets
            self.W_out = np.linalg.solve(A, B) 


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
                # If nothing left after washout, we do a fallback
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
        last_inputs = X[:, -1, :]
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
            preds[i] = y_seq[-1]  # last output as default return 
        return preds



