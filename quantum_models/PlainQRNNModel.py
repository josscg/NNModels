"""Plain quantum recurrent network using PennyLane and TensorFlow.

'paper' reproduces the displayed Figure 6 gate layout; 'paper_readout'
adds post-entangler RY rotations so hidden-state information can affect Z readout.
"""
import numpy as np
import pennylane as qml
import tensorflow as tf


class ReadIn:
    """Encode one scalar per step in the data register."""
    def __init__(self, n_feat, read_in_arch, wires):
        self.n_feat = n_feat
        self.arch = read_in_arch
        self.wires = list(wires)
        self.n_qbits = len(self.wires)
        self.encoding = self.arch.get('encoding', 'paper')
        if self.encoding == 'paper' and n_feat != 1:
            raise ValueError('Paper encoding requires n_feat=1.')

    def circuit(self, inputs):
        if self.encoding == 'paper':
            angle = qml.math.arccos(qml.math.clip(inputs[0], -1., 1.))
            for wire in self.wires:
                qml.RY(angle, wires=wire)
        elif self.encoding == 'angle':
            if self.n_feat != self.n_qbits:
                raise ValueError('Angle encoding requires n_feat == number of data wires.')
            for i, wire in enumerate(self.wires):
                qml.RY(inputs[i], wires=wire)
        else:
            raise ValueError(f'Unknown encoding: {self.encoding}')


class HiddenState:
    """Persistent hidden wires, never reinitialized within a sequence."""
    def __init__(self, hidden_arch, wires):
        self.arch = hidden_arch
        self.wires = list(wires)
        self.n_qbits = len(self.wires)


class Anzats:
    """RX-RZ-RX rotations and ring ZZ entanglers, optionally with readout mixing."""
    def __init__(self, n_qbits, anzats_arch, wires):
        self.n_qbits = n_qbits
        self.arch = anzats_arch
        self.wires = list(wires)
        self.gates = {'RX': qml.RX, 'RY': qml.RY, 'RZ': qml.RZ}
        self.control_gates = {'CNOT': qml.CNOT, 'CZ': qml.CZ}
        if self.arch in ('paper', 'paper_readout'):
            if n_qbits < 2:
                raise ValueError('At least two qubits are required.')
            self.pairs = [(self.wires[i], self.wires[(i + 1) % n_qbits])
                          for i in range(n_qbits)]
            if n_qbits == 2:
                self.pairs = self.pairs[:1]
            self.n_weights = 3 * n_qbits + len(self.pairs)
            if self.arch == 'paper_readout':
                self.n_weights += n_qbits
        elif isinstance(self.arch, list):
            self.n_weights = 0
            for layer in self.arch:
                for gate, targets in layer.items():
                    if gate in self.gates or gate == 'ZZ':
                        self.n_weights += len(targets)
                    elif gate not in self.control_gates:
                        raise ValueError(f'Unknown gate: {gate}')
        else:
            raise ValueError(f'Unknown ansatz: {self.arch}')

    def circuit(self, weights):
        index = 0
        if self.arch in ('paper', 'paper_readout'):
            for wire in self.wires:
                qml.RX(weights[index], wires=wire)
                qml.RZ(weights[index + 1], wires=wire)
                qml.RX(weights[index + 2], wires=wire)
                index += 3
            for a, b in self.pairs:
                qml.CNOT(wires=[a, b])
                qml.RZ(weights[index], wires=b)
                qml.CNOT(wires=[a, b])
                index += 1
            if self.arch == 'paper_readout':
                for wire in self.wires:
                    qml.RY(weights[index], wires=wire)
                    index += 1
        else:
            for layer in self.arch:
                for gate, targets in layer.items():
                    if gate in self.gates:
                        for wire in targets:
                            self.gates[gate](weights[index], wires=wire)
                            index += 1
                    elif gate == 'ZZ':
                        for pair in targets:
                            qml.IsingZZ(weights[index], wires=pair)
                            index += 1
                    else:
                        for pair in targets:
                            self.control_gates[gate](wires=pair)


class QuantumRecurrentBlock:
    """One time step: encode D, entangle D+H, then optionally reset D."""
    def __init__(self, n_feat, quantum_arch):
        self.arch = quantum_arch
        self.n_qbits_data = self.arch.get('data_register', {}).get('units', 3)
        self.n_qbits_hidden = self.arch.get('hidden_state', {}).get('units', 3)
        self.n_qbits = self.n_qbits_data + self.n_qbits_hidden
        self.data_wires = list(range(self.n_qbits_data))
        self.hidden_wires = list(range(self.n_qbits_data, self.n_qbits))
        self.read_in = ReadIn(n_feat, self.arch.get('read_in', {}), self.data_wires)
        self.hidden = HiddenState(self.arch.get('hidden_state', {}), self.hidden_wires)
        self.anzats = Anzats(self.n_qbits, self.arch.get('ansatz', 'paper'),
                             self.data_wires + self.hidden_wires)
        self.n_weights = self.anzats.n_weights

    def circuit(self, inputs, weights):
        self.read_in.circuit(inputs)
        self.anzats.circuit(weights)

    def reset_data(self):
        """
        Reset the data register to |000> using a quantum channel.

        This implements an unconditional reset while
        preserving the reduced density matrix of the
        hidden register.

        Requires a mixed-state simulator.
        """
        for wire in self.data_wires:
            qml.AmplitudeDamping(1.0, wires=wire)


class PlainQRNNModel(tf.keras.Model):
    """Sequence-to-one pQRNN; output shape (batch, 1).

    Input: (batch, n_steps, n_feat), with paper encoding inputs in [-1,1].
    Output: P(D[0]=1) by default, or values rescaled to target_range.
    """
    def __init__(self, n_feat, n_steps, architecture=None, target_range=None):
        super().__init__()
        architecture = architecture or {}
        self.n_feat = n_feat
        self.n_steps = n_steps
        self.arch = architecture
        self.block = QuantumRecurrentBlock(n_feat, architecture.get('quantum', {}))
        self.quantum_weights = self.add_weight(
            name='quantum_weights', shape=(self.block.n_weights,),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            trainable=True, dtype=tf.float64)
        self.target_range = None
        self.set_target_range(target_range)
        self.dev = qml.device('default.mixed', wires=self.block.n_qbits, shots=None)
        self.quantum_circuit = self._build_qnode()

    def set_target_range(self, target_range):
        """Set a fixed inverse-scaling range, fitted on training data only."""
        if target_range is None:
            self.target_range = None
            return
        low, high = map(float, target_range)
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError('target_range must satisfy finite min < max.')
        self.target_range = (low, high)

    def _build_qnode(self):
        """Build the QRNN circuit using density-matrix evolution."""

        @qml.qnode(
            self.dev,
            interface="tf",
            diff_method="best"
        )
        def circuit(sequence, weights):
            for t in range(self.n_steps):
                self.block.circuit(sequence[t], weights)

                if t < self.n_steps - 1:
                    self.block.reset_data()

            return qml.probs(wires=[self.block.data_wires[0]])

        return circuit

    def probabilities(self, inputs):
        """Return unscaled probabilities P(D[0]=1)."""
        inputs = tf.cast(inputs, tf.float64)

        if inputs.shape.rank != 3:
            raise ValueError('Expected (batch, time_steps, n_feat).')
        if inputs.shape[1] is not None and inputs.shape[1] != self.n_steps:
            raise ValueError('Input time_steps does not match n_steps.')
        if inputs.shape[2] is not None and inputs.shape[2] != self.n_feat:
            raise ValueError('Input feature count does not match n_feat.')

        predictions = [
            self.quantum_circuit(sequence, self.quantum_weights)[1]
            for sequence in tf.unstack(inputs, axis=0)
        ]

        return tf.reshape(tf.stack(predictions), (-1, 1))

    def call(self, inputs):
        """Return prediction in the original target scale, if configured."""
        probabilities = self.probabilities(inputs)
        if self.target_range is None:
            return probabilities
        low, high = self.target_range
        return probabilities * (high - low) + low


def make_forecasting_windows(series, n_steps=7, target_range=None):
    """Build day-(t+1) targets from preceding n_steps scalar observations.

    Fit the input range and target range using training data only. Pass the
    same input scaling to validation/test data; do not refit on test data.
    """
    values = np.asarray(series, dtype=np.float64).reshape(-1)
    if len(values) <= n_steps:
        raise ValueError('Series must contain more than n_steps elements.')
    if target_range is None:
        low, high = float(values.min()), float(values.max())
    else:
        low, high = map(float, target_range)
    if high <= low:
        raise ValueError('Scaling range must have max > min.')
    scaled = 2 * (values - low) / (high - low) - 1
    X = np.stack([scaled[i:i + n_steps] for i in range(len(values) - n_steps)])
    y = values[n_steps:]
    return X[..., None], y[:, None], (low, high)