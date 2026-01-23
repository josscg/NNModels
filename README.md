# Classical and Quantum Machine Learning Frameworks

This repository contains a modular framework for **classical and quantum machine learning**, designed for **deep learning, time-series forecasting (and extendable to classification tasks)**.  
The project integrates **classical neural architectures** and **quantum neural models** enhanced with classical pre- and post-processing.

The codebase emphasizes **modularity, extensibility, and reproducibility**.

## Key Features

- Unified framework for **classical and quantum machine learning**
- Modular architecture for easy model swapping and experimentation
- Support for:
  - Deep learning
  - Time-series forecasting


## Repository Structure
The framework is organized around interchangeable model components, allowing the same
data pipelines and evaluation metrics to be used for both classical and quantum models.


### Classical Machine Learning Models

Implemented using **TensorFlow** and custom Python modules:

- Feedforward Neural Networks (NN)
- Recurrent Neural Networks (RNN)
- Reservoir Computing
  - Echo State Networks
  - Custom reservoir implementations

These models are used as performance baselines and for direct comparison with quantum-enhanced approaches.

### Quantum Machine Learning Models

Quantum models are implemented using **PennyLane** with TensorFlow integration:

- Quantum Neural Networks (QNN)
- Quantum Recurrent Neural Networks (QRNN)
- Quantum Reservoir Computing

Quantum circuits are parameterized and trained using hybrid
classical–quantum optimization loops.  
Both simulated quantum backends and differentiable quantum nodes are supported.


