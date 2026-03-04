# NumPy-DL: A Modular Deep Learning Library from Scratch

**NumPy-DL-Library** is a modular deep learning library built entirely from scratch using NumPy. It includes every feature required to train and test using a three layer CNN (which will soon be updated to support arbitrary depth CNN and more).

The functionalities were originally written as part of the assignments of [cs231n](https://cs231n.stanford.edu/).

## Core Capabilities

- **Modular Layers (`dl_core/layers.py`):** Includes Fully-Connected (Affine), ReLU, Convolutional, Max Pooling, Dropout, Batch Normalization, and Layer Normalization. Each layer provides rigorous forward and backward passes.
- **Optimizers (`dl_core/optim.py`):** First-order update rules including SGD, SGD with Momentum, RMSProp, and Adam.
- **Architectures (`dl_core/classifiers/`):** Pre-assembled architectures such as `FullyConnectedNet` and `ThreeLayerConvNet` demonstrating the composition of the modular layers.
- **Solver (`dl_core/solver.py`):** A flexible training engine separate from the model architectures, handling the optimization loop, learning rate decay, validation checks, and checkpointing.

## Repository Structure

```
├── dl_core/          # Core neural network modules (layers, optimizers, solver, architectures)
├── utils/            # Utilities for data loading, image processing, and gradient checking
├── examples/         # Introductory tutorials and example usage of the library
├── scripts/          # Build scripts and Cython extensions for fast convolutions (im2col)
├── datasets/         # Directory for standard datasets (e.g., CIFAR-10)
└── README.md
```

## Setup and Installation

1. Clone the repository and navigate to the project root.
2. (Optional but Recommended) Compile the Cython extensions in `/scripts` for significantly faster convolutional operations:
   ```bash
   cd scripts
   python setup.py build_ext --inplace
   ```

## Usage Example

```python
from dl_core.classifiers.fc_net import FullyConnectedNet
from dl_core.solver import Solver
from utils.data_utils import get_CIFAR10_data

# Load data
data = get_CIFAR10_data()

# Initialize model
model = FullyConnectedNet([100, 100], input_dim=3*32*32, num_classes=10,
                          normalization='batchnorm', reg=1e-2)

# Train the model
solver = Solver(model, data,
                update_rule='adam',
                optim_config={'learning_rate': 1e-3},
                lr_decay=0.95,
                num_epochs=10, batch_size=100)
solver.train()
```
