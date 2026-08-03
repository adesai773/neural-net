# neural-net-from-scratch

[![PyPI](https://img.shields.io/pypi/v/neural-net-from-scratch.svg?refresh=1)](https://pypi.org/project/neural-net-from-scratch/)
[![Python](https://img.shields.io/pypi/pyversions/neural-net-from-scratch.svg?refresh=1)](https://pypi.org/project/neural-net-from-scratch/)
[![License](https://img.shields.io/pypi/l/neural-net-from-scratch.svg?refresh=1)](https://github.com/adesai773/neural-net/blob/main/LICENSE)

A tiny neural network library built from scratch in NumPy.

## Why does this exist?

I wanted to know how autograd actually works, not just "PyTorch handles it" but *how*: how the computation graph gets built, how gradients flow backwards, what an optimizer actually does when you call `.step()`. Best way to learn was to build it myself!

Everything here is pure NumPy. No PyTorch, no JAX, no C extensions. If you want a fast production framework, use PyTorch. If you want to read a few hundred lines of Python that spells out what those frameworks do under the hood, this might be useful.

## Installation

Requires Python 3.14+.

### Library use (in a project)

```bash
uv add neural-net-from-scratch
```

Or with the demo example included:

```bash
uv add "neural-net-from-scratch[examples]"
```

### Try the demo (no project needed)

```bash
uvx --from "neural-net-from-scratch[examples]" nn-regression-1d
```

`uvx` grabs the package into an ephemeral environment, runs the demo, and cleans up after itself.

## Quickstart

Fit a small MLP to `y = x² + 1`:

```python
import numpy as np

from neural_net.activation import ReLU
from neural_net.layer import Linear
from neural_net.loss_function import Mse
from neural_net.model import Model
from neural_net.node import Node


class MyModel(Model):
    def __init__(self):
        self.linear1 = Linear(1, 16, seed=42)
        self.relu = ReLU()
        self.linear2 = Linear(16, 1, seed=43)

    def forward(self, x: Node) -> Node:
        return self.linear2(self.relu(self.linear1(x)))


rng = np.random.default_rng(7)
X_train = rng.uniform(0, 1, size=(500, 1))
y_true = X_train**2 + 1 + rng.normal(scale=0.03, size=X_train.shape)

model = MyModel()
model.train(
    x_train=X_train,
    y_true=y_true,
    loss=Mse(),
    optimizer_key="sgd",
    num_epochs=2000,
    batch_size=40,
    learning_rate=0.04,
    shuffle=True,
)

X_test = np.linspace(0, 1, 100).reshape(-1, 1)
y_pred = model.predict(X_test)[0]
```

That's the whole thing. Subclass `Model`, define your layers in `__init__`, wire them together in `forward`, call `.train(...)`.

## Examples

Once you install with the `examples` extra, you get a CLI demo:

```bash
uv run nn-regression-1d
```

That runs a 1D regression sweep across a few hidden layer sizes (4, 8, 32, 256) and plots them side by side so you can see how capacity affects the fit. Source lives at [`src/neural_net/examples/regression_1d.py`](src/neural_net/examples/regression_1d.py) if you want to poke at it.

## Core concepts

A quick tour of the building blocks:

- **`Node`** — a NumPy array with autograd metadata attached (who created it, what its parents are, its accumulated gradient, whether it needs one). The autograd graph is a graph of Nodes.
- **`Layer`** — a stateless-ish transformation with a `forward()` and a `backward()`. Calling one builds a new Node and hooks it into the graph. `Linear` and `Add` live here.
- **`Activation`** — same as Layer, just semantically for non-parametric nonlinearities. `ReLU` lives here.
- **`LossFunction`** — takes `(y_pred, y_true)`, returns a scalar Node you can call `.backward()` on. `Mse` is included.
- **`Optimizer`** — walks the model's parameters and applies updates. `Sgd` is included; new optimizers auto-register themselves via `__init_subclass__`.
- **`Model`** — subclass this, drop your layers into `__init__`, wire them in `forward()`. Get `parameters()`, `predict()`, and a full `train()` loop for free.

The flow when you train:

1. Forward pass through your `Model` builds a computation graph of `Node`s.
2. The loss node sits at the root of that graph.
3. `.backward()` on the root does a topological sort and walks backwards, accumulating gradients on every Node with `requires_grad=True`.
4. The optimizer applies those gradients to your parameters.
5. Repeat.

That's it! No magic layers, no framework internals hiding anything. Read the source in [`src/neural_net/`](src/neural_net/) and you can trace every step.

## License

MIT. See [LICENSE](LICENSE).
