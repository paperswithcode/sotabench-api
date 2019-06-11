# sotabench

Easily benchmark PyTorch models on selected tasks and datasets.
## Example Usage

```python
from sotabench.vision.image_classification.cifar10 import evaluate_cifar10

# model = ... (returns a nn.Module object)

evaluate_cifar10(model=model)
# returns a dictionary with evaluation information
```
