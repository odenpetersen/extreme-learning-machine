# Extreme Learning Machine
[Wikipedia Description](https://en.wikipedia.org/wiki/Extreme_learning_machine)

To predict $\mathbf y \vert \mathbf x$:
1. Make random matrix $\mathbf W_1$
2. Return the least squares estimate with feature vector $\sigma\left(\mathbf W_1 \mathbf x\right)$, where $\sigma$ is an activation function.

![Example Plot](https://github.com/odenpetersen/extreme-learning-machine/blob/main/plot.png?raw=true)

Turns out this has already been done [here](https://github.com/wdm0006/sklearn-extensions/tree/master/sklearn_extensions/extreme_learning_machines). That version seems more compliant with sklearn conventions, but doesn't seem to have a regularisation option. Regularisation is particularly interesting in that it allows for model overparameterisation (more features than training samples).
