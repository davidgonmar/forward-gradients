# Training DNNs with forward mode auto-differentiation

This repository explores different ideas to train neural networks using forward mode automatic differentiation (instead of the usual reverse mode autodiff). It is just a playground for me to try stuff, don't expect it to be polished or work well.

Currently, two basic approaches are implemented.

### JVP-based coordinate descent
This method uses the Jacobian-vector product (JVP, equal to the directional derivative with direction $v$) to compute gradients and updates the parameters only at a reduced set of coordinates.
Particularly, it randomly selects some parameters of the network, and computes the partial derivatives (gradient) only w.r.t. those parameters. For a single coordinate, it be formalized as:

$$
\begin{aligned}
i &\sim \mathcal{U}(1,\ldots,d) \\
v &\leftarrow \mathrm{onehot}(i,d) \\
\Delta &\leftarrow \text{JVP}(f, x, v)
      = \frac{\partial f(x)}{\partial x}^T v
      = \frac{\partial f(x)}{\partial x_i} \\
x_i &\leftarrow x_i - \eta\, \Delta
\end{aligned}
$$

This is usually done with a batch of data and several parameters at the same time.

### Unbiased gradient estimator through JVP
This method uses the JVP to compute an unbiased gradient estimator. It can be formalized as:

$$
\begin{aligned}
v_1,\ldots,v_n &\sim P \\
\Delta_1,\ldots,\Delta_n &\leftarrow \text{JVP}(f,x,v_1),\ldots,\text{JVP}(f,x,v_n) \\
\hat{\nabla} f(x) &\leftarrow \frac{1}{n}\sum_{i=1}^{n} \Delta_i \cdot v_i
\end{aligned}
$$

For the unbiasedness to hold, the only requirement is that $\mathbb{E}_{v\sim P}[v \cdot v^T] = I$. Note that the $v_i$ need not be i.i.d. samples, they can be interdependent. We can prove it as follows:

$$
\mathbb{E}_{v\sim P}[\text{JVP}(f,x,v) \cdot v] = \mathbb{E}_{v\sim P}\left[\left(\frac{\partial f(x)}{\partial x}^T v\right) \cdot v\right] = \mathbb{E}_{v\sim P}[v v^T] \frac{\partial f(x)}{\partial x} = I \frac{\partial f(x)}{\partial x} = \frac{\partial f(x)}{\partial x}
$$

I also tried a vector that orthonormalizes the $v$'s (but may break the condition for unbiasedness). Also tried to use the sign of the estimated gradients instead of the sign and magnitude.

### Requirements
JAX and Flax are used. Torch is difficult to work with when using forward mode autodiff. The exact versions used are in `requirements.txt`. To run transformer experiments, the additional requirements are in `requirements-transformers.txt`. Python 3.11 is used. The code will probably work with other versions.

### Extra
This is just for me having fun experimenting, it is not optimized nor tried extensively. Right now, the unbiased gradient estimator seems to work for small MLPs. The coordinate descent optimizer does work on a decent CNN but is not close to being competitive with regular optimization, albeit being a lot slower. Tried to train a transformer, does not work well at all.

### License
This code is released under the MIT license. See [LICENSE](LICENSE) for details.
