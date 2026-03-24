# RNN Eigenspace Explorer

An interactive tool for **computational neuroscience** students to build intuition for how the eigenvalues and eigenvectors of a weight matrix govern the dynamics of a small recurrent neural network — in both discrete and continuous time.

**[Live demo →](https://MRIO.github.io/rnn-eigenspace-explorer/)**

---

## What is this?

A 3-neuron recurrent network is small enough that we can compute its eigendecomposition analytically and watch the results in real time. Every slider you move — gain, leak, update fraction — changes the eigenvalues, and the dynamics panel instantly shows what that means for how activity evolves. This bridges the gap between the abstract linear algebra and the felt experience of a network converging, oscillating, or diverging.

---

## The network

Three rate-coded neurons, indexed $i \in \{1, 2, 3\}$, connected by a $3 \times 3$ weight matrix $W$. Entry $W_{ij}$ is the synaptic weight from neuron $j$ to neuron $i$. The matrix is fully configurable in the UI: each cell cycles through $0$, $+1$, $-1$ on click.

The **default connectivity** is a forward ring with self-excitation:

$$W = \begin{pmatrix} 0 & 0 & 1 \\ 1 & 0 & 0 \\ 0 & 1 & 0 \end{pmatrix}$$

This means N1 receives from N3, N2 from N1, N3 from N2 — activity rotates around the ring. The eigenvalues of this $W$ are $\{1,\, \omega,\, \omega^*\}$ where $\omega = e^{2\pi i/3}$, a complex conjugate pair with argument $\pm 120°$.

---

## Two dynamical regimes

The tool supports two formulations of RNN dynamics, selectable via the **Discrete / Continuous** toggle.

---

### Discrete-time mode

$$\mathbf{r}[t+1] = (1 - \lambda)\,\mathbf{r}[t] + \lambda\,\varphi\!\left(g\,W\,\mathbf{r}[t]\right)$$

where:

| Symbol | Meaning |
|--------|---------|
| $\mathbf{r}[t] \in \mathbb{R}^3$ | firing rate vector at step $t$ |
| $g > 0$ | global gain |
| $\lambda \in (0, 1]$ | update fraction (leak) |
| $\varphi$ | pointwise nonlinearity ($\tanh$ or identity) |

**What $\lambda$ controls.** At $\lambda = 1$ the state is fully replaced each step: $\mathbf{r}[t+1] = \varphi(g W \mathbf{r}[t])$. With the forward ring and $g = 2$, this is a pure rotation — activity hops one neuron per step, giving a perfect period-3 limit cycle. As $\lambda \to 0$, more of the old state is retained; the update fraction acts as an exponential leak that slows and eventually kills the oscillation as the symmetric fixed point takes over.

#### Linearised stability

Linearising around the origin gives the update matrix:

$$M = (1 - \lambda)\,I + \lambda\,g\,W$$

Its eigenvalues $\mu_k$ determine stability:

$$\mu_k = (1 - \lambda) + \lambda\,g\,\nu_k$$

where $\nu_k$ are the eigenvalues of $W$. The criterion is simple:

$$|\mu_k| \begin{cases} > 1 & \text{unstable (grows)} \\ = 1 & \text{marginal (sustained oscillation)} \\ < 1 & \text{stable (decays to fixed point)} \end{cases}$$

For the forward ring, $\nu_{\text{complex}} = -\tfrac{1}{2} \pm \tfrac{\sqrt{3}}{2}i$, so:

$$|\mu_{\text{complex}}|^2 = \left(1 - \lambda - \tfrac{\lambda g}{2}\right)^2 + \tfrac{3}{4}(\lambda g)^2$$

At $\lambda = 1$: $|\mu_{\text{complex}}| = g$, so for any $g > 1$ these modes are unstable at the origin — the nonlinearity must bound the trajectory, creating the limit cycle.

---

### Continuous-time mode

$$\tau\,\dot{\mathbf{r}} = -\alpha\,\mathbf{r} + g\,\varphi(W\,\mathbf{r})$$

or equivalently:

$$\dot{\mathbf{r}} = \frac{1}{\tau}\left(-\alpha\,\mathbf{r} + g\,\varphi(W\,\mathbf{r})\right)$$

| Symbol | Meaning |
|--------|---------|
| $\alpha > 0$ | leak / passive decay rate |
| $\tau > 0$ | membrane time constant |

This is the standard **leaky rate model** from computational neuroscience (Wilson & Cowan 1972, Abbott 1994). Each neuron integrates its recurrent input and decays back to zero with time constant $\tau / \alpha$.

**What $\alpha$ controls.** A larger $\alpha$ means faster passive decay — the network forgets its state more quickly. At small $\alpha$ the network is near-lossless and oscillations persist. At large $\alpha$ they are heavily damped.

The **default continuous-mode connectivity** is the antisymmetric ring:

$$W_{\text{anti}} = \begin{pmatrix} 0 & -1 & +1 \\ +1 & 0 & -1 \\ -1 & +1 & 0 \end{pmatrix}$$

This has purely imaginary eigenvalues $\nu_k \in \{0,\, \pm i\sqrt{3}\}$, which is key: with purely imaginary $\nu$ the oscillatory content is encoded in $W$ itself, and $\alpha$ purely controls the decay envelope.

#### Linearised stability

The Jacobian at the origin is:

$$J = \frac{1}{\tau}\left(-\alpha\,I + g\,W\right)$$

Its eigenvalues are:

$$\lambda_k = \frac{-\alpha + g\,\nu_k}{\tau}$$

The stability criterion is:

$$\text{Re}(\lambda_k) \begin{cases} > 0 & \text{unstable} \\ = 0 & \text{marginal} \\ < 0 & \text{stable (decays)} \end{cases}$$

For the antisymmetric ring with $\nu_{\text{complex}} = \pm i\sqrt{3}$:

$$\text{Re}(\lambda_{\text{complex}}) = \frac{-\alpha}{\tau} < 0 \quad \text{always}$$

So the antisymmetric ring always produces **damped oscillations** — the oscillatory frequency is set by $g$ and $\nu$, while $\alpha/\tau$ sets the decay rate. This cleanly separates the two concepts.

The oscillation frequency (imaginary part) and decay rate (real part) are:

$$f = \frac{1}{2\pi} \cdot \frac{g\sqrt{3}}{\tau}, \qquad \text{decay rate} = \frac{\alpha}{\tau}$$

---

## The nonlinearity

The toggle switches between:

- **tanh:** $\varphi(x) = \tanh(x)$, bounded output $\in (-1, 1)$. The standard nonlinearity in computational neuroscience models (Sompolinsky, Crisanti & Sommers 1988). Its derivative at zero is 1, so the linearisation at the origin is exact. Away from zero, $|\varphi'(x)| < 1$ — the effective gain decreases, which is what bounds growing modes into limit cycles.

- **linear:** $\varphi(x) = x$, i.e. no nonlinearity. In this mode the system is purely linear: eigenvalues fully determine everything and activity either grows without bound, decays, or oscillates forever at fixed amplitude. Good for studying the eigenvalue picture in isolation.

---

## The eigenspace viewport

The 3D panel shows the **eigenvectors** of the effective linearised matrix — either $M$ (discrete) or $J$ (continuous) — as arrows in the 3-dimensional rate space $\mathbf{r} \in \mathbb{R}^3$. The current state vector $\mathbf{r}(t)$ is shown as a white arrow, normalised to unit length for visibility.

**Why eigenvectors matter.** Any initial condition $\mathbf{r}_0$ can be decomposed into eigenvector components. Each component evolves independently:

$$\mathbf{r}(t) = \sum_k c_k\,e^{\lambda_k t}\,\mathbf{v}_k \quad \text{(continuous)}$$
$$\mathbf{r}[t] = \sum_k c_k\,\mu_k^t\,\mathbf{v}_k \quad \text{(discrete)}$$

where $c_k = \langle \mathbf{v}_k^*, \mathbf{r}_0 \rangle$ is the projection of the initial condition onto eigenvector $k$. The component along the most unstable eigenvector dominates at long times.

In discrete mode, a **unit circle** is drawn in the 3D view as a reference — eigenvectors whose corresponding $|\mu_k| > 1$ point "outside" the stability boundary.

---

## The initial condition $[1, 0, 0]$

The default IC places all activity in N1 and none in N2 or N3. For the forward ring, this is a **maximally asymmetric** state: it has large projections onto the complex eigenvectors $\mathbf{v}_2, \mathbf{v}_3$ (the oscillatory modes), which is why you immediately see sequential 120° activation. If you started at $[1, 1, 1]$ instead, you would project entirely onto the symmetric eigenvector $\mathbf{v}_1$ and see no rotation at all.

---

## Key intuitions to build

| Experiment | What to observe |
|---|---|
| Discrete, $\lambda = 1$, forward ring | Perfect period-3 rotation: $|{\mu_{\text{complex}}}| = g > 1$ |
| Reduce $\lambda$ below 1 | Oscillation damps as $|\mu|$ shrinks toward the unit circle |
| Linear mode, gain sweep | Watch eigenvalues cross $|\mu|=1$ (discrete) or $\text{Re}(\lambda)=0$ (continuous) |
| Continuous, sweep $\alpha$ | Decay rate changes; frequency unchanged — $\alpha$ and $g$ are separable |
| Add self-connections ($W_{ii} = 1$) | Real eigenvalue shifts right; symmetric mode destabilises first |
| Add backward inhibition ($W_{ij} = -1$) | Purely imaginary spectrum; rotation without fixed-point competition |
| Randomise IC | Different eigenmode projections $c_k$ — watch which component dominates |

---

## Implementation notes

**Discrete step** uses the leaky update rule directly, one step per animation frame (speed-controlled).

**Continuous step** uses a 4th-order Runge-Kutta integrator with adaptive sub-stepping: the internal step size is capped at $\tau/4$ to maintain stability regardless of how small $\tau$ is set.

**Eigendecomposition** is computed analytically via the characteristic polynomial of the $3 \times 3$ effective matrix. The cubic is solved by the trigonometric method (three real roots) or Cardano's formula (one real + complex conjugate pair), giving exact eigenvalues with no numerical iteration.

---

## References

- Wilson, H.R. & Cowan, J.D. (1972). Excitatory and inhibitory interactions in localized populations of model neurons. *Biophysical Journal*, 12(1), 1–24.
- Abbott, L.F. (1994). Decoding neuronal firing and modelling neural networks. *Quarterly Reviews of Biophysics*, 27(3), 291–331.
- Sompolinsky, H., Crisanti, A. & Sommers, H.J. (1988). Chaos in random neural networks. *Physical Review Letters*, 61(3), 259.
- Strogatz, S.H. (2015). *Nonlinear Dynamics and Chaos*. Westview Press. (Chapters 7–8 for Hopf bifurcations.)
- Dayan, P. & Abbott, L.F. (2001). *Theoretical Neuroscience*. MIT Press. (Chapter 7 for network models.)

---

## Running locally

No build step. Open `index.html` in any modern browser.

```bash
git clone https://github.com/MRIO/rnn-eigenspace-explorer.git
cd rnn-eigenspace-explorer
open index.html   # macOS
# or: xdg-open index.html  (Linux)
# or just drag the file into your browser
```
