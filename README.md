# RNN-Based Channel Estimation for Time-Varying Wireless Channels

This project implements Recurrent Neural Network (RNN) based channel estimation for a time-varying FIR wireless communication channel using MATLAB.

The work compares data-driven channel estimation using RNNs against the classical Kalman filter estimator under dynamic fading conditions.

---

## Overview

The wireless channel is modeled as a two-tap time-varying FIR filter whose coefficients evolve according to a Gauss–Markov process.

The received signal model is:

```math
x[n] = v^T[n]h[n] + w[n]
```

where:

- `v[n]` is the transmitted signal vector
- `h[n]` is the unknown channel coefficient vector
- `w[n]` is additive white Gaussian noise (AWGN)

The channel evolution follows:

```math
h[n] = Ah[n-1] + u[n]
```

Instead of explicitly modeling the channel statistics, the RNN learns the temporal channel dynamics directly from training data.

---


## RNN Formulation

The RNN treats channel estimation as a sequential learning problem.

At each time step, the network input is:

```math
z[n] = [x[n], v[n], v[n-1]]
```

The hidden state evolves recursively as:

```math
s[n] = f_\theta(s[n-1], z[n])
```

The estimated channel coefficients are produced through:

```math
\hat{h}[n] = g_\theta(s[n])
```

The network parameters are trained by minimizing the mean-square error loss:

```math
L =
\frac{1}{N}
\sum_{n=1}^{N}
\|h[n]-\hat{h}[n]\|^2
```

---

## Neural Network Architecture

### LSTM-Based Estimator

```matlab
layers = [
    sequenceInputLayer(3)
    lstmLayer(15,'OutputMode','sequence')
    fullyConnectedLayer(2)
    regressionLayer];
```

### GRU-Based Estimator

```matlab
layers = [
    sequenceInputLayer(3)
    gruLayer(15,'OutputMode','sequence')
    fullyConnectedLayer(2)
    regressionLayer];
```

---

## Simulations Performed

### Channel Tracking

The RNN estimates are compared against:

- True channel coefficients
- Kalman filter estimates

The simulations evaluate:

- Tracking accuracy
- Initial convergence behavior
- Steady-state estimation performance

### MSE Analysis

The following metrics are evaluated:

- Initial MSE
- Steady-state MSE
- Average MSE
- Convergence time

---

## Key Observations

### Initial Transient Performance

- Kalman filter starts with large estimation error due to zero initialization
- RNN estimators achieve significantly lower initial MSE
- GRU generally converges faster than LSTM

### Steady-State Performance

- Kalman filter achieves the lowest steady-state MSE when the channel model is perfectly known
- RNN estimators remain near-optimal
- RNN outputs appear smoother due to learned temporal averaging

### Impact of Training Dataset Size

Experiments were performed with:

- 100 training sequences
- 300 training sequences
- 1000 training sequences

Increasing the dataset size mainly improves:

- Initial transient estimation
- Robustness
- Smoother convergence

Steady-state performance changes only slightly after sufficient training.

### Optimizer Comparison

The following optimizers were evaluated:

- Adam
- RMSProp
- SGDM

Observations:

- RMSProp provides stable and fast convergence
- SGDM converges extremely quickly but may degrade steady-state performance
- LSTM is generally more stable under aggressive optimization
- GRU achieves lower initial MSE in several experiments

---

## Kalman Filter Baseline

The classical Kalman filter is implemented for comparison using the state-space model:

### Prediction Step

```math
\hat{h}[n|n-1] = A\hat{h}[n-1|n-1]
```

### Update Step

```math
\hat{h}[n|n] =
\hat{h}[n|n-1]
+ K[n](x[n]-v^T[n]\hat{h}[n|n-1])
```

The Kalman filter provides the MMSE-optimal solution when channel statistics are perfectly known.

---

## Main Findings

- RNNs outperform Kalman filtering during the initial transient phase
- Kalman filtering achieves superior steady-state MMSE
- GRU models provide faster convergence
- LSTM models provide improved robustness and stability
- RNN estimators are effective when channel statistics are unknown or difficult to model

---

## Files

- `rnn...m` — MATLAB implementation
- `RNN_Report.pdf` — Full derivations, experiments, plots, and analysis :contentReference[oaicite:0]{index=0}

---

## References

1. S. M. Kay, *Fundamentals of Statistical Signal Processing: Estimation Theory*, Prentice Hall, 1993.

2. K. P. Murphy, *Machine Learning: A Probabilistic Perspective*, MIT Press, 2012.
