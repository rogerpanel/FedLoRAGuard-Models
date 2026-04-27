# Algorithms

Reference algorithms implemented in this repository, traceable to the
manuscript.

## Algorithm 1 — FedLoRAGuard Training Round

```
input : clients {C_i}_{i=1}^{N}, global params w_t, clip S, noise sigma,
        learning rate eta, root set R_0, RDP accountant A, sampling rate q
output: w_{t+1}, cumulative budget eps_t, certified radius k*

server samples S_t ⊆ {C_i} of size m = ceil(q * N)
server computes reference gradient g_0 = grad_w  L(w_t; R_0)        # FLTrust
for each C_i in S_t in parallel:
    sample minibatch from local subgraph G_i(t)
    g_i  = grad_w  L_i(w_t; G_i(t))                                 # local
    g_i  = g_i * min(1, S / ||g_i||_2)                              # clip
    g_i~ = g_i + N(0, sigma^2 S^2 I)                                # noise
    upload via SecAgg
server reconstructs sum of g_i~                                     # SecAgg
server computes tau_i = ReLU(cos(g_i~, g_0)) for i in S_t           # FLTrust
g_t  = sum tau_i g_i~ / sum tau_i                                   # weighted
w_{t+1} = w_t - eta * g_t                                           # apply
eps_t = A.compose(t, q, sigma)                                      # PRV
k*    = floor(N * (Phi^-1(p1) - Phi^-1(p2)) / (2 e^{eps_t}))        # Thm. 2
return (w_{t+1}, eps_t, k*)
```

Implementation: `fedloraguard.federated.runtime.run_federated_training`.

## Theorem 1 — Gradient Sensitivity Bound

```
Delta_2 <= 2 S (rho * B_W * d_max)^L * sqrt(|R|) / |D_i|
```

Implementation: `fedloraguard.privacy.sensitivity.gradient_sensitivity_bound`.

## Theorem 2 — Certified Poisoning Radius via DP Composition

```
k* = floor( N * (Phi^-1(p_hat_1) - Phi^-1(p_hat_2)) / (2 * exp(epsilon_T)) )
```

Implementation: `fedloraguard.privacy.certified_radius.certified_poisoning_radius`.

## Multimodal node-feature encoder (Eq. 3)

```
phi(v_theta) = W_fuse * [ phi^wt ; phi^txt ; phi^beh ]
```

Implementation: `fedloraguard.encoders.multimodal.MultimodalEncoder`.

## HGT relation-aware attention (Eq. 4)

```
alpha_{u,v,r}
    = softmax( (W_Q^{tau(v)} h_v)^T  W_R^r  (W_K^{tau(u)} h_u) / sqrt(d/H) )
```

Implementation: `fedloraguard.models.hgt_attention.HGTRelationAwareAttention`.

## DyG-Mamba Lipschitz-constrained alternative

A spectral-norm-bounded SSM block whose Lipschitz constant <= 1, plugged into
the verifier in place of DyGFormer for the *DyG-Mamba* ablation row.

Implementation: `fedloraguard.models.dyg_mamba.DyGMambaEncoder`.

## RL-based active investigation

DQN with experience replay and a six-action fuzzing space (scan_spectrum,
inject_random_trigger, merge_with_benign, differential_test, flag_malicious,
clear_suspicion). Cited in the paper as our prior ElCon 2025 detector.

Implementation: `fedloraguard.investigation.dqn_investigator.DQNInvestigator`.
