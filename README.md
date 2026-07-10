# Brain Criticality & Network Topology

This repository holds the code, data, and figures for the master's thesis of
**Eliecer Diaz** (Aalto University). The project builds different network
topologies and runs large-scale oscillator simulations on them, searching for
signatures of **brain criticality** — the hypothesis that healthy brain
dynamics operate near a critical point between disorder and full synchrony.

## Idea in brief

Each network is a graph of `N = 1000` nodes. Every node is itself a *nested*
population of `300` phase oscillators, and the nodes are coupled through the
network's weighted adjacency matrix. The dynamics follow a **nested Kuramoto
model**, integrated with a 4th-order Runge–Kutta scheme and phase noise.

By sweeping the global coupling strength `k` across the synchronization
transition, we look for the point where the system sits at criticality —
identified through long-range temporal correlations (DFA) and other order
metrics.

## Network topologies

Simulations are run over 11 topologies (each with 3 replicas), covering both
control models and mean-degree variants (generated using Negative Binomial distribution):

- `LL` (Low mean and Low Variance), `LH` (Low mean and High), `HL`, `HH` — the four thesis conditions (low/high combinations)
- Watts–Strogatz (small-world), low and high mean
- Erdős–Rényi (random), low and high mean
- Barabási–Albert (scale-free), low and high mean
- Fully connected

The adjacency matrices are precomputed and stored in
[adjacency_matrices.npz](adjacency_matrices.npz) with shape
`(11 networks, 3 replicas, 1000, 1000)`.

## Metrics

For every network / replica / coupling value the simulation computes:

- **Order parameter** — mean and variability of the Kuramoto order parameter
- **PLV** — phase-locking value across node pairs (raw phasor)
- **DFA** — detrended fluctuation analysis exponent on the amplitude envelope
  (scales `2^8`–`2^10`); the DFA exponent peaks at criticality
- **FA** — fluctuation-analysis exponent

Simulation parameters: sampling rate `200 Hz`, center frequency `10 Hz` with a
Gaussian spread (std `1 Hz`), `50 s` total (first `10 s` discarded as
transient), coupling `k` swept over 35 values from 1 to 60.

## Running on Triton (Aalto HPC)

The simulations were run on **Triton**, Aalto University's high-performance
computing cluster, using GPU nodes. The job is a SLURM array — one array task
per network — submitted with [kuramoto_triton.sh](kuramoto_triton.sh):

```bash
sbatch kuramoto_triton.sh
```

Key resources requested per task: `gpu-v100-32g` partition, 1 GPU, 4 CPUs,
32 GB RAM, up to 60 h wall time. The model runs on CUDA via PyTorch when a GPU
is available and falls back to CPU otherwise.

## Repository layout

| File | Description |
| --- | --- |
| [kuramoto_and_metrics.py](kuramoto_and_metrics.py) | Nested Kuramoto model (PyTorch), metric functions, and the batch simulation entry point |
| [kuramoto_triton.sh](kuramoto_triton.sh) | SLURM array job script for Triton |
| [concatenation.py](concatenation.py) | Merges per-batch result `.npz` files into a single combined dataset |
| [adjacency_matrices.npz](adjacency_matrices.npz) | Precomputed adjacency matrices for all networks and replicas |
| [network_builder_final.ipynb](network_builder_final.ipynb) | Notebook for constructing the network topologies |
| [network_generationN400N.ipynb](network_generationN400N.ipynb) | Alternative network-generation notebook (N = 400) |
| [PlotterNetworks.ipynb](PlotterNetworks.ipynb) | Notebook for plotting networks and results |
| `plot_*.png` | Rendered figures of networks and Kuramoto dynamics |

## Reproducing locally

```bash
# Run a single network (index 0) without the cluster
python kuramoto_and_metrics.py --i-start 0 --i-end 1
```

Requires `numpy`, `torch`, and `tqdm`. Results are written as
`results_final_linspace_<i_start>_to_<i_end>_final_forever.npz` and can be
combined with `concatenation.py`.

## License

BSD 3-Clause — see [LICENSE](LICENSE).
