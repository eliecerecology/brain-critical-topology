import argparse
import numpy as np
import torch
import os
from numpy.polynomial.polynomial import polyfit, polyval
from typing import List
import tqdm


# ============================================================
#  KURAMOTO MODEL
# ============================================================

class KuramotoFast:
    def __init__(self, n_nodes, n_oscillators, sampling_rate, k_list,
             weight_matrix, frequency_spread, noise_scale=1.0,
             use_cuda=True, use_tqdm=True, node_frequencies=None,
             L=5.0, **kwargs):  # ADD L parameter
        """
        Nested Kuramoto model using PyTorch.
        N nodes each with M oscillators.
        Oscillator frequencies assigned gaussian
        """
        self._check_parameters(n_nodes, k_list, weight_matrix)

        if use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
            device_name = "GPU (CUDA/ROCm)"
        else:
            self.device = torch.device("cpu")
            device_name = "CPU"
        print(f"Using device: {device_name}")

        self.n_nodes = n_nodes
        self.n_oscillators = n_oscillators
        self.k_list = k_list
        self.noise_scale = 2 * np.pi * noise_scale / sampling_rate
        self.frequency_spread = frequency_spread
        self.node_frequencies = node_frequencies

        self.weight_matrix = torch.tensor(weight_matrix, dtype=torch.float32, device=self.device)
        torch.diagonal(self.weight_matrix).fill_(0)
        self.weight_matrix = self.weight_matrix.T * L # ADDED for L

        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate
        self.use_cuda = use_cuda
        self.disable_tqdm = not(use_tqdm)

        self._init_parameters_gaussian() 
        self._preallocate()

    def _check_parameters(self, n_nodes, k_list, weight_matrix):
        if len(k_list) != n_nodes:
            raise RuntimeError(f'Size of k_list ({len(k_list)}) != n_nodes ({n_nodes}).')
        if np.ndim(weight_matrix) != 2 or weight_matrix.shape[0] != weight_matrix.shape[1]:
            raise RuntimeError(f'weight_matrix must be 2d square, got {weight_matrix.shape}.')
        if weight_matrix.shape[0] != n_nodes:
            raise RuntimeError(f'weight_matrix must be N_nodes x N_nodes, got {weight_matrix.shape}.')

    def _init_parameters_gaussian(self):
        # GAUSSIAN frequency assignment — matches Palva, avoids beating artifact
        omegas = torch.zeros((self.n_nodes, self.n_oscillators), dtype=torch.float32, device=self.device)
        for idx, frequency in enumerate(self.node_frequencies):
            # CHANGED: torch.normal instead of torch.linspace
            omegas[idx] = torch.normal(
                mean=float(frequency),
                std=self.frequency_spread,
                size=(self.n_oscillators,),
                device=self.device,
                dtype=torch.float32
            )

        # NO jitter needed — Gaussian already has natural spread
        # CHANGED: removed the ±0.1 jitter line
        self.omegas = omegas * 2 * np.pi  # convert to rad/s

        C = torch.tensor(self.k_list, dtype=torch.float32, device=self.device) / self.n_oscillators
        self.shift_coeffs = C.view(-1, 1)

        thetas = torch.rand(omegas.shape, device=self.device, dtype=torch.float32) * 2 * np.pi - np.pi
        self.phases = torch.exp(1j * thetas).to(torch.complex64)

        self._complex_dtype = torch.complex64
        self._float_dtype = torch.float32

    def _preallocate(self):
        n_nodes, n_osc = self.phases.shape
        self._phase_conj = torch.empty_like(self.phases)
        self._external_buffer = torch.empty((n_nodes, n_nodes, n_osc),
                                            dtype=self.phases.dtype, device=self.device)

    def _compute_rhs(self, phases):
        mean_phase = torch.mean(phases, dim=1)
        self._phase_conj = torch.conj(phases)

        self._external_buffer = torch.tensordot(self._phase_conj, mean_phase, dims=0).permute(0, 2, 1)
        weight_expanded = self.weight_matrix[:, :, None].expand(-1, -1, self.n_oscillators)
        self._external_buffer *= weight_expanded
        external = self._external_buffer.sum(dim=1)
        external_rhs = external.imag / self.n_nodes

        self._phase_conj = phases * torch.sum(self._phase_conj, dim=1, keepdim=True)
        self._phase_conj = torch.conj(self._phase_conj)
        internal_rhs = self._phase_conj.imag * self.shift_coeffs

        rhs = self.omegas + internal_rhs + external_rhs
        return rhs

    def _rotate(self, dtheta):
        return torch.polar(torch.ones_like(dtheta), dtheta)

    def simulate(self, time: float, noise_realisations: int=100, random_seed: int=42) -> np.ndarray:
        torch.manual_seed(random_seed)
        n_iters = int(time * self.sampling_rate)
        history = torch.zeros((self.n_nodes, n_iters + 1), dtype=self._complex_dtype, device=self.device)
        history[:, 0] = self.phases.mean(dim=1)
        for i in tqdm.trange(1, n_iters + 1, leave=False, desc='Kuramoto model is running...', disable=self.disable_tqdm):
            k1 = self._compute_rhs(self.phases)
            phases2 = self.phases * self._rotate((self.dt / 2) * k1)
            k2 = self._compute_rhs(phases2)
            phases3 = self.phases * self._rotate((self.dt / 2) * k2)
            k3 = self._compute_rhs(phases3)
            phases4 = self.phases * self._rotate(self.dt * k3)
            k4 = self._compute_rhs(phases4)
            rhs = (k1 + 2 * k2 + 2 * k3 + k4) / 6
            shift_noise = torch.normal(mean=0.0, std=self.noise_scale,
                                       size=rhs.shape, device=self.device, dtype=torch.float32)
            rhs += shift_noise
            self.phases = self.phases * self._rotate(self.dt * rhs)
            history[:, i] = self.phases.mean(dim=1)
        history = history.cpu().numpy()
        return history


# ============================================================
#  METRIC FUNCTIONS
# ============================================================
def calc_detrened(data):
    x = np.abs(data)
    y = np.cumsum(x - np.mean(x))
    return y


def dfa_rms(y, scale):
    n_windows = len(y) // scale
    if n_windows == 0:
        return np.nan
    shape = (n_windows, scale)
    Y = np.lib.stride_tricks.as_strided(y, shape=shape)
    rms = np.zeros(n_windows)
    scale_axis = np.arange(scale)
    for i, window in enumerate(Y):
        coeff = np.polyfit(scale_axis, window, 1)
        trend = np.polyval(coeff, scale_axis)
        rms[i] = np.sqrt(np.mean((window - trend) ** 2))
    return np.mean(rms)


def dfa_scales(min_exp=8, max_exp=10, step=0.25):
    scales = np.round(2 ** np.arange(min_exp, max_exp, step)).astype(int)
    scales = np.unique(scales)
    return scales


def DFA(data):
    """DFA on amplitude envelope — peaks at criticality."""
    y = calc_detrened(data)
    scales = dfa_scales()
    F = []
    for s in scales:
        rms_val = dfa_rms(y, s)
        F.append(rms_val if not np.isnan(rms_val) else np.nan)
    F = np.array(F)
    mask = ~np.isnan(F)
    scales = scales[mask]
    F = F[mask]
    coeff = np.polyfit(np.log2(scales), np.log2(F), 1)
    alpha = coeff[0]
    return alpha, scales, F

def FA_metric(phasor, scales):
    y = calc_detrened(phasor)
    F_fa = np.zeros(len(scales))
    for i, s in enumerate(scales):
        diffs = y[s:] - y[:-s]
        F_fa[i] = np.sqrt(np.mean(diffs**2))
    coeff_fa = np.polyfit(np.log2(scales), np.log2(F_fa), 1)
    alpha_fa = coeff_fa[0]
    fit_fa = 2 ** np.polyval(coeff_fa, np.log2(scales))
    print(f"Estimated FA exponent alpha = {alpha_fa:.3f}")
    return fit_fa, alpha_fa


def plv_matrix_vectorized(inst_theta):
    """Raw phasor PLV — v1 style."""
    X = np.exp(1j * inst_theta)   # (T, N)
    M = np.dot(X.conj().T, X) / X.shape[0]
    return np.abs(M)


# ============================================================
#  SIMULATION BATCH
# ============================================================

def run_simulation_batch(i_start, i_end, all_adj, k_values, n_nodes, n_oscillators,
                          sampling_rate, frequency_spread, node_frequencies,
                          use_cuda, sim_time, network_names, output_dir, num_replicas):
    """
    Frequency distribution : Gaussian (std=1.0 Hz)
    DFA                    : amplitude envelope, scales 2^8 to 2^10
    PLV                    : raw phasor angle
    Transient removed      : first 10 seconds = 2000 samples = 100 oscillations
    Analysis window        : last 40 seconds  = 8000 samples = 400 oscillations
    """
    n_batch = i_end - i_start
    n_k = len(k_values)

    order_matrix       = np.zeros((n_batch, num_replicas, n_k))
    variability_matrix = np.zeros((n_batch, num_replicas, n_k))
    plv_matrix         = np.zeros((n_batch, num_replicas, n_k))
    dfa_matrix         = np.zeros((n_batch, num_replicas, n_k))
    fa_matrix          = np.zeros((n_batch, num_replicas, n_k))

    # Transient: first 10 seconds = 2000 samples = 100 oscillations
    transient_samples = int(10 * sampling_rate)
    print(f"Removing {transient_samples} samples as transient "
          f"({transient_samples/sampling_rate:.0f} seconds, "
          f"{transient_samples/sampling_rate * node_frequencies[0]:.0f} oscillations)")

    for local_i, i in enumerate(range(i_start, i_end)):
        print(f"\nNetwork {i}: {network_names[i]}")
        for r in range(num_replicas):
            print(f"  replica {r + 1}/{num_replicas}")
            W = all_adj[i, r]
            mean_weight = W[W > 0].mean()  # mean of non-zero weights
            W_norm = W / mean_weight       # normalize

            for k_idx, k in enumerate(k_values):
                random_seed = int(np.random.randint(0, 1_000_000))

                model = KuramotoFast(
                    n_nodes=n_nodes,
                    n_oscillators=n_oscillators,
                    k_list=[k] * n_nodes,
                    weight_matrix=W_norm, ## added for L
                    node_frequencies=node_frequencies,
                    sampling_rate=sampling_rate,
                    frequency_spread=frequency_spread,
                    use_cuda=use_cuda,
                    L=5.0,  # ADD THIS        
                )
                phase_data = model.simulate(time=sim_time, random_seed=random_seed)
                del model
                if use_cuda:
                    torch.cuda.empty_cache()

                # Remove transient
                phase_data = phase_data[:, transient_samples:]

                # --- Order parameter ---
                order_ts   = np.abs(np.mean(phase_data, axis=0))
                order_mean = order_ts.mean()
                order_std  = order_ts.std()

                # --- PLV (raw phasor angle — v1 style) ---
                inst_theta = np.angle(phase_data).T
                plv_mat    = plv_matrix_vectorized(inst_theta)
                np.fill_diagonal(plv_mat, 0)
                plv_order  = plv_mat[np.triu_indices_from(plv_mat, k=1)].mean()

                # --- DFA and FA on amplitude envelope ---
                scales        = dfa_scales()
                global_phasor = np.abs(phase_data).mean(axis=0)
                alpha         = DFA(global_phasor) ### REPLACES
                fa            = FA_metric(global_phasor, scales)

                # --- Store ---
                order_matrix[local_i, r, k_idx]       = order_mean
                variability_matrix[local_i, r, k_idx] = order_std
                plv_matrix[local_i, r, k_idx]         = plv_order
                dfa_matrix[local_i, r, k_idx]         = alpha[0]
                fa_matrix[local_i, r, k_idx]          = fa[1]

    results = {
        "order_matrix":       order_matrix,
        "variability_matrix": variability_matrix,
        "plv_matrix":         plv_matrix,
        "dfa_matrix":         dfa_matrix,
        "fa_matrix":          fa_matrix,
        "k_values":           k_values,
        "i_start":            i_start,
        "i_end":              i_end,
        "network_names":      np.array(network_names[i_start:i_end], dtype=object),
    }

    output_path = os.path.join(output_dir, f"results_final_linspace_{i_start}_to_{i_end}_final_forever.npz")
    np.savez(output_path, **results)
    print(f"\nResults saved to {output_path}")
    return results


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    ## ADDING ARGUMENTS FOR THE SLURM ARRAY JOB
    parser = argparse.ArgumentParser(description="Kuramoto simulation batch (gaussian frequencies)")
    parser.add_argument("--i-start", type=int, default=0, help="Start network index (inclusive)")
    parser.add_argument("--i-end", type=int, default=1, help="End network index (exclusive)")
    args = parser.parse_args()

    print("CUDA:", torch.cuda.is_available())

    k_values = np.concatenate([
        np.linspace(1, 8, 10),    # fine — pre-transition
        np.linspace(8, 30, 10),   # fine — transition region
        np.linspace(30, 60, 7),   # coarse — early supercritical
    ])  # total = 35 k values

    n_nodes          = 1000 # BIG CHANGE
    n_oscillators    = 300          # compromise between v1 (100) and Palva (500)
    sampling_rate    = 200
    frequency_spread = 1.0  # Gaussian std=1.0 Hz around 10 Hz center frequency
    node_frequencies = [10.0] * n_nodes
    use_cuda         = torch.cuda.is_available()
    # analysis  = last 40 sec  = 8000 samples = 400 oscillations
    # transient = first 10 sec = 2000 samples = 100 oscillations
    # analysis  = last 20 sec  = 4000 samples = 200 oscillations
    sim_time         = 50.0  # seconds
    output_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Results will be saved to: {output_dir}")

    network_names = [
        "LL", "LH", "HL", "HH",
        "Watts-Strogatz Low mean", "Watts-Strogatz High mean",
        "Erdos-Renyi Low mean", "Erdos-Renyi High mean",
        "Fully Connected",
        "Barabasi-Albert Low mean", "Barabasi-Albert High mean",
    ]

    # Load adjacency matrices
    data    = np.load(os.path.join(output_dir, "adjacency_matrices.npz"))
    all_adj = data["adjacency_matrices"]
    num_networks, num_replicas, N, _ = all_adj.shape
    assert num_networks == 11 and num_replicas == 3
    assert 0 <= args.i_start < args.i_end <= num_networks

    print(f"Running networks [{args.i_start}, {args.i_end})")

    run_simulation_batch(
        i_start=args.i_start, i_end=args.i_end,
        all_adj=all_adj,
        k_values=k_values,
        n_nodes=n_nodes,
        n_oscillators=n_oscillators,
        sampling_rate=sampling_rate,
        frequency_spread=frequency_spread,
        node_frequencies=node_frequencies,
        use_cuda=use_cuda,
        sim_time=sim_time,
        network_names=network_names,
        output_dir=output_dir,
        num_replicas=num_replicas,
    )
    print("All simulations complete.")
